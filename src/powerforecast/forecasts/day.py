"""Forecast one delivery day — the single implementation.

This module exists because there are now two callers that must never disagree:
the HTTP service, which answers a request, and the daily job, which runs before
the bid deadline and writes the result to disk. If each built its own design
matrix, the two would drift — and the drift would be invisible, because both
would keep returning numbers.

So the rule is: **anything that decides what a forecast is** lives here.
`serving/app.py` wraps this in HTTP, `jobs/daily_forecast.py` wraps it in a
schedule, and neither may reimplement it.

Three constraints run through the code, all of them consequences of the fact
that serving inverts the assumptions the backtest was built on:

1. *The target is unknown.* `features.build.usable_rows` requires the target to
   be present, which is right for training and wrong here — the target for a
   future delivery day is the thing being predicted. See `complete_rows`.
2. *The delivery day may not be in the panel.* Storage holds observed hours;
   tomorrow has none. See `extended_panel`.
3. *Partial is worse than nothing.* A day short one hour is a position nobody
   bid, so an incomplete day raises rather than returning what it has.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from powerforecast.features.availability import LOCAL_TZ, forecast_origins
from powerforecast.features.build import build_design_matrix
from powerforecast.models.persistence import SavedModel

# The columns every produced forecast carries, in order. The store, the service
# and the monitoring layer all agree on this tuple; changing it is a schema
# change and has to be made here.
FORECAST_COLUMNS = (
    "forecast_mwh",
    "model_name",
    "model_version",
    "forecast_origin",
    "generated_at",
)


class IncompleteForecastError(RuntimeError):
    """A delivery day cannot be forecast in full.

    Carries the diagnosis as data rather than only as a message, because the two
    callers render it differently: the service turns it into a JSON body, the job
    prints it and exits non-zero. Formatting it as prose here would force one of
    them to parse the other's sentences.
    """

    def __init__(
        self,
        delivery_date: date,
        *,
        expected: int,
        available: int,
        missing_features: list[str],
        data_available_until: pd.Timestamp | None,
    ) -> None:
        self.delivery_date = delivery_date
        self.expected = expected
        self.available = available
        self.missing_features = missing_features
        self.data_available_until = data_available_until
        super().__init__(
            f"cannot forecast {delivery_date.isoformat()}: "
            f"{available} of {expected} hours have a complete feature set"
        )

    def as_dict(self) -> dict[str, object]:
        """The diagnosis as data, for a JSON response or a log line."""
        return {
            "message": str(self),
            "missing_features": self.missing_features,
            "data_available_until": (
                self.data_available_until.isoformat()
                if self.data_available_until is not None
                else None
            ),
            "hint": (
                "Ingest the missing history and weather first: "
                "`uv run python -m powerforecast.data.backfill` and "
                "`uv run python -m powerforecast.data.backfill_weather`."
            ),
        }


@dataclass(frozen=True)
class DayForecast:
    """A full delivery day and the provenance of the numbers in it."""

    delivery_date: date
    forecast_origin: pd.Timestamp
    model_name: str
    model_version: str
    generated_at: datetime
    values: pd.Series

    def to_frame(self) -> pd.DataFrame:
        """The canonical record layout, ready for the store.

        The four provenance columns repeat identically down all 24 rows, and
        that repetition is deliberate. Three months from now the question about a
        stored forecast is "which model produced this, and what could it see?" —
        and the answer has to be in the row itself. A separate runs table would
        require a join, and a join that has to be remembered is a join that will
        not be made.
        """
        frame = pd.DataFrame(
            {
                "forecast_mwh": self.values.astype("float64"),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "forecast_origin": self.forecast_origin,
                "generated_at": pd.Timestamp(self.generated_at),
            },
            index=self.values.index,
        )
        frame.index.name = "timestamp"
        return frame[list(FORECAST_COLUMNS)]


def forecast_day(
    model: SavedModel,
    panel: pd.DataFrame,
    delivery_date: date,
    *,
    generated_at: datetime | None = None,
) -> DayForecast:
    """Predict every hour of one delivery day, or refuse.

    Args:
        model: A loaded model; its card fixes both the feature spec and the
            column order the estimator was fitted on.
        panel: Observed history. Extended internally to cover the delivery day.
        delivery_date: Local calendar date to forecast.
        generated_at: Injectable clock, so a test does not depend on when it runs.

    Raises:
        IncompleteForecastError: if any hour of the day lacks a complete feature
            set. Never returns a partial day.
    """
    card = model.card
    hours = delivery_hours(delivery_date)

    full = extended_panel(panel, through=hours[-1])
    features, _ = build_design_matrix(full, card.spec())

    # Reindex rather than filter. Filtering returns whatever happens to be
    # present; reindexing asks for exactly these 24 hours and leaves a visible
    # NaN wherever one is absent — the difference between a missing hour being
    # discovered and being noticed.
    day = features.reindex(hours)
    complete = complete_rows(day)

    if len(complete) < len(hours):
        observed = pd.DatetimeIndex(full.index)[full.notna().any(axis=1)]
        raise IncompleteForecastError(
            delivery_date,
            expected=len(hours),
            available=len(complete),
            missing_features=sorted(day.columns[day.isna().any()].tolist()),
            data_available_until=observed.max() if len(observed) else None,
        )

    return DayForecast(
        delivery_date=delivery_date,
        forecast_origin=pd.Timestamp(forecast_origins(hours).iloc[0]),
        model_name=card.name,
        model_version=card.version,
        generated_at=generated_at or datetime.now(UTC),
        values=model.predict(day.loc[complete]),
    )


def delivery_hours(delivery_date: date, *, tz: str = LOCAL_TZ) -> pd.DatetimeIndex:
    """Every hour of one local calendar day, as UTC timestamps.

    Not `range(24)`. A calendar day is 24 hours only when no daylight saving
    transition falls inside it; where one does, it is 23 or 25. Türkiye has been
    on permanent UTC+3 since 2016, so today the answer is always 24 — but this
    project has a European leg waiting on an ENTSO-E token, and the day the first
    German delivery day arrives, `range(24)` would silently drop or duplicate an
    hour twice a year.

    `DateOffset(days=1)` rather than `Timedelta(hours=24)` for the same reason:
    the offset means "the same wall-clock time tomorrow", which is what is meant,
    while the timedelta means "24 hours later", which on a transition day is a
    different moment.
    """
    start = pd.Timestamp(delivery_date, tz=tz)
    end = start + pd.DateOffset(days=1)
    local = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DatetimeIndex(local).tz_convert("UTC")


def extended_panel(panel: pd.DataFrame, *, through: pd.Timestamp) -> pd.DataFrame:
    """Extend the hourly grid so the delivery day has rows to predict from.

    Storage holds observed hours, so tomorrow is not in the panel and a design
    matrix built from it contains no row for any hour of tomorrow. That is not a
    failure the code would report — it is simply an empty result.

    Reindexing onto a longer grid adds those rows with every column null, and
    why that suffices is the availability discipline from ADR 0004 paying off:
    calendar and holiday features are computed from the index itself, and every
    lag is at least 48 hours and reads from history that exists.

    What it does *not* cover is weather, which has to be fetched for the delivery
    day. When it has not been, those columns stay null, the completeness check
    fails, and the caller is refused with the column names in the error. That is
    the correct outcome — a temperature-blind forecast for a hot August day is
    not a slightly worse forecast, and dropping the feature to force an answer
    would hide the real problem, which is that the weather step did not run.
    """
    index = pd.DatetimeIndex(panel.index)
    if through <= index.max():
        return panel
    return panel.reindex(pd.date_range(index.min(), through, freq="h", tz="UTC"))


def complete_rows(features: pd.DataFrame) -> pd.DatetimeIndex:
    """Hours where every feature is present.

    Deliberately *not* `features.build.usable_rows`, which also requires the
    target. That is right for training — a row with no target teaches nothing —
    and exactly wrong here, because the target for a future delivery day is what
    is being predicted. Reusing it would produce a system able to forecast only
    days whose answer was already known: every test written against historical
    dates would pass, and the first real request would fail.
    """
    return pd.DatetimeIndex(features.index[features.notna().all(axis=1)])
