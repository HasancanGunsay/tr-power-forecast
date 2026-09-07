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
from pathlib import Path

import pandas as pd

from powerforecast.features.availability import LOCAL_TZ, forecast_origins
from powerforecast.features.build import build_design_matrix
from powerforecast.forecasts.bias import BiasOffsets, estimate_offsets, should_correct
from powerforecast.forecasts.store import FORECAST_COLUMNS, latest_run, read_forecasts
from powerforecast.models.persistence import ModelCard, SavedModel
from powerforecast.targets import Target, for_column, resolve


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
    bias_offset: pd.Series
    target: Target
    bias_reason: str = ""

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
                "forecast_value": self.values.astype("float64"),
                "bias_offset": self.bias_offset.astype("float64"),
                # Repeated down all 24 rows like the provenance columns, and for
                # the same reason: a frame lifted out of its directory must still
                # be able to say whether it holds MWh or lira.
                "unit": self.target.unit,
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
    correct_bias: bool | None = None,
    forecasts_root: Path | None = None,
) -> DayForecast:
    """Predict every hour of one delivery day, or refuse.

    Args:
        model: A loaded model; its card fixes both the feature spec and the
            column order the estimator was fitted on.
        panel: Observed history. Extended internally to cover the delivery day.
        delivery_date: Local calendar date to forecast.
        generated_at: Injectable clock, so a test does not depend on when it runs.
        correct_bias: Apply the rolling bias correction. `None` means "ask the
            target", which is the right default because the answer differs by
            target and was measured separately for each. On load it is worth
            -6.3% of MAE on a retrained model (ADR 0010); on price the same
            machinery is worth -0.4% at best and +2.5% *worse* in load's own
            configuration, so it is off there (ADR 0015). Pass a bool only to
            override deliberately, as an experiment does.
        forecasts_root: Where past forecasts live, for the error history.

    Raises:
        IncompleteForecastError: if any hour of the day lacks a complete feature
            set. Never returns a partial day.
    """
    card = model.card
    # Derived, never passed. The card records the spec, the spec names the target
    # column, so the unit and the store partition follow from the model itself —
    # a caller cannot put a price forecast in the load directory by mistake.
    target = for_column(card.spec().target)
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

    origin = pd.Timestamp(forecast_origins(hours).iloc[0])
    raw = model.predict(day.loc[complete])

    # Asking the target rather than defaulting to True. The correction's value
    # was measured separately for each and the answers disagree - and not only
    # in size: on load it pays on a fresh model and hurts on a stale one, while
    # on price that is inverted. Inheriting load's setting was worth +2.5% of
    # price MAE until it was measured.
    apply_correction = target.correct_bias if correct_bias is None else correct_bias

    if apply_correction:
        offsets, reason = delivery_offsets(card, panel, origin, root=forecasts_root, target=target)
    else:
        why = (
            "correction disabled by the caller"
            if correct_bias is False
            else f"no bias correction is configured for {target.name}"
        )
        offsets, reason = BiasOffsets(method="none", reason=why), "disabled"
    applied = offsets.offsets_for(pd.DatetimeIndex(complete))

    return DayForecast(
        delivery_date=delivery_date,
        forecast_origin=origin,
        model_name=card.name,
        model_version=card.version,
        generated_at=generated_at or datetime.now(UTC),
        values=raw + applied,
        bias_offset=applied,
        target=target,
        bias_reason=f"{offsets.method}: {reason}" if reason else offsets.reason,
    )


def delivery_offsets(
    card: ModelCard,
    panel: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    root: Path | None = None,
    target: Target | str | None = None,
) -> tuple[BiasOffsets, str]:
    """Estimate the correction for one delivery day from forecasts already stored.

    Lives here rather than in the job because the service produces forecasts too,
    and a correction applied on one path and not the other would make the two
    disagree — which is the divergence this whole module exists to prevent.

    The error history is **what the system actually forecast**, read back from the
    store and joined to what happened. It cannot be recomputed: a forecast
    recalculated today from today's data is a different, much better forecast.

    Refuses on a stale model. That is not a detail — the same correction is worth
    −6.3% of MAE under regular retraining and **+2.6%** on a model trained once
    and never refreshed. See `forecasts.bias`.
    """
    fresh, why = should_correct(card.train_end, origin)
    if not fresh:
        return BiasOffsets(method="none", reason=why), why

    resolved = for_column(card.spec().target) if target is None else resolve(target)

    stored = read_forecasts(end=origin, model_name=card.name, root=root, target=resolved)
    if stored.empty:
        return BiasOffsets(method="none", reason="no stored forecasts yet"), why

    stored = latest_run(stored)
    actual = panel[resolved.column].reindex(stored.index)
    # `residual = actual - forecast`, the convention `estimate_offsets` expects.
    # Note it uses the *uncorrected* forecast: correcting from already-corrected
    # errors would estimate the residual of the correction, not of the model.
    raw = stored["forecast_value"] - stored["bias_offset"]
    errors = (actual - raw).dropna()

    return estimate_offsets(errors, origin=origin), why


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
