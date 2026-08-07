"""The forecasting service.

    uv run uvicorn powerforecast.serving.app:app --reload

Two things are happening in this file, and it is worth separating them before
reading a single line of code.

**FastAPI is not a server.** It is a library for *describing* an application:
which URLs exist, what they accept, what they return. Something else has to
listen on a TCP port, parse HTTP, and call that description. That something is
`uvicorn`. The split is why the command above names both — `uvicorn` is the
program being run, `powerforecast.serving.app:app` is the object it is being
handed. Confusing the two is a common interview stumble; they are as separate as
a recipe and a stove.

**Serving inverts every assumption the backtest made.** The backtest fits a
model, scores it, throws it away, and knows the answer while it asks the
question. A service does the opposite: it loads a model somebody else fitted,
weeks ago, and is asked about a day whose true demand nobody knows yet. Three
consequences run through everything below.

1. *The target is unknown.* `usable_rows` — the helper the training code uses —
   requires the target to be present, and for a future delivery day it never is.
   Using it here would reject every real request. See `_complete_rows`.
2. *The delivery day may not be in the panel at all.* Storage holds observed
   hours; tomorrow has none. The design matrix therefore has to be built over a
   grid that has been *extended* to cover the requested day, or there are no
   rows to predict from. See `_extended_panel`.
3. *Nothing may fail silently.* A backtest that quietly drops an hour loses a
   little precision in a number. A service that quietly drops an hour returns
   23 values where the market needs 24, and the missing hour is a position
   nobody bid. Every incomplete case here is an error, never a short answer.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from powerforecast.data.panel import load_panel
from powerforecast.features.availability import LOCAL_TZ, forecast_origins
from powerforecast.features.build import build_design_matrix
from powerforecast.models.persistence import ModelStoreError, SavedModel, load_model

DEFAULT_MODEL_NAME = "load-lightgbm"

# How long a loaded panel may be reused before it is read from disk again.
#
# This number encodes a judgement about *lifetimes*, which is the interesting
# design question in this file. The model and the data do not change on the same
# clock: the model is refit deliberately, perhaps monthly, while the panel gains
# rows every day when the ingestion job runs. Treating them the same way would be
# wrong in one direction or the other — reloading the model per request wastes
# hundreds of milliseconds on something that has not changed, and caching the
# panel forever means a service that has been up for a week is forecasting from
# week-old history and reporting no error at all.
#
# So: the model is loaded once, at startup, and the panel is cached with an
# expiry. Fifteen minutes is short enough that a freshly ingested day appears
# almost immediately and long enough that a burst of requests reads the parquet
# files once rather than once each.
#
# Note precisely what this is: an *expiry*, not invalidation. Nothing tells the
# service that new data arrived; it simply stops trusting what it holds after a
# while. The guarantee is therefore "at most fifteen minutes stale", and
# `/health` publishes the actual age so the guarantee can be checked from
# outside rather than believed.
PANEL_TTL_SECONDS = 15 * 60


# ---------------------------------------------------------------------------
# What the service holds while it is running
# ---------------------------------------------------------------------------


@dataclass
class PanelCache:
    """The panel, plus how old it is.

    A plain module-level global would work and is what most examples do. It is
    avoided here because a global cannot be replaced in a test without leaking
    into the next one, and because two apps in the same process — which is
    exactly what a test suite creates — would silently share it.
    """

    loader: Callable[[], pd.DataFrame]
    ttl_seconds: float = PANEL_TTL_SECONDS
    _panel: pd.DataFrame | None = field(default=None, repr=False)
    _loaded_at: float = field(default=0.0, repr=False)

    def get(self) -> pd.DataFrame:
        """Return the panel, reading it again if the cached copy has expired."""
        # `>=`, not `>`. With a TTL of zero the two are not the same thing: on a
        # clock whose resolution is coarser than the work being timed — roughly
        # 15 ms on Windows — a fresh read can report an age of exactly 0.0, and
        # `0.0 > 0` is false, so a cache configured never to cache would cache
        # forever. `>=` makes "reuse while age is below the TTL" say what it
        # means at the boundary as well as away from it.
        if self._panel is None or self.age_seconds() >= self.ttl_seconds:
            self._panel = self.loader()
            self._loaded_at = time.monotonic()
        return self._panel

    def age_seconds(self) -> float:
        """Seconds since the panel was last read from disk.

        `time.monotonic`, not `time.time`. A wall clock can jump — NTP
        corrections, daylight saving on a badly configured host — and a jump
        backwards would make a stale cache look fresh. Monotonic time only ever
        moves forward, which is the only property this measurement needs.
        """
        if self._panel is None:
            return float("inf")
        return time.monotonic() - self._loaded_at


@dataclass
class ServiceState:
    """Everything the request handlers need, assembled once at startup.

    Handlers reach this through `Depends(get_state)` rather than by importing a
    global. That indirection is what lets a test build an app around a fake
    panel and a model in a temporary directory, without monkey-patching
    anything.
    """

    model: SavedModel
    panel: PanelCache


# ---------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------
#
# These are pydantic models rather than plain dicts, for three reasons that are
# worth stating because "use a schema" on its own is cargo cult:
#
# 1. FastAPI reads them to generate the OpenAPI document, which is what makes
#    /docs an interactive page rather than a blank one. The documentation is
#    then derived from the code and cannot drift away from it.
# 2. The response is validated on the way out. A handler that returns a string
#    where a float belongs fails here, in this process, rather than in whatever
#    consumes the JSON.
# 3. They name the contract. `forecast_mwh` says both what the number is and
#    what unit it is in — and a unit that lives only in someone's head is the
#    kind of thing that eventually gets multiplied by a thousand.


class HourlyForecast(BaseModel):
    """One delivery hour."""

    timestamp: datetime = Field(description="Start of the delivery hour, UTC.")
    local_hour: int = Field(ge=0, le=23, description="Hour of the day in Europe/Istanbul.")
    forecast_mwh: float = Field(description="Forecast consumption for that hour, in MWh.")


class ForecastResponse(BaseModel):
    """A full delivery day, and the provenance of the numbers in it."""

    delivery_date: date
    forecast_origin: datetime = Field(
        description=(
            "The newest observation the forecast is allowed to use: 11:00 local on the "
            "previous day, because bids close at 12:30 and the value stamped 12:00 is "
            "still accumulating."
        )
    )
    model_name: str
    model_version: str
    generated_at: datetime
    hours: list[HourlyForecast]


class HealthResponse(BaseModel):
    """Liveness, and — more usefully — identity."""

    status: str
    model_name: str
    model_version: str
    model_trained_rows: int
    model_trained_until: datetime
    panel_end: datetime = Field(description="Last hour present in the cached panel, UTC.")
    panel_age_seconds: float


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------


def create_app(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = "latest",
    model_directory: Path | None = None,
    panel_loader: Callable[[], pd.DataFrame] | None = None,
    panel_ttl_seconds: float = PANEL_TTL_SECONDS,
    require_environment: bool = True,
) -> FastAPI:
    """Build the application.

    A factory rather than a module-level `app = FastAPI()` with globals around
    it. Every dependency the service has — which model, from where, which panel
    — is an argument, so a test can supply small fakes and get a real app rather
    than a patched one.

    Args:
        require_environment: Refuse to start when the model was fitted under
            different library versions. **On by default here**, unlike in
            `load_model`, and the asymmetry is deliberate. Opening an old model
            in a notebook to look at it is a different risk from answering
            requests with it. A service should fail at startup, loudly, in front
            of whoever is deploying it — rather than at three in the morning,
            subtly, in the numbers.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load the model once, before the first request.

        This is the whole reason `lifespan` exists. Loading inside a handler
        would unpickle a LightGBM model on every single request — hundreds of
        milliseconds of work whose result is identical every time. Loading at
        import time instead would be worse in a different way: the module could
        no longer be imported on a machine without a model store, which breaks
        tooling, tests, and `--help`.

        Failing here is also the point. If the model is missing or was fitted
        under other libraries, the process refuses to come up. A service that
        starts and then returns errors looks healthy to everything watching it.
        """
        try:
            model = load_model(
                model_name,
                model_version,
                directory=model_directory,
                require_environment=require_environment,
            )
        except ModelStoreError as error:
            # Re-raised with the fix in the message. A stack trace tells the
            # reader what broke; this tells them what to do about it.
            raise RuntimeError(
                f"the service cannot start without a model: {error}\n"
                "Train one with: uv run python -m powerforecast.models.train"
            ) from error

        app.state.service = ServiceState(
            model=model,
            panel=PanelCache(loader=panel_loader or load_panel, ttl_seconds=panel_ttl_seconds),
        )
        yield
        # Nothing to tear down: no sockets, no pools, no threads. The block is
        # kept anyway so that when something does need closing, there is an
        # obvious place for it rather than an `atexit` handler nobody finds.

    app = FastAPI(
        title="tr-power-forecast",
        version="0.1.0",
        summary="Day-ahead electricity demand forecasts for the Turkish grid.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(state: Annotated[ServiceState, Depends(get_state)]) -> HealthResponse:
        """Report that the service is up — and, more importantly, *what* is up.

        "Is the service running?" is rarely the question that matters. After a
        deployment the question is "is the model I just shipped the one
        answering?", and the cheapest way to catch a rollout that silently kept
        the old version is to have the version in the health response.

        Reading the panel here is deliberate: a health check that touches
        nothing proves only that the process exists. This one proves the data
        can be read, and publishes how stale it is.
        """
        card = state.model.card
        panel = state.panel.get()
        return HealthResponse(
            status="ok",
            model_name=card.name,
            model_version=card.version,
            model_trained_rows=card.n_train_rows,
            model_trained_until=datetime.fromisoformat(card.train_end),
            panel_end=pd.DatetimeIndex(panel.index).max().to_pydatetime(),
            panel_age_seconds=round(state.panel.age_seconds(), 1),
        )

    @app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
    def forecast(
        state: Annotated[ServiceState, Depends(get_state)],
        delivery_date: Annotated[
            date,
            Query(
                alias="date",
                description="Delivery day, local calendar date, as YYYY-MM-DD.",
                examples=["2026-08-01"],
            ),
        ],
    ) -> ForecastResponse:
        """Forecast every hour of one delivery day.

        The parameter is typed `date`, not `str`, and that single choice removes
        a whole category of handler code. FastAPI parses and validates it before
        this function is entered, so `?date=elma` never reaches here — it comes
        back as a 422 naming the field and the expected format. Accepting a
        string and parsing it by hand would mean writing that error message
        badly, once per endpoint.
        """
        return _forecast_day(state, delivery_date)

    return app


def get_state(request: Request) -> ServiceState:
    """Hand the handler the objects built at startup.

    FastAPI resolves this for every request that declares it. It exists as a
    named dependency, rather than the handler reaching into `request.app.state`
    directly, so that the state is a typed argument — which the type checker can
    see and a test can replace.
    """
    state: ServiceState | None = getattr(request.app.state, "service", None)
    if state is None:  # pragma: no cover — unreachable unless lifespan is skipped
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the service has not finished starting",
        )
    return state


# ---------------------------------------------------------------------------
# The forecast itself
# ---------------------------------------------------------------------------


def _forecast_day(state: ServiceState, delivery_date: date) -> ForecastResponse:
    """Build features for one delivery day, predict, and refuse anything partial."""
    card = state.model.card
    hours = _delivery_hours(delivery_date)

    panel = _extended_panel(state.panel.get(), through=hours[-1])
    features, _ = build_design_matrix(panel, card.spec())

    # Reindex, rather than filter. Filtering would return whatever happens to be
    # present; reindexing asks for exactly the 24 hours wanted and leaves a
    # visible hole — as NaN — wherever one is absent. The difference is whether
    # a missing hour is something to be discovered or something to be noticed.
    day = features.reindex(hours)
    complete = _complete_rows(day)

    if len(complete) < len(hours):
        raise HTTPException(
            # 422, the same code FastAPI returns for a malformed parameter. The
            # request was well formed but cannot be acted on, which is exactly
            # what 422 means; 400 would suggest the caller typed something wrong,
            # and 500 would suggest the service is broken. Neither is true here —
            # the data has not arrived yet.
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_incomplete_detail(delivery_date, day, complete, panel),
        )

    predictions = state.model.predict(day.loc[complete])
    local_hours = pd.DatetimeIndex(complete).tz_convert(LOCAL_TZ).hour

    return ForecastResponse(
        delivery_date=delivery_date,
        forecast_origin=forecast_origins(hours).iloc[0].to_pydatetime(),
        model_name=card.name,
        model_version=card.version,
        generated_at=datetime.now(UTC),
        hours=[
            HourlyForecast(
                timestamp=timestamp.to_pydatetime(),
                local_hour=int(local_hour),
                forecast_mwh=float(value),
            )
            for timestamp, local_hour, value in zip(complete, local_hours, predictions, strict=True)
        ],
    )


def _delivery_hours(delivery_date: date) -> pd.DatetimeIndex:
    """Every hour of one local calendar day, as UTC timestamps.

    Not `range(24)`. A calendar day is 24 hours only when there is no daylight
    saving transition in it; where there is, it is 23 or 25. Türkiye has been on
    permanent UTC+3 since 2016, so today the answer is always 24 — but this
    project also has a European leg waiting on an ENTSO-E token, and the day the
    first German delivery day arrives, `range(24)` would drop or duplicate an
    hour twice a year without saying so.

    `DateOffset(days=1)` rather than `Timedelta(hours=24)` for the same reason:
    the offset means "the same wall-clock time tomorrow", which is the thing
    actually meant, while the timedelta means "24 hours later", which on a
    transition day is a different moment.
    """
    start = pd.Timestamp(delivery_date, tz=LOCAL_TZ)
    end = start + pd.DateOffset(days=1)
    local = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DatetimeIndex(local).tz_convert("UTC")


def _extended_panel(panel: pd.DataFrame, *, through: pd.Timestamp) -> pd.DataFrame:
    """Extend the hourly grid so the delivery day has rows to predict from.

    Storage holds observed hours. Tomorrow has none — so tomorrow is not in the
    panel, and a design matrix built from the panel as stored contains no row
    for any hour of it. Asking the model to predict for a day that does not
    appear in its input is not a failure the code would report; it is simply an
    empty result.

    Reindexing onto a longer grid adds those rows with every column null. That
    is enough for most of the design matrix, and *why* it is enough is the
    availability discipline from ADR 0004 paying off:

    * calendar and holiday features are computed from the index itself, and a
      calendar is known years ahead;
    * every lag is at least 48 hours and is read from history, which is present.

    What it is *not* enough for is weather, which has to be fetched for the
    delivery day. When it has not been, the columns stay null, the completeness
    check fails, and the request is refused with those column names in it. That
    is the correct outcome: a temperature-blind forecast for a hot August day is
    not a slightly worse forecast, and dropping the feature to force an answer
    would hide the real problem, which is that the weather job did not run.
    """
    index = pd.DatetimeIndex(panel.index)
    if through <= index.max():
        return panel
    extended = pd.date_range(index.min(), through, freq="h", tz="UTC")
    return panel.reindex(extended)


def _complete_rows(features: pd.DataFrame) -> pd.DatetimeIndex:
    """Hours where every feature is present.

    Deliberately *not* `features.build.usable_rows`, which also requires the
    target to be present. That is right for training — a row with no target
    teaches nothing — and exactly wrong here, because the target for a future
    delivery day is the thing being predicted. Reusing it would make the service
    able to forecast only days whose answer was already known.

    Worth pausing on, because it is the shape of a mistake that is easy to make
    and produces a service that passes every test written against historical
    dates and fails on the first real request.
    """
    return pd.DatetimeIndex(features.index[features.notna().all(axis=1)])


def _incomplete_detail(
    delivery_date: date,
    day: pd.DataFrame,
    complete: pd.Index,
    panel: pd.DataFrame,
) -> dict[str, object]:
    """Explain a refusal well enough that the reader knows what to fix.

    A bare "422 Unprocessable Entity" tells an operator that something is wrong
    and nothing about what. The three facts below turn a support question into a
    self-service fix: how many hours were short, which columns were missing, and
    how far the data actually reaches. The last one usually is the answer —
    the ingestion job has not run.
    """
    missing_columns = sorted(day.columns[day.isna().any()].tolist())
    observed = pd.DatetimeIndex(panel.index)[panel.notna().any(axis=1)]
    return {
        "message": (
            f"cannot forecast {delivery_date.isoformat()}: "
            f"{len(complete)} of {len(day)} hours have a complete feature set"
        ),
        "missing_features": missing_columns,
        "data_available_until": observed.max().isoformat() if len(observed) else None,
        "hint": (
            "Ingest the missing history and weather first: "
            "`uv run python -m powerforecast.data.backfill` and "
            "`uv run python -m powerforecast.data.backfill_weather`."
        ),
    }


# The object uvicorn is pointed at. Built at import time, but note that nothing
# expensive happens yet: `create_app` only describes the application, and the
# model is not touched until `lifespan` runs at startup.
app = create_app()
