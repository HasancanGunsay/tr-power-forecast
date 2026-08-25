"""Produce and store tomorrow's forecast, unattended.

    uv run python -m powerforecast.jobs.daily_forecast

Run it before the 12:30 bid deadline. Three steps:

    fetch tomorrow's weather  ->  forecast  ->  store

It reads the panel from disk and does **not** ingest. Ingestion is a separate job
with separate failure modes — credentials, a cumulative API quota, a platform
that is sometimes simply down — and folding it in here would mean one broken
thing stops two. `scripts/daily.ps1` runs them in order; the scheduler is the
orchestrator, not this file.

Nothing here is clever. That is the point: this is the first part of the system
that runs without a person watching, and unattended code is judged by how it
behaves when something is missing rather than by what it does when everything is
present. Three decisions follow from that.

**Exit codes are the interface.** A scheduler cannot read prose. It sees an
integer, and 0 means "the forecast exists". Every failure path here returns
non-zero, and no failure path returns 0 with nothing written — a job that exits
successfully having done nothing is worse than one that crashes, because the
crash is visible and the silence is not.

**The forecast is either whole or absent.** `forecast_day` refuses a partial
day, and this job does not catch that refusal and store what it has. Storing 20
of 24 hours would leave four hours nobody bid, and the gap would be discovered
by the market rather than by us.

**The weather step may not silently fall back.** Tomorrow's temperature comes
from the live forecast run — the same object training used, seen from the other
side of the day (see `data.weather.fetch_live_all_cities`). If that call fails,
the job fails. Reusing yesterday's temperature to keep the pipeline green would
produce a forecast that looks exactly like a good one and is not.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from powerforecast.data.panel import load_panel
from powerforecast.data.weather import WeatherError, fetch_live_all_cities, weather_features
from powerforecast.forecasts.day import IncompleteForecastError, forecast_day
from powerforecast.forecasts.store import write_forecasts
from powerforecast.models.persistence import DEFAULT_MODEL_NAME, ModelStoreError, load_model

logger = logging.getLogger("daily_forecast")

# Exit codes, distinct so a scheduler's alert can say which stage failed without
# anyone opening a log. A single non-zero code would make "the model is missing"
# and "the weather API is down" indistinguishable, and those need different people.
EXIT_OK = 0
EXIT_MODEL = 2
EXIT_WEATHER = 3
EXIT_INCOMPLETE = 4
EXIT_UNEXPECTED = 5


def run(
    delivery_date: date | None = None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = "latest",
    model_directory: Path | None = None,
    processed_root: Path | None = None,
    today: date | None = None,
    fetch_weather: bool = True,
) -> int:
    """Produce and store one delivery day. Returns the process exit code."""
    today = today or datetime.now(UTC).astimezone().date()
    delivery_date = delivery_date or today + timedelta(days=1)

    logger.info("delivery day %s (today %s)", delivery_date, today)

    try:
        model = load_model(
            model_name,
            model_version,
            directory=model_directory,
            # On by default in unattended code, for the same reason as in the
            # service: a version mismatch here would produce numbers rather than
            # an error, and nobody is watching.
            require_environment=True,
        )
    except ModelStoreError as error:
        logger.error(
            "no usable model: %s\nTrain one with: uv run python -m powerforecast.models.train",
            error,
        )
        return EXIT_MODEL

    logger.info("model %s/%s", model.card.name, model.card.version)

    panel = load_panel()
    logger.info("panel covers %s .. %s", panel.index.min(), panel.index.max())

    if fetch_weather:
        try:
            panel = _with_tomorrow_weather(panel, delivery_date, today=today)
        except WeatherError as error:
            logger.error("weather for %s unavailable: %s", delivery_date, error)
            return EXIT_WEATHER

    try:
        produced = forecast_day(model, panel, delivery_date)
    except IncompleteForecastError as error:
        for key, value in error.as_dict().items():
            logger.error("%s: %s", key, value)
        return EXIT_INCOMPLETE

    rows = write_forecasts(produced.to_frame(), root=processed_root)
    logger.info(
        "stored %d hours for %s (origin %s); %d rows now in the affected months",
        len(produced.values),
        delivery_date,
        produced.forecast_origin,
        rows,
    )
    return EXIT_OK


def _with_tomorrow_weather(
    panel: pd.DataFrame, delivery_date: date, *, today: date
) -> pd.DataFrame:
    """Attach the delivery day's temperature to the panel, in memory only.

    Deliberately not written to `data/raw`. The stored weather series is the
    *archive* — what was being forecast for a past day, as recorded afterwards —
    and mixing a live forecast into it would blur the distinction between a
    prediction and a record. The archive backfill will pick this same day up
    later, from the proper source, and that is the copy that should persist.

    The panel gains rows for the delivery day here rather than in `forecast_day`,
    because the weather columns have to exist before the design matrix is built.
    """
    hours = pd.date_range(
        pd.Timestamp(delivery_date, tz="Europe/Istanbul"),
        pd.Timestamp(delivery_date, tz="Europe/Istanbul") + pd.DateOffset(days=1),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")

    raw = fetch_live_all_cities(delivery_date, today=today)
    weather = weather_features(raw).reindex(hours)

    if weather.isna().any().any():
        raise WeatherError(
            f"live weather for {delivery_date.isoformat()} is incomplete: "
            f"{int(weather.isna().any(axis=1).sum())} of {len(hours)} hours missing"
        )

    grid = pd.date_range(panel.index.min(), hours[-1], freq="h", tz="UTC")
    extended = panel.reindex(grid)
    # `.loc` assignment rather than concat/join: the columns already exist in the
    # panel and only the delivery day's rows are being filled. A join would
    # duplicate the columns with suffixes and the design matrix would then be
    # built from the wrong one — silently, since both would be present.
    for column in weather.columns:
        extended.loc[hours, column] = weather[column].to_numpy()
    return extended


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="delivery day (default: tomorrow)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default="latest")
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="skip the live weather fetch; only useful for a past day already in the panel",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    try:
        return run(
            args.date,
            model_name=args.model,
            model_version=args.model_version,
            fetch_weather=not args.no_weather,
        )
    except Exception:
        # Last line of defence. An unhandled traceback would still exit non-zero,
        # but it would exit with 1 — indistinguishable from a usage error — and
        # the log would end mid-sentence. Logging it here keeps the exit codes
        # meaningful and puts the traceback where the scheduler collects output.
        logger.exception("daily forecast failed unexpectedly")
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    raise SystemExit(main())
