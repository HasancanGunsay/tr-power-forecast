"""Assemble the stored series into one analysis-ready table.

Raw storage keeps each series separate, which is right for ingestion: they are
fetched independently and revised independently. Analysis wants the opposite —
one row per hour, one column per series — so that is built here rather than
being re-derived in every notebook and script.

The index stays in UTC. Calendar features are derived from **local** time by
`add_local_calendar`, because electricity demand follows human routine: people
wake, work and cook on local clocks. An "hour of day" feature taken from UTC
would smear the evening peak across two different hours.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from powerforecast.data.epias import ALL_SERIES, SeriesSpec
from powerforecast.data.ingest import read_monthly, read_raw

LOCAL_TZ = "Europe/Istanbul"


WEATHER_COLUMNS = ("temperature_c", "hdd", "cdd")

# Conditions at the generators rather than at the consumers. Kept as its own
# tuple because the two datasets have different site lists and are backfilled
# by different scripts.
SUPPLY_WEATHER_COLUMNS = ("solar_index", "wind_index")


def load_panel(
    specs: tuple[SeriesSpec, ...] = ALL_SERIES,
    *,
    root: Path | None = None,
    with_weather: bool = True,
    with_supply_weather: bool = True,
) -> pd.DataFrame:
    """Load every series into a single hourly frame indexed in UTC.

    A series that is missing hours contributes nulls rather than removing the
    hour: an unpublished load plan is a hole in one column, not a reason to
    discard the consumption and price actually recorded for that hour.

    Raises:
        FileNotFoundError: if a requested series has not been backfilled, named
            so the fix is obvious.
    """
    frames: dict[str, pd.Series] = {}
    for spec in specs:
        stored = read_raw(spec, root=root)
        if stored.empty:
            raise FileNotFoundError(
                f"series {spec.name!r} has no stored data; run "
                f"`python -m powerforecast.data.backfill --series {spec.name}` first"
            )
        frames[spec.column] = stored[spec.column]

    panel = pd.concat(frames, axis=1, join="outer").sort_index()

    # Reindex onto a continuous hourly grid so a gap stays visible as a null row
    # instead of becoming an invisible jump between two adjacent timestamps.
    complete = pd.date_range(panel.index[0], panel.index[-1], freq="h", tz="UTC")
    panel = panel.reindex(complete)
    panel.index.name = "timestamp"

    if with_weather:
        panel = _join_weather(panel, root=root)

    if with_supply_weather:
        panel = _join_supply_weather(panel, root=root)

    return panel


def _join_weather(panel: pd.DataFrame, *, root: Path | None) -> pd.DataFrame:
    """Attach temperature columns if they have been backfilled.

    Absence is tolerated rather than fatal. The archive starts in 2022 while the
    electricity series starts in 2021, so weather is genuinely missing for the
    first year — and a project should still run end to end before every optional
    dataset has been downloaded. Missing columns become nulls, and the backtest
    drops incomplete rows inside each fold, which keeps the evaluation window
    identical to the weather-free run.
    """
    stored = read_monthly("weather", root=root)
    if stored.empty:
        return panel

    available = [column for column in WEATHER_COLUMNS if column in stored.columns]
    return panel.join(stored[available])


def _join_supply_weather(panel: pd.DataFrame, *, root: Path | None) -> pd.DataFrame:
    """Attach the solar and wind indices if they have been backfilled.

    Tolerates absence for the same reason `_join_weather` does, and it matters
    more here: this dataset was added late, so any clone of the repository has
    the electricity series long before it has these columns.
    """
    stored = read_monthly("supply_weather", root=root)
    if stored.empty:
        return panel

    available = [column for column in SUPPLY_WEATHER_COLUMNS if column in stored.columns]
    if not available:
        return panel
    return panel.join(stored[available])


def add_local_calendar(panel: pd.DataFrame, *, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Attach calendar columns derived from local time.

    Deliberately excludes holidays. Turkish religious holidays follow the lunar
    calendar and move roughly eleven days earlier each year, so they cannot be
    derived from the timestamp alone and need a proper calendar source. Adding a
    naive fixed-date holiday flag would be worse than none: it would mark the
    wrong days and look correct.
    """
    local = pd.DatetimeIndex(panel.index).tz_convert(tz)

    annotated = panel.copy()
    annotated["local_time"] = local
    annotated["local_date"] = local.date
    annotated["hour"] = local.hour
    annotated["dayofweek"] = local.dayofweek
    annotated["month"] = local.month
    annotated["year"] = local.year
    annotated["is_weekend"] = local.dayofweek >= 5
    return annotated
