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
from powerforecast.data.ingest import read_raw

LOCAL_TZ = "Europe/Istanbul"


def load_panel(
    specs: tuple[SeriesSpec, ...] = ALL_SERIES,
    *,
    root: Path | None = None,
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
    return panel


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
