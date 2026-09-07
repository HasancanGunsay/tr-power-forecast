"""Where produced forecasts are kept.

A forecast that is not written down cannot be checked. The whole point of the
monitoring layer is to ask, days later, "what did we say would happen, and what
happened?" — and that question is unanswerable unless the answer was recorded at
the moment it was still a prediction. Recording it afterwards is not recording it.

Layout mirrors `data/raw`: one directory per target, one parquet file per
month inside it, UTC-partitioned.

    data/processed/forecasts/<target>/YYYY-MM.parquet

**The target is a directory, not a column.** Load is measured in MWh and price
in TRY/MWh, and a reader that spans both and forgets to filter would average
the two into a number that means nothing and raises nothing. Separate
directories make that impossible rather than merely detectable — the same
reason `data/raw` partitions by series. Every row still carries its `unit`, so
a frame that has been read out of its directory can still say what it is.

**The merge-on-write rule is not optional here.** Writing a month file by
replacing it is correct only when the write covers the whole month, and this job
writes 24 rows a day. Replacing would keep the newest day and silently discard
every earlier one — the same failure that once took `data/raw` from 49,008 rows
to 297 without raising anything. It was caught only because validation ran
against disk rather than memory, which is why `write_forecasts` reads back what
it wrote and returns the count.

The identity of a row is `(timestamp, model_version)`:

* Re-running the job on the same day **updates** rather than duplicates. A
  scheduler retries — after a network error, after a manual trigger — and a job
  that is not idempotent is a job that corrupts data on its second run.
* Running a **different** model version keeps both. Comparing what two versions
  predicted for the same hour is exactly what the monitoring layer is for, and
  that comparison is impossible if the newer write erases the older.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from powerforecast.config import PATHS
from powerforecast.targets import Target, resolve

FORECAST_DIR = "forecasts"

# The columns every produced forecast carries, in order. This is the contract
# between the job that writes forecasts and the monitoring layer that reads them,
# so it lives with the storage rather than with the model code — and changing it
# is a schema change, made here.
# Unit-neutral names. `forecast_mwh` was honest while load was the only target
# and becomes a lie the moment a price lands in it, so the unit moved out of the
# column name and into a column of its own where it can be read.
FORECAST_COLUMNS = (
    "forecast_value",
    "bias_offset",
    "unit",
    "model_name",
    "model_version",
    "forecast_origin",
    "generated_at",
)

# What the columns were called when load was the only target. Read support is
# kept so the 24 rows written before this change stay readable; nothing writes
# these names any more.
LEGACY_COLUMNS = {"forecast_mwh": "forecast_value", "bias_offset_mwh": "bias_offset"}

# What makes two rows the same row. Not just the timestamp: one hour may legally
# hold several forecasts, one per model version.
KEY_COLUMNS = ("timestamp", "model_version")


def forecasts_root(root: Path | None = None, *, target: Target | str | None = None) -> Path:
    """Directory holding one target's forecasts."""
    return (root or PATHS.processed) / FORECAST_DIR / resolve(target).name


def month_path(month: str, *, root: Path | None = None, target: Target | str | None = None) -> Path:
    """Path of the file holding one UTC month, e.g. ``"2026-08"``."""
    return forecasts_root(root, target=target) / f"{month}.parquet"


def write_forecasts(
    frame: pd.DataFrame,
    *,
    root: Path | None = None,
    target: Target | str | None = None,
) -> int:
    """Merge `frame` into the store and return the number of rows now on disk.

    The return value is read back **from disk**, not counted in memory. That
    distinction is the entire reason the earlier silent data loss in `data/raw`
    was noticed at all: a function that reports what it believes it wrote will
    confirm its own bug.

    Args:
        frame: Rows in the canonical layout — a UTC `DatetimeIndex` named
            ``timestamp`` plus `FORECAST_COLUMNS`.
        root: Processed-data root. Defaults to `PATHS.processed`.

    Returns:
        Total rows in the affected month files after the merge.
    """
    _validate(frame)

    index = pd.DatetimeIndex(frame.index)
    written = 0

    # Group by UTC month, because that is how the files are partitioned. Note
    # that a *local* delivery day straddles two UTC days and can straddle two UTC
    # months — 1 August local starts at 21:00 UTC on 31 July — so a single day's
    # 24 rows may legitimately land in two files. Grouping rather than assuming
    # one file per call is what makes that a non-event.
    for month, group in frame.groupby(index.strftime("%Y-%m")):
        path = month_path(str(month), root=root, target=target)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            group = _merge(pd.read_parquet(path), group)

        group.to_parquet(path, index=True)
        written += len(pd.read_parquet(path))

    return written


def read_forecasts(
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    root: Path | None = None,
    target: Target | str | None = None,
) -> pd.DataFrame:
    """Read stored forecasts, optionally narrowed by time or model version.

    Returns an empty frame with the right columns when nothing matches, rather
    than raising. A monitoring run over a period with no forecasts is a real
    situation with a real answer — "nothing was forecast" — not an error.
    """
    directory = forecasts_root(root, target=target)
    if not directory.is_dir():
        return _empty()

    files = sorted(directory.glob("*.parquet"))
    if not files:
        return _empty()

    frame = pd.concat([pd.read_parquet(path) for path in files]).sort_index()

    # Two generations of older files are tolerated on read, both by filling in
    # what they could not have carried. A reader that had to handle three shapes
    # would grow the same three branches in every caller.
    frame = frame.rename(columns=LEGACY_COLUMNS)

    # Files written before the correction existed have no offset column. Zero is
    # exactly right: those forecasts were uncorrected.
    if "bias_offset" not in frame.columns:
        frame["bias_offset"] = 0.0

    # Files written before the store was partitioned by target are load, because
    # load was the only target that existed when they were written.
    if "unit" not in frame.columns:
        frame["unit"] = resolve(target).unit

    frame = frame[list(FORECAST_COLUMNS)]

    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end)]
    if model_name is not None:
        frame = frame[frame["model_name"] == model_name]
    if model_version is not None:
        frame = frame[frame["model_version"] == model_version]

    return frame


def latest_run(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per hour: the most recently generated forecast.

    Needed because the store deliberately keeps several model versions per hour.
    Error metrics computed over that raw frame would count some hours twice and
    weight whichever hour happened to be forecast by more versions — a bias with
    no relation to how the models actually perform.
    """
    if frame.empty:
        return frame
    ordered = frame.sort_values("generated_at")
    return ordered[~ordered.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _merge(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Combine an existing month with new rows, newest wins on a key collision.

    `keep="last"` after concatenating in that order means the incoming row wins,
    which is what makes a re-run an update. Deduplication is on
    `(timestamp, model_version)` — deduplicating on the index alone would delete
    every version but one and quietly destroy the comparison the store exists to
    support.
    """
    combined = pd.concat([existing, incoming])
    keys = (
        combined.index.to_series().rename("timestamp").astype(str)
        + "|"
        + combined["model_version"].astype(str)
    )
    return combined[~keys.duplicated(keep="last")].sort_index()


def _validate(frame: pd.DataFrame) -> None:
    """Reject anything that is not the canonical layout, before it reaches disk.

    Storage is the one place where a wrong shape is expensive: a bad row written
    today is read back as truth for months. Validating on the way in costs one
    comparison and saves an archaeology session.
    """
    missing = [column for column in FORECAST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"forecast frame is missing columns: {missing}")

    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"forecast frame must be indexed by timestamp, got {type(index).__name__}")
    if index.tz is None:
        raise ValueError("forecast index must be timezone-aware; the store partitions on UTC")
    if str(index.tz) != "UTC":
        raise ValueError(f"forecast index must be UTC, got {index.tz}")
    if frame.empty:
        raise ValueError("refusing to write an empty forecast frame")


def _empty() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in FORECAST_COLUMNS}, index=index
    )
