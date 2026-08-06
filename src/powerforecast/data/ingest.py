"""Turn EPİAŞ responses into a validated, stored hourly series.

Three ideas carry this module.

**Timestamps are stored in UTC.** The Turkish market runs on a fixed UTC+03:00,
so local time would be harmless here — but European bidding zones observe
daylight saving, where one day a year has 23 hours and another has 25. A series
indexed in local time then contains a duplicated hour and a missing one, and
every lag feature built on it is quietly wrong. Storing UTC makes the hourly grid
exactly hourly, always. Local time is reintroduced later, deliberately, when
calendar features are built.

**Validation happens before storage.** A duplicated or missing hour raises
nothing by itself; it shifts lags and biases the backtest that follows. Checking
after the fact means the corrupt data has already been used.

**Raw data is written per month.** Re-fetching one month rewrites one file rather
than the whole history.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from powerforecast.config import PATHS
from powerforecast.data.epias import ISTANBUL_OFFSET, EpiasClient, SeriesSpec

INDEX_NAME = "timestamp"


class DataQualityError(RuntimeError):
    """Raised when data fails a check that would corrupt downstream features."""


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split an inclusive date range into non-overlapping monthly chunks.

    The API caps how much a single request may return, so long backfills have to
    be split. Both ends of each chunk are inclusive, matching the platform's own
    treatment of `endDate` — requesting 1 Aug to 2 Aug returns 48 hours, not 24.
    Getting that wrong produces overlapping chunks and duplicated rows.
    """
    if end < start:
        raise ValueError(f"end ({end}) must not be before start ({start})")

    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        last_of_month = date(
            cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1]
        )
        chunks.append((cursor, min(last_of_month, end)))
        cursor = last_of_month + timedelta(days=1)
    return chunks


def _as_request_bound(day: date) -> str:
    return f"{day.isoformat()}T00:00:00{ISTANBUL_OFFSET}"


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def to_frame(items: list[dict[str, Any]], spec: SeriesSpec) -> pd.DataFrame:
    """Convert raw response items into a UTC-indexed single-column frame.

    The `date` field already carries the full hourly timestamp with its offset,
    so the separate `time` / `hour` field is redundant and is dropped — two
    representations of the same instant only ever diverge.
    """
    empty = pd.DataFrame(
        {spec.column: pd.Series(dtype="float64")},
        index=pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME),
    )
    if not items:
        return empty

    missing = [item for item in items if spec.value_field not in item]
    if missing:
        raise DataQualityError(
            f"{len(missing)} item(s) for series {spec.name!r} are missing the "
            f"{spec.value_field!r} field; the endpoint may have changed shape"
        )

    frame = pd.DataFrame(
        {
            INDEX_NAME: pd.to_datetime([item["date"] for item in items], utc=True),
            spec.column: pd.to_numeric([item[spec.value_field] for item in items], errors="coerce"),
        }
    )
    return frame.set_index(INDEX_NAME).sort_index()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_hourly(frame: pd.DataFrame, *, name: str, max_reported: int = 5) -> None:
    """Assert that a frame is a complete, gap-free, hourly UTC series.

    Raises:
        DataQualityError: naming the specific timestamps at fault, because
            "there is a gap" is not actionable while "2026-03-31 01:00 is
            missing" points straight at the cause.
    """
    if frame.empty:
        raise DataQualityError(
            f"series {name!r} is empty; this usually means a wrong date range or a "
            "request that failed without raising"
        )

    index = pd.DatetimeIndex(frame.index)

    duplicates = index[index.duplicated()]
    if len(duplicates) > 0:
        raise DataQualityError(
            f"series {name!r} has {len(duplicates)} duplicate timestamp(s), "
            f"first: {_sample(duplicates, max_reported)}"
        )

    expected = pd.date_range(index[0], index[-1], freq="h", tz="UTC")
    gaps = expected.difference(index)
    if len(gaps) > 0:
        raise DataQualityError(
            f"series {name!r} is missing {len(gaps)} hour(s), first: {_sample(gaps, max_reported)}"
        )

    nulls = frame[frame.isna().any(axis=1)]
    if not nulls.empty:
        raise DataQualityError(
            f"series {name!r} has {len(nulls)} null value(s), "
            f"first: {_sample(nulls.index, max_reported)}"
        )


def _sample(index: pd.Index, limit: int) -> str:
    shown = [str(ts) for ts in index[:limit]]
    suffix = ", ..." if len(index) > limit else ""
    return ", ".join(shown) + suffix


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def fetch_chunk(client: EpiasClient, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
    """Fetch a single inclusive date range in one request."""
    body = client.post(
        spec.path,
        {"startDate": _as_request_bound(start), "endDate": _as_request_bound(end)},
    )
    return to_frame(body.get("items", []), spec)


def fetch_series(
    client: EpiasClient,
    spec: SeriesSpec,
    start: date,
    end: date,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Fetch one series over an inclusive date range, month by month."""
    frames = [
        fetch_chunk(client, spec, chunk_start, chunk_end)
        for chunk_start, chunk_end in month_chunks(start, end)
    ]

    combined = pd.concat(frames).sort_index() if frames else to_frame([], spec)
    if validate:
        validate_hourly(combined, name=spec.name)
    return combined


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def _series_dir(spec: SeriesSpec, root: Path | None) -> Path:
    return (root or PATHS.raw) / spec.name


def write_monthly(frame: pd.DataFrame, name: str, *, root: Path | None = None) -> list[Path]:
    """Write any hourly frame to `<root>/<name>/YYYY-MM.parquet`, one file per month.

    An existing month is **merged with**, not replaced by, the incoming rows: on
    a conflicting timestamp the new value wins, and rows the caller did not
    supply are kept.

    Replacing outright would be simpler, and it is correct when a whole series
    is written in one call. It is silently destructive when months arrive one at
    a time: storage partitions on UTC while requests are made in local time, so a
    request for local January also returns hours belonging to the UTC December
    and February files. Under replace semantics, each write would truncate its
    neighbour's file to the handful of overlapping hours — which is exactly the
    bug this function was rewritten to fix.

    Merging keeps re-runs idempotent, which is the property that mattered in the
    first place. This is deliberately the only implementation of that logic in
    the project; a second copy is a second chance to get it wrong.
    """
    directory = (root or PATHS.raw) / name
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    # Group by the UTC calendar month. `to_period("M")` would do the same but
    # discards the timezone with a warning; formatting the index states plainly
    # that partitioning follows UTC, not local time.
    months = pd.DatetimeIndex(frame.index).strftime("%Y-%m")
    for period, group in frame.groupby(months):
        path = directory / f"{period}.parquet"

        if path.exists():
            existing = pd.read_parquet(path)
            # `keep="last"` puts the incoming rows after the stored ones, so a
            # re-fetch overwrites a revised value rather than preserving a stale one.
            group = pd.concat([existing, group])
            group = group[~group.index.duplicated(keep="last")].sort_index()

        group.to_parquet(path, index=True)
        written.append(path)
    return written


def read_monthly(name: str, *, root: Path | None = None) -> pd.DataFrame:
    """Read every stored month of a named dataset back into one frame."""
    directory = (root or PATHS.raw) / name
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in files]).sort_index()


def write_raw(frame: pd.DataFrame, spec: SeriesSpec, *, root: Path | None = None) -> list[Path]:
    """Write one EPİAŞ series, partitioned by month. See `write_monthly`."""
    return write_monthly(frame, spec.name, root=root)


def stored_months(spec: SeriesSpec, *, root: Path | None = None) -> set[str]:
    """Return the `YYYY-MM` labels already on disk for a series."""
    directory = _series_dir(spec, root)
    return {path.stem for path in directory.glob("*.parquet")}


def read_raw(spec: SeriesSpec, *, root: Path | None = None) -> pd.DataFrame:
    """Read every stored month of a series back into one frame."""
    directory = _series_dir(spec, root)
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return to_frame([], spec)

    return pd.concat([pd.read_parquet(path) for path in files]).sort_index()
