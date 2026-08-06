"""Download archived day-ahead temperature forecasts.

    uv run python -m powerforecast.data.backfill_weather

Fetched a year at a time and written as it lands, so an interrupted run keeps
what it already has. Unlike the EPİAŞ backfill this needs no credentials, but it
shares the same storage layout and the same merge-on-write semantics.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from powerforecast.data.ingest import read_monthly, write_monthly
from powerforecast.data.weather import ARCHIVE_START, CITIES, degree_days, fetch_all_cities

DATASET = "weather"


def year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split a range into calendar-year pieces, both ends inclusive."""
    if end < start:
        raise ValueError(f"end ({end}) must not be before start ({start})")

    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        year_end = date(cursor.year, 12, 31)
        chunks.append((cursor, min(year_end, end)))
        cursor = date(cursor.year + 1, 1, 1)
    return chunks


def backfill(start: date, end: date) -> pd.DataFrame:
    for chunk_start, chunk_end in year_chunks(start, end):
        frame = fetch_all_cities(chunk_start, chunk_end)
        frame = frame.join(degree_days(frame["temperature_c"]))

        written = write_monthly(frame, DATASET)
        coverage = 100 * frame["temperature_c"].notna().mean()
        print(
            f"  {chunk_start} .. {chunk_end}  {len(frame):,} hours  "
            f"{coverage:5.1f}% covered  -> {len(written)} files"
        )

    return read_monthly(DATASET)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=ARCHIVE_START)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    start = max(args.start, ARCHIVE_START)
    if start != args.start:
        print(f"note: the archive begins {ARCHIVE_START}, starting there instead\n")

    print(f"cities: {', '.join(c.name for c in CITIES)}")
    print(f"range : {start} .. {args.end}\n")

    stored = backfill(start, args.end)

    print(f"\nstored {len(stored):,} hours, {stored.index[0]} .. {stored.index[-1]}")
    missing = stored["temperature_c"].isna().sum()
    if missing:
        print(f"warning: {missing:,} hours have no temperature")


if __name__ == "__main__":
    main()
