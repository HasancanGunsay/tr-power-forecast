"""Download a date range of EPİAŞ series into `data/raw/`.

Run as:

    uv run python -m powerforecast.data.backfill --start 2021-01-01

Each month is written as soon as it arrives, so an interrupted run keeps
everything it already fetched. Re-running skips months that are already on disk
unless `--refresh` is given — the platform's quota is cumulative across
endpoints, so re-downloading data we already hold is not merely wasteful, it is
what pushes the next request into a 429.

Deliberately **does not** abort on a data-quality problem. `data/raw/` holds what
the API actually returned, gaps included; discovering later that a month is short
is far better than a backfill that stops halfway and leaves a partial dataset
looking complete. Problems are reported at the end and resolved in the processing
step, where the choice — interpolate, flag, or drop — is a modelling decision
rather than a download detail.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from powerforecast.config import get_settings
from powerforecast.data.epias import ALL_SERIES, EpiasAuth, EpiasClient, SeriesSpec
from powerforecast.data.ingest import (
    DataQualityError,
    fetch_chunk,
    month_chunks,
    read_raw,
    stored_months,
    validate_hourly,
    write_raw,
)

DEFAULT_START = date(2021, 1, 1)

# Bulk downloads get a wider default gap than interactive use. Being throttled
# mid-backfill costs far more than the extra seconds spent avoiding it.
BACKFILL_INTERVAL = 1.0
BACKFILL_ATTEMPTS = 6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill EPİAŞ series into data/raw/.")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        # Yesterday: today's series is still being published and would land as a
        # short final day that never gets refilled.
        default=date.today() - timedelta(days=1),
    )
    parser.add_argument(
        "--series",
        nargs="*",
        choices=[spec.name for spec in ALL_SERIES],
        help="Series to download (default: all).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download months that are already stored.",
    )
    return parser.parse_args(argv)


def selected_series(names: list[str] | None) -> tuple[SeriesSpec, ...]:
    if not names:
        return ALL_SERIES
    return tuple(spec for spec in ALL_SERIES if spec.name in names)


def _months_covered(start: date, end: date) -> set[str]:
    """UTC month labels a local date range can touch.

    Storage partitions on UTC, and Istanbul is UTC+3, so a range starting on the
    1st of a month reaches back into the previous UTC month. Treating a month as
    already downloaded therefore requires *both* labels to be present.
    """
    return {
        stamp.strftime("%Y-%m")
        for stamp in pd.date_range(
            pd.Timestamp(start, tz="Europe/Istanbul"),
            pd.Timestamp(end, tz="Europe/Istanbul") + pd.Timedelta(days=1),
            freq="h",
        ).tz_convert("UTC")
    }


def backfill_series(
    client: EpiasClient,
    spec: SeriesSpec,
    start: date,
    end: date,
    *,
    refresh: bool,
) -> pd.DataFrame:
    """Download and store one series month by month, returning what was fetched."""
    on_disk = set() if refresh else stored_months(spec)
    fetched: list[pd.DataFrame] = []

    for chunk_start, chunk_end in month_chunks(start, end):
        if _months_covered(chunk_start, chunk_end) <= on_disk:
            continue

        frame = fetch_chunk(client, spec, chunk_start, chunk_end)
        write_raw(frame, spec)
        fetched.append(frame)
        print(
            f"  {chunk_start:%Y-%m}  {len(frame):>4} rows  (interval {client.interval:.1f}s)",
            flush=True,
        )

    return pd.concat(fetched).sort_index() if fetched else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    username, password = settings.require_epias()

    auth = EpiasAuth(username=username, password=password)
    problems: list[str] = []

    with EpiasClient(
        auth, min_interval=BACKFILL_INTERVAL, max_attempts=BACKFILL_ATTEMPTS
    ) as client:
        for spec in selected_series(args.series):
            print(f"{spec.name}: {args.start} .. {args.end}", flush=True)

            backfill_series(client, spec, args.start, args.end, refresh=args.refresh)

            # Validate everything on disk, not just what this run fetched, so a
            # resumed run still checks the series as a whole.
            try:
                validate_hourly(read_raw(spec), name=spec.name)
            except DataQualityError as exc:
                problems.append(str(exc))
                print(f"  quality: {exc}", flush=True)
            else:
                print("  quality: complete hourly coverage, no nulls", flush=True)

    if problems:
        print(f"\nCompleted with {len(problems)} quality issue(s) — raw data was still written.")
    else:
        print("\nCompleted with no quality issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
