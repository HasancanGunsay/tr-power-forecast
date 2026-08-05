"""Download a date range of EPİAŞ series into `data/raw/`.

Run as:

    uv run python -m powerforecast.data.backfill --start 2021-01-01

Deliberately **does not** fail on a data-quality problem. `data/raw/` holds what
the API actually returned, gaps included; discovering later that a month is short
is far better than having a backfill abort halfway and leave a partial dataset
that looks complete. Problems are reported at the end and dealt with explicitly
in the processing step, where the choice — interpolate, flag, or drop — is a
modelling decision rather than a download detail.

Re-running is safe: months are rewritten in place, never appended to.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from powerforecast.config import get_settings
from powerforecast.data.epias import ALL_SERIES, EpiasAuth, EpiasClient, SeriesSpec
from powerforecast.data.ingest import (
    DataQualityError,
    fetch_series,
    validate_hourly,
    write_raw,
)

DEFAULT_START = date(2021, 1, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        # Yesterday: today's series is still being published and would land as a
        # short final day that then never gets refilled.
        default=date.today() - timedelta(days=1),
    )
    parser.add_argument(
        "--series",
        nargs="*",
        choices=[spec.name for spec in ALL_SERIES],
        help="Series to download (default: all).",
    )
    return parser.parse_args(argv)


def selected_series(names: list[str] | None) -> tuple[SeriesSpec, ...]:
    if not names:
        return ALL_SERIES
    return tuple(spec for spec in ALL_SERIES if spec.name in names)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    username, password = settings.require_epias()

    auth = EpiasAuth(username=username, password=password)
    problems: list[str] = []

    with EpiasClient(auth) as client:
        for spec in selected_series(args.series):
            print(f"{spec.name}: {args.start} .. {args.end}", flush=True)

            frame = fetch_series(client, spec, args.start, args.end, validate=False)
            files = write_raw(frame, spec)

            print(f"  {len(frame):,} rows -> {len(files)} monthly file(s)", flush=True)

            try:
                validate_hourly(frame, name=spec.name)
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
