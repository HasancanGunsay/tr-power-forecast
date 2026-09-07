"""Download archived day-ahead irradiance and wind forecasts at the generators.

    uv run python -m powerforecast.data.backfill_supply_weather

Mirrors `backfill_weather` deliberately: same year-at-a-time chunking, same
merge-on-write storage, same "written as it lands so an interrupted run keeps
what it has". The differences are the site lists and the two aggregates, and
both are explained in `data.supply_weather`.

Stored as its own dataset rather than joined onto `weather`, because the two
have different site lists and a future change to one should not silently
rewrite the other's files.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from powerforecast.data.backfill_weather import year_chunks
from powerforecast.data.ingest import read_monthly, write_monthly
from powerforecast.data.supply_weather import (
    SOLAR_SITES,
    SOLAR_VARIABLE,
    WIND_SITES,
    WIND_VARIABLE,
    fetch_sites,
    solar_index,
    wind_index,
)
from powerforecast.data.weather import ARCHIVE_START

DATASET = "supply_weather"


def backfill(start: date, end: date) -> pd.DataFrame:
    for chunk_start, chunk_end in year_chunks(start, end):
        solar = fetch_sites(SOLAR_SITES, SOLAR_VARIABLE, chunk_start, chunk_end)
        wind = fetch_sites(WIND_SITES, WIND_VARIABLE, chunk_start, chunk_end)

        # Prefix the per-site columns. İzmir appears in both lists — it has coast
        # and it has sun — and without a prefix the two frames would collide on
        # that one column and silently drop a site.
        frame = pd.concat(
            [solar.add_prefix("ghi_"), wind.add_prefix("wind_")],
            axis=1,
        ).sort_index()
        frame["solar_index"] = solar_index(solar)
        frame["wind_index"] = wind_index(wind)

        written = write_monthly(frame, DATASET)
        coverage = 100 * frame["solar_index"].notna().mean()
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

    print(f"solar: {', '.join(s.name for s in SOLAR_SITES)}")
    print(f"wind : {', '.join(s.name for s in WIND_SITES)}")
    print(f"range: {start} .. {args.end}\n")

    stored = backfill(start, args.end)

    print(f"\nstored {len(stored):,} hours, {stored.index[0]} .. {stored.index[-1]}")
    for column in ("solar_index", "wind_index"):
        missing = stored[column].isna().sum()
        if missing:
            print(f"warning: {missing:,} hours have no {column}")


if __name__ == "__main__":
    main()
