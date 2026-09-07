"""Tests for the forecast store.

The store is where a bug is most expensive: a row written wrongly today is read
back as truth for months, and the read is what the monitoring layer builds on.
So the cases here are mostly about *not losing anything* — the failure mode this
project has already hit once, in `data/raw`, where a month file was replaced
rather than merged and 49,008 rows became 297 without a single exception.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from powerforecast.forecasts.day import FORECAST_COLUMNS, DayForecast
from powerforecast.forecasts.store import (
    latest_run,
    month_path,
    read_forecasts,
    write_forecasts,
)
from powerforecast.targets import LOAD


def make_day(
    delivery_date: str,
    *,
    version: str = "20260807T070528Z",
    base: float = 40_000.0,
    offset: float = 0.0,
    generated_at: datetime | None = None,
) -> pd.DataFrame:
    """One delivery day in the canonical layout."""
    start = pd.Timestamp(delivery_date, tz="Europe/Istanbul")
    hours = pd.DatetimeIndex(
        pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left")
    ).tz_convert("UTC")
    # `values` is the *corrected* forecast, which is what gets stored; the offset
    # is carried alongside so the raw number stays recoverable.
    raw = base + pd.Series(range(len(hours)), index=hours) * 10.0
    values = pd.Series(raw + offset, index=hours)

    return DayForecast(
        delivery_date=date.fromisoformat(delivery_date),
        forecast_origin=hours[0] - pd.Timedelta(hours=13),
        model_name="load-lightgbm",
        model_version=version,
        generated_at=generated_at or datetime(2026, 8, 7, 9, 0, tzinfo=UTC),
        values=values,
        bias_offset=pd.Series(offset, index=hours, dtype="float64"),
        target=LOAD,
    ).to_frame()


def test_the_applied_offset_travels_with_the_forecast(tmp_path):
    """A correction that cannot be audited later is a correction nobody trusts.

    Storing the offset next to the value means "was this number corrected, and by
    how much?" is answerable from the row — rather than by rerunning an estimator
    against a history that has since changed underneath it.
    """
    write_forecasts(make_day("2026-08-01", offset=250.0), root=tmp_path)

    stored = read_forecasts(root=tmp_path)

    assert (stored["bias_offset"] == 250.0).all()
    # The stored value is the corrected one; the raw forecast is recoverable.
    raw = stored["forecast_value"] - stored["bias_offset"]
    assert raw.iloc[0] == 40_000.0


def test_a_written_day_reads_back_whole(tmp_path):
    rows = write_forecasts(make_day("2026-08-01"), root=tmp_path)
    assert rows == 24

    stored = read_forecasts(root=tmp_path)
    assert len(stored) == 24
    assert list(stored.columns) == list(FORECAST_COLUMNS)
    assert str(stored.index.tz) == "UTC"


def test_a_second_day_is_added_not_substituted(tmp_path):
    """The failure this store exists to prevent: month files being replaced."""
    write_forecasts(make_day("2026-08-05"), root=tmp_path)
    write_forecasts(make_day("2026-08-06"), root=tmp_path)

    assert len(read_forecasts(root=tmp_path)) == 48


def test_rerunning_the_same_day_updates_rather_than_duplicates(tmp_path):
    """A scheduler retries. A job that is not idempotent corrupts data on retry."""
    write_forecasts(make_day("2026-08-05", base=40_000.0), root=tmp_path)
    write_forecasts(
        make_day(
            "2026-08-05",
            base=41_000.0,
            generated_at=datetime(2026, 8, 7, 11, 0, tzinfo=UTC),
        ),
        root=tmp_path,
    )

    stored = read_forecasts(root=tmp_path)
    assert len(stored) == 24
    assert stored["forecast_value"].iloc[0] == 41_000.0  # the newer write won


def test_a_different_model_version_is_kept_alongside(tmp_path):
    """Comparing versions on the same hour is the point of the store."""
    write_forecasts(make_day("2026-08-05", version="v1"), root=tmp_path)
    write_forecasts(make_day("2026-08-05", version="v2", base=41_000.0), root=tmp_path)

    stored = read_forecasts(root=tmp_path)
    assert len(stored) == 48
    assert set(stored["model_version"]) == {"v1", "v2"}
    assert len(read_forecasts(model_version="v2", root=tmp_path)) == 24


def test_a_local_day_that_straddles_two_utc_months_lands_in_both_files(tmp_path):
    """1 August local starts at 21:00 UTC on 31 July — two files, no rows lost."""
    write_forecasts(make_day("2026-08-01"), root=tmp_path)

    assert month_path("2026-07", root=tmp_path).exists()
    assert month_path("2026-08", root=tmp_path).exists()
    assert len(read_forecasts(root=tmp_path)) == 24


def test_latest_run_collapses_versions_to_one_row_per_hour(tmp_path):
    write_forecasts(make_day("2026-08-05", version="v1"), root=tmp_path)
    write_forecasts(
        make_day(
            "2026-08-05",
            version="v2",
            base=41_000.0,
            generated_at=datetime(2026, 8, 7, 12, 0, tzinfo=UTC),
        ),
        root=tmp_path,
    )

    collapsed = latest_run(read_forecasts(root=tmp_path))
    assert len(collapsed) == 24
    assert set(collapsed["model_version"]) == {"v2"}


def test_reading_an_empty_store_returns_a_frame_not_an_error(tmp_path):
    """No forecasts is a real answer to a real question, not a failure."""
    empty = read_forecasts(root=tmp_path)
    assert empty.empty
    assert list(empty.columns) == list(FORECAST_COLUMNS)


def test_a_frame_in_the_wrong_shape_is_refused_before_it_reaches_disk(tmp_path):
    good = make_day("2026-08-01")

    with pytest.raises(ValueError, match="missing columns"):
        write_forecasts(good.drop(columns=["model_version"]), root=tmp_path)

    with pytest.raises(ValueError, match="UTC"):
        write_forecasts(good.tz_convert("Europe/Istanbul"), root=tmp_path)

    with pytest.raises(ValueError, match="empty"):
        write_forecasts(good.iloc[:0], root=tmp_path)


def test_time_filters_narrow_the_read(tmp_path):
    write_forecasts(make_day("2026-08-05"), root=tmp_path)
    write_forecasts(make_day("2026-08-06"), root=tmp_path)

    narrowed = read_forecasts(start="2026-08-05T21:00:00Z", root=tmp_path)
    assert len(narrowed) == 24  # only the 6th local, which begins 21:00 UTC on the 5th
