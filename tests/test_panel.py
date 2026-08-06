"""Tests for assembling the three series into one aligned table."""

from __future__ import annotations

import pandas as pd
import pytest

from powerforecast.data.epias import CONSUMPTION, DAY_AHEAD_PRICE, LOAD_PLAN
from powerforecast.data.ingest import to_frame, write_raw
from powerforecast.data.panel import add_local_calendar, load_panel


def _write(spec, start: str, hours: int, field: str, root) -> None:
    stamps = pd.date_range(start, periods=hours, freq="h", tz="Europe/Istanbul")
    items = [
        {"date": ts.isoformat(), "time": ts.strftime("%H:%M"), field: float(i)}
        for i, ts in enumerate(stamps)
    ]
    write_raw(to_frame(items, spec), spec, root=root)


def test_joins_the_series_on_the_shared_index(tmp_path) -> None:
    _write(CONSUMPTION, "2026-03-01 00:00", 48, "consumption", tmp_path)
    _write(DAY_AHEAD_PRICE, "2026-03-01 00:00", 48, "price", tmp_path)
    _write(LOAD_PLAN, "2026-03-01 00:00", 48, "lep", tmp_path)

    panel = load_panel(root=tmp_path)

    assert list(panel.columns) == ["consumption_mwh", "price_try_mwh", "load_plan_mwh"]
    assert len(panel) == 48
    assert str(panel.index.tz) == "UTC"


def test_a_gap_in_one_series_becomes_a_null_not_a_dropped_row(tmp_path) -> None:
    _write(CONSUMPTION, "2026-03-01 00:00", 48, "consumption", tmp_path)
    _write(DAY_AHEAD_PRICE, "2026-03-01 00:00", 48, "price", tmp_path)
    _write(LOAD_PLAN, "2026-03-01 00:00", 24, "lep", tmp_path)  # one day short

    panel = load_panel(root=tmp_path)

    # Dropping the row would throw away consumption and price we do have; the
    # unpublished load plan is a hole in one column, not in the hour itself.
    assert len(panel) == 48
    assert panel["load_plan_mwh"].isna().sum() == 24
    assert panel["consumption_mwh"].notna().all()


def test_the_index_is_a_complete_hourly_grid(tmp_path) -> None:
    _write(CONSUMPTION, "2026-03-01 00:00", 3, "consumption", tmp_path)
    _write(DAY_AHEAD_PRICE, "2026-03-01 05:00", 3, "price", tmp_path)

    panel = load_panel((CONSUMPTION, DAY_AHEAD_PRICE), root=tmp_path)

    # The two series do not overlap in time at all. Reindexing onto a continuous
    # grid keeps the missing hours visible instead of producing a series with a
    # silent jump in it.
    expected = pd.date_range(panel.index[0], panel.index[-1], freq="h", tz="UTC")
    assert panel.index.equals(expected)


def test_missing_series_raises_with_a_useful_message(tmp_path) -> None:
    _write(CONSUMPTION, "2026-03-01 00:00", 24, "consumption", tmp_path)

    with pytest.raises(FileNotFoundError, match="day_ahead_price"):
        load_panel(root=tmp_path)


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #


def test_calendar_columns_use_local_time_not_utc(tmp_path) -> None:
    _write(CONSUMPTION, "2026-03-01 00:00", 24, "consumption", tmp_path)
    _write(DAY_AHEAD_PRICE, "2026-03-01 00:00", 24, "price", tmp_path)
    _write(LOAD_PLAN, "2026-03-01 00:00", 24, "lep", tmp_path)

    panel = add_local_calendar(load_panel(root=tmp_path))

    # Storage is UTC, but consumption follows human routine: people wake, work
    # and cook on local time. An "hour" feature in UTC would smear the evening
    # peak across two different hours of the day.
    first = panel.iloc[0]
    assert first["hour"] == 0
    assert panel.index[0].hour == 21  # 21:00 UTC == 00:00 in Istanbul


def test_weekend_flag_follows_the_local_date(tmp_path) -> None:
    # 2026-03-07 is a Saturday.
    _write(CONSUMPTION, "2026-03-06 00:00", 72, "consumption", tmp_path)
    _write(DAY_AHEAD_PRICE, "2026-03-06 00:00", 72, "price", tmp_path)
    _write(LOAD_PLAN, "2026-03-06 00:00", 72, "lep", tmp_path)

    panel = add_local_calendar(load_panel(root=tmp_path))
    by_date = panel.groupby("local_date")["is_weekend"].first()

    assert not by_date[pd.Timestamp("2026-03-06").date()]
    assert by_date[pd.Timestamp("2026-03-07").date()]
    assert by_date[pd.Timestamp("2026-03-08").date()]
