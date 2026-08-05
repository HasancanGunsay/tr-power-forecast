"""Tests for turning API responses into a validated hourly series.

The failures worth guarding against here are quiet ones. A duplicated hour or a
missing hour does not raise anything on its own — it silently shifts every lag
feature built on top of it and biases the backtest that follows. So the checks
have to be explicit, and they have to run before the data is ever written.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pandas as pd
import pytest

from powerforecast.data.epias import CONSUMPTION, DAY_AHEAD_PRICE
from powerforecast.data.ingest import (
    DataQualityError,
    month_chunks,
    read_raw,
    to_frame,
    validate_hourly,
    write_raw,
)


def _items(start: str, hours: int, field: str = "consumption") -> list[dict[str, object]]:
    stamps = pd.date_range(start, periods=hours, freq="h", tz="Europe/Istanbul")
    return [
        {"date": ts.isoformat(), "time": ts.strftime("%H:%M"), field: 100.0 + i}
        for i, ts in enumerate(stamps)
    ]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def test_single_month_is_one_chunk() -> None:
    assert month_chunks(date(2026, 3, 1), date(2026, 3, 31)) == [
        (date(2026, 3, 1), date(2026, 3, 31))
    ]


def test_partial_months_keep_the_requested_edges() -> None:
    chunks = month_chunks(date(2026, 1, 15), date(2026, 3, 10))

    assert chunks == [
        (date(2026, 1, 15), date(2026, 1, 31)),
        (date(2026, 2, 1), date(2026, 2, 28)),
        (date(2026, 3, 1), date(2026, 3, 10)),
    ]


def test_chunks_never_overlap() -> None:
    # Overlapping chunks would silently duplicate rows: the API treats endDate as
    # inclusive, so an off-by-one here reappears as duplicated hours later.
    chunks = month_chunks(date(2024, 11, 5), date(2026, 2, 20))

    for (_, first_end), (second_start, _) in pairwise(chunks):
        assert second_start > first_end


def test_rejects_a_reversed_range() -> None:
    with pytest.raises(ValueError, match="before"):
        month_chunks(date(2026, 5, 1), date(2026, 4, 1))


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_builds_a_utc_indexed_frame() -> None:
    frame = to_frame(_items("2026-08-01 00:00", 3), CONSUMPTION)

    assert list(frame.columns) == ["consumption_mwh"]
    assert frame.index.name == "timestamp"
    assert str(frame.index.tz) == "UTC"
    # 00:00 in Istanbul (UTC+3) is 21:00 UTC on the previous day.
    assert frame.index[0] == pd.Timestamp("2026-07-31 21:00", tz="UTC")


def test_drops_the_redundant_time_field() -> None:
    # `time` restates the hour already present in `date`; carrying it forward
    # invites the two disagreeing after a timezone conversion.
    frame = to_frame(_items("2026-08-01 00:00", 2), CONSUMPTION)

    assert "time" not in frame.columns


def test_reads_the_field_named_by_the_spec() -> None:
    items = _items("2026-08-01 00:00", 2, field="price")
    frame = to_frame(items, DAY_AHEAD_PRICE)

    assert list(frame.columns) == ["price_try_mwh"]
    assert frame["price_try_mwh"].iloc[0] == 100.0


def test_empty_response_gives_an_empty_typed_frame() -> None:
    frame = to_frame([], CONSUMPTION)

    assert frame.empty
    assert list(frame.columns) == ["consumption_mwh"]


def test_missing_value_field_is_reported_with_the_field_name() -> None:
    with pytest.raises(DataQualityError, match="consumption"):
        to_frame([{"date": "2026-08-01T00:00:00+03:00", "time": "00:00"}], CONSUMPTION)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_a_clean_series_passes() -> None:
    validate_hourly(to_frame(_items("2026-08-01 00:00", 48), CONSUMPTION), name="consumption")


def test_duplicate_hours_are_rejected() -> None:
    items = _items("2026-08-01 00:00", 3)
    items.append(items[1])  # the API returned the same hour twice

    with pytest.raises(DataQualityError, match="duplicate"):
        validate_hourly(to_frame(items, CONSUMPTION), name="consumption")


def test_missing_hours_are_rejected_and_named() -> None:
    items = _items("2026-08-01 00:00", 5)
    del items[2]

    with pytest.raises(DataQualityError, match="2026-07-31 23:00"):
        validate_hourly(to_frame(items, CONSUMPTION), name="consumption")


def test_null_values_are_rejected() -> None:
    items = _items("2026-08-01 00:00", 3)
    items[1]["consumption"] = None

    with pytest.raises(DataQualityError, match="null"):
        validate_hourly(to_frame(items, CONSUMPTION), name="consumption")


def test_validation_of_an_empty_frame_fails_loudly() -> None:
    # An empty result usually means a wrong date range or a silently failed
    # request. Writing it would poison the dataset with a gap.
    with pytest.raises(DataQualityError, match="empty"):
        validate_hourly(to_frame([], CONSUMPTION), name="consumption")


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def test_raw_data_round_trips_through_parquet(tmp_path) -> None:
    frame = to_frame(_items("2026-08-01 00:00", 72), CONSUMPTION)

    write_raw(frame, CONSUMPTION, root=tmp_path)
    restored = read_raw(CONSUMPTION, root=tmp_path)

    pd.testing.assert_frame_equal(frame, restored)


def test_raw_data_is_partitioned_by_month(tmp_path) -> None:
    frame = to_frame(_items("2026-01-30 00:00", 24 * 5), CONSUMPTION)

    write_raw(frame, CONSUMPTION, root=tmp_path)

    written = sorted(p.name for p in (tmp_path / "consumption").glob("*.parquet"))
    # Monthly files keep a re-fetch of one month from rewriting the whole series.
    assert written == ["2026-01.parquet", "2026-02.parquet"]


def test_rewriting_the_same_data_is_idempotent(tmp_path) -> None:
    first = to_frame(_items("2026-08-01 00:00", 24), CONSUMPTION)
    write_raw(first, CONSUMPTION, root=tmp_path)
    write_raw(first, CONSUMPTION, root=tmp_path)

    restored = read_raw(CONSUMPTION, root=tmp_path)
    assert len(restored) == 24


def test_a_partial_write_does_not_truncate_the_month(tmp_path) -> None:
    # This is the shape of a real bug: chunks are requested in Istanbul local
    # time but stored per UTC month, so consecutive chunks each land a few hours
    # in the neighbour's file. Replacing the file outright would leave only
    # those few hours behind.
    full_month = to_frame(_items("2026-06-01 03:00", 700), CONSUMPTION)
    write_raw(full_month, CONSUMPTION, root=tmp_path)

    spillover = to_frame(_items("2026-06-30 22:00", 5), CONSUMPTION)
    write_raw(spillover, CONSUMPTION, root=tmp_path)

    restored = read_raw(CONSUMPTION, root=tmp_path)
    assert len(restored) == 705
    assert restored.index.is_monotonic_increasing


def test_refetched_values_win_over_stored_ones(tmp_path) -> None:
    original = to_frame(_items("2026-08-01 00:00", 3), CONSUMPTION)
    write_raw(original, CONSUMPTION, root=tmp_path)

    revised = original.copy()
    revised.iloc[1, 0] = 999.0
    write_raw(revised, CONSUMPTION, root=tmp_path)

    # The platform revises published figures; a stale value must not survive.
    restored = read_raw(CONSUMPTION, root=tmp_path)
    assert restored.iloc[1, 0] == 999.0
    assert len(restored) == 3


def test_reading_an_absent_series_gives_an_empty_frame(tmp_path) -> None:
    assert read_raw(CONSUMPTION, root=tmp_path).empty
