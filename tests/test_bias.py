"""Tests for the deployable bias correction.

The one that matters is `test_an_error_from_the_origins_own_day_is_never_used`.
Everything else here is arithmetic; that one is the leak, and a leak in a bias
correction is particularly nasty because it improves every backtest metric while
being impossible to reproduce in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.forecasts.bias import (
    BiasOffsets,
    estimate_offsets,
    usable_error_cutoff,
)

TZ = "Europe/Istanbul"


def hourly_index(start: str, days: int) -> pd.DatetimeIndex:
    begin = pd.Timestamp(start, tz=TZ)
    return pd.DatetimeIndex(
        pd.date_range(begin, begin + pd.DateOffset(days=days), freq="h", inclusive="left")
    ).tz_convert("UTC")


def origin_for(delivery_date: str) -> pd.Timestamp:
    """11:00 local on the day before delivery — the rule from ADR 0004."""
    return pd.Timestamp(delivery_date, tz=TZ) - pd.Timedelta(hours=13)


def test_the_cutoff_is_local_midnight_of_the_origins_day():
    origin = origin_for("2026-03-10")  # 2026-03-09 11:00 local

    cutoff = usable_error_cutoff(origin)

    assert cutoff.tz_convert(TZ) == pd.Timestamp("2026-03-09 00:00", tz=TZ)


def test_an_error_from_the_origins_own_day_is_never_used():
    """The leak. At 11:00 on D-1, D-1 is still in progress.

    An implementation that averaged "everything up to the origin" would pull in
    that morning's errors — real data, wrongly timed — and the correction would
    be better than any live system could produce.
    """
    index = hourly_index("2026-03-01", 12)
    errors = pd.Series(0.0, index=index)
    # A single enormous error inside the origin's own day. If it leaks, it moves
    # the constant by thousands; if it does not, the constant stays at zero.
    poisoned = index[
        (index >= pd.Timestamp("2026-03-09", tz=TZ)) & (index < pd.Timestamp("2026-03-10", tz=TZ))
    ]
    errors.loc[poisoned] = 100_000.0

    offsets = estimate_offsets(errors, origin=origin_for("2026-03-10"))

    assert offsets.method != "none"
    assert offsets.constant == pytest.approx(0.0)


def test_a_constant_offset_is_recovered():
    errors = pd.Series(750.0, index=hourly_index("2026-03-01", 20))

    offsets = estimate_offsets(errors, origin=origin_for("2026-03-21"))

    assert offsets.method == "hourly"
    assert offsets.constant == pytest.approx(750.0)
    assert set(offsets.hourly) == set(range(24))
    assert all(value == pytest.approx(750.0) for value in offsets.hourly.values())


def test_an_hourly_pattern_is_recovered_hour_by_hour():
    index = hourly_index("2026-03-01", 20)
    local_hour = pd.DatetimeIndex(index).tz_convert(TZ).hour
    errors = pd.Series(np.where(local_hour < 12, -400.0, 900.0), index=index, dtype="float64")

    offsets = estimate_offsets(errors, origin=origin_for("2026-03-21"))

    assert offsets.method == "hourly"
    assert offsets.hourly[3] == pytest.approx(-400.0)
    assert offsets.hourly[18] == pytest.approx(900.0)


def test_applying_offsets_moves_the_forecast_the_right_way():
    """`error = actual - forecast`, so a positive mean error means we under-forecast."""
    index = hourly_index("2026-04-01", 1)
    errors = pd.Series(600.0, index=hourly_index("2026-03-01", 20))
    offsets = estimate_offsets(errors, origin=origin_for("2026-03-21"))

    corrected = offsets.apply(pd.Series(40_000.0, index=index))

    assert corrected.round(6).eq(40_600.0).all()


def test_offsets_are_exposed_so_a_stored_forecast_can_record_them():
    index = hourly_index("2026-04-01", 1)
    errors = pd.Series(600.0, index=hourly_index("2026-03-01", 20))

    applied = estimate_offsets(errors, origin=origin_for("2026-03-21")).offsets_for(index)

    assert len(applied) == 24
    assert applied.round(6).eq(600.0).all()


def test_only_the_window_is_averaged():
    """A regime change six months ago must not steer today's correction."""
    old = pd.Series(10_000.0, index=hourly_index("2026-01-01", 30))
    recent = pd.Series(500.0, index=hourly_index("2026-03-01", 20))

    offsets = estimate_offsets(
        pd.concat([old, recent]), origin=origin_for("2026-03-21"), window_days=28
    )

    assert offsets.constant == pytest.approx(500.0)


def test_thin_hours_fall_back_to_a_constant_and_say_so():
    """An hourly correction fitted on three observations is a confident new error."""
    index = hourly_index("2026-03-01", 20)
    local_hour = pd.DatetimeIndex(index).tz_convert(TZ).hour
    errors = pd.Series(500.0, index=index)
    # Hour 4 observed on only two days; every other hour is complete.
    keep = (local_hour != 4) | (pd.DatetimeIndex(index).tz_convert(TZ).day <= 2)

    offsets = estimate_offsets(errors[keep], origin=origin_for("2026-03-21"))

    assert offsets.method == "constant"
    assert offsets.constant == pytest.approx(500.0)
    assert "below" in offsets.reason


def test_too_little_history_applies_nothing_at_all():
    errors = pd.Series(500.0, index=hourly_index("2026-03-05", 2))

    offsets = estimate_offsets(errors, origin=origin_for("2026-03-08"))

    assert offsets.method == "none"
    assert offsets.constant == 0.0
    assert "below the" in offsets.reason

    unchanged = offsets.apply(pd.Series(40_000.0, index=hourly_index("2026-03-08", 1)))
    assert (unchanged == 40_000.0).all()


def test_an_empty_history_is_a_no_op_rather_than_an_error():
    """The first day of a deployment has no history, and that is not a failure."""
    offsets = estimate_offsets(
        pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC")),
        origin=origin_for("2026-03-08"),
    )

    assert offsets == BiasOffsets(method="none", reason="no error history")


def test_the_reason_is_always_populated():
    """A correction that cannot explain itself is one nobody will trust or check."""
    cases = [
        pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC")),
        pd.Series(500.0, index=hourly_index("2026-03-05", 2)),
        pd.Series(500.0, index=hourly_index("2026-03-01", 20)),
    ]
    for errors in cases:
        assert estimate_offsets(errors, origin=origin_for("2026-03-21")).reason
