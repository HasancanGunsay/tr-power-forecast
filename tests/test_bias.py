"""Tests for the bias correction.

Two of these carry the weight.

`test_an_error_from_the_origins_own_day_is_never_used` is the leak. A leak in a
bias correction is particularly nasty because it improves every backtest metric
while being impossible to reproduce in production.

`test_a_stale_model_is_refused` is the condition. The correction is worth −6.3%
under regular retraining and **+2.6%** — worse than nothing — on a model trained
once and never refreshed. That coupling lives in code rather than in a comment,
so it gets a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.forecasts.bias import (
    DEFAULT_WINDOW_DAYS,
    MAX_MODEL_AGE_DAYS,
    BiasOffsets,
    estimate_offsets,
    should_correct,
    usable_error_cutoff,
)

TZ = "Europe/Istanbul"

# Long enough to satisfy the per-hour minimum with room to spare: the default
# window is 119 days and an hourly correction wants 30 samples in every hour.
HISTORY_DAYS = 140


def hourly_index(start: str, days: int) -> pd.DatetimeIndex:
    begin = pd.Timestamp(start, tz=TZ)
    return pd.DatetimeIndex(
        pd.date_range(begin, begin + pd.DateOffset(days=days), freq="h", inclusive="left")
    ).tz_convert("UTC")


def origin_for(delivery_date: str) -> pd.Timestamp:
    """11:00 local on the day before delivery — the rule from ADR 0004."""
    return pd.Timestamp(delivery_date, tz=TZ) - pd.Timedelta(hours=13)


def history(days: int = HISTORY_DAYS, start: str = "2026-01-01") -> pd.DatetimeIndex:
    return hourly_index(start, days)


DELIVERY = "2026-05-25"  # comfortably after 140 days from 1 January


# --------------------------------------------------------------------------- #
# Availability — the leak
# --------------------------------------------------------------------------- #


def test_the_cutoff_is_local_midnight_of_the_origins_day():
    cutoff = usable_error_cutoff(origin_for("2026-03-10"))  # origin 2026-03-09 11:00

    assert cutoff.tz_convert(TZ) == pd.Timestamp("2026-03-09 00:00", tz=TZ)


def test_an_error_from_the_origins_own_day_is_never_used():
    """At 11:00 on D-1, D-1 is still in progress.

    An implementation averaging "everything up to the origin" would pull in that
    morning's errors — real data, wrongly timed — and produce a correction better
    than any live system could.
    """
    index = history()
    errors = pd.Series(0.0, index=index)
    own_day = index[
        (index >= pd.Timestamp("2026-05-24", tz=TZ)) & (index < pd.Timestamp("2026-05-25", tz=TZ))
    ]
    errors.loc[own_day] = 100_000.0

    offsets = estimate_offsets(errors, origin=origin_for(DELIVERY))

    assert offsets.method != "none"
    assert offsets.constant == pytest.approx(0.0)


def test_only_the_window_is_averaged():
    """A regime six months ago must not steer today's correction."""
    old = pd.Series(10_000.0, index=hourly_index("2025-06-01", 60))
    recent = pd.Series(500.0, index=history())

    offsets = estimate_offsets(pd.concat([old, recent]), origin=origin_for(DELIVERY))

    assert offsets.constant == pytest.approx(500.0)
    assert offsets.n_days <= DEFAULT_WINDOW_DAYS


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #


def test_a_constant_offset_is_recovered():
    offsets = estimate_offsets(pd.Series(750.0, index=history()), origin=origin_for(DELIVERY))

    assert offsets.method == "hourly"
    assert offsets.constant == pytest.approx(750.0)
    assert set(offsets.hourly) == set(range(24))
    assert all(value == pytest.approx(750.0) for value in offsets.hourly.values())


def test_an_hourly_pattern_is_recovered_hour_by_hour():
    index = history()
    local_hour = pd.DatetimeIndex(index).tz_convert(TZ).hour
    errors = pd.Series(np.where(local_hour < 12, -400.0, 900.0), index=index, dtype="float64")

    offsets = estimate_offsets(errors, origin=origin_for(DELIVERY))

    assert offsets.method == "hourly"
    assert offsets.hourly[3] == pytest.approx(-400.0)
    assert offsets.hourly[18] == pytest.approx(900.0)


def test_the_statistic_is_a_parameter_rather_than_an_assumption():
    """Mean and median disagree on skewed error, and which one is right was measured."""
    index = history()
    errors = pd.Series(100.0, index=index)
    errors.iloc[-200:] = 10_000.0  # a heavy right tail, inside the window

    by_mean = estimate_offsets(errors, origin=origin_for(DELIVERY), statistic="mean")
    by_median = estimate_offsets(errors, origin=origin_for(DELIVERY), statistic="median")

    assert by_mean.constant > by_median.constant
    assert by_median.constant == pytest.approx(100.0)
    assert by_mean.statistic == "mean" and by_median.statistic == "median"

    with pytest.raises(ValueError, match=r"must be .median. or .mean."):
        estimate_offsets(errors, origin=origin_for(DELIVERY), statistic="mode")


def test_applying_offsets_moves_the_forecast_the_right_way():
    """`error = actual - forecast`, so a positive mean error means we forecast low."""
    offsets = estimate_offsets(pd.Series(600.0, index=history()), origin=origin_for(DELIVERY))

    corrected = offsets.apply(pd.Series(40_000.0, index=hourly_index(DELIVERY, 1)))

    assert corrected.round(6).eq(40_600.0).all()


def test_offsets_are_exposed_so_a_stored_forecast_can_record_them():
    offsets = estimate_offsets(pd.Series(600.0, index=history()), origin=origin_for(DELIVERY))

    applied = offsets.offsets_for(hourly_index(DELIVERY, 1))

    assert len(applied) == 24
    assert applied.round(6).eq(600.0).all()


# --------------------------------------------------------------------------- #
# Degrading rather than guessing
# --------------------------------------------------------------------------- #


def test_thin_hours_fall_back_to_a_constant_and_say_so():
    """An hourly offset fitted on a handful of observations is a confident new error."""
    index = history()
    local = pd.DatetimeIndex(index).tz_convert(TZ)
    errors = pd.Series(500.0, index=index)
    # Hour 4 observed on only three days; every other hour is complete.
    keep = (local.hour != 4) | (local.day <= 3)

    offsets = estimate_offsets(errors[keep], origin=origin_for(DELIVERY))

    assert offsets.method == "constant"
    assert offsets.constant == pytest.approx(500.0)
    assert "below" in offsets.reason


def test_too_little_history_applies_nothing_at_all():
    offsets = estimate_offsets(
        pd.Series(500.0, index=hourly_index("2026-05-20", 2)), origin=origin_for("2026-05-23")
    )

    assert offsets.method == "none"
    assert offsets.constant == 0.0
    assert "below the" in offsets.reason
    assert (
        offsets.apply(pd.Series(40_000.0, index=hourly_index("2026-05-23", 1))).eq(40_000.0).all()
    )


def test_an_empty_history_is_a_no_op_rather_than_an_error():
    """The first days of a deployment have no history, and that is not a failure."""
    offsets = estimate_offsets(
        pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC")),
        origin=origin_for(DELIVERY),
    )

    assert offsets == BiasOffsets(method="none", reason="no error history")


def test_the_reason_is_always_populated():
    """A correction that cannot explain itself is one nobody will check."""
    for errors in (
        pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC")),
        pd.Series(500.0, index=hourly_index("2026-05-20", 2)),
        pd.Series(500.0, index=history()),
    ):
        assert estimate_offsets(errors, origin=origin_for(DELIVERY)).reason


# --------------------------------------------------------------------------- #
# The condition
# --------------------------------------------------------------------------- #


def test_a_fresh_model_is_allowed():
    ok, reason = should_correct("2026-05-01", origin_for(DELIVERY))

    assert ok
    assert "days old" in reason


def test_a_stale_model_is_refused():
    """The measurement: −6.3% when retrained, +2.6% when stale. Encoded, not noted."""
    ok, reason = should_correct("2026-01-01", origin_for(DELIVERY))

    assert not ok
    assert "stale" in reason and str(MAX_MODEL_AGE_DAYS) in reason


def test_a_model_trained_past_the_origin_is_refused():
    """Correcting from its own future would not be a correction."""
    ok, reason = should_correct("2026-06-01", origin_for(DELIVERY))

    assert not ok
    assert "future" in reason


def test_the_age_limit_is_adjustable_but_has_a_default():
    origin = origin_for(DELIVERY)

    assert not should_correct("2026-04-01", origin)[0]  # ~54 days, past the default
    assert should_correct("2026-04-01", origin, max_age_days=90)[0]
