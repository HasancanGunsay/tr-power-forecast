"""Tests for the descriptive summaries behind the exploratory figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.analysis.profiles import (
    daily_profile,
    monthly_stats,
    peak_hours,
    tail_summary,
    weekday_weekend_gap,
)


def _panel(values: list[float], start: str = "2026-01-01 00:00") -> pd.DataFrame:
    local = pd.date_range(start, periods=len(values), freq="h", tz="Europe/Istanbul")
    frame = pd.DataFrame({"value": values}, index=local.tz_convert("UTC"))
    frame["local_time"] = local
    frame["hour"] = local.hour
    frame["is_weekend"] = local.dayofweek >= 5
    return frame


def test_daily_profile_averages_each_local_hour() -> None:
    # Two identical days: hour 0 is 10, hour 1 is 20, and so on.
    values = list(np.tile([10.0, 20.0, 30.0, 40.0], 2))
    frame = _panel(values)
    frame["hour"] = [0, 1, 2, 3] * 2

    profile = daily_profile(frame, "value")

    assert profile.loc[0, "value"] == pytest.approx(10.0)
    assert profile.loc[3, "value"] == pytest.approx(40.0)


def test_daily_profile_can_be_split_by_a_column() -> None:
    frame = _panel([10.0, 20.0, 30.0, 40.0])
    frame["hour"] = [0, 1, 0, 1]
    frame["season"] = ["winter", "winter", "summer", "summer"]

    profile = daily_profile(frame, "value", by="season")

    assert profile.loc[0, "winter"] == pytest.approx(10.0)
    assert profile.loc[1, "summer"] == pytest.approx(40.0)


def test_peak_hours_are_ranked_by_average() -> None:
    frame = _panel([5.0, 50.0, 30.0, 5.0])
    frame["hour"] = [0, 1, 2, 3]

    assert peak_hours(frame, "value", top=2) == [1, 2]


def test_weekday_weekend_gap_is_a_fraction_of_the_weekday_level() -> None:
    frame = _panel([100.0, 100.0, 80.0, 80.0])
    frame["is_weekend"] = [False, False, True, True]

    assert weekday_weekend_gap(frame, "value") == pytest.approx(0.2)


def test_monthly_stats_uses_local_months() -> None:
    # 22:00 UTC on 31 January is already February in Istanbul (UTC+3).
    frame = _panel([1.0] * 5, start="2026-01-31 23:00")

    stats = monthly_stats(frame, "value")

    assert list(stats.index.month) == [1, 2]
    assert stats["n"].sum() == 5


def test_monthly_stats_reports_spread_not_just_the_centre() -> None:
    frame = _panel([10.0, 10.0, 10.0, 1000.0])

    stats = monthly_stats(frame, "value")

    # The median ignores the spike while the mean is dragged by it; reporting
    # both is what makes a regime change visible rather than debatable.
    assert stats["median"].iloc[0] == pytest.approx(10.0)
    assert stats["mean"].iloc[0] > stats["median"].iloc[0]
    assert stats["max"].iloc[0] == pytest.approx(1000.0)


def test_tail_summary_flags_a_heavy_upper_tail() -> None:
    # The spike has to be wide enough to reach the 99th percentile: a single
    # outlier in 100 points is interpolated away, which is precisely why p99 is
    # a more stable tail measure than the maximum.
    ordinary = pd.Series([100.0] * 95 + [110.0] * 5)
    spiky = pd.Series([100.0] * 95 + [10_000.0] * 5)

    assert tail_summary(ordinary)["p99_over_median"] < 1.5
    assert tail_summary(spiky)["p99_over_median"] > 50


def test_tail_summary_counts_zero_prices() -> None:
    # Zero-priced hours are exactly the ones that make MAPE undefined, so their
    # share decides whether MAPE can be used on this series at all.
    summary = tail_summary(pd.Series([0.0, 0.0, 100.0, 200.0]))

    assert summary["share_at_zero"] == pytest.approx(0.5)


def test_tail_summary_ignores_nulls() -> None:
    summary = tail_summary(pd.Series([10.0, np.nan, 30.0]))

    assert summary["n"] == 2
    assert summary["median"] == pytest.approx(20.0)
