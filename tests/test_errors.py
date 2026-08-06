"""Tests for the error diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.analysis.errors import (
    compare_on_worst,
    error_by,
    error_concentration,
    error_frame,
    worst_days,
)


def _pair(actual: list[float], forecast: list[float], start: str = "2026-03-02 00:00"):
    local = pd.date_range(start, periods=len(actual), freq="h", tz="Europe/Istanbul")
    index = local.tz_convert("UTC")
    return (
        pd.Series(actual, index=index, name="actual"),
        pd.Series(forecast, index=index, name="forecast"),
    )


def test_error_is_forecast_minus_actual() -> None:
    actual, forecast = _pair([100.0, 100.0], [110.0, 90.0])

    errors = error_frame(actual, forecast)

    # Positive means the forecast ran high; the sign has to survive so bias
    # stays interpretable downstream.
    assert list(errors["error"]) == [10.0, -10.0]
    assert list(errors["abs_error"]) == [10.0, 10.0]


def test_calendar_columns_use_local_time() -> None:
    actual, forecast = _pair([100.0], [100.0], start="2026-03-02 00:00")

    errors = error_frame(actual, forecast)

    assert errors["hour"].iloc[0] == 0
    assert errors.index[0].hour == 21  # 21:00 UTC is midnight in Istanbul


def test_only_overlapping_complete_rows_are_kept() -> None:
    actual, forecast = _pair([100.0, 100.0, 100.0], [110.0, np.nan, 90.0])

    assert len(error_frame(actual, forecast)) == 2


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #


def test_evenly_spread_error_is_not_concentrated() -> None:
    n = 100
    actual, forecast = _pair([100.0] * n, [110.0] * n)

    concentration = error_concentration(error_frame(actual, forecast))

    # With identical errors the worst 1% holds 1% of the squared error.
    assert concentration.loc[1.0, "of_squared_error_%"] == pytest.approx(1.0, abs=0.5)


def test_a_few_huge_misses_dominate_squared_error() -> None:
    n = 100
    forecast_values = [100.0] * n
    actual_values = [100.0] * n
    for i in range(n):
        actual_values[i] = 99.0  # small error everywhere
    actual_values[0] = 0.0  # one enormous error
    actual, forecast = _pair(actual_values, forecast_values)

    concentration = error_concentration(error_frame(actual, forecast))

    # This is what a tie on RMSE beside a win on MAE looks like underneath.
    assert concentration.loc[1.0, "of_squared_error_%"] > 90


# --------------------------------------------------------------------------- #
# Worst days
# --------------------------------------------------------------------------- #


def test_worst_days_are_ranked_by_daily_mae() -> None:
    # Three days: the middle one is badly wrong.
    actual = [100.0] * 72
    forecast = [101.0] * 24 + [150.0] * 24 + [102.0] * 24
    a, f = _pair(actual, forecast)

    worst = worst_days(error_frame(a, f), top=3)

    assert worst.index[0] == pd.Timestamp("2026-03-03").date()
    assert worst["mae"].iloc[0] == pytest.approx(50.0)


def test_worst_days_report_direction_and_weekday() -> None:
    actual = [100.0] * 24
    forecast = [80.0] * 24
    a, f = _pair(actual, forecast)

    worst = worst_days(error_frame(a, f), top=1)

    # Knowing the model ran *low* on a day, and which weekday it was, is what
    # turns a bad number into a hypothesis.
    assert worst["bias"].iloc[0] == pytest.approx(-20.0)
    assert worst["weekday"].iloc[0] == "Monday"


def test_error_by_hour_groups_correctly() -> None:
    actual = [100.0] * 48
    forecast = [110.0] * 24 + [120.0] * 24
    a, f = _pair(actual, forecast)

    by_hour = error_by(error_frame(a, f), "hour")

    assert len(by_hour) == 24
    assert by_hour["hours"].eq(2).all()


# --------------------------------------------------------------------------- #
# Tail comparison
# --------------------------------------------------------------------------- #


def test_models_are_compared_on_one_fixed_set_of_hard_hours() -> None:
    n = 200
    actual_values = [100.0] * n
    good = [100.5] * n
    bad = [100.5] * n
    for i in range(n):
        if i < 4:
            good[i] = 200.0  # the reference model's worst hours
            bad[i] = 150.0  # a rival that does better on exactly those hours

    a, f_good = _pair(actual_values, good)
    _, f_bad = _pair(actual_values, bad)

    frames = {"reference": error_frame(a, f_good), "rival": error_frame(a, f_bad)}
    comparison = compare_on_worst(frames, reference="reference", quantile=0.98)

    # Each model's own worst hours are a different set; fixing the set by one
    # model's difficulty is what makes the tail comparable at all.
    assert comparison.index[0] == "rival"
    assert comparison.loc["rival", "MAE"] < comparison.loc["reference", "MAE"]
