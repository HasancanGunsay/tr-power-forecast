"""Tests for forecast error metrics.

Most of these guard against metrics that look fine on clean data and mislead on
real data: MAPE exploding near zero, RMSE and MAE disagreeing about which
forecast is better, and a bias that cancels itself out in the average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.evaluation.metrics import (
    ErrorSummary,
    bias,
    mae,
    mape,
    rmse,
    smape,
    summarize,
)


def _series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def test_perfect_forecast_scores_zero() -> None:
    actual = _series([10.0, 20.0, 30.0])

    assert mae(actual, actual) == 0.0
    assert rmse(actual, actual) == 0.0
    assert smape(actual, actual) == 0.0
    assert bias(actual, actual) == 0.0


def test_mae_is_the_mean_absolute_error() -> None:
    actual = _series([10.0, 20.0])
    forecast = _series([12.0, 16.0])

    assert mae(actual, forecast) == pytest.approx(3.0)


def test_rmse_punishes_a_single_large_error_more_than_mae() -> None:
    actual = _series([10.0, 10.0, 10.0, 10.0])
    spread = _series([12.0, 12.0, 12.0, 12.0])  # four errors of 2
    concentrated = _series([10.0, 10.0, 10.0, 18.0])  # one error of 8

    # Same MAE, very different RMSE. Which one you optimise is a business
    # decision: in a market with punitive imbalance settlement, one huge miss
    # can cost more than many small ones.
    assert mae(actual, spread) == pytest.approx(mae(actual, concentrated))
    assert rmse(actual, concentrated) > rmse(actual, spread)


def test_bias_separates_over_from_under_forecasting() -> None:
    actual = _series([10.0, 10.0])
    over = _series([12.0, 13.0])

    # Positive bias means the forecast runs high. MAE hides the direction, and
    # direction is what tells you whether a model is systematically wrong.
    assert bias(actual, over) == pytest.approx(2.5)
    assert bias(actual, _series([8.0, 7.0])) == pytest.approx(-2.5)


def test_bias_cancels_out_while_mae_does_not() -> None:
    actual = _series([10.0, 10.0])
    forecast = _series([15.0, 5.0])

    # This is why bias is never reported alone.
    assert bias(actual, forecast) == pytest.approx(0.0)
    assert mae(actual, forecast) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# MAPE — the metric that quietly breaks on price data
# --------------------------------------------------------------------------- #


def test_mape_is_a_percentage_of_the_actual() -> None:
    actual = _series([100.0, 200.0])
    forecast = _series([110.0, 180.0])

    assert mape(actual, forecast) == pytest.approx(10.0)


def test_mape_excludes_rows_below_the_floor() -> None:
    actual = _series([100.0, 0.0, 100.0])
    forecast = _series([110.0, 50.0, 90.0])

    # Dividing by zero would give inf and poison the mean. Excluding the row is
    # honest as long as the exclusion is reported — see mape_coverage below.
    assert mape(actual, forecast, floor=1.0) == pytest.approx(10.0)


def test_mape_without_a_floor_is_undefined_when_actuals_reach_zero() -> None:
    actual = _series([100.0, 0.0])
    forecast = _series([110.0, 50.0])

    assert np.isnan(mape(actual, forecast))


def test_smape_stays_bounded_where_mape_explodes() -> None:
    actual = _series([1.0])
    forecast = _series([100.0])

    # MAPE reports 9900%; sMAPE is capped at 200% and stays comparable across
    # series with different scales.
    assert mape(actual, forecast) == pytest.approx(9900.0)
    assert smape(actual, forecast) <= 200.0


# --------------------------------------------------------------------------- #
# Alignment and missing data
# --------------------------------------------------------------------------- #


def test_only_overlapping_timestamps_are_compared() -> None:
    actual = _series([10.0, 20.0, 30.0])
    # Overlaps the last two hours only; the first actual has no counterpart.
    forecast = _series([22.0, 32.0], start="2026-01-01 01:00")

    summary = summarize(actual, forecast)

    assert summary.n == 2
    assert summary.mae == pytest.approx(2.0)


def test_rows_with_a_missing_value_are_dropped_and_counted() -> None:
    actual = _series([10.0, 20.0, 30.0])
    forecast = _series([12.0, np.nan, 33.0])

    summary = summarize(actual, forecast)

    # The load plan has a genuinely unpublished day; silently treating it as
    # zero would understate the error, and imputing it would invent a result.
    assert summary.n == 2
    assert summary.dropped == 1


def test_no_overlap_raises_rather_than_returning_nan() -> None:
    actual = _series([10.0])
    forecast = _series([10.0], start="2027-01-01")

    with pytest.raises(ValueError, match="overlap"):
        summarize(actual, forecast)


def test_summary_reports_mape_coverage() -> None:
    actual = _series([100.0, 0.0, 100.0, 100.0])
    forecast = _series([110.0, 5.0, 90.0, 100.0])

    summary = summarize(actual, forecast, mape_floor=1.0)

    # A MAPE computed on three quarters of the data must say so.
    assert summary.mape_coverage == pytest.approx(0.75)


def test_summary_is_immutable() -> None:
    summary = summarize(_series([10.0, 20.0]), _series([11.0, 19.0]))

    assert isinstance(summary, ErrorSummary)
    with pytest.raises(AttributeError):
        summary.mae = 0.0  # type: ignore[misc]
