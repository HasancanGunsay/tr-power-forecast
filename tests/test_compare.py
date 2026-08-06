"""Tests for the forecast leaderboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.evaluation.compare import availability, common_index, compare_forecasts


def _series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def test_scores_every_forecast_on_the_same_hours() -> None:
    actual = _series([10.0, 20.0, 30.0, 40.0])
    forecasts = {
        # Perfect, but only on the second half.
        "narrow": _series([np.nan, np.nan, 30.0, 40.0]),
        # Wrong by 10 everywhere, but complete.
        "wide": _series([20.0, 30.0, 40.0, 50.0]),
    }

    table = compare_forecasts(actual, forecasts)

    # If each were scored on its own coverage, "narrow" would look flawless.
    # Restricted to the shared hours, both are judged on the same two points.
    assert (table["n"] == 2).all()
    assert table.loc["narrow", "MAE"] == pytest.approx(0.0)
    assert table.loc["wide", "MAE"] == pytest.approx(10.0)


def test_leaderboard_is_sorted_best_first() -> None:
    actual = _series([10.0, 20.0])
    forecasts = {
        "bad": _series([50.0, 60.0]),
        "good": _series([11.0, 21.0]),
    }

    table = compare_forecasts(actual, forecasts)

    assert list(table.index) == ["good", "bad"]


def test_sort_column_can_be_chosen() -> None:
    actual = _series([10.0, 10.0, 10.0, 10.0])
    forecasts = {
        "spread": _series([12.0, 12.0, 12.0, 12.0]),
        "concentrated": _series([10.0, 10.0, 10.0, 18.0]),
    }

    by_mae = compare_forecasts(actual, forecasts, sort_by="MAE")
    by_rmse = compare_forecasts(actual, forecasts, sort_by="RMSE")

    # Equal MAE, different RMSE: which forecast is "better" depends on whether
    # one large miss costs more than several small ones.
    assert by_mae.loc["spread", "MAE"] == pytest.approx(by_mae.loc["concentrated", "MAE"])
    assert by_rmse.index[0] == "spread"


def test_common_index_excludes_hours_any_forecast_is_missing() -> None:
    actual = _series([1.0, 2.0, 3.0])
    forecasts = {
        "a": _series([1.0, np.nan, 3.0]),
        "b": _series([1.0, 2.0, np.nan]),
    }

    assert len(common_index(actual, forecasts)) == 1


def test_no_shared_hour_raises() -> None:
    actual = _series([1.0, 2.0])
    forecasts = {
        "a": _series([1.0, np.nan]),
        "b": _series([np.nan, 2.0]),
    }

    with pytest.raises(ValueError, match="no hour"):
        compare_forecasts(actual, forecasts)


def test_availability_is_reported_separately_from_accuracy() -> None:
    forecasts = {
        "half": _series([1.0, np.nan, 3.0, np.nan]),
        "full": _series([1.0, 2.0, 3.0, 4.0]),
    }

    table = availability(forecasts)

    # A method that only covers half the delivery day is not interchangeable
    # with one that covers all of it, however well it scores where it exists.
    assert table.loc["half", "coverage_%"] == pytest.approx(50.0)
    assert table.loc["full", "coverage_%"] == pytest.approx(100.0)
