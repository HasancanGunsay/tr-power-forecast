"""Tests for rolling-origin backtesting.

The load-bearing tests are the ones about the train/test boundary. A backtest
that leaks does not fail — it reports a better number, which is the outcome
everyone was hoping for anyway.
"""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from powerforecast.evaluation.backtest import (
    rolling_origin_folds,
    run_backtest,
)
from powerforecast.features.availability import forecast_origins


def _index(hours: int) -> pd.DatetimeIndex:
    local = pd.date_range("2026-01-01 00:00", periods=hours, freq="h", tz="Europe/Istanbul")
    return pd.DatetimeIndex(local.tz_convert("UTC"))


def _data(hours: int) -> tuple[pd.DataFrame, pd.Series]:
    index = _index(hours)
    x = pd.DataFrame({"a": np.arange(hours, dtype="float64")}, index=index)
    y = pd.Series(np.arange(hours, dtype="float64") * 2, index=index, name="y")
    return x, y


class RecordingEstimator:
    """A trivial estimator that remembers what it was shown."""

    instances: ClassVar[list[RecordingEstimator]] = []

    def __init__(self) -> None:
        self.train_index: pd.DatetimeIndex | None = None
        RecordingEstimator.instances.append(self)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RecordingEstimator:
        self.train_index = pd.DatetimeIndex(X.index)
        self.coefficient = float((y / X["a"].replace(0, np.nan)).median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["a"] * self.coefficient).to_numpy()


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    RecordingEstimator.instances = []


# --------------------------------------------------------------------------- #
# Fold construction
# --------------------------------------------------------------------------- #


def test_folds_walk_forward_without_overlapping() -> None:
    folds = list(
        rolling_origin_folds(_index(24 * 60), initial_train_hours=24 * 30, test_hours=24 * 7)
    )

    assert len(folds) >= 3
    for earlier, later in pairwise(folds):
        assert later.test_start > earlier.test_end


def test_training_never_reaches_into_the_test_window() -> None:
    folds = list(
        rolling_origin_folds(_index(24 * 60), initial_train_hours=24 * 30, test_hours=24 * 7)
    )

    for fold in folds:
        assert fold.train_end < fold.test_start


def test_training_ends_at_the_origin_of_the_first_test_hour() -> None:
    """The property that replaces a hand-tuned embargo."""
    folds = list(
        rolling_origin_folds(_index(24 * 60), initial_train_hours=24 * 30, test_hours=24 * 7)
    )

    for fold in folds:
        origin = pd.Timestamp(forecast_origins(fold.test[:1]).iloc[0]).tz_convert("UTC")
        # Everything up to the origin is fair game; nothing after it is.
        assert fold.train_end <= origin
        assert fold.train_end > origin - pd.Timedelta(hours=2)


def test_the_gap_is_at_least_the_lead_time() -> None:
    folds = list(
        rolling_origin_folds(_index(24 * 60), initial_train_hours=24 * 30, test_hours=24 * 7)
    )

    for fold in folds:
        gap = fold.test_start - fold.train_end
        # A delivery day beginning at local midnight is bid 13 hours earlier.
        assert gap >= pd.Timedelta(hours=13)


def test_expanding_windows_grow() -> None:
    folds = list(
        rolling_origin_folds(_index(24 * 60), initial_train_hours=24 * 20, test_hours=24 * 7)
    )

    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_sliding_windows_stay_the_same_size() -> None:
    folds = list(
        rolling_origin_folds(
            _index(24 * 60), initial_train_hours=24 * 20, test_hours=24 * 7, expanding=False
        )
    )

    sizes = [len(f.train) for f in folds]
    window = 24 * 20

    # A fixed window is the tool for a series with regime changes, where old
    # data actively misleads rather than merely diluting. The cap is the window;
    # the first fold falls short of it by the lead time, because training stops
    # at the forecast origin rather than at the test boundary.
    assert max(sizes) == window
    assert len(set(sizes[1:])) == 1
    assert window - sizes[0] <= 36


def test_step_controls_how_far_each_fold_advances() -> None:
    kwargs = {"initial_train_hours": 24 * 20, "test_hours": 24 * 7}
    partition = list(rolling_origin_folds(_index(24 * 60), **kwargs))
    overlapping = list(rolling_origin_folds(_index(24 * 60), **kwargs, step_hours=24))

    assert len(overlapping) > len(partition)


def test_too_little_data_is_rejected_with_the_requirement() -> None:
    with pytest.raises(ValueError, match="at least"):
        list(rolling_origin_folds(_index(100), initial_train_hours=90, test_hours=50))


# --------------------------------------------------------------------------- #
# Running a backtest
# --------------------------------------------------------------------------- #


def test_predictions_cover_every_test_hour_exactly_once() -> None:
    x, y = _data(24 * 60)
    folds = list(rolling_origin_folds(x.index, initial_train_hours=24 * 30, test_hours=24 * 7))

    result = run_backtest(RecordingEstimator, x, y, folds)

    expected = pd.DatetimeIndex(np.concatenate([f.test.to_numpy() for f in folds]))
    assert result.predictions.index.is_unique
    assert set(result.predictions.index) == set(expected)


def test_each_fold_gets_its_own_estimator() -> None:
    x, y = _data(24 * 60)
    folds = list(rolling_origin_folds(x.index, initial_train_hours=24 * 30, test_hours=24 * 7))

    run_backtest(RecordingEstimator, x, y, folds)

    # Refitting one object across folds carries warm starts and fitted scalers
    # backwards in time. Nothing in the metrics would show it.
    assert len(RecordingEstimator.instances) == len(folds)


def test_no_estimator_is_shown_data_from_its_own_test_window() -> None:
    x, y = _data(24 * 60)
    folds = list(rolling_origin_folds(x.index, initial_train_hours=24 * 30, test_hours=24 * 7))

    run_backtest(RecordingEstimator, x, y, folds)

    for fold, estimator in zip(folds, RecordingEstimator.instances, strict=True):
        assert estimator.train_index is not None
        assert estimator.train_index.max() < fold.test_start


def test_per_fold_scores_are_reported() -> None:
    x, y = _data(24 * 60)
    folds = list(rolling_origin_folds(x.index, initial_train_hours=24 * 30, test_hours=24 * 7))

    result = run_backtest(RecordingEstimator, x, y, folds)

    # A single aggregate hides a model that works in summer and fails in
    # winter, which is exactly what a long backtest is for.
    assert len(result.folds) == len(folds)
    assert {"MAE", "RMSE", "test_start"} <= set(result.folds.columns)


def test_a_perfect_model_scores_zero() -> None:
    x, y = _data(24 * 60)
    folds = list(rolling_origin_folds(x.index, initial_train_hours=24 * 30, test_hours=24 * 7))

    result = run_backtest(RecordingEstimator, x, y, folds)

    # y is exactly 2*a, so the estimator recovers the relationship exactly.
    assert result.overall.mae == pytest.approx(0.0, abs=1e-6)


def test_rows_with_missing_features_are_skipped_not_imputed() -> None:
    x, y = _data(24 * 60)
    x.iloc[24 * 35 : 24 * 35 + 5, 0] = np.nan
    folds = list(rolling_origin_folds(x.index, initial_train_hours=24 * 30, test_hours=24 * 7))

    result = run_backtest(RecordingEstimator, x, y, folds)

    assert len(result.predictions) < sum(len(f.test) for f in folds)


def test_empty_result_raises_rather_than_returning_nothing() -> None:
    x, y = _data(24 * 60)
    x["a"] = np.nan
    folds = list(rolling_origin_folds(x.index, initial_train_hours=24 * 30, test_hours=24 * 7))

    with pytest.raises(ValueError, match="no fold"):
        run_backtest(RecordingEstimator, x, y, folds)
