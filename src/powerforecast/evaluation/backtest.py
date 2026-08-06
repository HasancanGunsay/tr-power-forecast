"""Rolling-origin backtesting.

Built before any model, deliberately. Choosing the evaluation after seeing
results is how a project ends up with the scheme that flatters it most, and the
choice is never conscious — it just feels like the earlier one had a flaw.

Two properties matter and both are enforced rather than assumed.

**Training data ends where the forecast begins.** Most implementations bolt on
an "embargo": a hand-tuned gap between the end of training and the start of
testing, there to stop the model learning from something it could not have seen.
That number is unnecessary here. The first test hour has a forecast origin, and
training simply ends at it. The gap falls out of the availability rule instead
of being guessed, and it is automatically correct for whichever hour a fold
happens to start on.

**Every fold gets a fresh model.** The estimator is built by a factory, once per
fold. Refitting one object across folds looks harmless and leaks: warm starts,
retained scalers and cached state all carry information backwards in time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from powerforecast.evaluation.metrics import ErrorSummary, summarize
from powerforecast.features.availability import (
    LAST_OBSERVED_HOUR,
    LOCAL_TZ,
    forecast_origins,
)


class Estimator(Protocol):
    """Minimal fit/predict interface. Any scikit-learn regressor satisfies it."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> object: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


EstimatorFactory = Callable[[], Estimator]


@dataclass(frozen=True)
class Fold:
    """One train/test split, with the boundary that produced it."""

    number: int
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> dict[str, object]:
        return {
            "fold": self.number,
            "train_hours": len(self.train),
            "test_hours": len(self.test),
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def rolling_origin_folds(
    index: pd.DatetimeIndex,
    *,
    initial_train_hours: int,
    test_hours: int,
    step_hours: int | None = None,
    expanding: bool = True,
    last_observed_hour: int = LAST_OBSERVED_HOUR,
    tz: str = LOCAL_TZ,
) -> Iterator[Fold]:
    """Generate chronological folds that walk forward through the series.

    Args:
        index: Complete hourly index of the data.
        initial_train_hours: Size of the first training window.
        test_hours: Length of each test window.
        step_hours: How far to advance between folds. Defaults to `test_hours`,
            which makes the test windows a partition of the evaluation period —
            every hour is scored exactly once.
        expanding: Keep all history (`True`) or slide a fixed window (`False`).
            Expanding is the default because more data usually helps; a sliding
            window is the tool for a series with regime changes, where old data
            actively misleads. A sliding window holds *at most*
            `initial_train_hours`: the first fold is shorter by the lead time,
            because training stops at the origin rather than at the test
            boundary.

    Yields:
        Folds in chronological order. Training always ends at the forecast
        origin of the first test hour, so no fold can train on something that
        was not observable when its first prediction was made.
    """
    index = pd.DatetimeIndex(index).sort_values()
    step = step_hours or test_hours

    if initial_train_hours <= 0 or test_hours <= 0 or step <= 0:
        raise ValueError("initial_train_hours, test_hours and step_hours must be positive")
    if initial_train_hours + test_hours > len(index):
        raise ValueError(
            f"not enough data: need at least {initial_train_hours + test_hours} hours, "
            f"got {len(index)}"
        )

    number = 0
    test_start_position = initial_train_hours

    while test_start_position + test_hours <= len(index):
        test = index[test_start_position : test_start_position + test_hours]

        # The boundary that makes the split honest: training may use anything
        # observed by the moment the first test-day bid was submitted.
        origin = forecast_origins(test[:1], last_observed_hour=last_observed_hour, tz=tz).iloc[0]
        train_end = pd.Timestamp(origin).tz_convert("UTC")

        train = index[index <= train_end]
        if not expanding:
            train = train[-initial_train_hours:]

        if len(train) > 0:
            number += 1
            yield Fold(
                number=number,
                train=train,
                test=test,
                train_end=train[-1],
                test_start=test[0],
                test_end=test[-1],
            )

        test_start_position += step


@dataclass(frozen=True)
class BacktestResult:
    """Predictions and per-fold scores from one backtest."""

    predictions: pd.Series
    folds: pd.DataFrame
    overall: ErrorSummary

    def summary_row(self, name: str) -> dict[str, object]:
        return self.overall.as_row(name)


def run_backtest(
    make_estimator: EstimatorFactory,
    features: pd.DataFrame,
    target: pd.Series,
    folds: Iterator[Fold] | list[Fold],
    *,
    mape_floor: float | None = None,
) -> BacktestResult:
    """Fit and predict across folds, returning out-of-sample predictions.

    Rows with missing features or a missing target are dropped inside each fold
    rather than up front, so the warm-up cost of a long lag reduces the training
    set without silently shrinking the evaluation period as well.
    """
    predictions: list[pd.Series] = []
    reports: list[dict[str, object]] = []

    for fold in folds:
        train_x, train_y = _complete(features, target, fold.train)
        test_x, _ = _complete(features, target, fold.test)

        if train_x.empty or test_x.empty:
            continue

        # A fresh estimator per fold. Reusing one carries fitted state forward,
        # which is leakage that no metric would reveal.
        estimator = make_estimator()
        estimator.fit(train_x, train_y)
        predicted = pd.Series(estimator.predict(test_x), index=test_x.index, name="prediction")
        predictions.append(predicted)

        fold_summary = summarize(target.loc[predicted.index], predicted, mape_floor=mape_floor)
        reports.append({**fold.describe(), **fold_summary.as_row(f"fold_{fold.number}")})

    if not predictions:
        raise ValueError("no fold produced predictions; check feature completeness")

    combined = pd.concat(predictions).sort_index()
    return BacktestResult(
        predictions=combined,
        folds=pd.DataFrame(reports).set_index("fold"),
        overall=summarize(target.loc[combined.index], combined, mape_floor=mape_floor),
    )


def _complete(
    features: pd.DataFrame,
    target: pd.Series,
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series]:
    subset_x = features.loc[features.index.intersection(index)]
    subset_y = target.loc[subset_x.index]
    keep = subset_x.notna().all(axis=1) & subset_y.notna()
    return subset_x[keep], subset_y[keep]
