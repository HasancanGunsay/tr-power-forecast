"""Model factories for the backtest.

Factories, not instances. `run_backtest` builds a fresh estimator per fold, and
handing it a pre-built object would silently carry fitted state across folds.

The ladder is deliberate and each rung has to earn its place:

1. **Ridge** — linear, scaled, regularised. Fast, hard to overfit, and if it
   beats the naive baselines then most of the signal is additive and a more
   complex model is buying very little.
2. **LightGBM** — gradient boosted trees. The usual strongest option on tabular
   time series, because it finds interactions (hot summer *afternoon*, cold
   winter *evening*) that a linear model cannot express without being told.

Both are wrapped so that the same feature matrix works for either: ridge needs
scaling and cannot see raw missing values, trees need neither.
"""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from powerforecast.config import RANDOM_SEED


def make_ridge(alpha: float = 1.0) -> Pipeline:
    """Ridge regression on standardised features.

    Scaling matters here for a reason that is easy to miss: ridge penalises the
    size of coefficients, so a feature measured in tens of thousands of MWh and
    one measured in [-1, 1] are penalised on completely different scales. Without
    standardisation the penalty is effectively arbitrary.

    Fitting the scaler inside the pipeline keeps it inside the fold, which is
    what stops the training statistics being computed over test data.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
        ]
    )


def make_lightgbm(**overrides: Any) -> LGBMRegressor:
    """Gradient boosted trees with conservative defaults.

    The defaults are deliberately unambitious. Tuning against the backtest is
    tempting and is how a project ends up reporting the best of fifty
    configurations as though it were a single honest result. Any tuning should
    happen on an inner split, and the number that goes in the README should come
    from a configuration chosen before it was scored.
    """
    params: dict[str, Any] = {
        "objective": "regression",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(overrides)
    return LGBMRegressor(**params)
