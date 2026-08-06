"""Tests for the model factories.

Not tests of scikit-learn or LightGBM — those are somebody else's job. What is
checked here is the wiring: that each call produces a *new* estimator, that
scaling happens inside the fold, and that repeated runs give the same answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.models.estimators import make_lightgbm, make_ridge


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 500
    x = pd.DataFrame(
        {
            "small": rng.normal(0, 1, n),
            # Deliberately on a wildly different scale, to exercise the scaler.
            "large": rng.normal(40_000, 5_000, n),
        }
    )
    y = pd.Series(3 * x["small"] + 0.001 * x["large"], name="y")
    return x, y


def test_each_call_returns_a_new_estimator() -> None:
    # run_backtest relies on this: a shared instance would carry fitted state
    # from one fold into the next.
    assert make_ridge() is not make_ridge()
    assert make_lightgbm() is not make_lightgbm()


def test_ridge_scales_inside_the_pipeline(data) -> None:
    x, y = data

    model = make_ridge()
    model.fit(x, y)

    # The scaler must be part of the estimator, not applied beforehand —
    # otherwise its statistics would be computed over the test data too.
    assert "scale" in model.named_steps
    assert model.named_steps["scale"].mean_ is not None


def test_ridge_recovers_a_linear_relationship(data) -> None:
    x, y = data

    model = make_ridge(alpha=0.01)
    model.fit(x, y)

    assert model.score(x, y) > 0.99


def test_ridge_is_deterministic(data) -> None:
    x, y = data

    first = make_ridge().fit(x, y).predict(x)
    second = make_ridge().fit(x, y).predict(x)

    np.testing.assert_allclose(first, second)


def test_lightgbm_is_deterministic(data) -> None:
    x, y = data

    first = make_lightgbm(n_estimators=50).fit(x, y).predict(x)
    second = make_lightgbm(n_estimators=50).fit(x, y).predict(x)

    # A model that cannot be reproduced cannot be defended.
    np.testing.assert_allclose(first, second)


def test_lightgbm_parameters_can_be_overridden() -> None:
    model = make_lightgbm(n_estimators=17, learning_rate=0.3)

    assert model.n_estimators == 17
    assert model.learning_rate == 0.3


def test_both_satisfy_the_backtest_interface(data) -> None:
    x, y = data

    for factory in (make_ridge, make_lightgbm):
        model = factory()
        model.fit(x, y)
        predictions = model.predict(x)
        assert len(predictions) == len(x)
