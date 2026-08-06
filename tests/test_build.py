"""Tests for the design matrix.

The important test here is the last one: a property check that no feature value
could have come from after its own forecast origin. It is the difference between
believing the feature layer is leak-free and demonstrating it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.features.availability import forecast_origins
from powerforecast.features.build import FeatureSpec, build_design_matrix, usable_rows


def _panel(hours: int = 24 * 40) -> pd.DataFrame:
    local = pd.date_range("2026-01-01 00:00", periods=hours, freq="h", tz="Europe/Istanbul")
    index = local.tz_convert("UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "consumption_mwh": np.arange(hours, dtype="float64"),
            "load_plan_mwh": np.arange(hours, dtype="float64") + rng.normal(0, 1, hours),
            "price_try_mwh": rng.normal(2000, 100, hours),
        },
        index=index,
    )


def test_returns_features_and_target_on_one_index() -> None:
    features, target = build_design_matrix(_panel())

    assert features.index.equals(target.index)
    assert target.name == "consumption_mwh"


def test_no_rows_are_dropped() -> None:
    panel = _panel()

    features, _ = build_design_matrix(panel)

    # Which rows are usable depends on the model and the evaluation window, so
    # the decision is left to the caller rather than baked in here.
    assert len(features) == len(panel)


def test_calendar_and_lag_columns_are_both_present() -> None:
    features, _ = build_design_matrix(_panel())

    assert "hour_sin" in features.columns
    assert "consumption_mwh_lag_168h" in features.columns
    assert "consumption_mwh_at_origin" in features.columns
    assert "consumption_mwh_mean_24h_at_origin" in features.columns


def test_load_plan_is_excluded_by_default() -> None:
    features, _ = build_design_matrix(_panel())

    # Including a professional forecast of the target changes the task from
    # "forecast demand" to "correct someone else's forecast". Legitimate, but it
    # has to be a deliberate choice.
    assert "load_plan_mwh" not in features.columns


def test_load_plan_can_be_opted_into() -> None:
    features, _ = build_design_matrix(_panel(), FeatureSpec(include_load_plan=True))

    assert "load_plan_mwh" in features.columns


def test_missing_target_fails_with_the_available_columns() -> None:
    with pytest.raises(KeyError, match="price_eur"):
        build_design_matrix(_panel(), FeatureSpec(target="price_eur"))


def test_usable_rows_excludes_the_warm_up() -> None:
    panel = _panel()
    features, target = build_design_matrix(panel)

    usable = usable_rows(features, target)

    # The 336-hour lag makes the first two weeks unusable. A model trained
    # without noticing this is training on less history than its author thinks.
    assert len(usable) < len(features)
    assert usable[0] >= panel.index[336]


def test_spec_is_immutable() -> None:
    spec = FeatureSpec()

    with pytest.raises(AttributeError):
        spec.target = "price_try_mwh"  # type: ignore[misc]


def test_no_feature_reaches_past_its_own_origin() -> None:
    """Property check across the whole matrix.

    The target is strictly increasing, so its value doubles as a clock: any
    feature derived from it that exceeds the value observed at the origin must
    have read something that had not happened yet.
    """
    panel = _panel()
    features, target = build_design_matrix(panel)

    index = pd.DatetimeIndex(features.index)
    origins = pd.DatetimeIndex(forecast_origins(index)).tz_convert("UTC")
    ceiling = target.reindex(origins).to_numpy()

    derived = [c for c in features.columns if c.startswith("consumption_mwh")]
    assert derived, "expected the matrix to contain target-derived features"

    for column in derived:
        values = features[column].to_numpy()
        observed = ~np.isnan(values) & ~np.isnan(ceiling)
        # `std` columns are dispersions, not levels, so the clock argument does
        # not apply to them.
        if column.endswith("std_24h_at_origin") or column.endswith("std_168h_at_origin"):
            continue
        assert np.all(values[observed] <= ceiling[observed]), f"{column} reads past its origin"
