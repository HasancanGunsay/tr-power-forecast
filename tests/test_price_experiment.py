"""Tests for the price experiment runner.

These do not check the model's accuracy — that is what the backtest is for.
They guard the three things that make price a different problem from load, each
of which is a silent failure rather than a crash if it regresses:

* training across the 2021 regime break,
* an unfloored MAPE on a target that reaches exactly zero,
* the price column quietly reverting to the load target.
"""

from __future__ import annotations

import pandas as pd
import pytest

from powerforecast.analysis import price_experiment as px
from powerforecast.evaluation.metrics import summarize
from powerforecast.features.build import FeatureSpec, build_design_matrix


def _panel(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    ramp = pd.Series(range(periods), index=index, dtype="float64")
    return pd.DataFrame(
        {
            "consumption_mwh": 30_000 + ramp,
            "price_try_mwh": 2_000 + ramp,
            "load_plan_mwh": 30_000 + ramp,
            "temperature_c": 15.0,
            "hdd": 0.0,
            "cdd": 0.0,
        },
        index=index,
    )


def test_regime_start_excludes_2021() -> None:
    """2021 is a different price regime, not early history.

    Pinned as a test rather than left to the constant's comment because the
    failure is invisible: training across the break produces a model that runs
    low, not one that raises.
    """
    assert pd.Timestamp(px.REGIME_START) == pd.Timestamp("2022-01-01")


def test_regime_start_filters_the_panel() -> None:
    panel = _panel("2021-11-01", periods=24 * 120)
    kept = panel[panel.index >= pd.Timestamp(px.REGIME_START, tz="UTC")]

    assert kept.index.min() >= pd.Timestamp("2022-01-01", tz="UTC")
    assert len(kept) < len(panel), "the filter must actually drop the 2021 rows"


def test_mape_floor_is_large_enough_to_survive_zero_prices() -> None:
    """The load runner's floor of 1.0 rescues MAPE arithmetically and not
    practically: an hour priced at 3 TRY/MWh still divides by 3.
    """
    assert px.PRICE_MAPE_FLOOR >= 100.0


def test_unfloored_mape_is_useless_on_zero_priced_hours() -> None:
    """The reason the floor exists, stated as an executable fact."""
    index = pd.date_range("2026-05-01", periods=3, freq="h", tz="UTC")
    actual = pd.Series([0.0, 0.0, 2_000.0], index=index)
    forecast = pd.Series([50.0, 50.0, 2_050.0], index=index)

    unfloored = summarize(actual, forecast, mape_floor=None)
    floored = summarize(actual, forecast, mape_floor=px.PRICE_MAPE_FLOOR)

    assert not pd.notna(unfloored.mape) or unfloored.mape > 1_000
    assert floored.mape < 100


def test_feature_spec_targets_price_not_load() -> None:
    """A copy-paste from the load runner would leave this pointing at
    consumption and every number in the table would be a load number.
    """
    panel = _panel("2022-01-01", periods=24 * 40)
    spec = FeatureSpec(target=px.PRICE_COLUMN)
    _, target = build_design_matrix(panel, spec)

    pd.testing.assert_series_equal(target, panel[px.PRICE_COLUMN])


def test_missing_price_column_raises_rather_than_warns() -> None:
    panel = _panel("2022-01-01", periods=24 * 40).drop(columns=[px.PRICE_COLUMN])

    with pytest.raises(KeyError, match=px.PRICE_COLUMN):
        build_design_matrix(panel, FeatureSpec(target=px.PRICE_COLUMN))


def test_baseline_is_the_weekly_lag() -> None:
    """168h beat 48h and 336h when measured; the constant records the winner."""
    assert px.BASELINE_SEASON_HOURS == 168


def test_supply_weather_start_is_pinned() -> None:
    """The supply features cost history, and the trade was measured at this date.

    If the constant drifts, the measured -3.4% no longer describes the model
    the runner actually builds.
    """
    assert pd.Timestamp(px.SUPPLY_WEATHER_START) == pd.Timestamp("2024-02-16")
    assert pd.Timestamp(px.SUPPLY_WEATHER_START) > pd.Timestamp(px.REGIME_START)


def test_supply_features_are_off_for_the_load_target() -> None:
    """Irradiance is a supply signal; the load target is a demand quantity.

    `include_supply_weather` defaults to False precisely so that switching to
    price is a deliberate act rather than something the load runner inherits.
    """
    assert FeatureSpec().include_supply_weather is False
