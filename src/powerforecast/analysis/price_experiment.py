"""Backtest the day-ahead price model and print one comparable table.

    uv run python -m powerforecast.analysis.price_experiment

Separate from `analysis.experiment` rather than a flag on it, because price is
not load with a different column name. Three things differ, and each one would
have to become a branch inside the load runner:

* **There is no operator forecast to beat.** The platform publishes a load plan
  but no price plan, so the strongest honest competitor is a seasonal naive.
  The load project's headline claim — "34% better than the forecast the market
  actually runs on" — has no counterpart here, and pretending otherwise by
  quoting a naive-relative number in the same breath would be dishonest.
* **The target reaches zero.** 0.91% of hours settle at exactly zero, so MAPE
  needs a floor large enough to matter (see `PRICE_MAPE_FLOOR`).
* **The usable history starts later.** 2021 is a different price regime
  entirely (see `REGIME_START`).

What is *not* here is as deliberate as what is. See ADR 0012 for the price cap
and the sliding window, both measured and both rejected.
"""

from __future__ import annotations

import time

import pandas as pd

from powerforecast.data.panel import load_panel
from powerforecast.evaluation.backtest import rolling_origin_folds, run_backtest
from powerforecast.evaluation.compare import compare_forecasts
from powerforecast.features.build import FeatureSpec, build_design_matrix
from powerforecast.models.baselines import seasonal_naive
from powerforecast.models.estimators import make_lightgbm, make_ridge

PRICE_COLUMN = "price_try_mwh"

# 2021 is not an early part of the current series; it is a different regime.
# Median price that year was 400 TRY/MWh against 2,345 the next — a factor of
# nearly six, driven by the 2022 energy shock and the currency, not by anything
# a forecaster could learn from. Training across the break teaches the model a
# level that never returns.
REGIME_START = "2022-01-01"

# MAPE divides by the actual, and 452 hours in the record settle at exactly
# zero. An unfloored MAPE on this series is infinite; a floor of 1.0 (the load
# runner's value) rescues it arithmetically while still letting a single hour
# priced at 3 TRY/MWh dominate the average.
#
# 100 TRY/MWh is chosen as roughly the smallest price anyone would act on
# differently from zero. It is a reporting choice and it is arbitrary in the way
# every MAPE floor is arbitrary, which is why **sMAPE is the scale-free metric
# to quote** for this target. MAPE is kept only for continuity with the load
# tables.
PRICE_MAPE_FLOOR = 100.0

# Seasonal naive at one week. Measured as the strongest of the three naive lags
# (48h: MAE 483, 168h: 428, 336h: 461), which matches the load side: at this
# horizon the weekly rhythm carries more than recency does.
BASELINE_SEASON_HOURS = 168

# Open-Meteo's archived *forecasts* of irradiance and wind begin here, two years
# later than its temperature archive. Turning the supply features on therefore
# costs history, and the two effects were measured separately on identical
# evaluation rows:
#
#   losing 2 years of history     421.3 -> 432.1 MAE  (+2.6%)
#   gaining the supply signal     432.1 -> 406.8 MAE  (-5.8%)
#   net against what came before  421.3 -> 406.8 MAE  (-3.4%)
#
# The signal is worth more than twice the data it costs, so it stays on. Worth
# rechecking as the archive lengthens: the cost term shrinks every month, which
# can only improve the trade.
SUPPLY_WEATHER_START = "2024-02-16"


def run(
    initial_train_days: int = 365,
    test_days: int = 30,
    *,
    regime_start: str = REGIME_START,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series]:
    """Backtest price models and return the leaderboard plus the predictions.

    The training window is **expanding**, not sliding. That is a measured
    choice, not an oversight: sliding windows of 3y, 2y, 18m and 1y all landed
    within 2% of expanding and not in monotonic order, i.e. inside the noise.
    ADR 0012 records the measurement.
    """
    panel = load_panel()
    panel = panel[panel.index >= pd.Timestamp(regime_start, tz="UTC")]

    folds = list(
        rolling_origin_folds(
            pd.DatetimeIndex(panel.index),
            initial_train_hours=initial_train_days * 24,
            test_hours=test_days * 24,
        )
    )
    print(f"{len(folds)} folds, {sum(len(f.test) for f in folds):,} test hours\n")

    variants = {
        "": FeatureSpec(target=PRICE_COLUMN, include_weather=False),
        " + weather": FeatureSpec(target=PRICE_COLUMN, include_weather=True),
        " + weather + supply": FeatureSpec(
            target=PRICE_COLUMN, include_weather=True, include_supply_weather=True
        ),
    }

    forecasts: dict[str, pd.Series] = {}

    for suffix, spec in variants.items():
        features, target = build_design_matrix(panel, spec)
        for name, factory in (("ridge", make_ridge), ("lightgbm", make_lightgbm)):
            started = time.perf_counter()
            result = run_backtest(factory, features, target, folds, mape_floor=PRICE_MAPE_FLOOR)
            forecasts[f"{name}{suffix}"] = result.predictions
            elapsed = time.perf_counter() - started
            print(f"  {name + suffix:22} {elapsed:6.1f}s  {len(result.predictions):,} rows")

    # Score everything on the hours every model could produce, so the table
    # answers one question. The supply variant is now the narrowest: irradiance
    # and wind forecasts only reach back to February 2024, so its folds train on
    # roughly two and a half years where the others get four and a half. That
    # truncation is not a flaw in the comparison, it is the thing being
    # compared — see `SUPPLY_WEATHER_START`.
    evaluated = forecasts["lightgbm + weather + supply"].index
    for name, forecast in forecasts.items():
        forecasts[name] = forecast.reindex(evaluated)

    actual = panel[PRICE_COLUMN].reindex(evaluated)

    for hours in (48, BASELINE_SEASON_HOURS, 336):
        forecasts[f"naive {hours}h"] = seasonal_naive(
            panel[PRICE_COLUMN], season_hours=hours
        ).reindex(evaluated)

    leaderboard = compare_forecasts(actual, forecasts, mape_floor=PRICE_MAPE_FLOOR)
    return leaderboard, forecasts, actual


def main() -> None:
    leaderboard, _, _ = run()
    print()
    print(leaderboard.round(1).to_string())


if __name__ == "__main__":
    main()
