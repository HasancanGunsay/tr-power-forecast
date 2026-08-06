"""Run the backtest for every model and print one comparable table.

    uv run python -m powerforecast.analysis.experiment

Everything is scored on the hours the backtest actually produced predictions
for, and the baselines are re-scored on exactly those hours too. Comparing a
model's backtest period against a baseline's full history would be comparing two
different questions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from powerforecast.config import PATHS
from powerforecast.data.panel import load_panel
from powerforecast.evaluation.backtest import rolling_origin_folds, run_backtest
from powerforecast.evaluation.compare import compare_forecasts
from powerforecast.evaluation.metrics import summarize_by
from powerforecast.features.build import FeatureSpec, build_design_matrix
from powerforecast.models.baselines import seasonal_naive
from powerforecast.models.estimators import make_lightgbm, make_ridge

MAPE_FLOOR = 1.0


def run(
    initial_train_days: int = 365 * 3,
    test_days: int = 30,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series]:
    """Backtest every model and return the leaderboard plus the predictions."""
    panel = load_panel()

    folds = list(
        rolling_origin_folds(
            pd.DatetimeIndex(panel.index),
            initial_train_hours=initial_train_days * 24,
            test_hours=test_days * 24,
        )
    )
    print(f"{len(folds)} folds, {sum(len(f.test) for f in folds):,} test hours\n")

    # Weather is missing before 2022 and for a few hundred hours since, so a
    # model that uses it is evaluated on slightly fewer rows. Running both
    # variants and scoring everything on the intersection is what keeps the
    # comparison an answer to one question rather than two.
    variants = {
        "": FeatureSpec(include_weather=False),
        " + weather": FeatureSpec(include_weather=True),
    }

    forecasts: dict[str, pd.Series] = {}

    for suffix, spec in variants.items():
        features, target = build_design_matrix(panel, spec)
        for name, factory in (("ridge", make_ridge), ("lightgbm", make_lightgbm)):
            started = time.perf_counter()
            result = run_backtest(factory, features, target, folds, mape_floor=MAPE_FLOOR)
            forecasts[f"{name}{suffix}"] = result.predictions
            elapsed = time.perf_counter() - started
            print(f"  {name + suffix:22} {elapsed:6.1f}s  {len(result.predictions):,} rows")

    # Score everything on the hours every model could produce. The weather
    # variant is the narrowest, so its index is the common denominator.
    evaluated = forecasts["lightgbm + weather"].index
    for name, forecast in forecasts.items():
        forecasts[name] = forecast.reindex(evaluated)

    # Read the target from the panel rather than from the loop variable: relying
    # on whichever spec happened to run last is the kind of thing that stays
    # correct only until a variant is reordered.
    actual = panel["consumption_mwh"].reindex(evaluated)
    plan = panel["load_plan_mwh"].reindex(evaluated)

    forecasts["official plan"] = plan
    forecasts["naive 168h"] = seasonal_naive(target, season_hours=168).reindex(evaluated)
    forecasts["naive 48h"] = seasonal_naive(target, season_hours=48).reindex(evaluated)

    forecasts.update(_bias_corrected(plan, actual, label="plan"))

    # The same courtesy for our own model. Correcting the competitor's bias but
    # not our own would be a comparison rigged in the opposite direction to the
    # usual one — and our best model has a larger bias than the corrected plan,
    # which is precisely what RMSE punishes.
    best = "lightgbm + weather"
    forecasts.update(_bias_corrected(forecasts[best], actual, label=best))

    leaderboard = compare_forecasts(actual, forecasts, mape_floor=MAPE_FLOOR)
    return leaderboard, forecasts, actual


def _bias_corrected(forecast: pd.Series, actual: pd.Series, *, label: str) -> dict[str, pd.Series]:
    """Remove a forecast's systematic offset, as a one-line competitor.

    The published plan runs systematically low. Adding that shortfall back needs
    no model, no training and no features — so the corrected plan, not the raw
    one, is what a learned model has to beat. A model that beats the raw plan but
    not this has not added skill; it has removed a constant anybody could remove.

    Applied to our own best model as well, for the same reason in reverse. Its
    bias is larger than the corrected plan's, and RMSE punishes exactly that, so
    correcting only the competitor would rig the comparison in the opposite
    direction to the usual one.

    These estimate the correction over the whole evaluation period, which makes
    them *optimistic*: a live system would have to estimate it from history. That
    is deliberate — they exist to be hard to beat, not to be deployed.
    """
    error = actual - forecast
    hours = pd.Series(
        pd.DatetimeIndex(actual.index).tz_convert("Europe/Istanbul").hour, index=actual.index
    )
    return {
        f"{label} + constant bias": forecast + error.mean(),
        f"{label} + hourly bias": forecast + hours.map(error.groupby(hours).mean()),
    }


def figure_model_comparison(
    forecasts: dict[str, pd.Series],
    actual: pd.Series,
    leaderboard: pd.DataFrame,
    directory: Path,
) -> Path:
    hours = pd.Series(
        pd.DatetimeIndex(actual.index).tz_convert("Europe/Istanbul").hour, index=actual.index
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    colours = [
        "#55A868"
        if name == "official plan"
        else "#DD8452"
        if name.startswith("plan +")
        else "#C44E52"
        if "weather" in name
        else "#4C72B0"
        for name in leaderboard.index
    ]
    axes[0].barh(leaderboard.index, leaderboard["MAPE_%"], color=colours)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAPE (%)")
    axes[0].set_title(f"Out-of-sample, {len(actual):,} hours")

    # Four lines, not eleven. The right-hand panel exists to show *where* the
    # error sits across the day; plotting every variant turns it into a smear
    # from which no comparison can be read.
    highlighted = [
        "lightgbm + weather + hourly bias",
        "lightgbm",
        "plan + hourly bias",
        "official plan",
    ]
    for name in highlighted:
        if name not in forecasts:
            continue
        by_hour = summarize_by(actual, forecasts[name], hours, mape_floor=MAPE_FLOOR)["MAPE_%"]
        axes[1].plot(by_hour.index, by_hour, label=name, linewidth=1.8)
    axes[1].set_xlabel("hour of day (Europe/Istanbul)")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title("Error by delivery hour")
    axes[1].legend(fontsize=8)

    fig.suptitle("Learned models against the published plan", fontsize=13)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model_comparison.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def save_predictions(
    forecasts: dict[str, pd.Series],
    actual: pd.Series,
    path: Path,
) -> Path:
    """Persist out-of-sample predictions so diagnostics need not refit.

    The backtest takes minutes; error analysis takes seconds. Keeping them apart
    means a question about *where* the model fails can be asked repeatedly and
    cheaply, which is the difference between investigating and guessing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"actual": actual, **forecasts})
    frame.to_parquet(path, index=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-days", type=int, default=365 * 3)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--out", type=Path, default=PATHS.reports / "figures")
    parser.add_argument("--predictions", type=Path, default=PATHS.reports / "predictions.parquet")
    args = parser.parse_args()

    leaderboard, forecasts, actual = run(args.train_days, args.test_days)

    print("\n" + "=" * 72)
    print(f"OUT-OF-SAMPLE LEADERBOARD  ({len(actual):,} hours)")
    print("=" * 72)
    print(leaderboard.to_string(float_format=lambda v: f"{v:,.1f}"))

    plan = leaderboard.loc["official plan", "MAE"]
    print(f"\nrelative to the published plan (MAE {plan:,.0f}):")
    for name, row in leaderboard.iterrows():
        if name == "official plan":
            continue
        change = 100 * (row["MAE"] - plan) / plan
        verdict = "better" if change < 0 else "worse"
        print(f"  {name:16} {change:+6.1f}%  {verdict}")

    print(f"\nfigure     : {figure_model_comparison(forecasts, actual, leaderboard, args.out)}")
    print(f"predictions: {save_predictions(forecasts, actual, args.predictions)}")


if __name__ == "__main__":
    main()
