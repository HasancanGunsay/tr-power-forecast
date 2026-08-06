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
    features, target = build_design_matrix(panel, FeatureSpec())

    folds = list(
        rolling_origin_folds(
            pd.DatetimeIndex(panel.index),
            initial_train_hours=initial_train_days * 24,
            test_hours=test_days * 24,
        )
    )
    print(f"{len(folds)} folds, {sum(len(f.test) for f in folds):,} test hours\n")

    forecasts: dict[str, pd.Series] = {}

    for name, factory in (("ridge", make_ridge), ("lightgbm", make_lightgbm)):
        started = time.perf_counter()
        result = run_backtest(factory, features, target, folds, mape_floor=MAPE_FLOOR)
        forecasts[name] = result.predictions
        print(
            f"  {name:10} {time.perf_counter() - started:6.1f}s  {len(result.predictions):,} rows"
        )

    # Score the baselines on the same hours the models were scored on.
    evaluated = forecasts["ridge"].index
    actual = target.reindex(evaluated)
    plan = panel["load_plan_mwh"].reindex(evaluated)

    forecasts["official plan"] = plan
    forecasts["naive 168h"] = seasonal_naive(target, season_hours=168).reindex(evaluated)
    forecasts["naive 48h"] = seasonal_naive(target, season_hours=48).reindex(evaluated)
    forecasts.update(_bias_corrected(plan, actual))

    leaderboard = compare_forecasts(target.reindex(evaluated), forecasts, mape_floor=MAPE_FLOOR)
    return leaderboard, forecasts, target.reindex(evaluated)


def _bias_corrected(plan: pd.Series, actual: pd.Series) -> dict[str, pd.Series]:
    """The corrections a learned model has to justify itself against.

    The published plan runs systematically low. Adding that shortfall back is a
    one-line change with no model, no training and no features — so it is the
    real competitor, not the raw plan. A model that beats the raw plan but not
    this is not adding skill, it is removing a constant that anybody could have
    removed.

    These use the whole evaluation period to estimate the correction, which
    means they are *optimistic*: a live system would have to estimate it from
    history. That bias is deliberate — the point is to make them hard to beat,
    not to propose them as a deployable forecast.
    """
    error = actual - plan
    hours = pd.Series(
        pd.DatetimeIndex(actual.index).tz_convert("Europe/Istanbul").hour, index=actual.index
    )
    return {
        "plan + constant bias": plan + error.mean(),
        "plan + hourly bias": plan + hours.map(error.groupby(hours).mean()),
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
        else "#4C72B0"
        for name in leaderboard.index
    ]
    axes[0].barh(leaderboard.index, leaderboard["MAPE_%"], color=colours)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAPE (%)")
    axes[0].set_title(f"Out-of-sample, {len(actual):,} hours")

    for name, forecast in forecasts.items():
        by_hour = summarize_by(actual, forecast, hours, mape_floor=MAPE_FLOOR)["MAPE_%"]
        axes[1].plot(by_hour.index, by_hour, label=name)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-days", type=int, default=365 * 3)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--out", type=Path, default=PATHS.reports / "figures")
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

    print(f"\nfigure: {figure_model_comparison(forecasts, actual, leaderboard, args.out)}")


if __name__ == "__main__":
    main()
