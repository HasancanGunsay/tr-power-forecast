"""Does a *deployable* bias correction recover the backtest's advantage?

    uv run python -m powerforecast.analysis.bias_correction

The backtest's headline model applies a correction estimated over the whole
evaluation period. That is a legitimate competitor — it is meant to be hard to
beat — and it is not shippable, because it sees the future. The deployed service
therefore carries no correction, and on the first genuinely out-of-sample window
it lost to the published plan.

This measures whether `forecasts.bias`, which only ever looks backwards, closes
that gap. The evaluation is **prequential**: the model is fitted once to a cut-off
date, then every delivery day after it is forecast in order, and the correction
applied to day D is estimated only from errors that were observable at D's
forecast origin. Day D's own error then joins the history for day D+1. No day is
scored with information from its own future.

Several variants are compared on identical hours, against the published plan.
Whatever the answer is, it goes in the README: a correction that does not help is
a result, and a correction adopted without measuring would be a decoration.

The answer, on 4,439 hours from February to August 2026, is that it barely helps
— and that *which* statistic is used decides the sign:

| forecast | MAE | RMSE |
|---|---|---|
| + hourly offset, **median** | **907.8** | 1,223.4 |
| + constant offset, **median** | 909.9 | **1,217.7** |
| raw, no correction | 914.5 | 1,233.6 |
| + hourly offset, mean | 918.1 | 1,230.4 |
| + constant offset, mean | 923.8 | 1,230.9 |
| published plan | 1,140.5 | 1,530.8 |

The mean makes MAE worse while improving RMSE, which is exactly what the two
metrics are defined to do: the mean minimises squared error, the median minimises
absolute error. Correcting with the mean optimises the metric this project does
not headline at the cost of the one it does.

Under one percent is the honest size of the win. The gap between the deployed
model and the backtest's corrected variant was never mostly a bias problem — the
backtest's correction looks powerful *because* it sees the whole period, not
because bias correction is powerful.

The other thing this run settles: **raw already beats the published plan by 19.8%
over these six months.** An earlier check on a single 30-day window in July had
put the deployed model behind the plan, and that reading does not survive a
longer window. July is a hard month for this model and an easy one for the plan;
one month is not a verdict.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from powerforecast.config import PATHS
from powerforecast.data.panel import load_panel
from powerforecast.evaluation.metrics import summarize
from powerforecast.features.availability import LOCAL_TZ, forecast_origins
from powerforecast.features.build import build_design_matrix, usable_rows
from powerforecast.forecasts.bias import (
    DEFAULT_WINDOW_DAYS,
    MIN_SAMPLES_PER_HOUR,
    estimate_offsets,
)
from powerforecast.forecasts.day import delivery_hours
from powerforecast.models.estimators import make_lightgbm

MAPE_FLOOR = 1.0


# name -> (statistic, min samples per hour). Forcing the per-hour minimum
# impossibly high is how a "constant only" variant is expressed without a second
# code path: the estimator falls back on its own, exercising the same fallback
# the deployed component would use.
VARIANTS: dict[str, tuple[str, int]] = {
    "constant, mean": ("mean", 10_000),
    "hourly, mean": ("mean", MIN_SAMPLES_PER_HOUR),
    "constant, median": ("median", 10_000),
    "hourly, median": ("median", MIN_SAMPLES_PER_HOUR),
}


@dataclass(frozen=True)
class Result:
    frame: pd.DataFrame
    leaderboard: pd.DataFrame
    methods: dict[str, pd.Series]


def run(
    *,
    train_until: str = "2026-01-31",
    window_days: int = DEFAULT_WINDOW_DAYS,
    panel: pd.DataFrame | None = None,
) -> Result:
    """Fit once, then walk forward day by day applying a backward-looking correction."""
    panel = load_panel() if panel is None else panel

    features, target = build_design_matrix(panel)
    usable = usable_rows(features, target)

    cutoff = pd.Timestamp(train_until, tz=LOCAL_TZ).tz_convert("UTC")
    train_index = usable[usable <= cutoff]
    if train_index.empty:
        raise ValueError(f"no usable rows on or before {cutoff}")

    model = make_lightgbm().fit(features.loc[train_index], target.loc[train_index])

    # Predict every evaluable hour once. Legitimate here because the model is
    # fixed and the panel is fixed, so predicting day by day would return exactly
    # these numbers — the walk-forward part of this experiment is the correction,
    # not the model.
    evaluate = usable[usable > cutoff]
    raw = pd.Series(model.predict(features.loc[evaluate]), index=evaluate, name="raw")

    actual = target.loc[evaluate]
    days = sorted({stamp.date() for stamp in pd.DatetimeIndex(evaluate).tz_convert(LOCAL_TZ)})

    corrected: dict[str, list[pd.Series]] = {name: [] for name in VARIANTS}
    methods: dict[str, list[str]] = {name: [] for name in VARIANTS}
    history = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))

    for day in days:
        hours = delivery_hours(day).intersection(pd.DatetimeIndex(evaluate))
        if hours.empty:
            continue

        origin = pd.Timestamp(forecast_origins(hours).iloc[0])
        for name, (statistic, per_hour) in VARIANTS.items():
            offsets = estimate_offsets(
                history,
                origin=origin,
                window_days=window_days,
                min_samples_per_hour=per_hour,
                statistic=statistic,
            )
            methods[name].append(offsets.method)
            corrected[name].append(offsets.apply(raw.loc[hours]))

        # Only now does this day's error become history. Appending before the
        # estimates would let a day correct itself — the whole point of walking
        # forward is that this cannot happen by accident.
        history = pd.concat([history, actual.loc[hours] - raw.loc[hours]])

    columns = {"actual": actual, "raw": raw, "plan": panel["load_plan_mwh"].reindex(evaluate)}
    for name, parts in corrected.items():
        columns[name] = pd.concat(parts).reindex(evaluate)

    frame = pd.DataFrame(columns).dropna()

    scored = ["raw", *VARIANTS, "plan"]
    leaderboard = (
        pd.DataFrame(
            [
                summarize(frame["actual"], frame[column], mape_floor=MAPE_FLOOR).as_row(name=column)
                for column in scored
            ]
        )
        .set_index("model")
        .sort_values("MAE")
    )

    return Result(
        frame=frame,
        leaderboard=leaderboard,
        methods={name: pd.Series(values) for name, values in methods.items()},
    )


def by_weekday(frame: pd.DataFrame, *, best: str = "hourly, median") -> pd.DataFrame:
    """Where the correction helps and where it does not.

    Split by weekday because the out-of-sample check found the deployed model's
    error concentrated on Saturdays and Sundays. An aggregate that improves while
    the weekend stays broken would be an aggregate hiding the actual problem.
    """
    local = pd.DatetimeIndex(frame.index).tz_convert(LOCAL_TZ)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grouped = frame.groupby(local.dayofweek)

    def mae_of(column: str) -> pd.Series:
        errors = (frame["actual"] - frame[column]).abs()
        return errors.groupby(local.dayofweek).mean()

    table = pd.DataFrame(
        {
            "hours": grouped.size(),
            "raw_MAE": mae_of("raw"),
            "corrected_MAE": mae_of(best),
            "plan_MAE": mae_of("plan"),
        }
    )
    table.index = pd.Index([names[i] for i in table.index], name="weekday")
    table["gain_%"] = 100 * (table["raw_MAE"] - table["corrected_MAE"]) / table["raw_MAE"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-until", default="2026-01-31")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--csv", type=Path, default=PATHS.reports / "bias_correction.csv")
    args = parser.parse_args()

    result = run(train_until=args.train_until, window_days=args.window_days)

    print(f"trained to {args.train_until}, window {args.window_days} days")
    print(f"{len(result.frame):,} out-of-sample hours\n")
    print(result.leaderboard.to_string(float_format=lambda v: f"{v:,.1f}"))

    maes = result.leaderboard["MAE"].astype(float)
    raw_mae, plan_mae = maes["raw"], maes["plan"]
    print(f"\nraw against the plan: {100 * (raw_mae - plan_mae) / plan_mae:+.1f}%")
    for name in VARIANTS:
        print(f"  {name:18} MAE {100 * (maes[name] - raw_mae) / raw_mae:+5.1f}% vs raw")
    print()
    print(by_weekday(result.frame).to_string(float_format=lambda v: f"{v:,.1f}"))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(args.csv)
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
