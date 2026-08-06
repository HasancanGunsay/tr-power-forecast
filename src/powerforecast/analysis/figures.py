"""Generate the exploratory figures and print the findings behind them.

Run with:

    uv run python -m powerforecast.analysis.figures

Deliberately a script rather than a notebook. A notebook is a good place to
explore and a bad place to keep a result: cells run out of order, outputs drift
from the code that produced them, and nothing guarantees the figure in the
README matches the data on disk. Everything here is regenerated from scratch on
every run, and the numbers come from `analysis.profiles`, which is tested.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Render without a display, so this works under CI and over SSH.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from powerforecast.analysis.profiles import (
    daily_profile,
    monthly_stats,
    peak_hours,
    tail_summary,
    weekday_weekend_gap,
)
from powerforecast.config import PATHS
from powerforecast.data.panel import add_local_calendar, load_panel
from powerforecast.evaluation.metrics import summarize, summarize_by

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 6: "Summer", 7: "Summer", 8: "Summer"}


def _save(fig: plt.Figure, name: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 1. The bar to beat
# --------------------------------------------------------------------------- #


def figure_benchmark(panel: pd.DataFrame, directory: Path) -> Path:
    """How accurate is the system operator's own published forecast?

    This is the single most important number in the project. Beating a
    seasonal-naive baseline proves very little; the published plan is the
    forecast the market actually runs on, so it is the honest bar.
    """
    actual = panel["consumption_mwh"]
    plan = panel["load_plan_mwh"]

    overall = summarize(actual, plan, mape_floor=1.0)
    by_hour = summarize_by(actual, plan, panel["hour"], mape_floor=1.0)
    by_month = summarize_by(
        actual, plan, panel["local_time"].dt.tz_localize(None).dt.to_period("M"), mape_floor=1.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].bar(by_hour.index, by_hour["MAPE_%"], color="#4C72B0")
    axes[0].axhline(
        overall.mape, color="#C44E52", linestyle="--", label=f"overall {overall.mape:.2f}%"
    )
    axes[0].set_xlabel("hour of day (Europe/Istanbul)")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_title("Official load plan error by hour")
    axes[0].legend()

    axes[1].plot(pd.PeriodIndex(by_month.index).to_timestamp(), by_month["MAPE_%"], color="#4C72B0")
    axes[1].set_xlabel("month")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title("Official load plan error over time")

    fig.suptitle("Benchmark: the forecast this project has to beat", fontsize=13)
    return _save(fig, "benchmark_load_plan.png", directory)


# --------------------------------------------------------------------------- #
# 2. Consumption shape
# --------------------------------------------------------------------------- #


def figure_consumption_profile(panel: pd.DataFrame, directory: Path) -> Path:
    seasonal = panel.assign(season=panel["month"].map(SEASONS)).dropna(subset=["season"])

    by_season = daily_profile(seasonal, "consumption_mwh", by="season")
    by_daytype = daily_profile(
        panel.assign(daytype=panel["is_weekend"].map({True: "Weekend", False: "Weekday"})),
        "consumption_mwh",
        by="daytype",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for label in by_season.columns:
        axes[0].plot(by_season.index, by_season[label] / 1000, label=label)
    axes[0].set_title("Daily profile by season")
    axes[0].set_ylabel("GWh per hour")

    for label in by_daytype.columns:
        axes[1].plot(by_daytype.index, by_daytype[label] / 1000, label=label)
    axes[1].set_title("Daily profile by day type")

    for ax in axes:
        ax.set_xlabel("hour of day (Europe/Istanbul)")
        ax.legend()

    fig.suptitle("Consumption follows human routine, not the clock in UTC", fontsize=13)
    return _save(fig, "consumption_profile.png", directory)


# --------------------------------------------------------------------------- #
# 3. Price regime
# --------------------------------------------------------------------------- #


def figure_price_regime(panel: pd.DataFrame, directory: Path) -> Path:
    """Is the 2022 price-cap regime visible in the data?

    If it is, a price model trained across the break is fitting two different
    processes at once and should either be restricted to one regime or told
    which one it is looking at.
    """
    stats = monthly_stats(panel, "price_try_mwh")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.fill_between(
        stats.index, stats["p05"], stats["p95"], alpha=0.25, color="#4C72B0", label="P05-P95"
    )
    ax.plot(stats.index, stats["median"], color="#4C72B0", label="median")
    ax.plot(stats.index, stats["max"], color="#C44E52", linewidth=0.8, label="monthly max")

    ax.set_xlabel("month")
    ax.set_ylabel("day-ahead price (TRY/MWh)")
    ax.set_title("Day-ahead price: level and spread by month")
    ax.legend()

    return _save(fig, "price_regime.png", directory)


# --------------------------------------------------------------------------- #
# 4. Price distribution
# --------------------------------------------------------------------------- #


def figure_price_distribution(panel: pd.DataFrame, directory: Path) -> Path:
    """What shape is the price distribution, and which metrics does it allow?

    The expectation going in was a heavy right tail, as in most liberalised
    power markets. The data says otherwise: skewness is near zero and the 99th
    percentile sits only about twice the median. The regulated ceiling truncates
    the distribution, which removes the scarcity spikes that produce fat tails
    elsewhere. Roughly 1% of hours still clear at exactly zero, so MAPE remains
    unusable without a floor.
    """
    price = panel["price_try_mwh"].dropna()
    recent = panel.loc[panel["year"] >= 2024, "price_try_mwh"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(recent, bins=80, color="#4C72B0")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("day-ahead price (TRY/MWh), 2024 onwards")
    axes[0].set_ylabel("hours (log scale)")
    axes[0].set_title("Truncated by the ceiling, with a spike at zero")

    ordered = price.sort_values(ascending=False).reset_index(drop=True)
    share = ordered.cumsum() / ordered.sum()
    axes[1].plot(100 * (ordered.index + 1) / len(ordered), 100 * share, color="#C44E52")
    axes[1].set_xlabel("share of hours, most expensive first (%)")
    axes[1].set_ylabel("share of total cost (%)")
    axes[1].set_xlim(0, 25)
    axes[1].set_title("A small share of hours carries much of the cost")
    axes[1].grid(alpha=0.3)

    return _save(fig, "price_distribution.png", directory)


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


def report(panel: pd.DataFrame) -> None:
    actual, plan = panel["consumption_mwh"], panel["load_plan_mwh"]
    benchmark = summarize(actual, plan, mape_floor=1.0)
    by_hour = summarize_by(actual, plan, panel["hour"], mape_floor=1.0)

    print("=" * 72)
    print(f"Coverage: {panel.index[0]} .. {panel.index[-1]}  ({len(panel):,} hours)")
    print("=" * 72)

    print("\n[1] BENCHMARK - official load plan vs actual consumption")
    print(f"  hours compared : {benchmark.n:,}   (dropped {benchmark.dropped} unpublished)")
    print(f"  MAE            : {benchmark.mae:,.0f} MWh")
    print(f"  RMSE           : {benchmark.rmse:,.0f} MWh")
    print(f"  MAPE           : {benchmark.mape:.2f} %")
    print(f"  bias           : {benchmark.bias:+,.0f} MWh  (positive = plan runs high)")
    worst = by_hour["MAPE_%"].idxmax()
    best = by_hour["MAPE_%"].idxmin()
    print(f"  worst hour     : {worst:02d}:00  ({by_hour.loc[worst, 'MAPE_%']:.2f} %)")
    print(f"  best hour      : {best:02d}:00  ({by_hour.loc[best, 'MAPE_%']:.2f} %)")

    print("\n[2] CONSUMPTION")
    print(f"  peak hours     : {peak_hours(panel, 'consumption_mwh')}")
    print(f"  weekday-weekend gap : {100 * weekday_weekend_gap(panel, 'consumption_mwh'):.1f} %")

    print("\n[3] PRICE")
    tail = tail_summary(panel["price_try_mwh"])
    print(f"  median         : {tail['median']:,.0f} TRY/MWh")
    print(f"  mean           : {tail['mean']:,.0f} TRY/MWh")
    print(f"  p99 / median   : {tail['p99_over_median']:.1f}x")
    print(f"  hours at zero  : {100 * tail['share_at_zero']:.2f} %")
    print(f"  skew           : {tail['skew']:.1f}")

    print("\n[4] PRICE BY YEAR (regime check)")
    yearly = panel.groupby("year")["price_try_mwh"].agg(["median", "mean", "max"])
    print(yearly.to_string(float_format=lambda v: f"{v:,.0f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PATHS.reports / "figures")
    args = parser.parse_args()

    panel = add_local_calendar(load_panel())
    report(panel)

    print("\nFigures written:")
    for builder in (
        figure_benchmark,
        figure_consumption_profile,
        figure_price_regime,
        figure_price_distribution,
    ):
        print(f"  {builder(panel, args.out)}")


if __name__ == "__main__":
    main()
