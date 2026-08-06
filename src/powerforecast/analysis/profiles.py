"""Descriptive summaries used by the exploratory figures.

Kept separate from the plotting code so the numbers can be tested. A figure is
hard to assert on; the table behind it is not, and a wrong figure is usually a
wrong table drawn faithfully.
"""

from __future__ import annotations

from typing import cast

import pandas as pd


def daily_profile(
    frame: pd.DataFrame,
    column: str,
    *,
    by: str | None = None,
) -> pd.DataFrame:
    """Average value for each local hour of day, optionally split by a column.

    This is the first thing to look at in any load series: it shows the shape of
    the day, where the peaks are, and whether they move with the season.
    """
    if by is None:
        return frame.groupby("hour")[column].mean().to_frame(column)

    grouped = frame.groupby([by, "hour"])[column].mean()
    # `by` becomes the columns and hour stays on the index, so each split is a
    # line that can be plotted directly against hour of day.
    return grouped.unstack(0)


def monthly_stats(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Monthly location and spread, for spotting regime changes.

    Median rather than mean as the headline: electricity prices are heavy-tailed
    and a handful of scarcity hours can move a monthly mean far away from what a
    typical hour actually cost. The quantiles carry the tail information that the
    median deliberately ignores.
    """
    # Drop the offset before converting to a period. The instant is unchanged —
    # `local_time` already carries local wall-clock time — and pandas warns when
    # a tz-aware index is periodised because a period has no timezone.
    local_months = frame["local_time"].dt.tz_localize(None).dt.to_period("M")
    grouped = frame.groupby(local_months)[column]

    stats = cast(
        "pd.DataFrame",
        grouped.agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            p05=lambda s: s.quantile(0.05),
            p95=lambda s: s.quantile(0.95),
            max="max",
        ),
    )
    stats.index = pd.PeriodIndex(stats.index).to_timestamp()
    stats.index.name = "month"
    return stats


def tail_summary(series: pd.Series) -> dict[str, float]:
    """Quantify how heavy the upper tail is.

    A distribution where the 99th percentile sits far above the median is one
    where squared-error metrics will be dominated by a few hours, and where MAPE
    on the low end is unstable. Both facts change which metric is worth
    optimising, so they are measured rather than assumed.
    """
    clean = series.dropna().astype("float64")
    median = float(clean.median())
    p99 = float(clean.quantile(0.99))

    return {
        "n": float(len(clean)),
        "median": median,
        "mean": float(clean.mean()),
        "p99": p99,
        "max": float(clean.max()),
        # How far the extreme hours sit above a typical one.
        "p99_over_median": p99 / median if median else float("nan"),
        "share_at_zero": float((clean == 0).mean()),
        # pandas-stubs types `skew()` for the general case, where it may return a
        # non-numeric summary; on a float64 Series it is always a float.
        "skew": float(cast("float", clean.skew())),
    }


def peak_hours(frame: pd.DataFrame, column: str, *, top: int = 3) -> list[int]:
    """The local hours with the highest average value."""
    profile = frame.groupby("hour")[column].mean()
    return [int(h) for h in profile.nlargest(top).index]


def weekday_weekend_gap(frame: pd.DataFrame, column: str) -> float:
    """Mean weekday value minus mean weekend value, as a fraction of the weekday mean.

    A large gap means a calendar feature will earn its place; a small one means
    the model should lean on weather and recent history instead.
    """
    weekday = frame.loc[~frame["is_weekend"], column].mean()
    weekend = frame.loc[frame["is_weekend"], column].mean()
    return float((weekday - weekend) / weekday)
