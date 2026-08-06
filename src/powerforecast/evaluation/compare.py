"""Score several forecasts against one another.

The only rule that matters here: **every forecast is scored on the same hours.**

Forecasts differ in availability. The weekly naive has no output for its first
week; the 24-hour naive is unusable for delivery hours after gate closure; the
published plan has days it was never issued for. Scoring each on whatever hours
it happens to cover produces a table that looks like a comparison and is not —
the method with the easiest subset wins, and the ranking says more about
coverage than about skill.

So the comparison runs on the intersection, and the number of hours dropped to
get there is reported next to it.
"""

from __future__ import annotations

import pandas as pd

from powerforecast.evaluation.metrics import summarize

COMPARISON_COLUMNS = ["n", "MAE", "RMSE", "MAPE_%", "sMAPE_%", "bias"]


def common_index(
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
) -> pd.DatetimeIndex:
    """Hours where the actual and every forecast are present."""
    combined = pd.concat({**forecasts, "__actual__": actual}, axis=1)
    return pd.DatetimeIndex(combined.dropna().index)


def compare_forecasts(
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
    *,
    mape_floor: float | None = None,
    sort_by: str = "MAE",
) -> pd.DataFrame:
    """Build a leaderboard over the hours all forecasts share.

    Raises:
        ValueError: if no hour is covered by everything, which usually means one
            forecast is empty or indexed in the wrong timezone.
    """
    index = common_index(actual, forecasts)
    if len(index) == 0:
        raise ValueError(
            "no hour is covered by the actual series and every forecast at once; "
            "check for an empty forecast or a timezone mismatch"
        )

    rows = [
        summarize(actual.loc[index], forecast.loc[index], mape_floor=mape_floor).as_row(name)
        for name, forecast in forecasts.items()
    ]

    table = pd.DataFrame(rows).set_index("model")[COMPARISON_COLUMNS]
    return table.sort_values(sort_by)


def availability(forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """How much of the horizon each forecast can actually produce.

    Reported alongside the leaderboard because a method that scores well on the
    half of the day it can cover is not competitive with one that covers all of
    it — and the table above deliberately hides that difference by scoring
    everything on the same subset.
    """
    return pd.DataFrame(
        {
            "hours": {name: int(f.notna().sum()) for name, f in forecasts.items()},
            "of_total": {name: len(f) for name, f in forecasts.items()},
            "coverage_%": {name: float(100 * f.notna().mean()) for name, f in forecasts.items()},
        }
    )
