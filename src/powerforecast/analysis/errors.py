"""Where a forecast fails, rather than how much.

An aggregate metric says a model is wrong by so much on average. It cannot say
whether that error is spread evenly across every hour — which would mean the
model is simply imprecise — or concentrated in a handful of days that share
something the model was never told about. Those two situations call for
completely different work, and only the second one can be fixed by a feature.

The headline result here is that the model ties the corrected plan on RMSE while
clearly beating it on MAE. Squared error weights large misses, so that pattern
says the remaining gap lives in the tail. These functions are for finding it.
"""

from __future__ import annotations

import pandas as pd

LOCAL_TZ = "Europe/Istanbul"


def error_frame(
    actual: pd.Series,
    forecast: pd.Series,
    *,
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    """Signed and absolute errors with local calendar columns attached."""
    aligned = pd.concat({"actual": actual, "forecast": forecast}, axis=1, join="inner").dropna()
    local = pd.DatetimeIndex(aligned.index).tz_convert(tz)

    frame = pd.DataFrame(index=aligned.index)
    frame["actual"] = aligned["actual"]
    frame["forecast"] = aligned["forecast"]
    frame["error"] = aligned["forecast"] - aligned["actual"]
    frame["abs_error"] = frame["error"].abs()
    frame["pct_error"] = 100 * frame["error"] / aligned["actual"]
    frame["local_date"] = local.date
    frame["hour"] = local.hour
    frame["dayofweek"] = local.dayofweek
    frame["month"] = local.month
    return frame


def error_concentration(
    errors: pd.DataFrame, shares: tuple[float, ...] = (0.01, 0.05, 0.10)
) -> pd.DataFrame:
    """How much of the total squared error the worst hours account for.

    RMSE is driven by squared error, so this answers directly what a tie on RMSE
    and a win on MAE implies. If 1% of hours carry a quarter of the squared
    error, improving the average hour cannot move RMSE — the tail has to be
    attacked specifically.
    """
    squared = (errors["error"] ** 2).sort_values(ascending=False)
    total = squared.sum()

    rows = []
    for share in shares:
        count = max(1, round(share * len(squared)))
        rows.append(
            {
                "worst_share_%": 100 * share,
                "hours": count,
                "of_squared_error_%": 100 * squared.iloc[:count].sum() / total,
            }
        )
    return pd.DataFrame(rows).set_index("worst_share_%")


def worst_days(errors: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """The delivery days with the largest mean absolute error.

    Aggregating to days rather than hours is deliberate. A single bad hour is
    usually noise; a bad *day* is a day the model misunderstood, and its date can
    be looked up — a holiday, a heatwave, a market event.
    """
    daily = errors.groupby("local_date").agg(
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
        max_abs=("abs_error", "max"),
        mean_actual=("actual", "mean"),
    )
    daily["mape_%"] = 100 * daily["mae"] / daily["mean_actual"]
    daily["weekday"] = pd.to_datetime(daily.index).day_name()
    return daily.sort_values("mae", ascending=False).head(top)


def error_by(errors: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean absolute error grouped by any calendar column."""
    grouped = errors.groupby(column).agg(
        hours=("abs_error", "size"),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
    )
    return grouped.sort_index()


def compare_on_worst(
    errors_by_model: dict[str, pd.DataFrame],
    reference: str,
    *,
    quantile: float = 0.99,
) -> pd.DataFrame:
    """Score every model on the hours the reference model finds hardest.

    Each model's own worst hours are a different set, so comparing them would
    compare different questions. Fixing the hours by one model's difficulty makes
    the tail comparable: it answers "on the hours this model struggles with, does
    anything else do better?"
    """
    hardest = errors_by_model[reference]["abs_error"]
    threshold = hardest.quantile(quantile)
    index = hardest[hardest >= threshold].index

    rows = []
    for name, frame in errors_by_model.items():
        subset = frame.reindex(index).dropna(subset=["abs_error"])
        rows.append(
            {
                "model": name,
                "hours": len(subset),
                "MAE": subset["abs_error"].mean(),
                "bias": subset["error"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("MAE")
