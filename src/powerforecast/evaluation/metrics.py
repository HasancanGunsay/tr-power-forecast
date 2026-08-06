"""Forecast error metrics.

No single number describes a forecast, so `summarize` reports several and says
how much data each one used:

* **MAE** — average error in the unit of the series. Easy to explain to someone
  who has to act on the forecast.
* **RMSE** — squares errors, so one large miss weighs more than several small
  ones. Whether that is desirable is a business question: under punitive
  imbalance settlement it usually is.
* **MAPE** — scale-free, and the metric most often misused. It divides by the
  actual value, so it explodes as the actual approaches zero. Turkish day-ahead
  prices do reach zero, which makes an unguarded MAPE on price data meaningless.
* **sMAPE** — bounded at 200%, so it survives near-zero actuals. The cost is
  that it is harder to interpret and asymmetric between over- and
  under-forecasting.
* **Bias** — mean signed error. MAE cannot tell a model that always runs 3%
  high from one that is randomly wrong by the same amount; bias can, and only
  the first is easy to fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ErrorSummary:
    """Errors for one forecast, with the sample sizes behind them."""

    n: int
    dropped: int
    mae: float
    rmse: float
    mape: float
    smape: float
    bias: float
    mape_coverage: float

    def as_row(self, name: str) -> dict[str, object]:
        """Flatten into a dict, for building comparison tables."""
        return {
            "model": name,
            "n": self.n,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE_%": self.mape,
            "sMAPE_%": self.smape,
            "bias": self.bias,
            "MAPE_coverage": self.mape_coverage,
        }


def _aligned(actual: pd.Series, forecast: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """Restrict both series to timestamps where both have a value.

    Comparing on the union and filling the holes would invent a result; padding
    with zeros would understate the error. Dropping is the only honest option,
    provided the number dropped is reported alongside the metric.
    """
    joined = pd.concat({"actual": actual, "forecast": forecast}, axis=1, join="inner")
    if joined.empty:
        raise ValueError(
            "actual and forecast have no overlapping timestamps; check the index "
            "timezone and the requested date range"
        )

    complete = joined.dropna()
    return complete["actual"], complete["forecast"], len(joined) - len(complete)


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    a, f, _ = _aligned(actual, forecast)
    return float((a - f).abs().mean())


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    a, f, _ = _aligned(actual, forecast)
    return float(np.sqrt(((a - f) ** 2).mean()))


def bias(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean signed error. Positive means the forecast runs high."""
    a, f, _ = _aligned(actual, forecast)
    return float((f - a).mean())


def mape(actual: pd.Series, forecast: pd.Series, floor: float | None = None) -> float:
    """Mean absolute percentage error, in percent.

    Args:
        floor: Rows where `abs(actual) < floor` are excluded. Without a floor,
            a single zero actual makes the result infinite, so this returns NaN
            instead of a number that would silently poison a comparison table.
    """
    a, f, _ = _aligned(actual, forecast)

    if floor is not None:
        keep = a.abs() >= floor
        a, f = a[keep], f[keep]
        if a.empty:
            return float("nan")
    elif (a == 0).any():
        return float("nan")

    return float(((a - f).abs() / a.abs()).mean() * 100)


def smape(actual: pd.Series, forecast: pd.Series) -> float:
    """Symmetric MAPE, in percent, bounded at 200%."""
    a, f, _ = _aligned(actual, forecast)
    denominator = (a.abs() + f.abs()) / 2

    # Both values zero means a perfect forecast, not an undefined one.
    ratio = np.where(denominator == 0, 0.0, (a - f).abs() / denominator.replace(0, np.nan))
    return float(np.nanmean(ratio) * 100)


def summarize(
    actual: pd.Series,
    forecast: pd.Series,
    *,
    mape_floor: float | None = None,
) -> ErrorSummary:
    """Compute every metric once, on a single aligned sample."""
    a, f, dropped = _aligned(actual, forecast)

    if mape_floor is not None:
        coverage = float((a.abs() >= mape_floor).mean())
    else:
        coverage = 1.0 if not (a == 0).any() else 0.0

    return ErrorSummary(
        n=len(a),
        dropped=dropped,
        mae=mae(a, f),
        rmse=rmse(a, f),
        mape=mape(a, f, floor=mape_floor),
        smape=smape(a, f),
        bias=bias(a, f),
        mape_coverage=coverage,
    )


def summarize_by(
    actual: pd.Series,
    forecast: pd.Series,
    grouper: pd.Series,
    *,
    mape_floor: float | None = None,
) -> pd.DataFrame:
    """Summarise errors within groups — by hour of day, month, or regime.

    An aggregate metric hides where a forecast actually fails. A load forecast
    that is excellent overnight and poor at the evening peak has the same MAE as
    one that is mediocre throughout, and only the first is worth fixing.
    """
    a, f, _ = _aligned(actual, forecast)
    groups = grouper.reindex(a.index)

    rows = []
    for key, index in a.groupby(groups).groups.items():
        summary = summarize(a.loc[index], f.loc[index], mape_floor=mape_floor)
        rows.append({"group": key, **summary.as_row(name=str(key))})

    return pd.DataFrame(rows).drop(columns=["model"]).set_index("group").sort_index()
