"""Calendar features derived from the target timestamp.

These are the only features in the project that carry no availability risk: the
calendar for a delivery day is known years in advance, so nothing here can look
at data that had not happened.

Cyclical encoding is used for hour and month. A model that sees hour as a plain
integer is told that 23:00 and 00:00 are 23 apart, when in load terms they are
adjacent — encoding each as a point on a circle removes that false distance.
Tree models can recover the wrap-around by splitting twice, but linear and
neural models cannot, and the encoding costs nothing either way.

Turkish public holidays are deliberately **not** included yet. The religious
ones follow the lunar calendar and move about eleven days earlier each year, so
they cannot be derived from the timestamp and need a proper calendar source. A
naive fixed-date flag would mark the wrong days while looking correct, which is
worse than having no holiday feature at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from powerforecast.features.availability import LOCAL_TZ


def _cyclical(values: pd.Index, period: int, name: str) -> pd.DataFrame:
    radians = 2 * np.pi * values.to_numpy() / period
    return pd.DataFrame({f"{name}_sin": np.sin(radians), f"{name}_cos": np.cos(radians)})


def calendar_features(
    index: pd.DatetimeIndex,
    *,
    tz: str = LOCAL_TZ,
    include_trend: bool = False,
) -> pd.DataFrame:
    """Build calendar features for the given target hours.

    Derived from local time: consumption follows human routine, so the relevant
    hour is the one on the wall clock, not in UTC.

    Args:
        include_trend: Add `years_elapsed`, a linear trend. **Off by default** —
            see the note below the code for the measurement that decided it.
    """
    local = pd.DatetimeIndex(index).tz_convert(tz)

    features = pd.DataFrame(index=index)
    features["hour"] = local.hour
    features["dayofweek"] = local.dayofweek
    features["month"] = local.month
    features["dayofyear"] = local.dayofyear
    features["is_weekend"] = (local.dayofweek >= 5).astype("int8")

    for frame, period, name in (
        (local.hour, 24, "hour"),
        (local.dayofweek, 7, "dayofweek"),
        (local.month, 12, "month"),
    ):
        cyclical = _cyclical(pd.Index(frame), period, name)
        cyclical.index = index
        features = features.join(cyclical)

    # A linear trend, so a model can express the fact that demand grows year on
    # year. Measured in years from the start of the series rather than as a raw
    # year number, which keeps the coefficient interpretable.
    #
    # OFF BY DEFAULT, and the reason is worth the paragraph. This feature is
    # monotonically increasing, so **every prediction row lies above every
    # training row** — on the current data, training spans 1.000 to 5.081 and
    # the evaluation window 5.082 to 5.591. A tree cannot extrapolate: it puts
    # all of them on one side of its highest split, which makes the prediction
    # inherit whatever the *last stretch of training data* happened to look
    # like.
    #
    # Usually that is harmless. Once it is not: with training ending 31 May
    # 2026 — four days after Kurban Bayrami — MAE on a July window was 1,764
    # against 1,076 for the same model without the feature. The top bin was
    # holiday-collapsed demand, and July inherited it, under-forecasting by
    # 1,551 MWh. Cutting off eleven days earlier, before the holiday, gave
    # 1,087. The feature turns *when you happened to stop training* into a
    # systematic level shift.
    #
    # Over a six-month window it is worth nothing to LightGBM either way (MAE
    # 912.4 with, 912.8 without), because the origin-rolling means already
    # carry the recent level in a range the trees can actually use. Neutral on
    # average, occasionally catastrophic, and redundant: removed. It helps
    # ridge, which can extrapolate — hence the flag rather than a deletion.
    #
    # See ADR 0009.
    if include_trend:
        elapsed = (index - index[0]).total_seconds() / (365.25 * 24 * 3600)
        features["years_elapsed"] = elapsed

    return features
