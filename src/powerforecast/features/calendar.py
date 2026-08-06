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


def calendar_features(index: pd.DatetimeIndex, *, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Build calendar features for the given target hours.

    Derived from local time: consumption follows human routine, so the relevant
    hour is the one on the wall clock, not in UTC.
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
    elapsed = (index - index[0]).total_seconds() / (365.25 * 24 * 3600)
    features["years_elapsed"] = elapsed

    return features
