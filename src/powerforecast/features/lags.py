"""Lag and rolling features, in two families.

**Target-relative** (`target_lag`) looks back a fixed number of hours from the
hour being predicted. It is the familiar one, and the one that quietly breaks:
the amount of history available depends on the delivery hour, so a lag that is
safe at 00:00 reaches into the future at 23:00. Anything unobservable becomes
`NaN` rather than a value.

**Origin-relative** (`origin_lag`, `origin_rolling`) looks back from the moment
the bid was submitted. Every hour of a delivery day gets the same value, and it
is observable by construction — there is no way to express an unavailable
feature in this family at all.

The second is how operational forecasting is usually set up, and it is the
family most often missing from portfolio projects. "What was demand when I
placed the bid" is a legitimate, powerful predictor; "what was demand one hour
before the hour I am predicting" is usually not knowable.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from powerforecast.features.availability import (
    LAST_OBSERVED_HOUR,
    LOCAL_TZ,
    forecast_origins,
    is_available,
    require_hourly,
)

Statistic = Literal["mean", "min", "max", "std"]


def target_lag(
    series: pd.Series,
    hours: int,
    *,
    last_observed_hour: int = LAST_OBSERVED_HOUR,
    tz: str = LOCAL_TZ,
) -> pd.Series:
    """Value `hours` before each target hour, blanked where it was not observable.

    A lag applied uniformly across a delivery day has to be at least 36 hours to
    be safe for every hour of it. Shorter lags are not rejected — they are
    genuinely useful for the early delivery hours — but the hours they cannot
    reach are left empty.
    """
    index = require_hourly(series)

    shifted = series.shift(hours)
    sources = index - pd.Timedelta(hours=hours)
    available = is_available(index, sources, last_observed_hour=last_observed_hour, tz=tz)

    lagged = shifted.where(available.to_numpy())
    lagged.name = f"{series.name}_lag_{hours}h"
    return lagged


def origin_lag(
    series: pd.Series,
    *,
    hours_before_origin: int = 0,
    last_observed_hour: int = LAST_OBSERVED_HOUR,
    tz: str = LOCAL_TZ,
) -> pd.Series:
    """Value at (or shortly before) the forecast origin.

    Constant across the 24 hours of a delivery day, because they share one
    origin. `hours_before_origin=0` is the freshest observation the forecaster
    had; larger values step further back into what was already known.
    """
    require_hourly(series)
    index = pd.DatetimeIndex(series.index)

    origins = forecast_origins(index, last_observed_hour=last_observed_hour, tz=tz)
    source = pd.DatetimeIndex(origins).tz_convert("UTC") - pd.Timedelta(hours=hours_before_origin)

    values = series.reindex(source).to_numpy()
    name = f"{series.name}_at_origin"
    if hours_before_origin:
        name += f"_minus_{hours_before_origin}h"
    return pd.Series(values, index=index, name=name)


def origin_rolling(
    series: pd.Series,
    window_hours: int,
    *,
    statistic: Statistic = "mean",
    last_observed_hour: int = LAST_OBSERVED_HOUR,
    tz: str = LOCAL_TZ,
) -> pd.Series:
    """Rolling statistic over the window ending at the forecast origin.

    The rolling window is computed on the full series first and only then
    sampled at each origin. Doing it the other way round — sampling first, then
    aggregating around the target — is how a "recent average" ends up including
    hours that had not happened yet.
    """
    require_hourly(series)
    index = pd.DatetimeIndex(series.index)

    rolled = getattr(series.rolling(window=window_hours, min_periods=window_hours), statistic)()

    origins = forecast_origins(index, last_observed_hour=last_observed_hour, tz=tz)
    source = pd.DatetimeIndex(origins).tz_convert("UTC")

    values = rolled.reindex(source).to_numpy()
    return pd.Series(
        values, index=index, name=f"{series.name}_{statistic}_{window_hours}h_at_origin"
    )
