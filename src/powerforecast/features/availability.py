"""What a forecaster could actually see, and when.

Every feature in this project has to answer one question: *was this observable
when the bid was submitted?* Getting that wrong does not raise an error — it
produces a better backtest and a worse model, which is the most expensive kind
of bug in forecasting.

The rule lives here, in one place, so that a new feature inherits it instead of
re-deriving it. See ADR 0004 for why the origin is expressed as a data timestamp
rather than as the deadline.
"""

from __future__ import annotations

import pandas as pd

LOCAL_TZ = "Europe/Istanbul"

# Day-ahead market timetable, local time on the day before delivery:
#   12:30  bid submission closes
#   13:00  bids verified
#   13:30  clearing prices and quantities determined
#   14:00  results published for all 24 hours of the delivery day
BID_DEADLINE = "12:30"

# The last hour whose observation is complete when bids close.
#
# Hourly data is stamped at the start of the hour, so the value stamped 12:00
# covers 12:00-13:00 and is still accumulating at the 12:30 deadline. The newest
# fully observed value is the one stamped 11:00.
LAST_OBSERVED_HOUR = 11


def require_hourly(series: pd.Series) -> pd.DatetimeIndex:
    """Reject anything that is not a continuous hourly index.

    Lag arithmetic here counts rows, not durations. On a series with gaps, a
    shift of 24 rows silently reaches somewhere other than 24 hours back, and
    every feature built on it is wrong in a way nothing will report.
    """
    index = pd.DatetimeIndex(series.index)
    if len(index) > 1:
        spacing = index.to_series().diff().dropna().unique()
        if not (len(spacing) == 1 and spacing[0] == pd.Timedelta(hours=1)):
            raise ValueError(
                "expected a continuous hourly index; reindex onto a complete "
                "hourly grid before building features"
            )
    return index


def forecast_origins(
    targets: pd.DatetimeIndex,
    *,
    last_observed_hour: int = LAST_OBSERVED_HOUR,
    tz: str = LOCAL_TZ,
) -> pd.Series:
    """The newest observation available when each target hour was forecast.

    All 24 hours of a delivery day share one origin, because one bid covers the
    whole day.
    """
    local = targets.tz_convert(tz)
    previous_day = local.normalize() - pd.Timedelta(days=1)
    origins = previous_day + pd.Timedelta(hours=last_observed_hour)
    return pd.Series(origins, index=targets, name="origin")


def is_available(
    targets: pd.DatetimeIndex,
    sources: pd.DatetimeIndex,
    *,
    last_observed_hour: int = LAST_OBSERVED_HOUR,
    tz: str = LOCAL_TZ,
) -> pd.Series:
    """Whether each source timestamp had been observed by its target's origin."""
    origins = forecast_origins(targets, last_observed_hour=last_observed_hour, tz=tz)
    return pd.Series(sources <= origins.to_numpy(), index=targets, name="available")


def minimum_safe_lag(
    local_hour: int,
    *,
    last_observed_hour: int = LAST_OBSERVED_HOUR,
) -> int:
    """The smallest target-relative lag that is observable for a delivery hour.

    Useful for reasoning rather than for computation: a delivery hour at 00:00
    can look back 13 hours, while 23:00 must look back 36. Any lag applied
    uniformly across the whole day therefore has to be at least 36 hours to be
    safe everywhere — which is why the 48-hour and weekly lags survive and the
    24-hour lag does not.
    """
    hours_from_origin_to_midnight = 24 - last_observed_hour
    return hours_from_origin_to_midnight + local_hour
