"""Naive baselines for day-ahead forecasting.

A baseline exists to make later results meaningful. A model that cannot beat
"the same hour last week" has not learned anything, and a headline metric with
nothing to compare it against says nothing at all.

The subtle part is not the arithmetic — it is *when* the forecast is made.

Bids in the day-ahead market close around midday on the day before delivery, so
every hour of delivery day D is forecast from a single moment on D-1. Lead time
therefore runs from roughly 12 hours for the first delivery hour to 35 for the
last, and only data that existed at that moment may be used.

This makes the obvious `lag(24)` baseline wrong. For a target at 20:00 on day D,
lag 24 reads 20:00 on D-1 — which had not happened when the bid was submitted.
Backtesting with it produces a flattering number that cannot be reproduced in
production, and the discrepancy usually surfaces only after the model is live.

Rather than documenting that trap, the functions here enforce it: a forecast
whose source was not yet observable at its own origin is `NaN`, never a value.
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
# Hourly data is stamped at the *start* of the hour, so the value stamped 12:00
# covers 12:00-13:00 and is still being accumulated at the 12:30 deadline. The
# newest fully observed value is therefore the one stamped 11:00. Off-by-one
# here is not cosmetic: it hands the model an hour of the future and inflates
# every backtest built on top of it.
LAST_OBSERVED_HOUR = 11


def _require_hourly(series: pd.Series) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(series.index)
    if len(index) > 1:
        spacing = index.to_series().diff().dropna().unique()
        if not (len(spacing) == 1 and spacing[0] == pd.Timedelta(hours=1)):
            raise ValueError(
                "seasonal_naive expects a continuous hourly index; reindex the "
                "series onto a complete hourly grid first"
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
    whole day. The origin is expressed as the timestamp of the last fully
    observed hour on the previous local day rather than as the deadline itself,
    so it can be compared directly against a data timestamp without a second
    off-by-one waiting to happen.

    Returns a series aligned to `targets`, usable as a mask or joined onto a
    feature frame.
    """
    local = targets.tz_convert(tz)
    previous_day = local.normalize() - pd.Timedelta(days=1)
    origins = previous_day + pd.Timedelta(hours=last_observed_hour)
    return pd.Series(origins, index=targets, name="origin")


def seasonal_naive(
    actual: pd.Series,
    *,
    season_hours: int,
    last_observed_hour: int = LAST_OBSERVED_HOUR,
    tz: str = LOCAL_TZ,
) -> pd.Series:
    """Forecast each hour as the observed value `season_hours` earlier.

    Hours whose source had not been observed by the forecast origin are left as
    `NaN`, so the caller cannot accidentally score against information that was
    not available.

    Practical consequences under the 12:30 bid deadline:

    * `season_hours=24` — usable only for delivery hours up to 11:00, so half
      the delivery day has no forecast at all. Included because it is the
      baseline everyone reaches for first, and watching half of it disappear is
      the clearest way to understand why the deadline matters.
    * `season_hours=48` — fully available: day D-2 is complete long before the
      deadline. This is the honest "yesterday-like" baseline.
    * `season_hours=168` — same hour last week. Fully available, and usually
      stronger than the 48-hour lag because it matches the day of the week.

    Args:
        actual: Observed series on a continuous hourly index.
        season_hours: How far back to copy from.
        last_observed_hour: Local hour of the newest complete observation at bid
            submission on the day before delivery.
        tz: Timezone whose calendar days define the delivery day.
    """
    index = _require_hourly(actual)

    shifted = actual.shift(season_hours)

    sources = index - pd.Timedelta(hours=season_hours)
    origins = forecast_origins(index, last_observed_hour=last_observed_hour, tz=tz)
    available = sources <= origins.to_numpy()

    forecast = shifted.where(available)
    forecast.name = f"seasonal_naive_{season_hours}h"
    return forecast
