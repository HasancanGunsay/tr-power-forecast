"""Naive baselines for day-ahead forecasting.

A baseline exists to make later results meaningful. A model that cannot beat
"the same hour last week" has not learned anything, and a headline metric with
nothing to compare it against says nothing at all.

The availability rule these depend on lives in `features.availability` and is
documented in ADR 0004: bids close at 12:30 on the day before delivery, so all
24 delivery hours are forecast from one origin and the newest fully observed
value is the one stamped 11:00 on D-1.
"""

from __future__ import annotations

import pandas as pd

from powerforecast.features.availability import LAST_OBSERVED_HOUR, LOCAL_TZ
from powerforecast.features.lags import target_lag


def seasonal_naive(
    actual: pd.Series,
    *,
    season_hours: int,
    last_observed_hour: int = LAST_OBSERVED_HOUR,
    tz: str = LOCAL_TZ,
) -> pd.Series:
    """Forecast each hour as the observed value `season_hours` earlier.

    A thin wrapper over `target_lag`: a seasonal naive forecast *is* a lag used
    directly as a prediction. Sharing the implementation means the baseline and
    the features it is compared against can never disagree about what was
    observable — which would otherwise be a subtle way to win a comparison.

    Practical consequences under the 12:30 deadline:

    * `season_hours=24` — usable only for delivery hours up to 11:00, so half
      the delivery day has no forecast at all. Included because it is the
      baseline everyone reaches for first, and watching half of it disappear is
      the clearest way to understand why the deadline matters.
    * `season_hours=48` — fully available: day D-2 is complete long before the
      deadline. The honest "yesterday-like" baseline.
    * `season_hours=168` — same hour last week. Fully available, and stronger
      than the 48-hour lag because it matches the day of the week.
    """
    forecast = target_lag(actual, season_hours, last_observed_hour=last_observed_hour, tz=tz)
    forecast.name = f"seasonal_naive_{season_hours}h"
    return forecast
