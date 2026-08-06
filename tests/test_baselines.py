"""Tests for the naive baselines.

Most of these are about *availability* rather than arithmetic. A day-ahead
forecast is made before gate closure on the previous day, so a lag that reaches
into hours which had not happened yet is leakage — it inflates the backtest and
cannot be reproduced in production. The baseline is built to make that
impossible rather than to be checked for afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.models.baselines import seasonal_naive


def _hourly(start: str, hours: int) -> pd.Series:
    """A series whose value encodes its own local hour, for easy assertions."""
    local = pd.date_range(start, periods=hours, freq="h", tz="Europe/Istanbul")
    return pd.Series(np.arange(hours, dtype="float64"), index=local.tz_convert("UTC"))


def _local(series: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(series.index).tz_convert("Europe/Istanbul")


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def test_yesterday_naive_is_unavailable_after_gate_closure() -> None:
    actual = _hourly("2026-08-01 00:00", 24 * 6)

    forecast = seasonal_naive(actual, season_hours=24, last_observed_hour=11)
    local = _local(forecast)

    last_day = forecast[local.date == pd.Timestamp("2026-08-06").date()]
    hours = pd.DatetimeIndex(last_day.index).tz_convert("Europe/Istanbul").hour

    # Hours up to and including 11:00 read from data that was complete when the
    # bid was submitted; later ones would read from the future. Exactly half the
    # delivery day is therefore unforecastable by this baseline.
    assert last_day[hours <= 11].notna().all()
    assert last_day[hours > 11].isna().all()


def test_two_day_lag_is_always_available() -> None:
    actual = _hourly("2026-08-01 00:00", 24 * 6)

    forecast = seasonal_naive(actual, season_hours=48, last_observed_hour=11)

    # Every hour of D-2 is complete well before gate closure on D-1, which is
    # what makes the 48-hour lag the honest "yesterday-like" baseline.
    assert forecast.iloc[48:].notna().all()


def test_weekly_lag_is_always_available() -> None:
    actual = _hourly("2026-08-01 00:00", 24 * 10)

    forecast = seasonal_naive(actual, season_hours=168, last_observed_hour=11)

    assert forecast.iloc[168:].notna().all()


def test_a_later_deadline_makes_more_of_yesterday_usable() -> None:
    actual = _hourly("2026-08-01 00:00", 24 * 6)

    early = seasonal_naive(actual, season_hours=24, last_observed_hour=8)
    late = seasonal_naive(actual, season_hours=24, last_observed_hour=17)

    assert late.notna().sum() > early.notna().sum()


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #


def test_forecast_repeats_the_value_one_season_earlier() -> None:
    actual = _hourly("2026-08-01 00:00", 24 * 9)

    forecast = seasonal_naive(actual, season_hours=168, last_observed_hour=11)
    target = pd.Timestamp("2026-08-08 05:00", tz="Europe/Istanbul").tz_convert("UTC")

    assert forecast[target] == actual[target - pd.Timedelta(hours=168)]


def test_forecast_is_indexed_like_the_input() -> None:
    actual = _hourly("2026-08-01 00:00", 24 * 3)

    forecast = seasonal_naive(actual, season_hours=48, last_observed_hour=11)

    assert forecast.index.equals(actual.index)
    assert forecast.name == "seasonal_naive_48h"


def test_the_first_season_has_no_history_to_copy() -> None:
    actual = _hourly("2026-08-01 00:00", 24 * 3)

    forecast = seasonal_naive(actual, season_hours=48, last_observed_hour=11)

    assert forecast.iloc[:48].isna().all()


def test_gaps_in_the_source_propagate_rather_than_being_filled() -> None:
    actual = _hourly("2026-08-01 00:00", 24 * 4)
    actual.iloc[10] = np.nan

    forecast = seasonal_naive(actual, season_hours=48, last_observed_hour=11)

    # Inventing a value here would hide a real hole from the error metrics.
    assert np.isnan(forecast.iloc[10 + 48])


def test_rejects_a_non_hourly_index() -> None:
    daily = pd.Series([1.0, 2.0], index=pd.date_range("2026-08-01", periods=2, freq="D", tz="UTC"))

    with pytest.raises(ValueError, match="hourly"):
        seasonal_naive(daily, season_hours=24)
