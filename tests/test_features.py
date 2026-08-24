"""Tests for the feature layer.

The arithmetic is easy; the availability is not. Most of these check that a
feature refuses to return a value the forecaster could not have seen, because
that failure is invisible in every other way — it makes the backtest better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from powerforecast.features.availability import (
    forecast_origins,
    is_available,
    minimum_safe_lag,
    require_hourly,
)
from powerforecast.features.calendar import calendar_features
from powerforecast.features.lags import origin_lag, origin_rolling, target_lag


def _series(start: str, hours: int, name: str = "load") -> pd.Series:
    local = pd.date_range(start, periods=hours, freq="h", tz="Europe/Istanbul")
    return pd.Series(np.arange(hours, dtype="float64"), index=local.tz_convert("UTC"), name=name)


def _local_hours(index: pd.Index) -> np.ndarray:
    return pd.DatetimeIndex(index).tz_convert("Europe/Istanbul").hour.to_numpy()


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def test_minimum_safe_lag_grows_across_the_delivery_day() -> None:
    # 00:00 can look back 13 hours; 23:00 must look back 36.
    assert minimum_safe_lag(0) == 13
    assert minimum_safe_lag(23) == 36


def test_a_uniform_lag_must_clear_the_worst_hour() -> None:
    worst = max(minimum_safe_lag(h) for h in range(24))

    # This is why 48 and 168 survive and 24 does not.
    assert worst == 36
    assert 24 < worst <= 48


def test_is_available_compares_against_the_shared_origin() -> None:
    targets = pd.date_range("2026-08-06 00:00", periods=24, freq="h", tz="Europe/Istanbul")
    targets_utc = pd.DatetimeIndex(targets.tz_convert("UTC"))
    sources = targets_utc - pd.Timedelta(hours=24)

    available = is_available(targets_utc, sources)

    assert available.to_numpy()[:12].all()
    assert not available.to_numpy()[12:].any()


def test_require_hourly_rejects_a_gap() -> None:
    series = _series("2026-08-01 00:00", 10)
    with_gap = series.drop(series.index[4])

    # Lags count rows. On a series with a hole, shift(24) silently lands
    # somewhere other than 24 hours back.
    with pytest.raises(ValueError, match="hourly"):
        require_hourly(with_gap)


# --------------------------------------------------------------------------- #
# Target-relative lags
# --------------------------------------------------------------------------- #


def test_short_lag_is_blank_for_late_delivery_hours() -> None:
    series = _series("2026-08-01 00:00", 24 * 6)

    lagged = target_lag(series, 24)
    hours = _local_hours(lagged.index)
    tail = lagged.iloc[48:]  # past the warm-up

    assert tail[hours[48:] <= 11].notna().all()
    assert tail[hours[48:] > 11].isna().all()


def test_lag_of_36_hours_is_safe_everywhere() -> None:
    series = _series("2026-08-01 00:00", 24 * 8)

    lagged = target_lag(series, 36)

    assert lagged.iloc[36:].notna().all()


def test_lag_reads_the_right_value() -> None:
    series = _series("2026-08-01 00:00", 24 * 10)
    target = pd.Timestamp("2026-08-08 05:00", tz="Europe/Istanbul").tz_convert("UTC")

    lagged = target_lag(series, 168)

    assert lagged[target] == series[target - pd.Timedelta(hours=168)]


def test_lag_is_named_after_its_source_and_offset() -> None:
    lagged = target_lag(_series("2026-08-01 00:00", 100), 48)

    assert lagged.name == "load_lag_48h"


# --------------------------------------------------------------------------- #
# Origin-relative features
# --------------------------------------------------------------------------- #


def test_origin_lag_is_constant_within_a_delivery_day() -> None:
    series = _series("2026-08-01 00:00", 24 * 6)

    at_origin = origin_lag(series)
    local_dates = pd.DatetimeIndex(at_origin.index).tz_convert("Europe/Istanbul").date

    # One bid covers the whole day, so every hour of it shares one observation.
    per_day = pd.Series(at_origin.to_numpy(), index=local_dates).groupby(level=0).nunique()
    assert (per_day.dropna() <= 1).all()


def test_origin_lag_reads_the_last_observed_hour() -> None:
    series = _series("2026-08-01 00:00", 24 * 6)
    target = pd.Timestamp("2026-08-05 19:00", tz="Europe/Istanbul").tz_convert("UTC")
    expected_source = pd.Timestamp("2026-08-04 11:00", tz="Europe/Istanbul").tz_convert("UTC")

    at_origin = origin_lag(series)

    assert at_origin[target] == series[expected_source]


def test_origin_lag_can_step_further_back() -> None:
    series = _series("2026-08-01 00:00", 24 * 6)
    target = pd.Timestamp("2026-08-05 19:00", tz="Europe/Istanbul").tz_convert("UTC")
    expected_source = pd.Timestamp("2026-08-04 08:00", tz="Europe/Istanbul").tz_convert("UTC")

    at_origin = origin_lag(series, hours_before_origin=3)

    assert at_origin[target] == series[expected_source]


def test_origin_rolling_averages_only_past_data() -> None:
    series = _series("2026-08-01 00:00", 24 * 6)
    target = pd.Timestamp("2026-08-05 19:00", tz="Europe/Istanbul").tz_convert("UTC")
    origin = pd.Timestamp("2026-08-04 11:00", tz="Europe/Istanbul").tz_convert("UTC")

    rolled = origin_rolling(series, 24)
    window = series.loc[origin - pd.Timedelta(hours=23) : origin]

    assert rolled[target] == pytest.approx(window.mean())


def test_origin_rolling_supports_other_statistics() -> None:
    series = _series("2026-08-01 00:00", 24 * 6)

    assert origin_rolling(series, 24, statistic="max").name == "load_max_24h_at_origin"
    assert origin_rolling(series, 24, statistic="std").notna().any()


def test_origin_features_never_reach_past_the_origin() -> None:
    """The property that makes this family safe by construction."""
    series = _series("2026-08-01 00:00", 24 * 8)
    index = pd.DatetimeIndex(series.index)
    origins = pd.DatetimeIndex(forecast_origins(index))

    # A monotonically increasing series encodes its own position, so a value
    # larger than the one at the origin could only have come from the future.
    at_origin = origin_lag(series)
    value_at_origin = series.reindex(origins.tz_convert("UTC")).to_numpy()

    assert np.all(np.nan_to_num(at_origin.to_numpy()) <= np.nan_to_num(value_at_origin))


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #


def test_calendar_uses_local_time() -> None:
    index = pd.DatetimeIndex(
        pd.date_range("2026-08-06 00:00", periods=3, freq="h", tz="Europe/Istanbul").tz_convert(
            "UTC"
        )
    )

    features = calendar_features(index)

    assert list(features["hour"]) == [0, 1, 2]
    assert index[0].hour == 21  # 21:00 UTC is midnight in Istanbul


def test_cyclical_encoding_puts_23_and_0_next_to_each_other() -> None:
    index = pd.DatetimeIndex(
        pd.date_range("2026-08-06 00:00", periods=24, freq="h", tz="Europe/Istanbul").tz_convert(
            "UTC"
        )
    )

    features = calendar_features(index)
    point = features[["hour_sin", "hour_cos"]].to_numpy()

    midnight_to_23 = np.linalg.norm(point[0] - point[23])
    midnight_to_12 = np.linalg.norm(point[0] - point[12])

    # As integers these are 23 and 12 apart; on the circle the first pair is
    # adjacent and the second is opposite.
    assert midnight_to_23 < midnight_to_12


def test_weekend_flag_matches_the_local_date() -> None:
    # 2026-08-08 is a Saturday.
    index = pd.DatetimeIndex(
        pd.date_range("2026-08-07 00:00", periods=48, freq="h", tz="Europe/Istanbul").tz_convert(
            "UTC"
        )
    )

    features = calendar_features(index)

    assert features["is_weekend"].iloc[0] == 0
    assert features["is_weekend"].iloc[24] == 1


def test_trend_is_measured_in_years_from_the_start() -> None:
    index = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=24 * 400, freq="h", tz="UTC"))

    features = calendar_features(index, include_trend=True)

    assert features["years_elapsed"].iloc[0] == pytest.approx(0.0)
    assert features["years_elapsed"].iloc[-1] == pytest.approx(1.09, abs=0.02)


def test_the_trend_is_off_by_default() -> None:
    """A monotonic feature is one every prediction row sits outside of.

    Trees cannot extrapolate, so all future rows land on one side of the highest
    split and inherit whatever the last stretch of training happened to look
    like. Measured: with training ending four days after Kurban Bayrami, July MAE
    was 1,764 with the feature and 1,076 without. See ADR 0009.
    """
    index = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC"))

    assert "years_elapsed" not in calendar_features(index).columns
