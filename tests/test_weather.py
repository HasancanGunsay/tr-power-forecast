"""Tests for temperature ingestion.

None of these touch the network. The behaviour worth pinning down is the
aggregation and the degree-day transform, plus the fact that the client asks for
the *archived forecast* variable rather than the realised temperature — which is
the difference between an honest feature and leakage.
"""

from __future__ import annotations

from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from powerforecast.data.weather import (
    ARCHIVE_URL,
    CITIES,
    FORECAST_URL,
    FORECAST_VARIABLE,
    LIVE_VARIABLE,
    City,
    WeatherError,
    degree_days,
    fetch_all_cities,
    fetch_city,
    fetch_live_all_cities,
    population_weighted,
    weather_features,
)

ISTANBUL = CITIES[0]


def _response(hours: int = 3, start: str = "2026-07-01T00:00") -> httpx.Response:
    times = pd.date_range(start, periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return httpx.Response(
        200,
        json={
            "hourly": {
                "time": times,
                FORECAST_VARIABLE: [20.0 + i for i in range(hours)],
            }
        },
    )


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


@respx.mock
def test_requests_the_archived_forecast_not_the_realised_temperature() -> None:
    route = respx.get(ARCHIVE_URL).mock(return_value=_response())

    fetch_city(ISTANBUL, date(2026, 7, 1), date(2026, 7, 1))

    params = route.calls[0].request.url.params
    # `temperature_2m` would be what the weather actually did — information no
    # bidder had on the previous day.
    assert params["hourly"] == FORECAST_VARIABLE
    assert "previous_day" in params["hourly"]


@respx.mock
def test_times_are_requested_and_returned_in_utc() -> None:
    route = respx.get(ARCHIVE_URL).mock(return_value=_response())

    series = fetch_city(ISTANBUL, date(2026, 7, 1), date(2026, 7, 1))

    assert route.calls[0].request.url.params["timezone"] == "UTC"
    assert str(series.index.tz) == "UTC"


@respx.mock
def test_series_is_named_after_the_city() -> None:
    respx.get(ARCHIVE_URL).mock(return_value=_response())

    assert fetch_city(ISTANBUL, date(2026, 7, 1), date(2026, 7, 1)).name == "istanbul"


@respx.mock
def test_http_error_is_reported_with_the_city() -> None:
    respx.get(ARCHIVE_URL).mock(return_value=httpx.Response(400, text="bad range"))

    with pytest.raises(WeatherError, match="istanbul"):
        fetch_city(ISTANBUL, date(2026, 7, 1), date(2026, 7, 1))


@respx.mock
def test_a_response_without_the_variable_fails_loudly() -> None:
    respx.get(ARCHIVE_URL).mock(return_value=httpx.Response(200, json={"hourly": {"time": []}}))

    with pytest.raises(WeatherError, match="previous_day1"):
        fetch_city(ISTANBUL, date(2026, 7, 1), date(2026, 7, 1))


@respx.mock
def test_all_cities_are_fetched_and_spaced_out() -> None:
    respx.get(ARCHIVE_URL).mock(return_value=_response())
    slept: list[float] = []

    frame = fetch_all_cities(
        date(2026, 7, 1), date(2026, 7, 1), sleep=slept.append, min_interval=0.5
    )

    assert set(c.name for c in CITIES) <= set(frame.columns)
    assert "temperature_c" in frame.columns
    # One pause between requests, none before the first.
    assert slept == [0.5] * (len(CITIES) - 1)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _frame(values: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.date_range(
        "2026-07-01", periods=len(next(iter(values.values()))), freq="h", tz="UTC"
    )
    return pd.DataFrame(values, index=index)


def test_weighting_follows_population_not_a_plain_average() -> None:
    cities = (City("big", 0, 0, 9.0), City("small", 0, 0, 1.0))
    frame = _frame({"big": [10.0], "small": [20.0]})

    weighted = population_weighted(frame, cities)

    # A plain mean would say 15; weighting by population says 11.
    assert weighted.iloc[0] == pytest.approx(11.0)


def test_a_missing_city_renormalises_rather_than_blanking_the_hour() -> None:
    cities = (City("big", 0, 0, 9.0), City("small", 0, 0, 1.0))
    frame = _frame({"big": [10.0], "small": [np.nan]})

    weighted = population_weighted(frame, cities)

    # Losing one city should shift the aggregate, not destroy it.
    assert weighted.iloc[0] == pytest.approx(10.0)


def test_all_cities_missing_gives_no_value() -> None:
    cities = (City("a", 0, 0, 1.0),)
    frame = _frame({"a": [np.nan]})

    assert population_weighted(frame, cities).isna().all()


# --------------------------------------------------------------------------- #
# Degree days
# --------------------------------------------------------------------------- #


def test_degree_days_split_the_v_shape_at_the_base() -> None:
    temperature = pd.Series([5.0, 18.0, 30.0])

    dd = degree_days(temperature, base_c=18.0)

    assert list(dd["hdd"]) == [13.0, 0.0, 0.0]
    assert list(dd["cdd"]) == [0.0, 0.0, 12.0]


def test_degree_days_are_never_negative() -> None:
    dd = degree_days(pd.Series([-10.0, 45.0]))

    assert (dd >= 0).all().all()


def test_base_temperature_can_be_changed() -> None:
    temperature = pd.Series([20.0])

    assert degree_days(temperature, base_c=22.0)["hdd"].iloc[0] == pytest.approx(2.0)
    assert degree_days(temperature, base_c=18.0)["cdd"].iloc[0] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# The live endpoint, used only for a delivery day that has not happened yet
# --------------------------------------------------------------------------- #


def _live_response(hours: int = 24, start: str = "2026-07-02T00:00") -> httpx.Response:
    times = pd.date_range(start, periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return httpx.Response(
        200,
        json={"hourly": {"time": times, LIVE_VARIABLE: [20.0] * hours}},
    )


@respx.mock
def test_live_fetch_asks_the_forecast_endpoint_for_the_plain_variable() -> None:
    """Train and serve must read the same quantity from two sides of the day.

    Training reads `temperature_2m_previous_day1` from the archive: the forecast
    for day D issued on D-1. Serving runs on D-1 and reads the live run for D,
    which is that same forecast before it became a record. Reading anything else
    here would put a different distribution in front of the model at serving
    time than the one it was fitted on.
    """
    route = respx.get(FORECAST_URL).mock(return_value=_live_response())

    fetch_live_all_cities(
        date(2026, 7, 2), cities=(ISTANBUL,), sleep=lambda _: None, today=date(2026, 7, 1)
    )

    params = route.calls[0].request.url.params
    assert params["hourly"] == LIVE_VARIABLE
    assert params["timezone"] == "UTC"
    # Two UTC days for one local day. Türkiye is UTC+3, so the local delivery day
    # begins at 21:00 UTC on the day before it; asking for the single UTC date
    # returns 21 of its 24 hours. Caught by running the daily job for a real
    # tomorrow, which refused the day with "3 of 24 hours missing".
    assert params["start_date"] == "2026-07-01"
    assert params["end_date"] == "2026-07-02"


@respx.mock
def test_live_fetch_refuses_a_day_that_is_not_the_next_one() -> None:
    """A forecast issued today for D+5 is not a day-ahead forecast.

    The endpoint would answer, the columns would fill and the service would
    return numbers — with an error larger than the backtest ever suggested and
    nothing to indicate why. Refusing is the only way that stays visible.
    """
    respx.get(FORECAST_URL).mock(return_value=_live_response())

    for day in (date(2026, 7, 6), date(2026, 7, 1), date(2026, 6, 30)):
        with pytest.raises(WeatherError, match="next delivery day"):
            fetch_live_all_cities(
                day, cities=(ISTANBUL,), sleep=lambda _: None, today=date(2026, 7, 1)
            )


@respx.mock
def test_live_fetch_aggregates_cities_like_the_archive_does() -> None:
    respx.get(FORECAST_URL).mock(return_value=_live_response())

    frame = fetch_live_all_cities(
        date(2026, 7, 2), cities=CITIES[:3], sleep=lambda _: None, today=date(2026, 7, 1)
    )

    assert "temperature_c" in frame.columns
    assert len(frame) == 24
    assert frame["temperature_c"].notna().all()


def test_weather_features_narrows_to_what_the_model_was_trained_on() -> None:
    """Per-city columns are for diagnosis; handing them to the model is a mismatch."""
    index = pd.date_range("2026-07-02", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "istanbul": [10.0, 20.0, 30.0],
            "izmir": [12.0, 22.0, 32.0],
            "temperature_c": [11.0, 21.0, 31.0],
        },
        index=index,
    )

    features = weather_features(frame)

    assert list(features.columns) == ["temperature_c", "hdd", "cdd"]
    assert features["hdd"].tolist() == [7.0, 0.0, 0.0]
    assert features["cdd"].tolist() == [0.0, 3.0, 13.0]
