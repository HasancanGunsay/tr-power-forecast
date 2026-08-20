"""Temperature forecasts, as they were known when bids were submitted.

The single most important decision in this module is which temperature to use.

**Realised temperature would be leakage.** Reanalysis products like ERA5 record
what the weather actually did, and a bidder on the previous day did not have
that. A model given realised temperature will look excellent in backtest and
degrade the moment it meets a real forecast, because it has been trained to rely
on information that does not exist at prediction time.

So this module reads Open-Meteo's **archived forecasts** instead: for each hour
of delivery day D, the temperature that was being forecast for it one day
earlier. That is genuinely what a bidder had. It is noisier than the truth, and
that noise is part of the problem being solved rather than something to remove.

Coverage note: the archive begins in 2022, while the electricity series starts in
2021. Weather features are simply absent before then, and the backtest drops
incomplete rows inside each fold — so the evaluation window is unchanged and the
comparison against the weather-free models stays like for like.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
import pandas as pd

ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# The live endpoint, used only when forecasting a day that has not happened yet.
# See `fetch_live_all_cities` for why it needs a different variable name and a
# guard on which day it may be asked about.
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LIVE_VARIABLE = "temperature_2m"

# Earliest date the archived-forecast API can serve.
ARCHIVE_START = date(2022, 1, 1)

# The variable name is Open-Meteo's: the value that was being forecast for this
# hour, as of one day earlier. `previous_day2` exists too and would be more
# conservative, but a bidder really did have the D-1 run.
FORECAST_VARIABLE = "temperature_2m_previous_day1"

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)

# Open-Meteo asks for modest request rates on the free tier.
MIN_REQUEST_INTERVAL = 1.0


@dataclass(frozen=True)
class City:
    """A demand centre, weighted by population.

    Weighting by population is a proxy for weighting by electricity demand.
    It is not exact — industrial load does not follow people — but it is far
    better than a plain average, which would let a sparsely populated but
    climatically extreme location dominate the national signal.
    """

    name: str
    latitude: float
    longitude: float
    population_millions: float


# The eight largest metropolitan areas, roughly 45% of Türkiye's population.
CITIES = (
    City("istanbul", 41.0138, 28.9497, 15.9),
    City("ankara", 39.9334, 32.8597, 5.8),
    City("izmir", 38.4237, 27.1428, 4.5),
    City("bursa", 40.1885, 29.0610, 3.2),
    City("antalya", 36.8969, 30.7133, 2.7),
    City("konya", 37.8746, 32.4932, 2.3),
    City("adana", 37.0000, 35.3213, 2.3),
    City("gaziantep", 37.0662, 37.3833, 2.2),
)


class WeatherError(RuntimeError):
    """Raised when temperature data cannot be retrieved."""


def fetch_city(
    city: City,
    start: date,
    end: date,
    *,
    client: httpx.Client | None = None,
) -> pd.Series:
    """Fetch archived day-ahead temperature forecasts for one city.

    Times are requested in UTC to match the project's storage convention;
    converting a local response afterwards would reintroduce exactly the
    daylight-saving ambiguity that UTC storage exists to avoid.
    """
    return _fetch_hourly(
        city,
        url=ARCHIVE_URL,
        variable=FORECAST_VARIABLE,
        start=start,
        end=end,
        client=client,
    )


def _fetch_hourly(
    city: City,
    *,
    url: str,
    variable: str,
    start: date,
    end: date,
    client: httpx.Client | None = None,
) -> pd.Series:
    """One city, one variable, one date range, from either Open-Meteo endpoint."""
    owned = client is None
    client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    try:
        response = client.get(
            url,
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": variable,
                "timezone": "UTC",
            },
        )
    except httpx.HTTPError as exc:
        raise WeatherError(f"could not reach Open-Meteo for {city.name}: {exc}") from exc
    finally:
        if owned:
            client.close()

    if response.status_code >= 400:
        raise WeatherError(
            f"Open-Meteo rejected the request for {city.name} "
            f"with HTTP {response.status_code}: {response.text[:200]}"
        )

    body = response.json()
    hourly = body.get("hourly")
    if not hourly or variable not in hourly:
        raise WeatherError(
            f"Open-Meteo returned no {variable!r} for {city.name}; got keys {list(body)}"
        )

    index = pd.to_datetime(hourly["time"], utc=True)
    values = pd.to_numeric(hourly[variable], errors="coerce")
    series = pd.Series(values, index=index, name=city.name, dtype="float64")
    series.index.name = "timestamp"
    return series


def fetch_all_cities(
    start: date,
    end: date,
    *,
    cities: tuple[City, ...] = CITIES,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    min_interval: float = MIN_REQUEST_INTERVAL,
) -> pd.DataFrame:
    """Fetch every city and return one frame, plus the weighted aggregate."""
    owned = client is None
    client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    try:
        columns = []
        for position, city in enumerate(cities):
            if position:
                sleep(min_interval)
            columns.append(fetch_city(city, start, end, client=client))
    finally:
        if owned:
            client.close()

    frame = pd.concat(columns, axis=1).sort_index()
    frame["temperature_c"] = population_weighted(frame, cities)
    return frame


def fetch_live_all_cities(
    day: date,
    *,
    cities: tuple[City, ...] = CITIES,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    min_interval: float = MIN_REQUEST_INTERVAL,
    today: date | None = None,
) -> pd.DataFrame:
    """Temperature for a delivery day that has not happened yet.

    Everything above this function reads the *archive*: for a past delivery day,
    what was being forecast for it one day earlier. That is the right quantity
    for training, and it is the reason the model is honest. But the archive
    cannot answer about tomorrow, because the forecast for tomorrow has not yet
    become a historical record — so a service asked to forecast tomorrow would
    have no temperature at all, and refuse.

    This closes that gap, and the important thing is that it closes it with
    **the same quantity**, not a similar one:

    * *Training* uses `temperature_2m_previous_day1` for delivery day D — the
      value that was being forecast for D by the run issued on D-1.
    * *Serving* runs on D-1, shortly before the 12:30 deadline, and reads the
      live forecast for D — which is the run issued on D-1.

    Those are the same object seen from two sides of the same day. Six months
    later, this very request is what the archive will return for D. Train and
    serve therefore see one distribution, not two, and the backtest keeps
    meaning what it claimed.

    Which is why `day` is checked rather than trusted. The live endpoint will
    cheerfully return a forecast for D+5 — issued today, five days out, far
    worse than anything the model was trained on. Nothing about that response
    would look wrong; the numbers arrive, the columns fill, the service answers,
    and the error is larger than the backtest ever suggested with no indication
    why. A day-ahead model must be fed a day-ahead forecast, so anything else is
    refused here rather than discovered later.

    Args:
        day: The delivery day. Must be exactly one day after `today`.
        today: Injectable clock. Tests must not depend on when they run.

    Raises:
        WeatherError: if `day` is not the day after `today`.
    """
    today = today or date.today()
    if day != today + timedelta(days=1):
        raise WeatherError(
            f"the live endpoint may only be used for the next delivery day: "
            f"asked for {day.isoformat()} on {today.isoformat()}. "
            "A forecast issued today for a day further out is not the day-ahead "
            "forecast the model was trained on; for a past day, use the archive."
        )

    owned = client is None
    client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    try:
        columns = []
        for position, city in enumerate(cities):
            if position:
                sleep(min_interval)
            columns.append(
                _fetch_hourly(
                    city,
                    url=FORECAST_URL,
                    variable=LIVE_VARIABLE,
                    start=day,
                    end=day,
                    client=client,
                )
            )
    finally:
        if owned:
            client.close()

    frame = pd.concat(columns, axis=1).sort_index()
    frame["temperature_c"] = population_weighted(frame, cities)
    return frame


def weather_features(frame: pd.DataFrame, *, base_c: float = 18.0) -> pd.DataFrame:
    """The three weather columns the design matrix expects, and nothing else.

    Both fetchers return one column per city alongside the aggregate, which is
    useful for diagnosis and wrong to hand to a model that was never trained on
    them. Narrowing here — rather than at each call site — means a caller cannot
    forget, and means the column set has one definition instead of several that
    slowly disagree.
    """
    temperature = frame["temperature_c"]
    return degree_days(temperature, base_c=base_c).join(temperature)[
        ["temperature_c", "hdd", "cdd"]
    ]


def population_weighted(frame: pd.DataFrame, cities: tuple[City, ...] = CITIES) -> pd.Series:
    """Combine city temperatures into one national signal.

    Missing cities are excluded and the remaining weights renormalised, so a gap
    in one city shifts the aggregate slightly rather than blanking the hour.
    """
    weights = pd.Series(
        {city.name: city.population_millions for city in cities if city.name in frame.columns}
    )
    subset = frame[list(weights.index)]

    available = subset.notna().mul(weights, axis=1)
    total = available.sum(axis=1)
    weighted = subset.mul(weights, axis=1).sum(axis=1, min_count=1)

    return (weighted / total.replace(0, pd.NA)).rename("temperature_c")


def degree_days(temperature: pd.Series, *, base_c: float = 18.0) -> pd.DataFrame:
    """Heating and cooling degrees relative to a comfort baseline.

    Electricity demand responds to temperature in a V shape: it rises when it is
    cold enough to heat and again when it is hot enough to cool, and is flat in
    between. A linear temperature term cannot express that — it has to choose a
    single slope and gets both ends wrong. Splitting into two one-sided variables
    lets even a linear model fit each arm separately, and gives a tree model the
    split point for free.

    18 °C is the conventional European base. It is a convention, not a
    measurement, and is worth re-checking against this series later.
    """
    return pd.DataFrame(
        {
            "hdd": (base_c - temperature).clip(lower=0),
            "cdd": (temperature - base_c).clip(lower=0),
        },
        index=temperature.index,
    )
