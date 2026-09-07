"""Solar and wind conditions at the generators, as forecast at bid time.

Price is set where supply meets demand, and until now the model has seen only
demand. This module supplies the other half — but it does so through *weather*,
not through the market's own supply schedules, and that choice is the whole
point of the file.

**Why not KGÜP.** The obvious supply feature is the finalised daily generation
programme. It is published, it is day-ahead, and it is broken down by source.
It is also submitted between 14:00 and 15:30 on D-1 — *after* the day-ahead
market closes at 12:30 — because it is a consequence of the market clearing
rather than an input to it. Using it to forecast the day-ahead price would feed
the model something derived from the answer. Same for EAK, on the same
timetable. Neither is admissible under ADR 0004, and both look excellent in
backtest, which is precisely what makes them dangerous.

**Why weather is admissible.** A weather forecast for delivery day D exists on
D-1 and is on the bidder's desk before the deadline. That is the identical
argument ADR 0005 already makes for temperature, and this module reuses the same
mechanism: Open-Meteo's *archived forecasts* (`_previous_day1`), never the
realised values.

**Why separate sites from `weather.py`.** The existing eight cities are weighted
by population, which is a sensible proxy for where electricity is *consumed*.
It is the wrong proxy for where it is *generated*: Turkish solar sits on the
Konya-Karaman plateau and in the southeast, wind on the Çanakkale-Balıkesir-İzmir
corridor. Population-weighted irradiance would give İstanbul's cloud more weight
than Konya's sun, which is backwards for a supply signal.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
import pandas as pd

from powerforecast.data.weather import (
    ARCHIVE_URL,
    DEFAULT_TIMEOUT,
    FORECAST_URL,
    MIN_REQUEST_INTERVAL,
    WeatherError,
)

# Global horizontal irradiance drives photovoltaic output; hub-height wind speed
# drives turbines. 10 m wind is the standard meteorological height and is the
# wrong one here — a modern turbine's hub is nearer 100 m, where the wind is
# both faster and less disturbed by terrain.
SOLAR_VARIABLE = "shortwave_radiation_previous_day1"
WIND_VARIABLE = "wind_speed_100m_previous_day1"


@dataclass(frozen=True)
class Site:
    """A generation region, weighted by roughly how much capacity sits there.

    The weights are **approximate and ordinal**. They encode "Konya carries more
    solar than Kayseri", not an audited megawatt figure, and they are not
    reported as a result anywhere. What they have to get right is the geography;
    getting the third significant figure right would not change a
    capacity-weighted average enough to matter, and pretending otherwise would
    put a precise-looking number in front of a rough one.
    """

    name: str
    latitude: float
    longitude: float
    weight: float


# The solar belt: the Konya-Karaman plateau, the southeast, and central Anatolia.
SOLAR_SITES = (
    Site("konya", 37.8746, 32.4932, 3.0),
    Site("karaman", 37.1811, 33.2150, 2.0),
    Site("sanliurfa", 37.1591, 38.7969, 2.0),
    Site("kayseri", 38.7312, 35.4787, 1.5),
    Site("nigde", 37.9698, 34.6766, 1.5),
    Site("ankara", 39.9334, 32.8597, 1.5),
)

# The wind corridor: the Marmara-Aegean coast, plus the Hatay-Osmaniye gap where
# the Amanos mountains funnel a persistent northerly.
WIND_SITES = (
    Site("canakkale", 40.1553, 26.4142, 3.0),
    Site("balikesir", 39.6484, 27.8826, 3.0),
    Site("izmir", 38.4237, 27.1428, 2.5),
    Site("manisa", 38.6191, 27.4289, 1.5),
    Site("hatay", 36.4018, 36.3498, 1.5),
    Site("osmaniye", 37.0748, 36.2464, 1.0),
)


def fetch_site(
    site: Site,
    variable: str,
    start: date,
    end: date,
    *,
    client: httpx.Client | None = None,
) -> pd.Series:
    """Archived day-ahead forecasts of one variable at one site.

    Deliberately not shared with `weather._fetch_hourly`: that function is
    keyed on a `City` and its population, and widening it to accept either type
    would make the demand path carry a parameter it never uses. Two short
    fetchers are easier to read than one general one.
    """
    return _fetch_one(site, ARCHIVE_URL, variable, start, end, client=client)


def _fetch_one(
    site: Site,
    url: str,
    variable: str,
    start: date,
    end: date,
    *,
    client: httpx.Client | None = None,
) -> pd.Series:
    """One site, one variable, one date range, from either Open-Meteo endpoint."""
    owned = client is None
    client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    try:
        response = client.get(
            url,
            params={
                "latitude": site.latitude,
                "longitude": site.longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": variable,
                "timezone": "UTC",
            },
        )
    except httpx.HTTPError as exc:
        raise WeatherError(f"could not reach Open-Meteo for {site.name}: {exc}") from exc
    finally:
        if owned:
            client.close()

    if response.status_code >= 400:
        raise WeatherError(
            f"Open-Meteo rejected the request for {site.name} "
            f"with HTTP {response.status_code}: {response.text[:200]}"
        )

    body = response.json()
    hourly = body.get("hourly")
    if not hourly or variable not in hourly:
        raise WeatherError(
            f"Open-Meteo returned no {variable!r} for {site.name}; got keys {list(body)}"
        )

    index = pd.to_datetime(hourly["time"], utc=True)
    values = pd.to_numeric(hourly[variable], errors="coerce")
    series = pd.Series(values, index=index, name=site.name, dtype="float64")
    series.index.name = "timestamp"
    return series


def fetch_sites(
    sites: tuple[Site, ...],
    variable: str,
    start: date,
    end: date,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    min_interval: float = MIN_REQUEST_INTERVAL,
) -> pd.DataFrame:
    """Every site for one variable, one column per site."""
    owned = client is None
    client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    try:
        columns = []
        for position, site in enumerate(sites):
            if position:
                sleep(min_interval)
            columns.append(fetch_site(site, variable, start, end, client=client))
    finally:
        if owned:
            client.close()

    return pd.concat(columns, axis=1).sort_index()


def capacity_weighted(frame: pd.DataFrame, sites: tuple[Site, ...]) -> pd.Series:
    """Combine site columns into one national signal.

    Missing sites are excluded and the remaining weights renormalised, so a gap
    at one site shifts the aggregate slightly rather than blanking the hour —
    the same rule `weather.population_weighted` follows, for the same reason.
    """
    weights = pd.Series({site.name: site.weight for site in sites if site.name in frame.columns})
    if weights.empty:
        raise WeatherError(f"none of the expected sites are present: {list(frame.columns)}")

    present = frame[list(weights.index)]
    normalised = weights / weights.sum()
    return (present * normalised).sum(axis=1, min_count=1)


def solar_index(frame: pd.DataFrame, sites: tuple[Site, ...] = SOLAR_SITES) -> pd.Series:
    """Capacity-weighted irradiance, in W/m^2.

    Photovoltaic output is close to linear in irradiance over the range that
    matters, so a weighted mean of GHI is a defensible proxy without modelling
    panel physics. It ignores temperature derating — panels lose efficiency when
    hot — which is a real effect and a deliberate omission: the panel
    temperature is not observable and adding an unvalidated correction would be
    inventing precision.
    """
    return capacity_weighted(frame, sites).rename("solar_index")


def wind_index(frame: pd.DataFrame, sites: tuple[Site, ...] = WIND_SITES) -> pd.Series:
    """Capacity-weighted wind *power* proxy, in (m/s)^3.

    The cube is the point. Available power in moving air is proportional to the
    cube of velocity, so averaging wind *speed* across sites and handing that to
    the model would compress exactly the variation that matters: 12 m/s carries
    roughly eight times the power of 6 m/s, not twice.

    Cubing before the weighted average rather than after also matters. A turbine
    responds to the wind at its own site, so the national signal is the sum of
    site powers, not the power of the average site — and those differ whenever
    the wind is unevenly distributed, which is most of the time.

    Left uncapped. A real turbine's power curve saturates at rated speed and
    cuts out entirely in a gale, so this overstates the high tail. Modelling the
    curve needs a rated speed and a cut-out this project has not measured, and a
    tree can learn a saturating response from an uncapped input anyway.
    """
    return capacity_weighted(frame**3, sites).rename("wind_index")


# The live endpoint, for a delivery day that has not happened yet. Same pair of
# variables without the `_previous_day1` suffix: on D-1 the live run *is* the
# D-1 run, which is exactly what the archive will hand back for D months later.
LIVE_SOLAR_VARIABLE = "shortwave_radiation"
LIVE_WIND_VARIABLE = "wind_speed_100m"


def fetch_live(
    day: date,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    min_interval: float = MIN_REQUEST_INTERVAL,
    today: date | None = None,
) -> pd.DataFrame:
    """Solar and wind for a delivery day that has not happened yet.

    The supply-side twin of `weather.fetch_live_all_cities`, and it exists for
    the same reason: the archive cannot answer about tomorrow, so without this
    a price model — which needs these columns — can never forecast the only day
    anyone wants forecast.

    Found by running the job rather than by reading it. Every test passed, the
    model was trained and saved, and the first real invocation refused the day
    with `missing_features: ['solar_index', 'wind_index']`.

    The same two guarantees the temperature path makes are kept here:

    * **Same quantity, not a similar one.** Training reads the D-1 run out of
      the archive; serving runs on D-1 and reads the live D-1 run. One
      distribution seen from two sides of the same day.
    * **Only ever the next day.** The live endpoint will happily return D+5,
      issued today and far worse than anything the model was trained on, and
      nothing about that response would look wrong. Refused here rather than
      discovered in the error weeks later.

    Two UTC days are requested for the same reason as the temperature path: a
    local delivery day starts at 21:00 UTC the day before, so a single UTC date
    covers only 21 of its hours.
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
    start, end = day - timedelta(days=1), day

    try:
        solar = _fetch_live_sites(
            SOLAR_SITES, LIVE_SOLAR_VARIABLE, start, end, client, sleep, min_interval
        )
        wind = _fetch_live_sites(
            WIND_SITES, LIVE_WIND_VARIABLE, start, end, client, sleep, min_interval
        )
    finally:
        if owned:
            client.close()

    frame = pd.concat([solar.add_prefix("ghi_"), wind.add_prefix("wind_")], axis=1).sort_index()
    frame["solar_index"] = solar_index(solar)
    frame["wind_index"] = wind_index(wind)
    return frame


def _fetch_live_sites(
    sites: tuple[Site, ...],
    variable: str,
    start: date,
    end: date,
    client: httpx.Client,
    sleep: Callable[[float], None],
    min_interval: float,
) -> pd.DataFrame:
    columns = []
    for position, site in enumerate(sites):
        if position:
            sleep(min_interval)
        columns.append(_fetch_one(site, FORECAST_URL, variable, start, end, client=client))
    return pd.concat(columns, axis=1).sort_index()
