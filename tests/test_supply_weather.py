"""Tests for the supply-side weather layer.

Two of these guard physics rather than plumbing. Cubing wind speed after the
weighted average instead of before is arithmetically tidy, silently wrong, and
would never raise — the sort of thing only a pinned test catches.
"""

from __future__ import annotations

import pandas as pd
import pytest

from powerforecast.data import supply_weather as sw
from powerforecast.data.weather import CITIES, WeatherError


def _frame(values: dict[str, list[float]]) -> pd.DataFrame:
    periods = len(next(iter(values.values())))
    index = pd.date_range("2026-05-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(values, index=index)


# --------------------------------------------------------------------------- #
# Availability — the reason this module exists at all
# --------------------------------------------------------------------------- #


def test_variables_are_day_ahead_forecasts_not_realised_values() -> None:
    """Realised irradiance is the single worst leak available here.

    A bidder at 12:30 on D-1 had a forecast, not a measurement. Both variables
    must carry Open-Meteo's `_previous_day1` suffix or the whole availability
    argument in ADR 0005 stops applying.
    """
    assert sw.SOLAR_VARIABLE.endswith("_previous_day1")
    assert sw.WIND_VARIABLE.endswith("_previous_day1")


def test_wind_is_measured_at_hub_height_not_ten_metres() -> None:
    """10 m is the meteorological standard and the wrong height for a turbine."""
    assert "100m" in sw.WIND_VARIABLE


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #


def test_wind_index_cubes_the_speed() -> None:
    """Power in moving air goes as v^3, so doubling the speed is eight times."""
    slow = _frame({site.name: [3.0] for site in sw.WIND_SITES})
    fast = _frame({site.name: [6.0] for site in sw.WIND_SITES})

    assert sw.wind_index(slow).iloc[0] == pytest.approx(27.0)
    assert sw.wind_index(fast).iloc[0] == pytest.approx(216.0)
    assert sw.wind_index(fast).iloc[0] == pytest.approx(8 * sw.wind_index(slow).iloc[0])


def test_wind_index_cubes_before_averaging_not_after() -> None:
    """A turbine responds to the wind at its own site.

    With one site becalmed and one blowing hard, the sum of site powers is far
    larger than the power of the average site. Cubing after the average would
    understate exactly the unevenness that drives real output.
    """
    sites = (sw.Site("a", 0.0, 0.0, 1.0), sw.Site("b", 0.0, 0.0, 1.0))
    frame = _frame({"a": [0.0], "b": [10.0]})

    cube_then_average = sw.wind_index(frame, sites).iloc[0]
    average_then_cube = ((0.0 + 10.0) / 2) ** 3

    assert cube_then_average == pytest.approx(500.0)
    assert average_then_cube == pytest.approx(125.0)
    assert cube_then_average > average_then_cube


def test_solar_index_is_linear_in_irradiance() -> None:
    """Unlike wind, PV output is close to linear in GHI — no cube here."""
    dim = _frame({site.name: [200.0] for site in sw.SOLAR_SITES})
    bright = _frame({site.name: [400.0] for site in sw.SOLAR_SITES})

    assert sw.solar_index(dim).iloc[0] == pytest.approx(200.0)
    assert sw.solar_index(bright).iloc[0] == pytest.approx(400.0)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_weights_renormalise_when_a_site_is_missing() -> None:
    """A gap at one site should shift the aggregate, not blank the hour."""
    sites = (sw.Site("a", 0.0, 0.0, 3.0), sw.Site("b", 0.0, 0.0, 1.0))
    both = _frame({"a": [100.0], "b": [200.0]})
    only_b = _frame({"b": [200.0]})

    assert sw.capacity_weighted(both, sites).iloc[0] == pytest.approx(125.0)
    assert sw.capacity_weighted(only_b, sites).iloc[0] == pytest.approx(200.0)


def test_missing_every_site_raises_rather_than_returning_nothing() -> None:
    sites = (sw.Site("a", 0.0, 0.0, 1.0),)

    with pytest.raises(WeatherError, match="none of the expected sites"):
        sw.capacity_weighted(_frame({"z": [1.0]}), sites)


def test_weighting_actually_favours_the_heavier_site() -> None:
    """Guards against a renormalisation bug that degrades to a plain mean."""
    sites = (sw.Site("heavy", 0.0, 0.0, 9.0), sw.Site("light", 0.0, 0.0, 1.0))
    frame = _frame({"heavy": [0.0], "light": [100.0]})

    weighted = sw.capacity_weighted(frame, sites).iloc[0]

    assert weighted == pytest.approx(10.0)
    assert weighted != pytest.approx(50.0), "fell back to an unweighted mean"


# --------------------------------------------------------------------------- #
# Siting
# --------------------------------------------------------------------------- #


def test_supply_sites_are_not_the_demand_cities() -> None:
    """Population weighting is right for demand and wrong for generation.

    If these lists ever collapse back onto `CITIES`, the supply signal would be
    telling the model about İstanbul's cloud cover instead of Konya's sun.
    """
    demand = {city.name for city in CITIES}
    solar = {site.name for site in sw.SOLAR_SITES}
    wind = {site.name for site in sw.WIND_SITES}

    assert solar != demand
    assert wind != demand
    assert "istanbul" not in solar, "İstanbul carries little solar capacity"


def test_solar_and_wind_lists_are_geographically_distinct() -> None:
    """The two resources are in different places, and İzmir is the exception.

    Overlap is expected and fine; identical lists would mean one of them was
    copied and never edited.
    """
    solar = {site.name for site in sw.SOLAR_SITES}
    wind = {site.name for site in sw.WIND_SITES}

    assert solar != wind
    assert len(solar & wind) < min(len(solar), len(wind))


def test_every_site_has_a_positive_weight() -> None:
    for site in sw.SOLAR_SITES + sw.WIND_SITES:
        assert site.weight > 0, f"{site.name} would silently drop out of the average"


def test_sites_are_inside_turkey() -> None:
    """A transposed latitude/longitude pair is a classic and silent error."""
    for site in sw.SOLAR_SITES + sw.WIND_SITES:
        assert 35.8 <= site.latitude <= 42.2, f"{site.name} latitude looks wrong"
        assert 25.5 <= site.longitude <= 45.0, f"{site.name} longitude looks wrong"
