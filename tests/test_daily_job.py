"""Tests for the daily forecast job and the shared day-forecast module.

Unattended code is judged by what it does when something is missing, so most of
these assert on **exit codes** rather than on output. A scheduler cannot read
prose; it sees an integer, and the integer has to distinguish "no model" from
"the weather API is down", because those need different people.

The one case worth naming: `test_a_second_run_updates_rather_than_duplicates`.
Schedulers retry. A job that is not idempotent does not merely waste a run — it
corrupts the record the monitoring layer is built on.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from powerforecast.features.build import FeatureSpec, build_design_matrix, usable_rows
from powerforecast.forecasts.day import (
    IncompleteForecastError,
    delivery_hours,
    forecast_day,
)
from powerforecast.forecasts.store import read_forecasts
from powerforecast.jobs import daily_forecast
from powerforecast.models.estimators import make_ridge
from powerforecast.models.persistence import save_model

PANEL_START = "2025-01-01"
PANEL_DAYS = 120
SERVED_DAY = date(2025, 4, 20)  # inside the panel, past the 336-hour lag
UNREACHED_DAY = date(2025, 9, 1)


def synthetic_panel(days: int = PANEL_DAYS) -> pd.DataFrame:
    index = pd.date_range(PANEL_START, periods=days * 24, freq="h", tz="UTC")
    hour = pd.Series(index.tz_convert("Europe/Istanbul").hour, index=index)
    load = 40_000 + 6_000 * ((hour - 4) % 24) / 24
    return pd.DataFrame(
        {
            "consumption_mwh": load,
            "load_plan_mwh": load * 0.99,
            "day_ahead_price_try_mwh": 2_000.0,
            "temperature_c": 15.0,
            "hdd": 3.0,
            "cdd": 0.0,
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def _isolate_from_real_data(monkeypatch):
    """Every job test reads the synthetic panel, never `data/`.

    Without this the suite passes or fails according to what was last
    downloaded, and the case that matters most — a delivery day the data does
    not reach — becomes untestable, because the real panel keeps growing past
    whatever date the test picked. Caught the hard way: the unreachable-day test
    initially passed with exit code 0, because the real panel did reach it.
    """
    monkeypatch.setattr(daily_forecast, "load_panel", synthetic_panel)


@pytest.fixture
def model_dir(tmp_path):
    """A fitted model saved in a temporary store."""
    panel = synthetic_panel()
    spec = FeatureSpec()
    features, target = build_design_matrix(panel, spec)
    usable = usable_rows(features, target)
    save_model(
        make_ridge().fit(features.loc[usable], target.loc[usable]),
        name="load-lightgbm",
        features=features.loc[usable],
        spec=spec,
        directory=tmp_path / "models",
    )
    return tmp_path / "models"


# --------------------------------------------------------------------------- #
# The shared module
# --------------------------------------------------------------------------- #


def test_delivery_hours_covers_the_local_day():
    hours = delivery_hours(SERVED_DAY)

    assert len(hours) == 24
    assert str(hours.tz) == "UTC"
    assert hours[0] == pd.Timestamp("2025-04-19T21:00:00Z")  # 00:00 Istanbul
    assert (hours.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_forecast_day_returns_a_whole_day_with_its_provenance(model_dir):
    from powerforecast.models.persistence import load_model

    produced = forecast_day(
        load_model("load-lightgbm", directory=model_dir),
        synthetic_panel(),
        SERVED_DAY,
        generated_at=datetime(2025, 4, 19, 9, tzinfo=UTC),
    )

    assert len(produced.values) == 24
    # Origin is 11:00 local on the previous day: bids close at 12:30 and the
    # value stamped 12:00 is still accumulating.
    assert produced.forecast_origin == pd.Timestamp("2025-04-19T11:00:00+03:00")

    frame = produced.to_frame()
    assert frame.index.name == "timestamp"
    assert frame["model_version"].nunique() == 1


def test_forecast_day_refuses_a_day_it_cannot_answer_in_full(model_dir):
    from powerforecast.models.persistence import load_model

    with pytest.raises(IncompleteForecastError) as caught:
        forecast_day(
            load_model("load-lightgbm", directory=model_dir), synthetic_panel(), UNREACHED_DAY
        )

    error = caught.value
    assert error.expected == 24
    assert error.available == 0
    assert error.missing_features
    assert "hint" in error.as_dict()


# --------------------------------------------------------------------------- #
# The job
# --------------------------------------------------------------------------- #


def test_a_successful_run_exits_zero_and_stores_the_day(tmp_path, model_dir):
    code = daily_forecast.run(
        SERVED_DAY,
        model_directory=model_dir,
        processed_root=tmp_path / "processed",
        fetch_weather=False,
    )

    assert code == daily_forecast.EXIT_OK
    assert len(read_forecasts(root=tmp_path / "processed")) == 24


def test_a_second_run_updates_rather_than_duplicates(tmp_path, model_dir):
    """Schedulers retry. Non-idempotent jobs corrupt data on the second run."""
    for _ in range(2):
        assert (
            daily_forecast.run(
                SERVED_DAY,
                model_directory=model_dir,
                processed_root=tmp_path / "processed",
                fetch_weather=False,
            )
            == daily_forecast.EXIT_OK
        )

    assert len(read_forecasts(root=tmp_path / "processed")) == 24


def test_a_missing_model_exits_with_its_own_code(tmp_path):
    code = daily_forecast.run(
        SERVED_DAY,
        model_directory=tmp_path / "empty",
        processed_root=tmp_path / "processed",
        fetch_weather=False,
    )

    assert code == daily_forecast.EXIT_MODEL


def test_an_unreachable_day_exits_incomplete_and_stores_nothing(tmp_path, model_dir):
    """Twenty of twenty-four hours would be four hours nobody bid."""
    code = daily_forecast.run(
        UNREACHED_DAY,
        model_directory=model_dir,
        processed_root=tmp_path / "processed",
        fetch_weather=False,
    )

    assert code == daily_forecast.EXIT_INCOMPLETE
    assert read_forecasts(root=tmp_path / "processed").empty


def test_a_weather_failure_exits_with_its_own_code(tmp_path, model_dir, monkeypatch):
    """Falling back to yesterday's temperature would look exactly like success."""
    from powerforecast.data.weather import WeatherError

    def unavailable(*args, **kwargs):
        raise WeatherError("Open-Meteo unreachable")

    monkeypatch.setattr(daily_forecast, "fetch_live_all_cities", unavailable)

    code = daily_forecast.run(
        SERVED_DAY,
        model_directory=model_dir,
        processed_root=tmp_path / "processed",
        fetch_weather=True,
    )

    assert code == daily_forecast.EXIT_WEATHER
    assert read_forecasts(root=tmp_path / "processed").empty


def test_every_failure_code_is_distinct():
    """One shared non-zero code would make every alert say the same thing."""
    codes = {
        daily_forecast.EXIT_OK,
        daily_forecast.EXIT_MODEL,
        daily_forecast.EXIT_WEATHER,
        daily_forecast.EXIT_INCOMPLETE,
        daily_forecast.EXIT_UNEXPECTED,
    }
    assert len(codes) == 5
