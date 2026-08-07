"""Tests for the forecasting service.

Two decisions shape this file.

**No server is started.** `TestClient` drives the application in-process,
through the same request pipeline uvicorn would use. There is no port to bind,
no race between "server is starting" and "test is asking", and no flake from a
port already in use. If a test needs a real socket to be meaningful, it is not
testing the application — it is testing uvicorn.

**No real data is read.** Every test builds its own panel with
`synthetic_panel`. Depending on `data/` would make the suite pass or fail
according to what somebody last downloaded, and would make it impossible to
test the interesting case — a delivery day the data does not reach — because
that case moves every time the ingestion job runs.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from powerforecast.features.build import FeatureSpec
from powerforecast.models.estimators import make_ridge
from powerforecast.models.persistence import save_model
from powerforecast.serving.app import create_app

# The synthetic panel covers this range; the tests refer to it rather than to
# today, so they do not start failing on a date in the future.
PANEL_START = "2025-01-01"
PANEL_DAYS = 120
SERVED_DAY = "2025-04-20"  # inside the panel, past the longest (336h) lag
UNREACHED_DAY = "2025-09-01"  # after the panel ends


def synthetic_panel(days: int = PANEL_DAYS) -> pd.DataFrame:
    """A panel with the shape of the real one and none of its size.

    The values need to be plausible rather than realistic: the service is being
    tested, not the model. What matters is that every column the design matrix
    expects exists, the index is a continuous hourly UTC grid, and there is
    enough history for a 336-hour lag to resolve.
    """
    index = pd.date_range(PANEL_START, periods=days * 24, freq="h", tz="UTC")
    hour = index.tz_convert("Europe/Istanbul").hour
    daily = 40_000 + 6_000 * pd.Series(hour, index=index).apply(lambda h: (h - 4) % 24 / 24)
    return pd.DataFrame(
        {
            "consumption_mwh": daily.to_numpy(),
            "load_plan_mwh": daily.to_numpy() * 0.99,
            "day_ahead_price_try_mwh": 2_000.0,
            "temperature_c": 15.0,
            "hdd": 3.0,
            "cdd": 0.0,
        },
        index=index,
    )


@pytest.fixture
def client(tmp_path) -> TestClient:
    """An app wired to a model and a panel that exist only for this test.

    This is what `create_app` being a factory buys. Nothing is patched, no
    global is mutated, and the application under test is the real one — it has
    simply been told where to look.
    """
    panel = synthetic_panel()
    spec = FeatureSpec(include_weather=True, include_holidays=True)

    from powerforecast.features.build import build_design_matrix, usable_rows

    features, target = build_design_matrix(panel, spec)
    usable = usable_rows(features, target)
    estimator = make_ridge().fit(features.loc[usable], target.loc[usable])
    save_model(
        estimator, name="test-model", features=features.loc[usable], spec=spec, directory=tmp_path
    )

    app = create_app(
        model_name="test-model",
        model_directory=tmp_path,
        panel_loader=synthetic_panel,
        # The model was fitted seconds ago by this very process, so the versions
        # match by construction and the check has nothing to catch. Left on
        # because turning it off in tests would mean the production default is
        # the one configuration never exercised.
        require_environment=True,
    )
    # The `with` block is not optional: it is what runs `lifespan`. Without it
    # the model is never loaded and every request fails on missing state — a
    # confusing failure that has nothing to do with the code being tested.
    with TestClient(app) as client:
        yield client


def test_health_reports_which_model_is_serving(client):
    """The point of /health is identity, not liveness."""
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_name"] == "test-model"
    assert body["model_version"]
    assert body["model_trained_rows"] > 0
    assert body["panel_end"].startswith("2025-04-30")


def test_forecast_returns_every_hour_of_the_delivery_day(client):
    response = client.get("/forecast", params={"date": SERVED_DAY})
    assert response.status_code == 200
    body = response.json()

    assert len(body["hours"]) == 24
    assert [h["local_hour"] for h in body["hours"]] == list(range(24))
    assert all(h["forecast_mwh"] > 0 for h in body["hours"])


def test_forecast_carries_its_own_provenance(client):
    """A forecast has to say which model produced it and what it could see."""
    body = client.get("/forecast", params={"date": SERVED_DAY}).json()

    assert body["model_name"] == "test-model"
    assert body["model_version"] == client.get("/health").json()["model_version"]
    # Origin is 11:00 local on the previous day — bids close at 12:30 and the
    # value stamped 12:00 is still accumulating.
    assert body["forecast_origin"].startswith("2025-04-19T11:00:00")
    assert body["delivery_date"] == SERVED_DAY


def test_hours_are_contiguous_and_start_at_local_midnight(client):
    body = client.get("/forecast", params={"date": SERVED_DAY}).json()
    stamps = pd.DatetimeIndex([h["timestamp"] for h in body["hours"]])

    assert stamps.is_monotonic_increasing
    assert (stamps.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert stamps[0] == pd.Timestamp("2025-04-19T21:00:00Z")  # 00:00 Istanbul


def test_a_day_the_data_does_not_reach_is_refused_not_shortened(client):
    """The failure that matters: 23 hours would be worse than an error."""
    response = client.get("/forecast", params={"date": UNREACHED_DAY})
    assert response.status_code == 422

    detail = response.json()["detail"]
    assert "0 of 24 hours" in detail["message"]
    assert detail["missing_features"]
    assert detail["data_available_until"].startswith("2025-04-30")


def test_a_malformed_date_is_rejected_by_the_framework(client):
    """Typing the parameter as `date` is what makes this a 422 and not a 500."""
    response = client.get("/forecast", params={"date": "elma"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "date"]


def test_the_service_refuses_to_start_without_a_model(tmp_path):
    """Failing at startup beats starting and then erroring on every request."""
    app = create_app(model_name="no-such-model", model_directory=tmp_path)

    with pytest.raises(RuntimeError, match="cannot start without a model"), TestClient(app):
        pass  # pragma: no cover — the context manager raises on entry


def test_the_panel_is_cached_and_then_refreshed(tmp_path):
    """The model and the data do not live on the same clock."""
    panel = synthetic_panel()
    spec = FeatureSpec()
    from powerforecast.features.build import build_design_matrix, usable_rows

    features, target = build_design_matrix(panel, spec)
    usable = usable_rows(features, target)
    save_model(
        make_ridge().fit(features.loc[usable], target.loc[usable]),
        name="test-model",
        features=features.loc[usable],
        spec=spec,
        directory=tmp_path,
    )

    reads = 0

    def counting_loader() -> pd.DataFrame:
        nonlocal reads
        reads += 1
        return panel

    app = create_app(
        model_name="test-model",
        model_directory=tmp_path,
        panel_loader=counting_loader,
        panel_ttl_seconds=1_000,
    )
    with TestClient(app) as client:
        for _ in range(3):
            client.get("/health")
        assert reads == 1, "the panel should be read once while the cache is warm"

        # Expire it. Setting the TTL to zero is how a test asserts on cache
        # behaviour without sleeping — a test that waits is a test that is slow
        # and, on a loaded machine, flaky.
        client.app.state.service.panel.ttl_seconds = 0
        client.get("/health")
        assert reads == 2
