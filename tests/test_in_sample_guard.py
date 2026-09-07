"""The guard that stops a fit being reported as a forecast.

This exists because the mistake was made. Backfilling ten delivery days through
the daily job produced MAE 285 against a backtest MAE of 914 — a threefold
improvement that was entirely an artefact of every one of those days falling
inside the model's training window. The number was excellent and worthless.

Nothing raised. Nothing could have: every component behaved correctly, and the
error was in what the aggregate *meant*. That is the expensive kind, so it gets
its own file.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from powerforecast.features.build import FeatureSpec, build_design_matrix, usable_rows
from powerforecast.models.estimators import make_ridge
from powerforecast.models.persistence import read_card, save_model
from powerforecast.models.train import train
from powerforecast.monitoring.verify import detect_drift, out_of_sample, overall, verify

PANEL_START = "2025-01-01"
PANEL_DAYS = 200


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


def forecasts_for(
    index: pd.DatetimeIndex,
    *,
    name: str,
    version: str,
    error: float = 400.0,
) -> pd.DataFrame:
    sign = pd.Series(np.where(np.arange(len(index)) % 2 == 0, 1.0, -1.0), index=index)
    return pd.DataFrame(
        {
            "forecast_value": 40_000.0 - sign * error,
            "model_name": name,
            "model_version": version,
            "forecast_origin": index[0],
            "generated_at": pd.Timestamp("2025-01-01", tz="UTC"),
        },
        index=index,
    )


@pytest.fixture
def store(tmp_path):
    """A model trained only on the first half of the panel."""
    panel = synthetic_panel()
    cutoff = panel.index[len(panel) // 2]
    saved = train(
        model="ridge",
        train_until=cutoff,
        panel=panel,
        name="dated",
        directory=tmp_path,
        notes="half the panel",
    )
    return tmp_path, saved.card.version, cutoff, panel


# --------------------------------------------------------------------------- #
# Training to a date
# --------------------------------------------------------------------------- #


def test_train_until_moves_the_recorded_boundary(store):
    directory, version, cutoff, _ = store
    card = read_card("dated", version, directory=directory)

    assert pd.Timestamp(card.train_end) <= cutoff
    assert card.n_train_rows > 0


def test_train_until_before_any_usable_row_is_refused(tmp_path):
    """Silently training on nothing would produce a model that predicts a constant."""
    with pytest.raises(ValueError, match="no usable rows"):
        train(
            model="ridge",
            train_until="2024-01-01",
            panel=synthetic_panel(),
            name="impossible",
            directory=tmp_path,
        )


def test_the_card_can_be_read_without_unpickling(store):
    """The audit path must be cheap, or it will not be run often enough."""
    directory, version, _, _ = store
    card = read_card("dated", version, directory=directory)

    assert card.name == "dated"
    assert card.feature_columns


# --------------------------------------------------------------------------- #
# The guard itself
# --------------------------------------------------------------------------- #


def test_hours_inside_the_training_window_are_marked(store):
    directory, version, cutoff, panel = store
    index = panel.index[::24]  # one hour a day across the whole panel, both sides of the cutoff

    verified = verify(
        forecasts_for(index, name="dated", version=version),
        panel,
        model_directory=directory,
    )

    assert verified["in_sample"].any()
    assert (~verified["in_sample"]).any()
    assert verified.loc[verified.index <= cutoff, "in_sample"].all()
    assert not verified.loc[verified.index > cutoff, "in_sample"].any()


def test_metrics_exclude_in_sample_hours_by_default(store):
    """The default has to be the honest one. Investigation is the opt-in."""
    directory, version, _cutoff, panel = store
    index = panel.index[::24]

    verified = verify(
        forecasts_for(index, name="dated", version=version),
        panel,
        model_directory=directory,
    )

    assert overall(verified).n == len(out_of_sample(verified))
    assert overall(verified, include_in_sample=True).n == len(verified)


def test_a_report_made_entirely_of_in_sample_hours_yields_no_verdict(store):
    """The exact situation that produced MAE 285: everything was a fit."""
    directory, version, cutoff, panel = store
    index = panel.index[panel.index <= cutoff][::24]

    verified = verify(
        forecasts_for(index, name="dated", version=version),
        panel,
        model_directory=directory,
    )

    assert verified["in_sample"].all()
    assert overall(verified) is None
    verdict = detect_drift(verified)
    assert verdict.status == "INSUFFICIENT"
    assert "training window" in verdict.detail


def test_each_model_version_gets_its_own_boundary(store):
    """A retrained model has a later boundary; the same hour differs between them."""
    directory, old_version, cutoff, panel = store
    newer = train(
        model="ridge",
        panel=panel,
        name="dated",
        directory=directory,
        notes="the whole panel",
    ).card.version

    index = panel.index[panel.index > cutoff][::24][:40]
    both = pd.concat(
        [
            forecasts_for(index, name="dated", version=old_version),
            forecasts_for(index, name="dated", version=newer, error=300.0).assign(
                # A later run, so `latest_run` has an unambiguous winner. Equal
                # timestamps would make the outcome depend on row order, and a
                # test whose result depends on row order is not testing anything.
                generated_at=pd.Timestamp("2025-06-01", tz="UTC")
            ),
        ]
    ).sort_index()

    verified = verify(both, panel, model_directory=directory)

    # `latest_run` keeps one row per hour and the newer model wins it — and for
    # the newer model these very hours *are* inside the training window.
    assert set(verified["model_version"]) == {newer}
    assert verified["in_sample"].all()


def test_an_unknown_version_is_kept_and_reported_rather_than_assumed(store, caplog):
    """Guessing "in-sample" would delete real evidence; guessing the other way keeps it."""
    directory, _, _, panel = store
    index = panel.index[::24][:40]

    with caplog.at_level("WARNING"):
        verified = verify(
            forecasts_for(index, name="dated", version="20990101T000000Z"),
            panel,
            model_directory=directory,
        )

    assert not verified["in_sample"].any()
    assert "no card for" in caplog.text


def test_the_saved_model_is_still_usable_after_a_dated_train(store):
    """A cut-off must not quietly produce a model that cannot predict."""
    directory, version, _, panel = store
    from powerforecast.models.persistence import load_model

    model = load_model("dated", version, directory=directory)
    features, target = build_design_matrix(panel, FeatureSpec())
    rows = usable_rows(features, target)[-24:]

    predictions = model.predict(features.loc[rows])

    assert len(predictions) == 24
    assert predictions.notna().all()


def test_a_model_fitted_here_and_now_matches_its_own_environment(tmp_path):
    """Sanity check on the guard's neighbour: the version check has nothing to catch."""
    panel = synthetic_panel(60)
    features, target = build_design_matrix(panel, FeatureSpec())
    usable = usable_rows(features, target)
    saved = save_model(
        make_ridge().fit(features.loc[usable], target.loc[usable]),
        name="fresh",
        features=features.loc[usable],
        spec=FeatureSpec(),
        directory=tmp_path,
    )

    assert saved.card.environment_differences() == {}
    assert datetime.fromisoformat(saved.card.created_at) <= datetime.now(UTC)
