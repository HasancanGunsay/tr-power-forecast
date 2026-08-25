"""Tests for the scheduled retrain.

The gate is the point. A store whose `latest` resolves by timestamp promotes
whatever was saved last, so saving *is* promotion — and a retrain has been
measured at 67% worse than its predecessor (ADR 0009). The check therefore has to
happen before the save, and it has to fail closed.

The second thing worth testing is the asymmetry: this gate blocks a clear
regression, not a candidate that merely fails to improve. An ageing model with its
bias correction switched off degrades on a schedule of its own, so refreshing is
the default.
"""

from __future__ import annotations

import pandas as pd

from powerforecast.features.build import FeatureSpec, build_design_matrix, usable_rows
from powerforecast.jobs import retrain
from powerforecast.models.estimators import make_ridge
from powerforecast.models.persistence import list_versions, load_model, save_model

PANEL_DAYS = 400


def synthetic_panel(days: int = PANEL_DAYS, start: str = "2025-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
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


def save_incumbent(directory, panel, *, train_until: str, name: str = "load-lightgbm"):
    """A model trained to a chosen boundary, so the holdout can be out-of-sample for it."""
    spec = FeatureSpec()
    features, target = build_design_matrix(panel, spec)
    usable = usable_rows(features, target)
    usable = usable[usable <= pd.Timestamp(train_until, tz="UTC")]
    return save_model(
        make_ridge().fit(features.loc[usable], target.loc[usable]),
        name=name,
        features=features.loc[usable],
        spec=spec,
        directory=directory,
    )


# --------------------------------------------------------------------------- #
# When a retrain is due
# --------------------------------------------------------------------------- #


def test_a_fresh_model_is_not_due():
    due, reason = retrain.is_due(
        "2026-08-01", now=pd.Timestamp("2026-08-10", tz="UTC"), after_days=21
    )
    assert not due
    assert "under the 21-day cycle" in reason


def test_an_old_model_is_due():
    due, reason = retrain.is_due("2026-07-01", now=pd.Timestamp("2026-08-10", tz="UTC"))

    assert due
    assert "past the" in reason


def test_the_cycle_sits_inside_the_correction_limit():
    """A retrain that slips a week must not switch the bias correction off.

    `should_correct` refuses past 35 days; retraining every 21 leaves two weeks of
    slack. If these two numbers ever cross, the correction turns itself off
    between scheduled runs and nothing says so except a log line.
    """
    from powerforecast.forecasts.bias import MAX_MODEL_AGE_DAYS

    assert retrain.RETRAIN_AFTER_DAYS < MAX_MODEL_AGE_DAYS


def test_a_reason_is_given_even_when_nothing_happens():
    """A scheduler run with no output is indistinguishable from a silent failure."""
    for trained in ("2026-08-01", "2026-06-01"):
        _, reason = retrain.is_due(trained, now=pd.Timestamp("2026-08-10", tz="UTC"))
        assert reason


# --------------------------------------------------------------------------- #
# The promotion gate
# --------------------------------------------------------------------------- #


def test_a_first_model_is_promoted_without_a_comparison(tmp_path):
    panel = synthetic_panel()

    comparison = retrain.compare_on_holdout(panel, None)

    assert comparison.verdict == "unevaluated"
    assert comparison.promote
    assert "first model" in comparison.detail


def test_an_incumbent_that_has_seen_the_holdout_is_not_scored_on_it(tmp_path):
    """Scoring it there would measure its fit, and block a refresh it might deserve to lose."""
    panel = synthetic_panel()
    incumbent = save_incumbent(tmp_path, panel, train_until="2026-01-01")

    comparison = retrain.compare_on_holdout(panel, incumbent, holdout_days=60)

    assert comparison.verdict == "unevaluated"
    assert comparison.promote
    assert "measure its fit" in comparison.detail


def test_a_comparable_incumbent_is_scored_and_the_verdict_is_reported(tmp_path):
    panel = synthetic_panel()
    # Trained well before the holdout, so the holdout is genuinely out-of-sample.
    incumbent = save_incumbent(tmp_path, panel, train_until="2025-09-01")

    comparison = retrain.compare_on_holdout(panel, incumbent, holdout_days=60)

    assert comparison.verdict in {"improved", "within tolerance", "regression"}
    assert comparison.incumbent_mae is not None
    assert comparison.hours >= 24 * 14
    assert "MAE" in comparison.detail


def test_a_clear_regression_is_refused(tmp_path, monkeypatch):
    """Fail closed: the incumbent stays, and the job says why."""
    panel = synthetic_panel()
    save_incumbent(tmp_path, panel, train_until="2025-09-01")

    def worse(*args, **kwargs):
        return retrain.Comparison(
            hours=1_440,
            incumbent_mae=800.0,
            candidate_mae=1_400.0,
            verdict="regression",
            detail="MAE 800 -> 1,400 (+75.0%)",
        )

    monkeypatch.setattr(retrain, "compare_on_holdout", worse)
    before = list_versions("load-lightgbm", directory=tmp_path)

    code = retrain.run(
        model_directory=tmp_path,
        panel=panel,
        now=pd.Timestamp("2027-01-01", tz="UTC"),
    )

    assert code == retrain.EXIT_REGRESSION
    assert list_versions("load-lightgbm", directory=tmp_path) == before


def test_a_candidate_that_is_slightly_worse_is_still_promoted(tmp_path, monkeypatch):
    """The asymmetry. Not refreshing costs more than 2% on one holdout."""
    panel = synthetic_panel()
    save_incumbent(tmp_path, panel, train_until="2025-09-01")

    monkeypatch.setattr(
        retrain,
        "compare_on_holdout",
        lambda *a, **k: retrain.Comparison(
            hours=1_440,
            incumbent_mae=800.0,
            candidate_mae=816.0,
            verdict="within tolerance",
            detail="MAE 800 -> 816 (+2.0%)",
        ),
    )
    before = len(list_versions("load-lightgbm", directory=tmp_path))

    code = retrain.run(
        model_directory=tmp_path, panel=panel, now=pd.Timestamp("2027-01-01", tz="UTC")
    )

    assert code == retrain.EXIT_OK
    assert len(list_versions("load-lightgbm", directory=tmp_path)) == before + 1


# --------------------------------------------------------------------------- #
# The job
# --------------------------------------------------------------------------- #


def test_an_up_to_date_model_exits_zero_and_changes_nothing(tmp_path):
    """Nothing to do is success. Non-zero here would alert every day."""
    panel = synthetic_panel()
    saved = save_incumbent(tmp_path, panel, train_until="2026-01-01")
    before = list_versions("load-lightgbm", directory=tmp_path)

    code = retrain.run(
        model_directory=tmp_path,
        panel=panel,
        now=pd.Timestamp(saved.card.train_end) + pd.Timedelta(days=3),
    )

    assert code == retrain.EXIT_OK
    assert list_versions("load-lightgbm", directory=tmp_path) == before


def test_force_retrains_an_up_to_date_model(tmp_path):
    panel = synthetic_panel()
    saved = save_incumbent(tmp_path, panel, train_until="2026-01-01")
    before = len(list_versions("load-lightgbm", directory=tmp_path))

    code = retrain.run(
        model_directory=tmp_path,
        panel=panel,
        force=True,
        now=pd.Timestamp(saved.card.train_end) + pd.Timedelta(days=3),
    )

    assert code == retrain.EXIT_OK
    assert len(list_versions("load-lightgbm", directory=tmp_path)) == before + 1


def test_the_promoted_model_records_why_it_was_promoted(tmp_path):
    """Six months on, "why is this the live model?" has to be answerable from the card."""
    panel = synthetic_panel()
    save_incumbent(tmp_path, panel, train_until="2025-09-01")

    retrain.run(model_directory=tmp_path, panel=panel, now=pd.Timestamp("2027-01-01", tz="UTC"))

    card = load_model("load-lightgbm", directory=tmp_path).card
    assert "scheduled retrain" in card.notes
    assert "holdout" in card.notes


def test_too_little_data_is_a_distinct_exit_code(tmp_path):
    """A scheduler alert should say which stage broke without anyone opening a log."""
    code = retrain.run(
        model_directory=tmp_path,
        panel=synthetic_panel(days=40),
        now=pd.Timestamp("2027-01-01", tz="UTC"),
    )

    assert code == retrain.EXIT_DATA


def test_every_exit_code_is_distinct():
    codes = {
        retrain.EXIT_OK,
        retrain.EXIT_REGRESSION,
        retrain.EXIT_DATA,
        retrain.EXIT_NO_NEW_DATA,
        retrain.EXIT_UNEXPECTED,
    }
    assert len(codes) == 5


def test_retraining_on_data_it_has_already_seen_is_refused(tmp_path):
    """Found by running it, not by reading it.

    With the panel not yet advanced, a retrain produces a model trained to exactly
    the same date — and resets its age. `should_correct` reads that age and would
    switch the bias correction back on for another 35 days, on a model whose
    knowledge had not moved at all.
    """
    panel = synthetic_panel()
    save_incumbent(tmp_path, panel, train_until=str(panel.index.max()))
    before = list_versions("load-lightgbm", directory=tmp_path)

    code = retrain.run(
        model_directory=tmp_path,
        panel=panel,
        force=True,
        now=pd.Timestamp("2027-01-01", tz="UTC"),
    )

    assert code == retrain.EXIT_NO_NEW_DATA
    assert list_versions("load-lightgbm", directory=tmp_path) == before
