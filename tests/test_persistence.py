"""Tests for the model store.

The interesting cases are all about *refusing*: a model that loads happily
under any circumstances is exactly the model that returns confident nonsense in
production. So most of what is tested here is that the wrong thing raises.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from powerforecast.features.build import FeatureSpec
from powerforecast.models.persistence import (
    CARD_FILE,
    MODEL_FILE,
    FeatureMismatchError,
    ModelEnvironmentError,
    ModelIntegrityError,
    ModelNotFoundError,
    ModelStoreError,
    latest_version,
    list_versions,
    load_model,
    save_model,
)


@pytest.fixture
def fitted() -> tuple[Ridge, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {"lag_48": rng.normal(size=48), "temperature_c": rng.normal(size=48)},
        index=index,
    )
    target = pd.Series(features["lag_48"] * 2.0 + 1.0, index=index)
    model = Ridge(alpha=1.0).fit(features, target)
    return model, features


def test_save_then_load_round_trips(tmp_path, fitted):
    model, features = fitted
    spec = FeatureSpec(include_weather=True)

    saved = save_model(model, name="load-ridge", features=features, spec=spec, directory=tmp_path)
    loaded = load_model("load-ridge", directory=tmp_path)

    assert loaded.card == saved.card
    pd.testing.assert_series_equal(
        loaded.predict(features), saved.predict(features), check_names=False
    )


def test_card_records_what_the_model_was_trained_on(tmp_path, fitted):
    model, features = fitted
    spec = FeatureSpec(target_lags=(48, 168), include_weather=True)

    card = save_model(
        model, name="load-ridge", features=features, spec=spec, directory=tmp_path, notes="baseline"
    ).card

    assert card.feature_columns == ("lag_48", "temperature_c")
    assert card.n_train_rows == 48
    assert card.train_start.startswith("2025-01-01")
    assert card.estimator_class == "sklearn.linear_model._ridge.Ridge"
    assert card.notes == "baseline"
    # The spec survives the JSON round trip as tuples, so it compares equal to
    # the original rather than merely looking like it.
    assert card.spec() == spec


def test_card_is_readable_without_python(tmp_path, fitted):
    """The card has to be plain JSON — that is most of its value on a server."""
    model, features = fitted
    saved = save_model(
        model, name="load-ridge", features=features, spec=FeatureSpec(), directory=tmp_path
    )

    payload = json.loads((saved.path / CARD_FILE).read_text(encoding="utf-8"))
    assert payload["name"] == "load-ridge"
    assert payload["feature_columns"] == ["lag_48", "temperature_c"]


def test_predict_rejects_missing_and_extra_columns(tmp_path, fitted):
    model, features = fitted
    saved = save_model(
        model, name="load-ridge", features=features, spec=FeatureSpec(), directory=tmp_path
    )

    with pytest.raises(FeatureMismatchError, match="missing"):
        saved.predict(features.drop(columns=["temperature_c"]))

    with pytest.raises(FeatureMismatchError, match="unexpected"):
        saved.predict(features.assign(surprise=1.0))


def test_predict_rejects_reordered_columns(tmp_path, fitted):
    """Reordering is the dangerous one: the shapes match and nothing else complains."""
    model, features = fitted
    saved = save_model(
        model, name="load-ridge", features=features, spec=FeatureSpec(), directory=tmp_path
    )

    with pytest.raises(FeatureMismatchError, match="different order"):
        saved.predict(features[["temperature_c", "lag_48"]])


def test_tampered_model_file_is_refused(tmp_path, fitted):
    model, features = fitted
    saved = save_model(
        model, name="load-ridge", features=features, spec=FeatureSpec(), directory=tmp_path
    )

    (saved.path / MODEL_FILE).write_bytes(b"not a pickle")

    with pytest.raises(ModelIntegrityError, match="modified or truncated"):
        load_model("load-ridge", directory=tmp_path)


def test_environment_check_is_opt_in(tmp_path, fitted):
    model, features = fitted
    saved = save_model(
        model, name="load-ridge", features=features, spec=FeatureSpec(), directory=tmp_path
    )

    card = json.loads((saved.path / CARD_FILE).read_text(encoding="utf-8"))
    card["library_versions"]["pandas"] = "0.0.1-from-the-past"
    (saved.path / CARD_FILE).write_text(json.dumps(card), encoding="utf-8")

    # Loading still works by default; exploration should not be blocked.
    assert load_model("load-ridge", directory=tmp_path).card.environment_differences()

    with pytest.raises(ModelEnvironmentError, match="pandas"):
        load_model("load-ridge", directory=tmp_path, require_environment=True)


def test_versions_accumulate_and_latest_is_the_newest(tmp_path, fitted):
    model, features = fitted
    for stamp in ("20250101T000000Z", "20250301T000000Z", "20250201T000000Z"):
        save_model(
            model,
            name="load-ridge",
            features=features,
            spec=FeatureSpec(),
            directory=tmp_path,
            version=stamp,
        )

    assert list_versions("load-ridge", directory=tmp_path) == [
        "20250101T000000Z",
        "20250201T000000Z",
        "20250301T000000Z",
    ]
    assert latest_version("load-ridge", directory=tmp_path) == "20250301T000000Z"
    assert load_model("load-ridge", directory=tmp_path).card.version == "20250301T000000Z"


def test_a_version_is_never_overwritten(tmp_path, fitted):
    """History that can be rewritten is not history."""
    model, features = fitted
    kwargs = {
        "name": "load-ridge",
        "features": features,
        "spec": FeatureSpec(),
        "directory": tmp_path,
        "version": "20250101T000000Z",
    }
    save_model(model, **kwargs)

    with pytest.raises(ModelStoreError, match="never overwritten"):
        save_model(model, **kwargs)


def test_missing_model_says_what_is_available(tmp_path, fitted):
    model, features = fitted
    save_model(
        model,
        name="load-ridge",
        features=features,
        spec=FeatureSpec(),
        directory=tmp_path,
        version="20250101T000000Z",
    )

    with pytest.raises(ModelNotFoundError, match="20250101T000000Z"):
        load_model("load-ridge", "20990101T000000Z", directory=tmp_path)

    with pytest.raises(ModelNotFoundError, match="no saved model"):
        latest_version("no-such-model", directory=tmp_path)

    assert list_versions("no-such-model", directory=tmp_path) == []


def test_unfitted_or_empty_input_is_rejected_at_save_time(tmp_path, fitted):
    model, features = fitted

    with pytest.raises(ModelStoreError, match="no predict"):
        save_model(
            object(), name="broken", features=features, spec=FeatureSpec(), directory=tmp_path
        )

    with pytest.raises(ModelStoreError, match="empty"):
        save_model(
            model, name="broken", features=features.iloc[:0], spec=FeatureSpec(), directory=tmp_path
        )


def test_pipeline_parameters_survive_json(tmp_path, fitted):
    """A Pipeline's params contain objects; the card must still be writable."""
    from powerforecast.models.estimators import make_ridge

    _, features = fitted
    pipeline = make_ridge().fit(features, pd.Series(1.0, index=features.index))

    saved = save_model(
        pipeline, name="load-pipeline", features=features, spec=FeatureSpec(), directory=tmp_path
    )
    payload = json.loads((saved.path / CARD_FILE).read_text(encoding="utf-8"))

    assert "steps" in payload["estimator_params"]
    assert "Ridge" in str(payload["estimator_params"]["steps"])
