"""Save a fitted model together with its identity.

A pickle on disk is an anonymous object. It does not know what it was trained
on, which columns it expects, in what order, or under which library versions it
was fitted — and every one of those is something a serving process has to get
right or it will return numbers that look plausible and are wrong.

So nothing here saves a model alone. A save produces a *version directory*
holding two files:

    models/<name>/<version>/model.pkl   the fitted estimator
    models/<name>/<version>/card.json   what it is, readable without Python

The card is the point. Three months from now the question is never "is this a
LightGBM?" — it is "was this the one trained before or after the holiday
features went in?", and the answer has to be on disk, not in someone's memory.

The two failure modes this guards against, both of which are silent:

1. **Column drift.** The design matrix gains a feature, the service keeps
   sending the old set, and a tree model happily predicts from whatever it is
   given. `ModelCard.assert_compatible` turns that into an exception at the
   first request rather than a slow drift in accuracy nobody attributes to it.
2. **Environment drift.** A pickle unpickled under a different scikit-learn or
   LightGBM than it was fitted under is not guaranteed to behave the same way.
   The card records the versions; `require_environment=True` refuses to load
   under different ones.

**A note on pickle, honestly stated:** unpickling executes code, so this module
is only safe for files this project wrote. The recorded checksum protects
against a truncated or corrupted file, not against a hostile one. If a model
ever has to cross a trust boundary, the format has to change — that is a real
limitation, not an oversight.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import powerforecast
from powerforecast.config import PATHS, RANDOM_SEED
from powerforecast.features.build import FeatureSpec

MODEL_FILE = "model.pkl"
CARD_FILE = "card.json"

# A version is a UTC timestamp: sortable as text, unambiguous across machines,
# and readable without a lookup table. "latest" is then just the last one.
VERSION_FORMAT = "%Y%m%dT%H%M%SZ"
VERSION_PATTERN = re.compile(r"^\d{8}T\d{6}Z(-\d+)?$")

# Only the libraries whose version can change what a loaded model *does*. Adding
# matplotlib here would produce mismatches that mean nothing, and a check that
# fires on things that do not matter is a check people learn to bypass.
TRACKED_LIBRARIES = ("pandas", "numpy", "scikit-learn", "lightgbm")


class ModelStoreError(RuntimeError):
    """Base class for every failure in this module."""


class ModelNotFoundError(ModelStoreError):
    """No saved model matches the requested name and version."""


class ModelIntegrityError(ModelStoreError):
    """The stored file does not match the checksum recorded when it was saved."""


class ModelEnvironmentError(ModelStoreError):
    """The model was fitted under library versions that are not the current ones."""


class FeatureMismatchError(ModelStoreError):
    """The features offered at prediction time are not the ones the model was fitted on."""


@dataclass(frozen=True)
class ModelCard:
    """Everything needed to say what a saved model is.

    Frozen, and written as JSON rather than pickled, so it can be read by
    anything — `cat`, a browser, a shell script in a container — without
    importing this project. A metadata format that needs the code to be readable
    is not much better than no metadata.
    """

    name: str
    version: str
    created_at: str
    target: str
    feature_columns: tuple[str, ...]
    feature_spec: dict[str, Any]
    train_start: str
    train_end: str
    n_train_rows: int
    estimator_class: str
    estimator_params: dict[str, Any]
    random_seed: int
    package_version: str
    python_version: str
    library_versions: dict[str, str]
    model_sha256: str = ""
    notes: str = ""

    # --- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["feature_columns"] = list(self.feature_columns)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            # A card written by a newer version of this module. Refusing beats
            # loading it while ignoring the fields that were added precisely
            # because they mattered.
            raise ModelStoreError(f"card has unrecognised fields: {sorted(unknown)}")
        payload = dict(data)
        payload["feature_columns"] = tuple(payload["feature_columns"])
        return cls(**payload)

    def spec(self) -> FeatureSpec:
        """Rebuild the `FeatureSpec` this model was trained with.

        Round-tripping through JSON turns every tuple into a list, and
        `FeatureSpec` is frozen and compared by value, so restoring the tuples
        is what makes `saved.spec() == original_spec` true rather than merely
        looking true.
        """
        payload = dict(self.feature_spec)
        for key in ("target_lags", "origin_windows", "origin_offsets", "extra"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return FeatureSpec(**payload)

    # --- checks -----------------------------------------------------------

    def assert_compatible(self, features: pd.DataFrame) -> None:
        """Fail unless `features` is exactly what this model was fitted on.

        Exactly, including order. scikit-learn and LightGBM both index the
        design matrix positionally once fitted, so a reordered frame is not a
        warning — it feeds temperature into the coefficient for hour-of-day and
        returns a number with no error attached to it.

        The message names what is missing, what is extra and whether the order
        moved, because the fix depends on which of the three it is.
        """
        offered = tuple(features.columns)
        if offered == self.feature_columns:
            return

        expected = set(self.feature_columns)
        given = set(offered)
        problems = []
        if missing := sorted(expected - given):
            problems.append(f"missing {missing}")
        if extra := sorted(given - expected):
            problems.append(f"unexpected {extra}")
        if not problems:
            problems.append(
                f"same columns in a different order: expected {list(self.feature_columns)}, "
                f"got {list(offered)}"
            )
        raise FeatureMismatchError(
            f"features do not match model {self.name}/{self.version}: " + "; ".join(problems)
        )

    def environment_differences(self) -> dict[str, tuple[str, str]]:
        """Libraries whose current version differs from the fitted one.

        Returns `{library: (fitted, current)}`, empty when everything matches.
        Reported rather than raised so a caller can decide: a notebook exploring
        an old model has different needs from a service answering requests.
        """
        current = _library_versions()
        return {
            library: (fitted, current.get(library, "absent"))
            for library, fitted in self.library_versions.items()
            if current.get(library, "absent") != fitted
        }


@dataclass(frozen=True)
class SavedModel:
    """A fitted estimator and its card, loaded together.

    They travel as one value because they are only meaningful as one: an
    estimator without its card cannot be checked, and a card without its
    estimator cannot predict.
    """

    estimator: Any
    card: ModelCard
    path: Path = field(compare=False)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict, refusing to guess when the features are not the fitted ones.

        The check is not optional and there is no flag to skip it. Prediction on
        mismatched columns is the failure this whole module exists to prevent,
        so leaving a door open would defeat the purpose of building the wall.
        """
        self.card.assert_compatible(features)
        values = self.estimator.predict(features)
        return pd.Series(values, index=features.index, name="forecast")


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def save_model(
    estimator: Any,
    *,
    name: str,
    features: pd.DataFrame,
    spec: FeatureSpec,
    directory: Path | None = None,
    version: str | None = None,
    notes: str = "",
) -> SavedModel:
    """Write a fitted estimator and its card to a new version directory.

    `features` is the design matrix the estimator was *actually fitted on* —
    passing anything else makes the card a lie, which is worse than no card at
    all because it is trusted. Column names, order, row count and training
    window are all read from it rather than taken on the caller's word.

    Args:
        estimator: A fitted estimator exposing `predict`.
        name: Logical model name, e.g. ``"load-lightgbm"``. Versions accumulate
            underneath it.
        features: The design matrix used for fitting.
        spec: The `FeatureSpec` that produced `features`.
        directory: Model store root. Defaults to `PATHS.models`.
        version: Override the generated timestamp. For tests and for restoring
            a model store; ordinary saves should let it be generated.
        notes: Free text — why this model was trained, what changed.

    Returns:
        The `SavedModel`, whose `path` is the version directory just written.
    """
    if not hasattr(estimator, "predict"):
        raise ModelStoreError(f"{type(estimator).__name__} has no predict(); it cannot be served")
    if features.empty:
        raise ModelStoreError("features is empty; a model card would record a training set of zero")

    root = (directory or PATHS.models) / name
    version = version or _next_version(root)
    target_dir = root / version
    if target_dir.exists():
        raise ModelStoreError(f"{target_dir} already exists; versions are never overwritten")

    index = pd.DatetimeIndex(features.index)
    card = ModelCard(
        name=name,
        version=version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        target=spec.target,
        feature_columns=tuple(features.columns),
        feature_spec=_spec_to_dict(spec),
        train_start=index.min().isoformat(),
        train_end=index.max().isoformat(),
        n_train_rows=len(features),
        estimator_class=f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        estimator_params=_json_safe(_estimator_params(estimator)),
        random_seed=RANDOM_SEED,
        package_version=powerforecast.__version__,
        python_version=platform.python_version(),
        library_versions=_library_versions(),
        notes=notes,
    )

    # Write the model first, then hash what actually landed on disk and record
    # that. Hashing the in-memory bytes would certify a file that was never
    # written — which is exactly the case the checksum exists to catch.
    target_dir.mkdir(parents=True)
    model_path = target_dir / MODEL_FILE
    model_path.write_bytes(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL))
    card = replace(card, model_sha256=_sha256(model_path))

    (target_dir / CARD_FILE).write_text(
        json.dumps(card.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return SavedModel(estimator=estimator, card=card, path=target_dir)


def load_model(
    name: str,
    version: str = "latest",
    *,
    directory: Path | None = None,
    require_environment: bool = False,
) -> SavedModel:
    """Load a saved model, verifying it is intact before returning it.

    Args:
        name: The logical model name given at save time.
        version: A version string, or ``"latest"`` for the most recent.
        directory: Model store root. Defaults to `PATHS.models`.
        require_environment: Refuse to load when the tracked library versions
            differ from the fitted ones. Off by default because exploration
            should stay possible; a **service should turn it on** — that is the
            one place where "probably the same" is not good enough.

    Raises:
        ModelNotFoundError: no such name or version.
        ModelIntegrityError: the file does not match its recorded checksum.
        ModelEnvironmentError: versions differ and `require_environment` is set.
    """
    root = (directory or PATHS.models) / name
    resolved = latest_version(name, directory=directory) if version == "latest" else version
    target_dir = root / resolved

    if not (target_dir / CARD_FILE).exists():
        available = list_versions(name, directory=directory)
        raise ModelNotFoundError(
            f"no model {name}/{resolved} under {root}; "
            + (f"available versions: {available}" if available else "the store is empty")
        )

    card = ModelCard.from_dict(json.loads((target_dir / CARD_FILE).read_text(encoding="utf-8")))

    model_path = target_dir / MODEL_FILE
    digest = _sha256(model_path)
    if card.model_sha256 and digest != card.model_sha256:
        raise ModelIntegrityError(
            f"{model_path} does not match its card: recorded {card.model_sha256[:12]}, "
            f"found {digest[:12]}. The file has been modified or truncated since it was saved."
        )

    if require_environment and (differences := card.environment_differences()):
        detail = ", ".join(
            f"{library}: fitted {fitted}, current {current}"
            for library, (fitted, current) in sorted(differences.items())
        )
        raise ModelEnvironmentError(
            f"{name}/{resolved} was fitted under different libraries ({detail}). "
            "Unpickling across versions is not guaranteed to reproduce the fitted behaviour."
        )

    # Unpickling executes code. Safe here only because the store holds files
    # this project wrote; see the module docstring for why that is a real limit.
    estimator = pickle.loads(model_path.read_bytes())
    return SavedModel(estimator=estimator, card=card, path=target_dir)


def list_versions(name: str, *, directory: Path | None = None) -> list[str]:
    """Every stored version of `name`, oldest first. Empty when there are none."""
    root = (directory or PATHS.models) / name
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and VERSION_PATTERN.match(entry.name) and (entry / CARD_FILE).exists()
    )


def latest_version(name: str, *, directory: Path | None = None) -> str:
    """The most recent stored version of `name`.

    Raises:
        ModelNotFoundError: when nothing has been saved under that name. The
            alternative — returning `None` and letting the caller dereference it
            — turns a missing model into an AttributeError three frames away.
    """
    versions = list_versions(name, directory=directory)
    if not versions:
        root = (directory or PATHS.models) / name
        raise ModelNotFoundError(f"no saved model named {name!r} under {root}")
    return versions[-1]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _next_version(root: Path) -> str:
    """A fresh, sortable version string.

    Two saves inside the same second would otherwise collide, and the loser
    would be an `already exists` error in the middle of a training script. The
    suffix keeps sort order intact: `...Z` sorts before `...Z-1`.
    """
    stamp = datetime.now(UTC).strftime(VERSION_FORMAT)
    if not (root / stamp).exists():
        return stamp
    for counter in range(1, 100):
        candidate = f"{stamp}-{counter}"
        if not (root / candidate).exists():
            return candidate
    raise ModelStoreError(f"cannot allocate a version under {root}: 100 saves in one second")


def _spec_to_dict(spec: FeatureSpec) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(spec).items()
    }


def _estimator_params(estimator: Any) -> dict[str, Any]:
    """Hyperparameters, when the estimator follows the scikit-learn interface."""
    if hasattr(estimator, "get_params"):
        try:
            return dict(estimator.get_params(deep=False))
        except Exception:  # pragma: no cover — third-party estimators may differ
            return {}
    return {}


def _json_safe(values: dict[str, Any]) -> dict[str, Any]:
    """Coerce a parameter dict into something JSON can hold.

    A `Pipeline`'s parameters include the steps themselves, which are objects.
    Falling back to `repr` keeps the record human-readable — the card is for a
    person asking "what was this?", not for reconstructing the estimator, which
    is what the pickle is for.
    """
    safe: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or isinstance(value, str | bool | int | float):
            safe[key] = value
        elif isinstance(value, list | tuple):
            safe[key] = [
                item if isinstance(item, str | bool | int | float) else repr(item) for item in value
            ]
        else:
            safe[key] = repr(value)
    return safe


def _library_versions() -> dict[str, str]:
    """Installed versions of the libraries that can change a model's behaviour."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for library in TRACKED_LIBRARIES:
        try:
            versions[library] = version(library)
        except PackageNotFoundError:  # pragma: no cover — every one is a hard dependency
            continue
    return versions


def _sha256(path: Path) -> str:
    """Checksum of a file, read in chunks so a large model does not need to fit twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
