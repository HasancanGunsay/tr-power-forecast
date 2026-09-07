"""Fit one model on all available history and save it.

    uv run python -m powerforecast.models.train

This is not the backtest and does not report a score. The backtest answers "how
well does this approach work?" by fitting many models on many windows; this
answers "which single object goes into production?" by fitting one on
everything, which is what you want when the goal is to predict tomorrow rather
than to measure yesterday.

Keeping them separate matters. A score printed by a script that trains on all
the data would be an in-sample score, and in-sample scores on a boosted tree
model are close to meaningless. The honest numbers live in the backtest, and
this script deliberately prints none.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from powerforecast.config import PATHS
from powerforecast.data.panel import load_panel
from powerforecast.features.build import FeatureSpec, build_design_matrix, usable_rows
from powerforecast.models.estimators import make_lightgbm, make_ridge
from powerforecast.models.persistence import SavedModel, save_model
from powerforecast.targets import TARGETS, Target, for_column, resolve

FACTORIES: dict[str, Callable[[], Any]] = {"lightgbm": make_lightgbm, "ridge": make_ridge}


def train(
    *,
    model: str = "lightgbm",
    train_until: pd.Timestamp | str | None = None,
    spec: FeatureSpec | None = None,
    panel: pd.DataFrame | None = None,
    name: str | None = None,
    directory: Path | None = None,
    notes: str = "",
) -> SavedModel:
    """Fit `model` on every usable row of the panel and save it with its card.

    Rows with an incomplete design matrix are dropped rather than imputed. The
    longest lag is 336 hours, so the first two weeks of history are unusable by
    construction, and weather is missing before 2022 — filling those in would
    invent training data for the period the model is least able to check.
    """
    if model not in FACTORIES:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(FACTORIES)}")

    spec = spec or FeatureSpec()
    panel = load_panel() if panel is None else panel

    features, target = build_design_matrix(panel, spec)
    usable = usable_rows(features, target)
    if usable.empty:
        raise ValueError("no complete rows in the design matrix; has the panel been backfilled?")

    if train_until is not None:
        # A deliberate cut-off, for two reasons that both matter.
        #
        # Retraining: production models are refit on a schedule, and the version
        # that replaces this one has to be trained on more history than it did —
        # which means the boundary has to be expressible rather than implicit.
        #
        # Honest monitoring: a forecast for a day inside the training window is
        # not a forecast, it is a fit. Being able to train to a date is what
        # makes it possible to produce genuinely out-of-sample forecasts for days
        # that have already happened, and therefore to test the monitoring layer
        # against known outcomes rather than waiting weeks for new ones.
        cutoff = pd.Timestamp(train_until)
        if cutoff.tz is None:
            cutoff = cutoff.tz_localize("UTC")
        usable = usable[usable <= cutoff]
        if usable.empty:
            raise ValueError(f"no usable rows on or before {cutoff}")

    features = features.loc[usable]
    target = target.loc[usable]

    estimator = FACTORIES[model]().fit(features, target)
    return save_model(
        estimator,
        # Derived from what the spec actually forecasts. Hardcoding "load" here
        # would have silently filed a price model under the load name, and the
        # store resolves `latest` by timestamp — so the next service restart
        # would have served a price model to a caller asking for demand.
        name=name or f"{for_column(spec.target).name}-{model}",
        features=features,
        spec=spec,
        directory=directory,
        notes=notes,
    )


def spec_for(target: Target, *, weather: bool = True) -> FeatureSpec:
    """The feature set a deployed model for `target` should be fitted on.

    Reads the preferences off the target rather than branching on its name, so
    adding a third target is a row in `powerforecast.targets` and not an edit
    here.
    """
    return FeatureSpec(
        target=target.column,
        include_weather=weather,
        include_supply_weather=target.include_supply_weather,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(FACTORIES), default="lightgbm")
    parser.add_argument("--name", default=None, help="model name in the store")
    parser.add_argument("--notes", default="", help="why this model was trained")
    parser.add_argument(
        "--train-until",
        default=None,
        help="last timestamp to train on, ISO; omit to use every usable row",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default="load",
        help="what to forecast; selects the target column and its feature preset",
    )
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--out", type=Path, default=PATHS.models)
    args = parser.parse_args()

    saved = train(
        model=args.model,
        train_until=args.train_until,
        spec=spec_for(resolve(args.target), weather=not args.no_weather),
        name=args.name,
        directory=args.out,
        notes=args.notes,
    )

    card = saved.card
    print(f"saved   : {saved.path}")
    print(f"trained : {card.n_train_rows:,} rows, {card.train_start[:10]} .. {card.train_end[:10]}")
    print(f"features: {len(card.feature_columns)}")


if __name__ == "__main__":
    main()
