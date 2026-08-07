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

FACTORIES: dict[str, Callable[[], Any]] = {"lightgbm": make_lightgbm, "ridge": make_ridge}


def train(
    *,
    model: str = "lightgbm",
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

    features = features.loc[usable]
    target = target.loc[usable]

    estimator = FACTORIES[model]().fit(features, target)
    return save_model(
        estimator,
        name=name or f"load-{model}",
        features=features,
        spec=spec,
        directory=directory,
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(FACTORIES), default="lightgbm")
    parser.add_argument("--name", default=None, help="model name in the store")
    parser.add_argument("--notes", default="", help="why this model was trained")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--out", type=Path, default=PATHS.models)
    args = parser.parse_args()

    saved = train(
        model=args.model,
        spec=FeatureSpec(include_weather=not args.no_weather),
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
