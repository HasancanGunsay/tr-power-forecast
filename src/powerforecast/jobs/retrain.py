"""Refresh the model on a schedule — and refuse to promote a worse one.

    uv run python -m powerforecast.jobs.retrain

Retraining is now load-bearing rather than housekeeping. The bias correction is
worth 6.3% of MAE while the model is fresh and **worse than nothing** once it is
stale (ADR 0010), and `should_correct` switches it off after 35 days. Without a
retrain on a shorter cycle than that, the system quietly loses the correction
about five weeks after each manual `train.py` run.

So this job exists. What it deliberately is *not* is "refit monthly and hope".

## Why a promotion gate

A retrain can produce a materially worse model, and that is measured rather than
feared. Scoring one fixed July window with models identical except for their
training cut-off (ADR 0009):

| cut-off | MAE |
|---|---|
| 31 January | 1,057 |
| 31 March | 1,136 |
| **31 May** | **1,764** |
| 30 June | 1,134 |

A 67% swing from nothing but the boundary. The cause there was a feature that has
since been removed, but the lesson survives its cause: **more recent data is not
automatically a better model**, and a store whose `latest` resolves by timestamp
promotes whatever was saved last. Saving is promotion, so the check has to happen
before the save.

## Retraining on data it has already seen

Found by running it rather than by reading it: with the panel not yet advanced,
a forced retrain produced a model trained to exactly the same date as the one it
replaced. Harmless in isolation — and not harmless at all in combination with
`should_correct`, which reads the model's age and would now believe the model
fresh while its knowledge is not. A version that learned nothing new would have
switched the bias correction back on for another 35 days on false pretences.

So a retrain requires the panel to have moved. `EXIT_NO_NEW_DATA` says so, and it
says so as a *failure* rather than a quiet success, because the reason it is due
and cannot proceed is that the ingestion job has not run.

## The asymmetry, stated

The gate blocks a *clear regression*, not a candidate that fails to improve.
Refreshing is the default because the alternative — an ageing model with its
correction switched off — degrades on a schedule of its own. So the candidate is
promoted unless it is worse by more than `REGRESSION_TOLERANCE`.

That is a judgement, and it is a judgement in one direction: this project would
rather ship a model that is 2% worse on one holdout than stop refreshing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from powerforecast.data.panel import load_panel
from powerforecast.features.availability import LOCAL_TZ
from powerforecast.features.build import FeatureSpec, build_design_matrix, usable_rows
from powerforecast.forecasts.bias import MAX_MODEL_AGE_DAYS
from powerforecast.models.persistence import (
    DEFAULT_MODEL_NAME,
    ModelStoreError,
    SavedModel,
    load_model,
)
from powerforecast.models.train import spec_for, train
from powerforecast.targets import TARGETS, resolve

logger = logging.getLogger("retrain")

# Comfortably inside the 35-day limit `should_correct` enforces, so a retrain can
# slip by a week without the bias correction switching itself off.
RETRAIN_AFTER_DAYS = 21

# How much worse a candidate may be on the holdout and still be promoted. Five
# percent is wide enough that one ordinary month of noise does not block a
# refresh, and narrow enough to catch the kind of regression measured in ADR 0009.
REGRESSION_TOLERANCE = 0.05

# Long enough that the comparison is not decided by a fortnight's weather. This
# project has twice mistaken one window for a distribution; two months is the
# minimum at which a model comparison is worth acting on.
HOLDOUT_DAYS = 60

EXIT_OK = 0
EXIT_REGRESSION = 2
EXIT_DATA = 3
EXIT_NO_NEW_DATA = 4
EXIT_UNEXPECTED = 5


@dataclass(frozen=True)
class Comparison:
    """What the holdout said, or why it could not say anything."""

    hours: int
    incumbent_mae: float | None
    candidate_mae: float
    verdict: str  # "improved" | "within tolerance" | "regression" | "unevaluated"
    detail: str

    @property
    def promote(self) -> bool:
        return self.verdict != "regression"


def is_due(
    trained_until: pd.Timestamp | str,
    *,
    now: pd.Timestamp | None = None,
    after_days: int = RETRAIN_AFTER_DAYS,
) -> tuple[bool, str]:
    """Whether the model is old enough to be worth refreshing.

    Returns `(due, reason)`. The reason is populated either way, so a job that
    does nothing still says why — a scheduler run with no output and no change is
    indistinguishable from one that failed silently.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    trained = pd.Timestamp(trained_until)
    trained = trained.tz_localize("UTC") if trained.tz is None else trained
    age = (now - trained).total_seconds() / 86400

    if age < after_days:
        return False, (
            f"model is {int(age)} days old, under the {after_days}-day cycle; "
            f"the bias correction stays on until {MAX_MODEL_AGE_DAYS} days"
        )
    return True, f"model is {int(age)} days old, past the {after_days}-day cycle"


def compare_on_holdout(
    panel: pd.DataFrame,
    incumbent: SavedModel | None,
    *,
    spec: FeatureSpec | None = None,
    holdout_days: int = HOLDOUT_DAYS,
    tolerance: float = REGRESSION_TOLERANCE,
    model: str = "lightgbm",
) -> Comparison:
    """Fit a candidate to the data before the holdout and score both on it.

    Both models are scored on hours neither was trained on — the candidate by
    construction, the incumbent only if its training ended before the holdout
    began. When it did not, there is nothing honest to compare and the comparison
    says so rather than inventing a verdict.
    """
    spec = spec or FeatureSpec()
    features, target = build_design_matrix(panel, spec)
    usable = usable_rows(features, target)
    if usable.empty:
        raise ValueError("no complete rows in the design matrix")

    boundary = usable.max() - pd.Timedelta(days=holdout_days)
    holdout = usable[usable > boundary]
    if len(holdout) < 24 * 14:
        raise ValueError(f"holdout is only {len(holdout)} hours; not enough to judge a model on")

    # `usable[usable <= boundary]`, not `features.loc[usable <= boundary]`. The
    # second passes a boolean array as long as `usable` into a frame as long as
    # the panel, which pandas reads as a mask of the wrong length. It raised here
    # rather than silently selecting the wrong rows, which was luck.
    train_index = usable[usable <= boundary]
    if len(train_index) < 24 * 90:
        raise ValueError(
            f"only {len(train_index)} hours before the holdout; a candidate fitted on "
            "that would be judged against a model trained on far more"
        )

    from powerforecast.models.train import FACTORIES

    candidate = FACTORIES[model]().fit(features.loc[train_index], target.loc[train_index])
    actual = target.loc[holdout]
    candidate_mae = float((actual - candidate.predict(features.loc[holdout])).abs().mean())

    if incumbent is None:
        return Comparison(
            hours=len(holdout),
            incumbent_mae=None,
            candidate_mae=candidate_mae,
            verdict="unevaluated",
            detail="no incumbent to compare against; this is the first model",
        )

    trained_until = pd.Timestamp(incumbent.card.train_end)
    if trained_until > boundary:
        # The incumbent has already seen the holdout, so scoring it there would
        # measure its fit rather than its forecast — flattering it, and blocking
        # a refresh it might well deserve to lose.
        return Comparison(
            hours=len(holdout),
            incumbent_mae=None,
            candidate_mae=candidate_mae,
            verdict="unevaluated",
            detail=(
                f"incumbent was trained to {trained_until.date()}, inside the holdout "
                f"beginning {boundary.date()}; scoring it there would measure its fit, "
                "not its forecast. Promoting without a comparison."
            ),
        )

    incumbent_features, _ = build_design_matrix(panel, incumbent.card.spec())
    scored = holdout.intersection(pd.DatetimeIndex(incumbent_features.dropna().index))
    if len(scored) < 24 * 14:
        return Comparison(
            hours=len(scored),
            incumbent_mae=None,
            candidate_mae=candidate_mae,
            verdict="unevaluated",
            detail="the incumbent's feature set does not cover enough of the holdout",
        )

    incumbent_mae = float(
        (target.loc[scored] - incumbent.predict(incumbent_features.loc[scored])).abs().mean()
    )
    candidate_mae = float(
        (target.loc[scored] - candidate.predict(features.loc[scored])).abs().mean()
    )
    change = (candidate_mae - incumbent_mae) / incumbent_mae

    if change > tolerance:
        verdict = "regression"
    elif change > 0:
        verdict = "within tolerance"
    else:
        verdict = "improved"

    return Comparison(
        hours=len(scored),
        incumbent_mae=incumbent_mae,
        candidate_mae=candidate_mae,
        verdict=verdict,
        detail=(
            f"MAE {incumbent_mae:,.0f} -> {candidate_mae:,.0f} ({change:+.1%}) on "
            f"{len(scored):,} holdout hours"
        ),
    )


def run(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_directory: Path | None = None,
    after_days: int = RETRAIN_AFTER_DAYS,
    holdout_days: int = HOLDOUT_DAYS,
    tolerance: float = REGRESSION_TOLERANCE,
    force: bool = False,
    now: pd.Timestamp | None = None,
    panel: pd.DataFrame | None = None,
    target: str | None = None,
) -> int:
    """Retrain if due and if the candidate does not regress. Returns the exit code."""
    try:
        incumbent: SavedModel | None = load_model(
            model_name, directory=model_directory, require_environment=False
        )
    except ModelStoreError:
        incumbent = None
        logger.info("no existing model under %r; training the first one", model_name)

    if incumbent is not None:
        due, reason = is_due(incumbent.card.train_end, now=now, after_days=after_days)
        logger.info("%s/%s: %s", incumbent.card.name, incumbent.card.version, reason)
        if not due and not force:
            # Nothing to do is a success. Exiting non-zero here would make a
            # scheduler treat an up-to-date model as a failure every day.
            return EXIT_OK

    # The candidate must be fitted on the *incumbent's* feature set, read off its
    # card. Falling back to `FeatureSpec()` here would have been silent and
    # expensive: a scheduled retrain of the price model would have trained a
    # **load** model, saved it under the price name, and compared it against the
    # price incumbent on a holdout — three wrong things, none of which raises.
    # The store resolves `latest` by timestamp, so saving is promoting.
    spec = incumbent.card.spec() if incumbent is not None else spec_for(resolve(target))
    logger.info("training on the %s target", spec.target)

    panel = load_panel() if panel is None else panel
    panel_end = pd.DatetimeIndex(panel.index).max()

    if incumbent is not None and panel_end <= pd.Timestamp(incumbent.card.train_end):
        # Resetting the age clock without new data would tell `should_correct`
        # the model is fresh while its knowledge is not — switching the bias
        # correction back on for another 35 days on false pretences.
        logger.error(
            "the panel ends at %s and the model was already trained to %s: there is "
            "nothing new to learn. Run the ingestion backfill first; retraining now "
            "would refresh the model's age without refreshing the model.",
            panel_end.date(),
            pd.Timestamp(incumbent.card.train_end).date(),
        )
        return EXIT_NO_NEW_DATA

    try:
        comparison = compare_on_holdout(
            panel, incumbent, holdout_days=holdout_days, tolerance=tolerance, spec=spec
        )
    except ValueError as error:
        logger.error("cannot evaluate a candidate: %s", error)
        return EXIT_DATA

    logger.info("holdout verdict: %s — %s", comparison.verdict, comparison.detail)

    if not comparison.promote:
        logger.error(
            "refusing to promote: the candidate is worse by more than %.0f%%. "
            "The incumbent stays. A retrain has been measured at 67%% worse before "
            "(ADR 0009), which is why this gate exists rather than trusting the date.",
            tolerance * 100,
        )
        return EXIT_REGRESSION

    saved = train(
        model="lightgbm",
        panel=panel,
        spec=spec,
        name=model_name,
        directory=model_directory,
        notes=f"scheduled retrain; holdout {comparison.verdict}: {comparison.detail}",
    )
    logger.info(
        "promoted %s/%s — %d rows, trained to %s",
        saved.card.name,
        saved.card.version,
        saved.card.n_train_rows,
        saved.card.train_end[:10],
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default=None,
        help=(
            "convenience for --model. Only consulted when there is no incumbent; "
            "otherwise the feature set comes from the incumbent's card."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--after-days", type=int, default=RETRAIN_AFTER_DAYS)
    parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    parser.add_argument("--tolerance", type=float, default=REGRESSION_TOLERANCE)
    parser.add_argument(
        "--force", action="store_true", help="retrain even if the model is not yet due"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )

    try:
        return run(
            model_name=args.model
            or (f"{args.target}-lightgbm" if args.target else DEFAULT_MODEL_NAME),
            target=args.target,
            after_days=args.after_days,
            holdout_days=args.holdout_days,
            tolerance=args.tolerance,
            force=args.force,
        )
    except Exception:
        logger.exception("retrain failed unexpectedly")
        return EXIT_UNEXPECTED


def delivery_day_of(timestamp: pd.Timestamp) -> date:
    """The local delivery day an hour belongs to. Exposed for callers and tests."""
    return pd.Timestamp(timestamp).tz_convert(LOCAL_TZ).date()


if __name__ == "__main__":
    raise SystemExit(main())
