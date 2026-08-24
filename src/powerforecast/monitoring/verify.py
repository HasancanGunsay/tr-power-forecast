"""Check what was forecast against what happened, and decide whether it is decaying.

    uv run python -m powerforecast.monitoring.verify

Most portfolio projects stop at "the model is behind an API". This is the part
that comes after, and it answers a different question from the backtest. A
backtest asks *did this approach ever work*. This asks *is it still working*, on
data nobody chose, with a model nobody refit.

The whole layer rests on one thing having been done correctly earlier: the
forecast was **written down while it was still a prediction**. Nothing here can
be reconstructed after the fact — a forecast recomputed today from today's data
is not the forecast that was made, it is a much better one.

## The design decision that matters: drift needs a control

The obvious drift check compares this week's error to last month's, and it is
wrong in a specific and expensive way. Forecast error moves for two completely
different reasons:

* the model got worse — features drifted, the relationship changed, the world
  moved away from the training set;
* the period got harder — a heatwave, a holiday cluster, unusual industrial
  demand. Every forecaster's error rose, ours included.

The first needs a retrain. The second needs nothing at all. A bare
recent-vs-baseline comparison cannot tell them apart, so it produces alerts that
are ignored — and an alert that is ignored is worse than no alert, because it
also teaches people to ignore the next one.

This project has a free control sitting in the panel: **the grid operator's own
published forecast**, for every hour, produced under the same conditions with
the same information. So drift is judged on *skill* — how much better than the
plan we are — rather than on error alone. Difference-in-differences, not
difference.

That is the same discipline as the "cheapest competitor" rule that shaped the
whole benchmark: never interpret one number without asking what an alternative
would have scored on exactly the same hours.

## The third decision: an in-sample hour is not evidence

A stored forecast whose delivery hour falls **inside its model's training
window** says nothing about live performance. The model has already seen that
hour's answer; scoring it measures the fit, not the forecast.

This is not hypothetical. Backfilling ten delivery days through the job to
exercise the pipeline produced MAE 285 against a backtest MAE of 914 — a
threefold "improvement" that was entirely an artefact of every one of those days
falling before the model's `train_end`. The number was beautiful and meaningless.

The model card records `train_end`, so the check costs one comparison. In-sample
hours are marked and **excluded from metrics and from drift by default**, and the
count of exclusions is printed rather than hidden: a report that quietly drops
rows is a report nobody can check.

## The second decision: refuse to judge on too little data

Every comparison here has a minimum sample size and returns `INSUFFICIENT`
rather than a verdict when it is not met. Daily MAE on 24 points is noisy enough
that a week of ordinary variation will produce a "degradation" if you let it.
A monitor that cries wolf on noise gets muted, and then it is not a monitor.
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
from powerforecast.evaluation.metrics import ErrorSummary, bias, summarize
from powerforecast.features.availability import LOCAL_TZ
from powerforecast.forecasts.store import latest_run, read_forecasts
from powerforecast.models.persistence import ModelStoreError, read_card

logger = logging.getLogger("verify")

TARGET_COLUMN = "consumption_mwh"
PLAN_COLUMN = "load_plan_mwh"

# Turkish demand never approaches zero, so MAPE needs no floor for its own sake.
# It is passed anyway so that a bad ingestion — a run of zeros where a series
# failed — cannot silently produce an infinite MAPE that looks like a modelling
# problem.
MAPE_FLOOR = 1.0

# Below this many verified hours, no verdict is issued. Seven days of a delivery
# day is 168 hours; asking for 168 in each window means the shortest comparison
# is a week against a week, which is roughly where daily noise stops dominating.
MIN_HOURS = 168

# How much worse the *skill* has to get before it is called degradation. Skill is
# the fraction of the plan's error we remove, so 0.05 means we used to beat the
# plan by, say, 25% and now beat it by 20%. Chosen to be well outside week-to-week
# wobble and well inside anything worth a retrain — and named here rather than
# buried, because it is a judgement, not a measurement.
SKILL_DROP_THRESHOLD = 0.05


@dataclass(frozen=True)
class DriftVerdict:
    """The outcome of one drift comparison, with the evidence attached.

    The numbers travel with the verdict on purpose. "DEGRADED" on its own invites
    the reader to trust or dismiss it; "DEGRADED, skill fell from 0.26 to 0.14 on
    336 hours" invites them to check it.
    """

    status: str  # OK | DEGRADED | HARDER_PERIOD | INSUFFICIENT
    detail: str
    recent_hours: int
    baseline_hours: int
    recent_mae: float
    baseline_mae: float
    recent_skill: float
    baseline_skill: float

    def __str__(self) -> str:
        return f"{self.status}: {self.detail}"


def verify(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    target: str = TARGET_COLUMN,
    plan: str = PLAN_COLUMN,
    model_directory: Path | None = None,
) -> pd.DataFrame:
    """Join stored forecasts to what actually happened.

    Only hours whose actual has arrived are returned. An hour still in the future
    is not a miss, and counting it as one would make every report look worse the
    closer it ran to the present.

    Returns a frame indexed by delivery hour with the forecast, the actual, the
    signed **residual**, an `in_sample` flag, and — where the operator published
    one — the plan and its residual.

    A note on signs, because this repository once held both conventions at once.
    `residual = actual - forecast`, so a positive residual means the forecast
    came in **low**. Reported `bias` uses the opposite convention, matching
    `evaluation.metrics.bias`: positive means the forecast **runs high**. Two
    columns, two meanings, and the names now differ so neither can be read as
    the other. The earlier collision — a column called `error` being reported
    under the heading `bias` — produced two confidently wrong sentences before
    anyone noticed, which is the whole argument for naming things apart.

    The plan columns are what make drift detection possible; `in_sample` is what
    stops a fitted hour being counted as a forecast. See the module docstring.
    """
    if forecasts.empty:
        return _empty_verification()

    # One row per hour. The store deliberately keeps several model versions per
    # hour, and averaging metrics over a frame that holds two versions for some
    # hours and one for others would weight those hours differently for a reason
    # that has nothing to do with performance.
    forecasts = latest_run(forecasts)

    actual = panel[target].reindex(forecasts.index)
    verified = forecasts.loc[actual.notna()].copy()
    if verified.empty:
        return _empty_verification()

    verified["actual_mwh"] = actual.loc[verified.index]
    verified["residual"] = verified["actual_mwh"] - verified["forecast_mwh"]
    verified["abs_error"] = verified["residual"].abs()

    if plan in panel.columns:
        verified["plan_mwh"] = panel[plan].reindex(verified.index)
        verified["plan_residual"] = verified["actual_mwh"] - verified["plan_mwh"]
        verified["plan_abs_error"] = verified["plan_residual"].abs()

    verified["delivery_date"] = pd.DatetimeIndex(verified.index).tz_convert(LOCAL_TZ).date
    verified["in_sample"] = _in_sample_flags(verified, directory=model_directory)
    return verified


def out_of_sample(verified: pd.DataFrame) -> pd.DataFrame:
    """Drop hours the model was trained on. Everything scored passes through here."""
    if verified.empty or "in_sample" not in verified.columns:
        return verified
    return verified[~verified["in_sample"].fillna(False).astype(bool)]


def daily_summary(verified: pd.DataFrame) -> pd.DataFrame:
    """One row per delivery day: how the day went, ours and the plan's.

    Per-day rather than per-hour because a delivery day is the unit that was
    actually bid. A bad afternoon inside an otherwise fine day is a modelling
    detail; a bad day is an operational event, and the two want different tables.
    """
    if verified.empty:
        return pd.DataFrame()

    rows = []
    for day, group in verified.groupby("delivery_date"):
        row: dict[str, object] = {
            "delivery_date": day,
            "hours": len(group),
            "MAE": float(group["abs_error"].mean()),
            "RMSE": float((group["residual"] ** 2).mean() ** 0.5),
            # Same convention as `evaluation.metrics.bias`: positive means the
            # forecast runs high. Computed through that function rather than
            # negated by hand, so the two can never drift apart.
            "bias": bias(group["actual_mwh"], group["forecast_mwh"]),
        }
        if "plan_abs_error" in group and group["plan_abs_error"].notna().any():
            plan_mae = float(group["plan_abs_error"].mean())
            row["plan_MAE"] = plan_mae
            row["skill_vs_plan"] = _skill(row["MAE"], plan_mae)  # type: ignore[arg-type]
        rows.append(row)

    return pd.DataFrame(rows).set_index("delivery_date").sort_index()


def overall(verified: pd.DataFrame, *, include_in_sample: bool = False) -> ErrorSummary | None:
    """Every out-of-sample hour as one summary, or `None` when there are none.

    In-sample hours are excluded by default; the flag exists only so a deliberate
    investigation can look at them. The default is the honest one — including
    them produces a number that flatters the model for a reason unrelated to
    forecasting.
    """
    scored = verified if include_in_sample else out_of_sample(verified)
    if scored.empty:
        return None
    return summarize(scored["actual_mwh"], scored["forecast_mwh"], mape_floor=MAPE_FLOOR)


def detect_drift(
    verified: pd.DataFrame,
    *,
    recent_days: int = 14,
    min_hours: int = MIN_HOURS,
    skill_drop: float = SKILL_DROP_THRESHOLD,
) -> DriftVerdict:
    """Compare a recent window against everything before it, controlling for difficulty.

    The window split is by delivery day, not by row count, so a day with missing
    hours does not silently pull days from the other side of the boundary.

    Four outcomes:

    * ``INSUFFICIENT`` — not enough verified hours on one side. No verdict.
    * ``DEGRADED`` — error rose **and** skill against the plan fell. This is the
      one worth acting on: we lost ground the plan did not.
    * ``HARDER_PERIOD`` — error rose but skill held. The period was harder for
      everyone; a retrain would be chasing weather.
    * ``OK`` — nothing to report.

    Note what is *not* here: no test statistic, no p-value. Two windows of
    autocorrelated hourly error do not satisfy the independence assumptions those
    would need, and a number that looks rigorous while resting on a violated
    assumption is worse than a threshold that is openly a judgement call.
    """
    if verified.empty:
        return _insufficient("no verified forecasts")

    verified = out_of_sample(verified)
    if verified.empty:
        return _insufficient(
            "every verified hour is inside the model's training window; "
            "in-sample error says nothing about live performance"
        )

    days = sorted(verified["delivery_date"].unique())
    if len(days) < 2:
        return _insufficient(f"only {len(days)} delivery day(s) verified")

    cutoff = days[-recent_days] if len(days) > recent_days else days[len(days) // 2]
    recent = verified[verified["delivery_date"] >= cutoff]
    baseline = verified[verified["delivery_date"] < cutoff]

    if len(recent) < min_hours or len(baseline) < min_hours:
        return _insufficient(
            f"need {min_hours} hours per window, have {len(baseline)} baseline "
            f"and {len(recent)} recent",
            recent_hours=len(recent),
            baseline_hours=len(baseline),
        )

    recent_mae = float(recent["abs_error"].mean())
    baseline_mae = float(baseline["abs_error"].mean())
    recent_skill = _window_skill(recent)
    baseline_skill = _window_skill(baseline)

    worse = recent_mae > baseline_mae
    skill_fell = (
        baseline_skill == baseline_skill  # not NaN
        and recent_skill == recent_skill
        and (baseline_skill - recent_skill) > skill_drop
    )

    if worse and skill_fell:
        status = "DEGRADED"
        detail = (
            f"MAE {baseline_mae:,.0f} -> {recent_mae:,.0f} and skill against the plan "
            f"{baseline_skill:.3f} -> {recent_skill:.3f}. The plan did not lose the same "
            "ground, so this is the model rather than the weather. Consider a retrain."
        )
    elif worse:
        status = "HARDER_PERIOD"
        detail = (
            f"MAE {baseline_mae:,.0f} -> {recent_mae:,.0f}, but skill against the plan held "
            f"({baseline_skill:.3f} -> {recent_skill:.3f}). The period was harder for every "
            "forecaster; retraining would be chasing conditions, not fixing a model."
        )
    else:
        status = "OK"
        detail = (
            f"MAE {baseline_mae:,.0f} -> {recent_mae:,.0f}, skill "
            f"{baseline_skill:.3f} -> {recent_skill:.3f}."
        )

    return DriftVerdict(
        status=status,
        detail=detail,
        recent_hours=len(recent),
        baseline_hours=len(baseline),
        recent_mae=recent_mae,
        baseline_mae=baseline_mae,
        recent_skill=recent_skill,
        baseline_skill=baseline_skill,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _in_sample_flags(frame: pd.DataFrame, *, directory: Path | None) -> pd.Series:
    """True where the delivery hour lies inside the producing model's training window.

    Looked up per `(model_name, model_version)`, because the store deliberately
    holds several versions and a boundary right for one is wrong for another: a
    model retrained last week has a later `train_end` than the one it replaced,
    so the same hour can be in-sample for one and genuinely out-of-sample for the
    other.

    A version whose card is missing — deleted, or produced elsewhere — is marked
    `False` rather than dropped. Guessing "in-sample" would silently discard real
    evidence; keeping it and logging the missing card leaves the uncertainty
    visible instead of assumed away.
    """
    flags = pd.Series(False, index=frame.index)
    pairs = frame[["model_name", "model_version"]].drop_duplicates().itertuples(index=False)

    for name, version in pairs:
        try:
            card = read_card(str(name), str(version), directory=directory)
        except ModelStoreError:
            logger.warning(
                "no card for %s/%s; its hours are treated as out-of-sample, which may "
                "flatter the report",
                name,
                version,
            )
            continue
        rows = (frame["model_name"] == name) & (frame["model_version"] == version)
        flags |= rows & (frame.index <= pd.Timestamp(card.train_end))

    return flags


def _skill(our_mae: float, plan_mae: float) -> float:
    """Fraction of the plan's error we remove. 0 means level, 1 means perfect.

    Negative when we are worse, which is the right behaviour: a metric that
    floors at zero would hide exactly the situation the monitor exists to catch.
    """
    if plan_mae <= 0:
        return float("nan")
    return (plan_mae - our_mae) / plan_mae


def _window_skill(window: pd.DataFrame) -> float:
    if "plan_abs_error" not in window or window["plan_abs_error"].isna().all():
        return float("nan")
    return _skill(float(window["abs_error"].mean()), float(window["plan_abs_error"].mean()))


def _insufficient(detail: str, *, recent_hours: int = 0, baseline_hours: int = 0) -> DriftVerdict:
    return DriftVerdict(
        status="INSUFFICIENT",
        detail=detail,
        recent_hours=recent_hours,
        baseline_hours=baseline_hours,
        recent_mae=float("nan"),
        baseline_mae=float("nan"),
        recent_skill=float("nan"),
        baseline_skill=float("nan"),
    )


def _empty_verification() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["forecast_mwh", "actual_mwh", "residual", "abs_error", "delivery_date"],
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    *,
    start: str | None = None,
    end: str | None = None,
    model_version: str | None = None,
    recent_days: int = 14,
    processed_root: Path | None = None,
    model_directory: Path | None = None,
) -> tuple[pd.DataFrame, DriftVerdict]:
    forecasts = read_forecasts(
        start=start, end=end, model_version=model_version, root=processed_root
    )
    verified = verify(forecasts, load_panel(), model_directory=model_directory)
    return verified, detect_drift(verified, recent_days=recent_days)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="first delivery hour, ISO")
    parser.add_argument("--end", default=None, help="last delivery hour, ISO")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--recent-days", type=int, default=14)
    parser.add_argument("--csv", type=Path, default=None, help="write the daily table here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    verified, verdict = run(
        start=args.start,
        end=args.end,
        model_version=args.model_version,
        recent_days=args.recent_days,
    )

    if verified.empty:
        print("nothing to verify: no stored forecast has a matching actual yet")
        # Not an error. Early in a deployment this is the expected state, and a
        # non-zero exit here would make the first fortnight of monitoring look
        # like a fortnight of failures.
        return 0

    scored = out_of_sample(verified)
    excluded = len(verified) - len(scored)

    if excluded:
        print(
            f"excluded {excluded:,} of {len(verified):,} hours: inside the model's training "
            "window. In-sample error measures the fit, not the forecast."
        )
    if scored.empty:
        print("nothing left to score once in-sample hours are removed")
        return 0

    summary = overall(verified)
    assert summary is not None
    daily = daily_summary(scored)

    print(f"verified {len(scored):,} out-of-sample hours across {len(daily)} delivery days")
    print(f"  MAE {summary.mae:,.0f}   RMSE {summary.rmse:,.0f}   MAPE {summary.mape:.2f}%")
    if "skill_vs_plan" in daily.columns:
        print(f"  mean daily skill against the published plan: {daily['skill_vs_plan'].mean():.3f}")
    print()
    print(daily.to_string(float_format=lambda v: f"{v:,.3f}"))
    print()
    print(verdict)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(args.csv)
        print(f"\nwrote {args.csv}")

    # A verdict is information, not a failure. Exiting non-zero on DEGRADED would
    # make a scheduler treat "the model needs attention" as "the monitor broke",
    # and the two need different responses.
    return 0


def delivery_day_of(timestamp: pd.Timestamp) -> date:
    """The local delivery day an hour belongs to. Exposed for callers and tests."""
    return pd.Timestamp(timestamp).tz_convert(LOCAL_TZ).date()


if __name__ == "__main__":
    raise SystemExit(main())
