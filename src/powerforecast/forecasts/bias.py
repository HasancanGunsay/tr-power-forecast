"""A bias correction that could actually be deployed.

The backtest's headline number uses a correction estimated over the **whole
evaluation period**, which is a competitor built to be hard to beat rather than a
component built to be shipped. It sees the future. Applied honestly, both to the
published plan and to our own model, it makes the comparison fair — and it makes
the absolute numbers optimistic on both sides. The README has always said so.

The consequence only became visible once forecasts were produced unattended and
verified: the *deployed* configuration carries no correction, and on the first
genuinely out-of-sample window it lost to the published plan. The gap between
"what the backtest reports" and "what the service does" is exactly this module.

## What makes a correction deployable

Two things, and only the second is difficult.

**It must be estimated from the past.** A rolling window of recent errors, not
the whole period.

**It must respect the forecast origin.** This is where a leak would hide. Bids
for delivery day D close at 12:30 on D-1, so the newest complete observation is
11:00 on D-1 (ADR 0004). Which past errors does that leave?

* Delivery day D-2 ended at midnight before the origin: **complete**.
* Delivery day D-1 is *in progress* at the origin — hours 00:00 to 11:00 are
  observed, the rest are not.

This module uses complete days only, so `D-2` and earlier. Including the observed
part of D-1 would be legitimate and would add up to twelve hours of the freshest
data; it is left out because a partial day is a different animal — it covers only
the night and morning, so an hourly offset estimated from it would be built from
a biased sample of hours. Throwing away half a day of a slow-moving quantity
costs little; getting the availability rule subtly wrong costs the whole result.

The cutoff is computed in code from the origin, not documented and hoped for.

## The median, not the mean — measured, not assumed

The obvious estimator is the mean error, and it was tried first. It made MAE
**worse** — 914.5 to 918.1 over 4,439 out-of-sample hours — while nudging RMSE
down from 1,233.6 to 1,230.4.

That split is not noise, it is the definition of the two metrics. Shifting a
forecast by the mean error minimises *squared* error; shifting by the **median**
error minimises *absolute* error. Correcting with the mean therefore optimises a
metric this project does not report as its headline, at the expense of the one it
does.

The median improves both: MAE 914.5 to 907.8, RMSE 1,233.6 to 1,223.4. That is
0.7%, and 0.7% is the honest description. A bias correction is not where the
remaining error lives. See `analysis/bias_correction.py` to reproduce.

## Finer grouping was tried and rejected

Given hourly offsets, hour-by-weekday offsets look like the obvious next step,
especially since the model's error concentrates on weekends. Measured on the same
walk-forward (a separate pass, so its own baseline, 4,463 hours):

| correction | MAE |
|---|---|
| hourly, median | **907.8** |
| none | 912.4 |
| hour x weekend, median | 922.7 |
| hour x weekday, median | 979.3 |

Both refinements are worse than doing nothing at all. The arithmetic says why: a
28-day window holds 28 samples per hour, but only 4 per hour-and-weekday cell.
An offset fitted on four observations is noise wearing a confident sign, and 168
of them are 168 chances to be confidently wrong. Neither is shipped — the code
for a rejected idea is a maintenance cost with no user.

This is the same limit reached with the holiday features: the data supports the
coarse pattern and not the fine one, and the way to tell is to measure rather
than to reason about which is more expressive.

## Graceful degradation, stated rather than silent

An hourly correction needs enough samples in **every** hour to mean anything.
When it does not have them the code falls back — hourly to constant, constant to
nothing — and the returned object says which happened. A correction quietly
fitted on three observations is how a model acquires a confident systematic error
it did not have before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from powerforecast.features.availability import LOCAL_TZ

# Long enough that each hour has a few weeks of examples, short enough to follow
# a regime that moves. Four weeks also lands on whole weeks, so every hour sees
# the same number of Saturdays as Tuesdays — a window of, say, 25 days would
# weight some weekdays more than others for no reason at all.
DEFAULT_WINDOW_DAYS = 28

# Per-hour and overall minimums before a correction is applied at all.
MIN_SAMPLES_PER_HOUR = 14
MIN_TOTAL_SAMPLES = 72

# Median by default. MAE is minimised by shifting to the median error and MSE by
# shifting to the mean, so the choice follows the metric being reported rather
# than habit — and it was measured, not assumed. See the module docstring.
DEFAULT_STATISTIC = "median"


@dataclass(frozen=True)
class BiasOffsets:
    """What to add to a forecast, and how much evidence stands behind it."""

    method: str  # "hourly" | "constant" | "none"
    statistic: str = DEFAULT_STATISTIC
    constant: float = 0.0
    hourly: dict[int, float] = field(default_factory=dict)
    n_samples: int = 0
    n_days: int = 0
    reason: str = ""

    def apply(self, forecasts: pd.Series, *, tz: str = LOCAL_TZ) -> pd.Series:
        """Return the corrected forecast. `method == "none"` returns it unchanged."""
        return forecasts + self.offsets_for(pd.DatetimeIndex(forecasts.index), tz=tz)

    def offsets_for(self, index: pd.DatetimeIndex, *, tz: str = LOCAL_TZ) -> pd.Series:
        """The amount added to each hour. Exposed so a stored forecast can record it.

        Keeping the offset alongside the corrected value is what makes the
        correction auditable: months later, "was this number corrected, and by how
        much?" is answerable from the row rather than by rerunning the estimator
        against a history that has since changed.
        """
        if self.method == "hourly":
            hours = pd.Index(index.tz_convert(tz).hour)
            return pd.Series(hours.map(self.hourly).to_numpy(), index=index, dtype="float64")
        value = self.constant if self.method == "constant" else 0.0
        return pd.Series(value, index=index, dtype="float64")


def usable_error_cutoff(origin: pd.Timestamp, *, tz: str = LOCAL_TZ) -> pd.Timestamp:
    """The first instant whose error may **not** be used at `origin`.

    Local midnight of the origin's own day. The origin is 11:00 on D-1, so this
    is the start of D-1 — leaving delivery day D-2 and everything before it, all
    of them complete days.
    """
    return pd.Timestamp(origin).tz_convert(tz).normalize().tz_convert("UTC")


def estimate_offsets(
    errors: pd.Series,
    *,
    origin: pd.Timestamp,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_samples_per_hour: int = MIN_SAMPLES_PER_HOUR,
    min_total_samples: int = MIN_TOTAL_SAMPLES,
    statistic: str = DEFAULT_STATISTIC,
    tz: str = LOCAL_TZ,
) -> BiasOffsets:
    """Estimate a correction from errors that were observable at `origin`.

    Args:
        errors: `actual - forecast`, indexed by delivery hour in UTC. Anything at
            or after the availability cutoff is dropped here rather than trusted
            to have been dropped by the caller.
        origin: The forecast origin of the day being corrected.
        window_days: How much recent history to average over.
        min_samples_per_hour: Below this in any hour, fall back to a constant.
        min_total_samples: Below this overall, apply no correction.
        statistic: `"median"` (default, minimises MAE) or `"mean"` (minimises
            MSE). Exposed so the choice can be re-measured rather than inherited.

    Returns:
        `BiasOffsets`, whose `method` says what was actually done and whose
        `reason` says why — including when the answer is "nothing".
    """
    if errors.empty:
        return BiasOffsets(method="none", reason="no error history")

    cutoff = usable_error_cutoff(origin, tz=tz)
    window_start = cutoff - pd.Timedelta(days=window_days)

    index = pd.DatetimeIndex(errors.index)
    usable = errors[(index < cutoff) & (index >= window_start)].dropna()

    if usable.empty:
        return BiasOffsets(
            method="none",
            reason=f"no observable errors in the {window_days} days before {cutoff.date()}",
        )

    if statistic not in {"median", "mean"}:
        raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}")

    local = pd.DatetimeIndex(usable.index).tz_convert(tz)
    n_days = len(set(local.date))
    constant = float(usable.median() if statistic == "median" else usable.mean())

    if len(usable) < min_total_samples:
        return BiasOffsets(
            method="none",
            statistic=statistic,
            n_samples=len(usable),
            n_days=n_days,
            reason=(
                f"only {len(usable)} observable errors, below the {min_total_samples} "
                "needed for any correction"
            ),
        )

    by_hour = usable.groupby(local.hour)
    counts = by_hour.size()
    centres = by_hour.median() if statistic == "median" else by_hour.mean()

    if len(counts) == 24 and int(counts.min()) >= min_samples_per_hour:
        return BiasOffsets(
            method="hourly",
            statistic=statistic,
            constant=constant,
            hourly=dict(zip(counts.index.astype(int), centres.astype(float), strict=True)),
            n_samples=len(usable),
            n_days=n_days,
            reason=f"{int(counts.min())}+ samples in every hour over {n_days} days",
        )

    thin = 24 - len(counts) if len(counts) < 24 else int((counts < min_samples_per_hour).sum())
    return BiasOffsets(
        method="constant",
        statistic=statistic,
        constant=constant,
        n_samples=len(usable),
        n_days=n_days,
        reason=(
            f"{thin} hour(s) below {min_samples_per_hour} samples, so an hourly "
            "correction would be fitted on noise"
        ),
    )
