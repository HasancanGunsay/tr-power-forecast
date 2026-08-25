"""Correcting the forecast's systematic bias, and the condition that makes it work.

The backtest's headline model applies a bias correction estimated over the whole
evaluation period. That is deliberate (ADR 0005): a competitor built to be hard to
beat. It is also not shippable, because it sees the future.

This module estimates the same correction **from the past only** — and, measured
properly, recovers almost all of the oracle's gain:

| forecast | MAE | RMSE |
|---|---|---|
| raw, no correction | 889.8 | 1,220.6 |
| corrected, 28-day window | 854.2 | 1,179.6 |
| **corrected, 120-day window** | **833.3** | **1,151.9** |
| oracle, whole-period correction | 830.6 | 1,148.0 |

**But only when the model is being retrained.** The same code, applied to a model
trained once and never refreshed, makes things worse:

| | raw | 28-day | 120-day |
|---|---|---|---|
| refit monthly | 889.8 | 854.2 (−4.0%) | **833.3 (−6.3%)** |
| trained once, never refreshed | 912.8 | 936.6 (**+2.6%**) | 916.4 (+0.4%) |

That table is the whole module. Two things had to be right at once, and an earlier
attempt (ADR 0008) got both wrong — a 28-day window on a stale model — and
concluded, reasonably but incorrectly, that bias correction was worthless here.

## Why the condition exists

Under regular retraining the model's bias is roughly stationary, so a long
trailing window estimates it well. A stale model's bias **drifts as the model
ages**: the world moves away from its training set, and a trailing estimate is
always describing a version of the error that has already passed. Adding a stale
offset to a drifting bias is worse than adding nothing.

So the retraining schedule and the bias correction are not two decisions. They are
one, and `should_correct` refuses the correction when the model is too old rather
than leaving that coupling to a comment nobody reads.

## Why 120 days, and why the mean

Both were swept rather than chosen. The window curve is flat from about 45 days
onward — 30 days gives −3.9%, 45 gives −5.8%, and everything from 60 to 365 sits
between −5.5% and −6.3%. 120 is comfortably inside the plateau and short enough to
follow a slow regime change. The earlier 28 days sat on the steep part of that
curve, which is the second half of why ADR 0008 measured what it did.

The mean beats the median here (−6.34% against −6.00%), reversing an earlier
finding taken under different conditions. Both are available; neither is assumed.

## Respecting the forecast origin

This is where a leak would hide. Bids for delivery day D close at 12:30 on D-1, so
the newest complete observation is 11:00 on D-1 (ADR 0004). Delivery day D-2 ended
at midnight before that origin and is complete; D-1 is still in progress. The
module uses complete days only, and the cutoff is computed from the origin rather
than trusted to the caller.

Including the observed part of D-1 would be legitimate and would add up to twelve
hours of the freshest data. It is left out because a partial day covers only night
and morning, so an hourly offset built from it would be estimated on a biased
sample of hours.

## Graceful degradation, stated rather than silent

An hourly correction needs enough samples in every hour to mean anything. When it
does not have them the code falls back — hourly to constant, constant to nothing —
and the returned object records which happened and why. A correction quietly
fitted on three observations is how a model acquires a confident systematic error
it did not have before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from powerforecast.features.availability import LOCAL_TZ

# Swept, not chosen. The curve is flat from about 45 days on; 120 sits inside the
# plateau and still tracks a slow regime change. A multiple of 7 keeps every hour
# seeing the same number of Saturdays as Tuesdays.
DEFAULT_WINDOW_DAYS = 119

# Per-hour and overall minimums before a correction is applied at all. Thirty
# samples per hour is roughly a month of history — enough that an hourly offset
# is describing a pattern rather than a fortnight's weather.
MIN_SAMPLES_PER_HOUR = 30
MIN_TOTAL_SAMPLES = 24 * MIN_SAMPLES_PER_HOUR

# Mean, measured: -6.34% against the median's -6.00% on the same hours. An earlier
# measurement under different conditions preferred the median, which is why this
# is a parameter and not a constant in the code.
DEFAULT_STATISTIC = "mean"

# How old a model may be before its bias correction is refused.
#
# The correction assumes the bias it estimates from the past still describes the
# present. That holds while the model is retrained; a stale model's bias drifts,
# and a trailing estimate then describes an error that has already moved. Measured
# on a model trained once and never refreshed, the correction made MAE *worse* by
# 2.6% at 28 days and was neutral at 120.
#
# 35 days allows a monthly retrain to slip by a few days without silently turning
# the correction off. Past that, `should_correct` returns False and says why.
MAX_MODEL_AGE_DAYS = 35


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


def should_correct(
    trained_until: pd.Timestamp | str,
    origin: pd.Timestamp | str,
    *,
    max_age_days: int = MAX_MODEL_AGE_DAYS,
) -> tuple[bool, str]:
    """Whether a model is fresh enough for its bias correction to be trusted.

    Returns `(ok, reason)` — the reason is populated either way, because a caller
    that skips the correction has to be able to say so in a log line rather than
    leaving a silent difference between two runs.

    Args:
        trained_until: End of the model's training window (`ModelCard.train_end`).
        origin: The forecast origin of the day being corrected.
        max_age_days: Refuse beyond this. See `MAX_MODEL_AGE_DAYS`.
    """
    # Cards store `train_end` as a UTC-aware ISO string, but a caller reading a
    # date off a config file will hand over a naive one. Localising here means the
    # guard cannot be defeated by the shape of its input.
    trained = pd.Timestamp(trained_until)
    at = pd.Timestamp(origin)
    trained = trained.tz_localize("UTC") if trained.tz is None else trained
    at = at.tz_localize("UTC") if at.tz is None else at

    age = (at - trained).total_seconds() / 86400
    if age > max_age_days:
        return False, (
            f"model is {age:.0f} days old at this origin, past the {max_age_days}-day "
            "limit; a stale model's bias drifts and a trailing estimate of it makes "
            "the forecast worse rather than better"
        )
    if age < 0:
        return False, (
            f"model was trained past this origin ({trained_until} > {origin}); "
            "correcting from its own future would not be a correction"
        )
    return True, f"model is {age:.0f} days old"


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
