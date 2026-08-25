# 0010 — The bias correction works, and what it depends on

**Status:** Accepted
**Date:** 2026-08-24
**Supersedes:** [0008](0008-deployable-bias-correction.md)

## Context

[ADR 0008](0008-deployable-bias-correction.md) built a rolling, leak-free bias
correction and measured it as worth +0.7% of MAE. [ADR 0009](0009-no-monotonic-trend-feature.md)
re-measured it after removing the trend feature, found every variant *worse* than
no correction, and concluded the residual bias carried no usable signal.

Both conclusions were wrong, and wrong for the same reason: each was measured
under one configuration that happened to be unfavourable, and the configuration
was never part of the question being asked.

The deployed model runs low. On the backtest its bias is −375 MWh against an MAE
of 890, and the whole-period oracle correction takes 890 to 831. That is a 6.6%
gain sitting in plain sight, and three separate attempts to reach it had failed:
a rolling correction, a level-relative target, and a sliding training window.

## What the measurement actually says

Two parameters decide the outcome. Both had been set wrongly.

**Window length**, swept on backtest predictions (21,849 out-of-sample hours):

| window | MAE | vs raw |
|---|---|---|
| none | 889.8 | — |
| 30 days | 855.4 | −3.9% |
| 45 days | 838.1 | −5.8% |
| 90 days | 835.6 | −6.1% |
| **120 days** | **833.9** | **−6.3%** |
| 365 days | 833.8 | −6.3% |
| *whole-period oracle* | *830.6* | *−6.7%* |

The curve is flat from about 45 days on. The 28 days used in ADR 0008 sat on the
steep part of it.

**Model freshness** — the same code, same windows, on a model trained once and
never refreshed:

| | raw | 28-day | 120-day |
|---|---|---|---|
| refit monthly (backtest) | 889.8 | 854.2 (−4.0%) | **833.3 (−6.3%)** |
| trained once, never refreshed | 912.8 | 936.6 (**+2.6%**) | 916.4 (+0.4%) |

Under regular retraining the model's bias is roughly stationary, so a long
trailing window estimates it well. A stale model's bias **drifts as the model
ages**, and a trailing estimate then describes an error that has already moved.

ADR 0008 measured a 28-day window on a stale model — both parameters wrong at
once — and drew a conclusion about bias correction in general.

## Decision

**Apply the correction, with the freshness condition enforced in code.**

* `DEFAULT_WINDOW_DAYS = 119` — 17 weeks, a multiple of 7 so every hour sees the
  same number of Saturdays as Tuesdays.
* `DEFAULT_STATISTIC = "mean"`, measured at −6.34% against the median's −6.00%.
  An earlier measurement under different conditions preferred the median, which
  is why this stays a parameter rather than becoming a constant.
* `MIN_SAMPLES_PER_HOUR = 30` — roughly a month of history in every hour.
* **`should_correct(train_end, origin)` refuses beyond `MAX_MODEL_AGE_DAYS = 35`**
  and returns its reason, which the job logs. The retraining schedule and the
  bias correction are one decision, so the coupling is a function rather than a
  comment.

**The correction lives in `forecasts/day.py`, not in the job.** The service
produces forecasts too, and a correction applied on one path and not the other
would make the two disagree — the exact divergence that module exists to prevent.

**The applied offset is stored.** `FORECAST_COLUMNS` gains `bias_offset_mwh`, so
the raw forecast stays recoverable and "was this number corrected, and by how
much?" is answerable from the row rather than by rerunning an estimator against a
history that has since changed. Files written before the column existed read back
as zero, which is accurate: those forecasts were uncorrected.

## Alternatives considered

**Correcting the hour-of-day shape only, leaving the level alone.** Motivated by
a stability measurement: across consecutive 30-day blocks the hourly profile
correlates +0.38 while the *level* correlates +0.05 — essentially nothing. The
obvious reading is that the shape is recoverable and the level is not.

Measured, it is the reverse. Shape-only correction is *worse* than doing nothing
(892–896 against 890), while the full correction — mostly a level shift — gives
833. The level correlation rises to +0.45 over 90-day blocks, which is why a long
window works where a short one does not. A plausible mechanism read off a
correlation table, contradicted by direct measurement; the measurement wins.

**Applying it only in the job, not the service.** Simpler, and it would leave two
paths producing different numbers for the same day.

**Dropping the freshness guard and documenting the constraint instead.** Rejected.
The failure mode is a *worse* forecast with no error and no signal, on a model
that quietly aged past the point where the correction helps. A property that must
be remembered is a property that will eventually be forgotten.

## Consequences

* **The deployed configuration should reach roughly the headline.** Backtest MAE
  890 → 833 against the oracle's 831: the deployable correction recovers about
  95% of a gain that was previously available only with hindsight.

* **That is an expectation, not an observation.** The −6.3% was measured on
  backtest predictions. The live store has no history yet, so what the running
  system achieves is unmeasured, and the README says so rather than borrowing the
  backtest number.

* **The correction has a warm-up.** It needs 30 samples in every hour — about a
  month of operation — before it does anything. Until then `method="none"` and
  the reason says why. The first month of any deployment is uncorrected.

* **Forecast history must survive retraining.** The estimate deliberately spans
  model versions of the same model name: in the backtest the error history
  crosses folds, which is the setting in which −6.3% was measured. The store
  keeps every version, so this holds by construction — but it is now a property
  the correction *depends* on rather than an incidental one.

* **A retraining cadence is load-bearing and does not yet exist.** Nothing
  enforces monthly retraining; the guard only detects its absence and switches
  the correction off. Until a retrain job exists, the correction will stop
  applying about five weeks after each manual `train.py` run — visibly in the
  job's log, but only to someone reading it.

* **Two earlier ADRs were wrong and are marked so.** 0008 concluded the correction
  was worth 0.7%; 0009 concluded it was worthless. Both measured one configuration
  and generalised from it. The pattern is now familiar enough to name: this
  project's mistakes have not been in its code, they have been in treating a
  single measurement as a distribution.
