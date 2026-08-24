# 0008 — A bias correction that can be deployed, and what it is worth

**Status:** Superseded by [0009](0009-no-monotonic-trend-feature.md)
**Date:** 2026-08-20

> **Superseded the same day.** Every measurement here was taken while the design
> matrix still contained `years_elapsed`. Removing that feature removed the
> persistent level error this correction was repairing, and re-measured without
> it, every variant is **worse than no correction**. The reasoning below about
> availability, the mean-versus-median distinction and the cost of fine
> grouping all still hold; the adoption does not. The record is left standing
> rather than rewritten — see ADR 0009 for what replaced it.

## Context

The backtest's headline model applies a bias correction estimated over the whole
evaluation period. That is deliberate and documented (ADR 0005): it exists to be
a competitor that is hard to beat, applied symmetrically to the published plan
and to our own model. It is also not shippable, because it sees the future.

So the served model carries no correction, and the README's 27% is a backtest
number the deployed system does not reproduce. ADR 0007 made that gap visible by
verifying real forecasts; this ADR is about closing it — or measuring that it
cannot be closed this way.

Two questions had to be answered before anything could be adopted:

1. Can the correction be estimated from the past **without** breaking the
   availability rule that governs everything else in this project?
2. Does it actually help?

## Decision

**`forecasts/bias.py` estimates offsets from a rolling window of past errors,
with the availability cutoff computed in code.** Bids for delivery day D close at
12:30 on D-1, so the newest complete observation is 11:00 on D-1. Delivery day
D-2 is complete at that moment; D-1 is in progress. The module therefore uses
complete days only — D-2 and earlier — and the cutoff is derived from the origin
rather than trusted to the caller.

**The statistic is the median, not the mean.** Measured over 4,439 out-of-sample
hours (February–August 2026, walk-forward):

| forecast | MAE | RMSE |
|---|---|---|
| + hourly offset, **median** | **907.8** | 1,223.4 |
| + constant offset, median | 909.9 | **1,217.7** |
| raw, no correction | 914.5 | 1,233.6 |
| + hourly offset, mean | 918.1 | 1,230.4 |
| + constant offset, mean | 923.8 | 1,230.9 |
| published plan | 1,140.5 | 1,530.8 |

The mean makes MAE **worse** while improving RMSE. That is not an anomaly, it is
the definition of the two metrics: shifting by the mean minimises squared error,
shifting by the median minimises absolute error. Correcting with the mean
optimises the metric this project does not headline, at the expense of the one it
does.

**Graceful degradation is explicit.** Hourly requires at least 14 samples in
every hour; below that it falls back to a constant, and below 72 samples overall
it applies nothing. The returned object records which happened and why.

## Alternatives considered

**Estimating from everything up to the origin, including the observed part of
D-1.** Legitimate — those hours really are observed — and rejected because a
partial day covers only night and morning, so an hourly offset built from it
would be estimated on a biased sample of hours. Twelve hours of a slow-moving
quantity is not worth the subtlety.

**Hour-by-weekday offsets.** The obvious refinement, especially since the error
concentrates on weekends. Measured on the same walk-forward:

| correction | MAE |
|---|---|
| hourly, median | **907.8** |
| none | 912.4 |
| hour × weekend, median | 922.7 |
| hour × weekday, median | 979.3 |

Both finer groupings are worse than doing nothing. A 28-day window holds 28
samples per hour but only 4 per hour-and-weekday cell, and 168 offsets fitted on
four observations each are 168 chances to be confidently wrong. Not shipped: the
code for a rejected idea is a maintenance cost with no user.

**A longer window to make finer grouping viable.** Would need months, over which
the bias being corrected is no longer the current one. The two requirements pull
in opposite directions, which is itself the finding.

**Applying the correction inside the served model.** Deferred. It changes what
the store records and needs a column for the offset so the correction stays
auditable. Worth doing — but a 0.7% gain does not justify rushing a storage
change, and the store's schema is the contract the monitoring layer reads.

## Consequences

* The correction is real, deployable, leak-free — and worth **0.7% of MAE**. That
  is the honest number, and it goes in the README next to the 27%.
* **The gap between the deployed model and the backtest was never mostly a bias
  problem.** The backtest's correction looks powerful because it sees the whole
  period, not because bias correction is powerful. Recognising that is worth more
  than the 0.7%, because it redirects the remaining work.
* **A previous conclusion is corrected.** ADR 0007 reported that the deployed
  model lost to the published plan, on a single 30-day window in July. Over
  February–August the raw model beats the plan by **19.8%**. July is a hard month
  for this model and an easy one for the plan; one month was not a verdict, and
  saying so at the time would have been better than saying it now.
* Training cut-off matters more than the correction does. On one fixed July
  window, MAE by cut-off: 31 Jan **1,057**, 31 Mar 1,136, 31 May **1,764**,
  30 Jun 1,134 — non-monotonic, and a 67% swing. Retraining cadence is therefore
  a first-order decision, not housekeeping, and *why* the end of May is so bad is
  an open question rather than a settled one.
* The weekend concentration survives the correction (Saturday −3.9%, Sunday
  −7.2% against Wednesday to Friday's +5.7% to +10.5%). It is a modelling problem,
  not a bias problem, and that is where the next real gain is.
