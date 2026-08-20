# 0007 — Recording forecasts, and how drift is judged

**Status:** Accepted
**Date:** 2026-08-20

## Context

With the model behind a service, the project could produce a forecast on demand
but could not answer the only question that matters after deployment: *is it
still working?*

That question cannot be answered retrospectively. A forecast recomputed today
from today's data is not the forecast that was made — it is a much better one,
built from information that did not exist at the time. So the forecast has to be
written down **while it is still a prediction**, or the evidence is gone.

Writing it down is the easy half. The hard half is interpreting it, and there are
two ways to get that wrong, both of which produce numbers rather than errors.

**Mistaking a harder period for a worse model.** Forecast error rises for two
unrelated reasons: the model drifted away from the data, or the period was
genuinely harder — a heatwave, a holiday cluster, unusual industrial demand. The
first needs a retrain; the second needs nothing. A recent-versus-baseline
comparison cannot separate them, so it raises alarms that turn out to be weather.
An alarm that is ignored is worse than no alarm, because it teaches people to
ignore the next one too.

**Scoring a fit as though it were a forecast.** This one was not hypothetical.
Backfilling ten delivery days through the new job to exercise the pipeline gave
MAE 285 against a backtest MAE of 914 — a threefold improvement, and entirely an
artefact: every one of those days lay inside the model's training window. Nothing
raised, and nothing could have. Every component behaved correctly; the error was
in what the aggregate meant.

## Decision

**A forecast store, merge-on-write, keyed on `(timestamp, model_version)`.**
Monthly parquet under `data/processed/forecasts/`, mirroring `data/raw`. Each row
carries its own provenance — model name, version, forecast origin, generation
time — repeated down all 24 rows of a day. Re-running a day updates it; running a
different model version keeps both.

**Drift is judged on skill, not on error.** The panel already holds the grid
operator's published forecast for every hour, produced under the same conditions
from the same information. That is a free control. Drift compares how much better
than the plan we are, before and after — a difference-in-differences. Three
outcomes are distinguished rather than one: `DEGRADED` (error rose, skill fell),
`HARDER_PERIOD` (error rose, skill held), `OK`.

**In-sample hours are excluded by default.** The model card records `train_end`,
so verification marks every delivery hour that falls inside the producing model's
training window and drops it from metrics and drift. The count of exclusions is
printed, never hidden. `overall(..., include_in_sample=True)` exists only for a
deliberate investigation.

**No verdict below 168 hours per window.** A week against a week is roughly where
daily noise stops dominating. Below that, `INSUFFICIENT` — no judgement at all.

**`train --train-until DATE`.** A model can be trained to a chosen boundary,
which is what makes both a dated retrain and an honest out-of-sample test of the
monitoring layer possible without waiting weeks for new days.

## Alternatives considered

**A statistical test on the two error windows.** Rejected. Hourly forecast errors
are strongly autocorrelated, so the independence assumption behind a t-test or a
Kolmogorov–Smirnov test is violated, and the resulting p-value would look
rigorous while meaning nothing. A stated threshold that is openly a judgement is
more honest than a statistic that is quietly invalid.

**Comparing live error to the backtest number.** Tempting — the backtest is
right there — and wrong, because the backtest covered five years including
winters, while any live window is one season. The comparison would report drift
every summer.

**Population-drift monitoring on the features (PSI, KL divergence).** Standard
practice, and rejected for now: it detects that the input distribution moved,
which is a proxy for the thing we can measure directly. Here the outcome arrives
within days, so measuring the error itself is strictly better than measuring a
predictor of it. Feature drift becomes worth adding only where labels are slow or
absent.

**Keeping one forecast per hour.** Simpler, and it would have made version
comparison impossible — which is one of the two things the store is for.

**Alerting by exit code on `DEGRADED`.** Rejected: a scheduler would treat "the
model needs attention" as "the monitor crashed", and those need different
responses. The verdict is information; the exit code is about whether the check
ran.

## Consequences

* The store is append-only and grows by 24 rows a day — about 9,000 a year. Small
  enough that pruning is a deliberate act rather than a scheduled one.
* Every model that ever produced a stored forecast must keep its card, or its
  hours can no longer be classified. Missing cards are treated as out-of-sample
  and logged, which errs toward keeping evidence rather than discarding it.
* **Corrected on 2026-08-20, see [ADR 0008](0008-deployable-bias-correction.md).**
  The paragraph below reports a single 30-day window and reads as a verdict on
  the deployed model. It is not one. Over February–August the raw model beats the
  published plan by 19.8%; July is a hard month for this model and an easy one
  for the plan. The measurement stands, the generalisation does not. It is left
  here rather than edited away, because a decision log that quietly revises its
  own conclusions is worth less than one that shows them being revised.

* **A genuinely out-of-sample check has now been run, and it is not flattering.**
  A model trained to 2026-06-30 and used for the 30 delivery days from 5 July to
  3 August scores MAE 1,165 against the published plan's 1,036 — mean daily skill
  **−0.36**. The deployed configuration loses to the operator's own forecast on
  this window. The error concentrates on weekends (Saturday MAE 1,667, Sunday
  1,507, both with strong negative bias — systematic over-forecasting) against
  roughly 800–1,270 on weekdays.
* That result is a finding, not a failure of the monitoring layer: the layer
  exists to produce exactly this kind of unwelcome, checkable number. The
  headline backtest figure (MAE 914) came from the **bias-corrected** variant,
  which is not what is deployed — closing that gap is now the top item of
  outstanding work, and until it is closed the README may not claim the deployed
  model beats the plan.
* `detect_drift` correctly reports `DEGRADED` on that window (skill 0.046 →
  −0.265), which is the first time the monitor has been exercised against real
  degradation rather than a fixture.
