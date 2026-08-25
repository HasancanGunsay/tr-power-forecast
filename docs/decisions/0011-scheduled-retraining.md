# 0011 — Scheduled retraining, with a gate

**Status:** Accepted
**Date:** 2026-08-25

## Context

[ADR 0010](0010-bias-correction-and-retraining.md) made retraining load-bearing.
The bias correction is worth 6.3% of MAE while the model is fresh and worse than
nothing once it is stale, and `should_correct` switches it off after 35 days.
Nothing was refreshing the model, so the system would quietly lose the correction
about five weeks after each manual `train.py` run — visibly in a log, but only to
someone reading it.

The obvious fix is a job that refits monthly. That fix is incomplete for a reason
this project has already measured.

**A retrain can produce a materially worse model.** Scoring one fixed July window
with models identical in every way except their training cut-off (ADR 0009):

| cut-off | MAE |
|---|---|
| 31 January | 1,057 |
| 31 March | 1,136 |
| **31 May** | **1,764** |
| 30 June | 1,134 |

A 67% swing from nothing but the boundary. That particular cause has been removed,
but the lesson outlives it: **more recent data is not automatically a better
model.** And the model store resolves `latest` by version timestamp, so *saving is
promotion*. A blind retrain therefore promotes whatever it produced.

## Decision

**`jobs/retrain.py`, on a 21-day cycle, with a promotion gate.**

**Twenty-one days, not thirty.** `should_correct` refuses past 35. A 21-day cycle
leaves two weeks of slack, so a retrain that slips a week does not silently switch
the bias correction off. A test asserts the two constants stay on the right side
of each other, because they are only meaningful in relation to one another.

**The gate is a holdout comparison, made before the save.** A candidate is fitted
to everything before a 60-day holdout and scored against the incumbent on hours
neither was trained on. Sixty days because this project has twice mistaken one
window for a distribution; two months is the least on which a model comparison is
worth acting.

**The gate blocks a regression, not a failure to improve.** The candidate is
promoted unless it is worse by more than 5%. That asymmetry is deliberate and it
is a judgement: the alternative to refreshing is an ageing model whose correction
switches itself off, which degrades on a schedule of its own. This project would
rather ship a model 2% worse on one holdout than stop refreshing.

**When the incumbent cannot be scored honestly, say so and promote.** If its
training ended inside the holdout, scoring it there measures its fit rather than
its forecast — flattering it, and blocking a refresh it might deserve to lose. The
comparison reports `unevaluated` with the reason instead of inventing a verdict.

**A retrain requires the panel to have advanced.** Found by running the job rather
than by reading it: with the data not yet backfilled, a forced retrain produced a
model trained to exactly the same date as the one it replaced — and reset its age.
`should_correct` reads that age, so a version that had learned nothing new would
have switched the correction back on for another 35 days on false pretences. This
now exits `EXIT_NO_NEW_DATA`, as a failure rather than a quiet success, because
the reason it cannot proceed is that the ingestion job has not run.

## Alternatives considered

**Refit on a schedule with no gate.** Simplest, and directly contradicted by the
67% measurement above. The failure mode is a worse model promoted silently, which
is the shape of failure this project spends most of its effort on.

**Requiring the candidate to beat the incumbent.** Symmetrical and wrong. On a
60-day holdout the two models are usually within noise of each other, so a
strict rule would block most refreshes and the correction would spend most of its
life switched off. The cost of not refreshing is not zero, so the gate should not
be priced as though it were.

**Retraining inside the daily job.** Convenient, and it merges two failure modes:
a bad retrain would then also stop the day's forecast. Different cadences,
different failures, different alerts.

**Automatic rollback if a promoted model turns out badly.** Attractive, and
premature: the monitoring layer already reports `DEGRADED` against the published
plan, and there is no live evidence yet that a rollback rule would fire on
anything real. It stays a manual decision until there is something to tune it on.

## Consequences

* The bias correction stays on, which is the whole point. Without this job it
  switches off five weeks after each manual training run.
* **Version count grows by roughly seventeen a year.** Each is a pickle plus a
  card, small enough that pruning stays a deliberate act. The store is
  append-only by design (ADR 0006), and the monitoring layer needs old cards to
  classify old forecasts as in-sample or not, so pruning would cost more than it
  saves.
* **The gate can be wrong in both directions and only one is loud.** A blocked
  refresh exits non-zero and says why. A promoted regression smaller than 5%
  passes silently and shows up later in the drift report. That is the intended
  trade, stated so it is not discovered.
* The comparison costs one extra model fit per run — about 90 seconds on this
  data, once every three weeks.
* `scripts/daily.ps1` runs it, in order, with the ingest and the daily forecast.
  Registered as a Windows scheduled task at 11:30 — after the 11:00 origin, an
  hour before the 12:30 deadline. See `scripts/README.md`.

## What running it actually found

Three defects, none of which reading the code had shown. All three shared a
shape: **the failure was loud somewhere far from its cause.**

**The data could never advance.** `backfill` skipped any month whose file
existed, so the first run of a month wrote a few days and every run afterwards
decided there was nothing to do. Under a scheduler this means the panel freezes
silently until the daily job starts refusing days for missing lags — a symptom
several steps removed from the cause. Fixed by always re-fetching the trailing
two months; `write_raw` merges rather than replaces (ADR 0003), so re-fetching is
safe and costs a handful of requests a day.

**Tomorrow's weather was 21 hours of 24.** The live fetch asked Open-Meteo for a
single UTC date, but a delivery day is *local*: Türkiye is UTC+3, so the local
day begins at 21:00 UTC on the day before. The first three local hours were never
requested. Now the request spans both UTC days and the caller reindexes. The same
UTC-versus-local subtlety as the month partitioning, in a different disguise.

**The ingest stopped a day short of what the forecast needs.** `--end` defaulted
to yesterday, to avoid writing a partial final day that would never be refilled —
sound reasoning while a stored month was skipped forever after, and obsolete the
moment trailing months were re-fetched. And it was actively wrong: the forecast
origin for tomorrow is 11:00 **today**, so a panel ending yesterday leaves every
origin-relative feature null. Fixed to today.

There was a fourth, in the shell rather than in Python: a PowerShell function
returns everything left on its pipeline, and `Tee-Object` writes to the pipeline
as well as to the file. `Invoke-Step` was returning kilobytes of job output with
the exit code on the end, so the script exited on a string and **reported success
while the forecast had failed.** That one is the reason the run is exercised by
hand before being scheduled.
