# 0015 — No bias correction on price, and a service that reads the store

**Status:** Accepted
**Date:** 2026-09-07

## Context

Two settings reached the price target by inheritance rather than by measurement, and
both turned out to be wrong. This ADR measures them.

### The bias correction was inherited, and it costs 2.5%

ADR 0010 established the correction on load: worth −6.3% of MAE on a freshly retrained
model, +2.6% *worse* on a stale one, so `should_correct` refuses it beyond 35 days. When
price was wired into the same pipeline it kept those settings — a 119-day window, the
mean, and the same age guard — because nothing said otherwise.

Measured prequentially over 435 delivery days and 10,439 hours, with the correction for
day D estimated only from errors observable at D's forecast origin:

**Model refreshed monthly** — which is what the 35-day guard enforces:

| window | statistic | MAE | bias | vs raw |
|---|---|---|---|---|
| raw | — | **405.6** | +50.1 | — |
| 60d | median | 404.0 | +32.8 | −0.4% |
| 60d | mean | 404.3 | +0.9 | −0.3% |
| **119d** | **mean** *(deployed)* | **415.8** | −4.8 | **+2.5%** |
| 119d | median | 418.2 | +44.5 | +3.1% |

**Model trained once, never refreshed:**

| window | statistic | MAE | bias | vs raw |
|---|---|---|---|---|
| raw | — | 507.7 | +146.2 | — |
| **60d** | **mean** | **433.8** | −8.4 | **−14.6%** |
| 119d | mean | 480.7 | +1.8 | −5.3% |

**The condition is inverted relative to load.** On load the correction pays on a fresh
model and hurts on a stale one. On price it does nothing on a fresh model and pays
handsomely on a stale one — and the age guard means only the useless case ever runs.

The mechanism is consistent with the target. A stale price model carries a large level
error (+146 bias) because the regime moved under it — the spring 2026 collapse and the
April 2026 ceiling change from 3,400 to 4,500. A fresh model's +50 bias on a ~400 MAE is
small against the noise, so removing it buys nothing.

The −14.6% is a trap worth naming: a corrected stale model still scores **433.8**, worse
than an uncorrected fresh one at **405.6**. The right response to a stale price model is
to retrain it, not to patch its level with an offset.

### A window under 30 days is silently a no-op

At 28 days both regimes score identically to raw, to the decimal. `MIN_SAMPLES_PER_HOUR`
is 30, so an hourly estimate needs at least 30 days of history and a 28-day window can
never qualify — it falls back to no correction and reports success. ADR 0008's original
28-day measurement was therefore doubly unlucky: the window was on the steep part of the
curve *and* below the sampling floor.

### The service could not answer the only day that matters

The service reads the panel from disk, and the panel holds observed history, so
tomorrow's weather is not in it and `forecast_day` correctly refuses. Both targets failed
identically; both answered for past days. The project's serving layer could not serve the
question the project exists to answer.

## Decision

### 1. The bias correction is a per-target setting, and it is off for price

`Target.correct_bias` — `True` for load, `False` for price. `forecast_day` asks the
target rather than defaulting to on; an explicit bool still overrides, for experiments.

Not the 60-day variant: −0.4% is noise, and carrying a correction mechanism for that is
machinery without a result.

### 2. The service falls back to the forecast store

When the panel cannot answer a delivery day, the service returns what the scheduled job
stored for it, and the response carries `source: "computed" | "stored"`.

**Fetching live weather inside the handler was rejected.** It puts twelve outbound HTTP
requests on the request path: responses go from milliseconds to tens of seconds, the
service stops answering whenever the weather provider is down, and ten callers asking
about the same day pay for the same fetch ten times.

Reading the store is better on every axis that matters here, and one that was not
obvious: it is **more auditable**. The stored number is the one that was actually
available when bids were submitted. A recomputation would use data that arrived
afterwards and would quietly be a different, better forecast — which is precisely the
distinction the whole availability layer exists to preserve.

A stored day is served only if it is complete. Returning 23 hours would be worse than
refusing, because nothing in a JSON list tells the caller that an hour is missing from
it.

The refusal names both attempts. A message mentioning only missing features sends the
reader to fix the panel when the real answer is that the daily job has not run.

## Alternatives considered

**Keep the correction on price with a 60-day window.** The best measured variant, at
−0.4%. Rejected: that is inside the noise, and a mechanism that has to be explained,
tested and reasoned about at every retrain needs to earn more than that.

**Lower `MAX_MODEL_AGE_DAYS` for price so the correction runs where it helps.** Backwards.
The −14.6% case is a model that should have been retrained; a corrected stale model is
still worse than an uncorrected fresh one. Optimising the patched-stale path would be
optimising the path we are trying not to be on.

**Live weather in the request handler.** Rejected above.

**Have the service call the daily job on demand.** Same network cost, plus it would write
to the store from a request handler — making a `GET` mutate state, and making two
concurrent requests race to write the same day.

## Consequences

Price forecasts are now raw model output. The `bias_offset` column stays in the schema
and is zero for them, so the store's shape does not change and a future decision to turn
it back on needs no migration.

Two of the settings price inherited from load have now been measured and both were wrong
— the supply-weather default was the third, and that one went the other way (ADR 0013).
The pattern is worth stating: **a shared pipeline makes inheritance the default, and
inheritance is not a measurement.** Every parameter in `targets.py` should be able to
name the run that set it.

The service now depends on the scheduled job having run for future days. That is a real
coupling and it is visible: `source` says which path answered, and a 422 says the job did
not run. Before this change the same situation produced a confusing message about
missing features.
