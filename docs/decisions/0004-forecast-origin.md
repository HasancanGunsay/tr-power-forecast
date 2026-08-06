# 0004 — Forecast origin and the information set

**Status:** Accepted
**Date:** 2026-08-06

## Context

"Forecast 24 hours ahead" is a convenient phrase and a misleading one.

In the Turkish day-ahead market a participant submits one bid covering all 24 hours of
delivery day D, and submission closes at 12:30 on D-1. Every hour of D is therefore
forecast from a single moment, and the lead time varies across the day: about 12 hours
for the first delivery hour, about 35 for the last. There is no point at which a
forecaster stands 24 hours before each individual target.

This matters because it defines what may be used. Hourly series are stamped at the
start of the hour, so the value stamped 12:00 covers 12:00–13:00 and is still
accumulating when bids close at 12:30. The newest complete observation available to a
bidder is the one stamped **11:00 on D-1**.

The consequence is that the standard `lag(24)` feature is not usable for half the
delivery day. For a target at 20:00 on D, it reads 20:00 on D-1 — nine hours after the
bid was submitted. A backtest built on it produces a number that cannot be reproduced
in production, and the discrepancy typically surfaces only after deployment.

## Decision

**Model the origin explicitly, and enforce availability in code.**

`forecast_origins` returns, for every target hour, the timestamp of the last fully
observed hour at bid submission: 11:00 on the previous local day. All 24 hours of a
delivery day share one origin.

`seasonal_naive` compares each source timestamp against that origin and yields `NaN`
where the source was not yet observable. Unavailability is represented as absence, not
as a value.

The origin is expressed as **the last observable data timestamp** rather than as the
12:30 deadline, so it can be compared directly against a data index. Encoding the
deadline instead would leave the hour-stamping correction to be reapplied at every
call site — an off-by-one waiting to recur.

`LAST_OBSERVED_HOUR` is a parameter with a default, not a constant. It is a market
rule that differs between markets and has been changed before.

## Alternatives considered

**Use `lag(24)` and note the caveat in the README.** Simplest, and standard practice in
a great deal of published work. Rejected: a caveat in prose does not stop the feature
being used, and the resulting metric is not merely optimistic but unreproducible. The
project's whole claim is honest evaluation.

**Shift the whole series by the lead time and treat the horizon as flat.** Makes the
arithmetic uniform. Rejected: it misrepresents the problem. The first and last delivery
hours genuinely differ in difficulty, and flattening that hides where a model actually
fails.

**Drop the 24-hour naive entirely.** It covers only half the horizon, so it is arguably
not a baseline at all. Rejected: it is the first thing any reader would reach for, and
publishing it *with* its 50% coverage is the clearest available demonstration of why
the origin matters. Removing it would leave the trap undocumented.

**Assume the data is available the instant the hour begins.** Would make 12:00 usable
and raise coverage to 54%. Rejected: it is wrong twice over — the hour is not complete,
and real-time metering is itself published with a lag.

## Consequences

* The 24-hour naive covers exactly 50% of delivery hours, and comparisons that include
  it lose the other half. The leaderboard therefore reports coverage next to accuracy.
* Every feature added later — lags, rolling statistics, weather — must pass the same
  availability check. The intention is to generalise this into the feature layer rather
  than to re-derive it per feature.
* `LAST_OBSERVED_HOUR = 11` assumes realised consumption is published promptly. If
  metering lags by more than half an hour, even 11:00 is optimistic and the constant
  should move earlier.
* The published load plan is used as a benchmark on the assumption that it is issued
  before the same deadline. It has been confirmed ex-ante and never revised, but its
  publication time within D-1 is not yet established; if it is issued after 12:30 it
  has an information advantage over everything measured against it here.
