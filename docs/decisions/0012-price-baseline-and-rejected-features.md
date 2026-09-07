# 0012 — Price modelling: the baseline, the window, and two rejected features

**Status:** Accepted
**Date:** 2026-08-25

## Context

The same infrastructure that forecasts load can forecast day-ahead price: the series is
already in the panel, and `FeatureSpec.target` already selects it. That makes it cheap to
produce a number and correspondingly easy to produce a meaningless one.

Three questions had to be settled before any of it was worth building, and the project's
own history says why. ADR 0009 removed a feature that was *plausible* and wrong, and the
reason it survived as long as it did was that nobody had measured it in more than one
configuration. So each of the three was measured.

Backtests below use rolling-origin folds, 30-day test windows, LightGBM with weather,
scored on hours where every variant produced a prediction.

### 1. There is no operator forecast to beat

The platform publishes a **load** plan. It publishes no price plan. The load project's
headline — 34% better than the forecast the market actually runs on — has no counterpart
here, and quoting a naive-relative number in the same tone would misrepresent it.

Measured over 31,680 hours (2023-01 → 2026-08):

| forecast | MAE | RMSE | sMAPE |
|---|---|---|---|
| seasonal naive 48h | 483 | 753 | 31.3% |
| **seasonal naive 168h** | **428** | **671** | **28.5%** |
| seasonal naive 336h | 461 | 709 | 30.0% |
| lightgbm | **352** | **501** | **24.4%** |

The weekly lag is the strongest naive, matching the load side: at a ~24-hour horizon the
weekly rhythm carries more than recency does. The model beats it by **17.7% on MAE** —
roughly half the margin available on load, which is the honest characterisation of price
as the harder target.

### 2. The price has an administrative ceiling, and it moves

The realised maximum is a step function, and the steps are hard: a large share of hours
settle at *exactly* the ceiling.

| period | ceiling | hours at exactly that value |
|---|---|---|
| Jul 2023 – Jun 2024 | 2,700 | — |
| Jul 2024 – Mar 2025 | 3,000 | 8.19% of 2024 |
| Apr 2025 – Mar 2026 | 3,400 | **11.68% of 2025** |
| Apr 2026 – | 4,500 | 0.76% of 2026 so far |

In 2025 roughly one hour in nine settled at 3,400.00. This has the shape of the failure
ADR 0009 removed: a tree cannot predict above the highest value in its training data, so
a model trained under a 3,400 ceiling should systematically under-forecast once the
ceiling moves to 4,500.

That reasoning is sound and the measurement does not support it:

| | MAE | RMSE | bias |
|---|---|---|---|
| lightgbm, raw target | 352.3 | 500.9 | +24 |
| lightgbm + ceiling as a feature | 350.7 | 499.9 | +20 |
| lightgbm, target normalised by ceiling | **456.9** | 605.6 | **−145** |

Normalising is far worse than doing nothing — worse than the naive baseline. Supplying
the ceiling as a feature changes MAE by 0.5%, and in the two years where the ceilings are
independently confirmed by mass-at-value the difference vanishes entirely: 2024 gives 282
against 284, 2025 gives 321 against 321.

The likely reason is that the ceiling is already visible. `origin_lag` and
`origin_rolling` carry the recent level, and when the market is pressed against the
ceiling those lags sit against it too. Unlike `years_elapsed`, the regime is transmitted
by **observable** features rather than by an unobservable trend, so the model needs no
separate channel for it.

Nor does the transition hurt. At the April 2025 change, MAE rises from 320 to 429 — but
the seasonal naive, which knows nothing of ceilings, rises from 437 to 580 over the same
months. Skill against the naive is unchanged (0.73 → 0.74). It was a volatile month, not
a ceiling artefact.

### 3. A sliding window does not help

Evaluated on a fixed window (2024-07 onward, 18,000 hours) so that every fold sits in the
confirmed-ceiling era:

| training window | MAE | RMSE | sMAPE |
|---|---|---|---|
| **expanding (2022+)** | **370.1** | 543.2 | 28.0% |
| sliding 3y | 371.9 | 542.9 | 28.0% |
| sliding 2y | 373.7 | 549.2 | 28.2% |
| sliding 18m | 370.5 | 548.2 | 28.1% |
| sliding 1y | 378.0 | 556.6 | 28.5% |

The spread is 2% and it is not monotonic — 18 months matches expanding while 2 years is
worse than both. That is noise, not a trend.

One month argues actively against a short window. In August 2026, after the spring
dislocation reverted, expanding scored 347 and sliding-1y scored 467 — **35% worse**,
because the short window contained only the dislocation. A sliding window is not merely
unhelpful here; around a regime that reverts it is a liability.

## Decision

1. **The baseline is `seasonal_naive` at 168 hours**, and every price claim is stated
   relative to it. No price result will be phrased so that it can be mistaken for a
   comparison against a published operator forecast, because none exists.
2. **Training data starts 2022-01-01** (`REGIME_START`). 2021's median price was 400
   against 2,345 the following year.
3. **The training window expands.** Sliding windows were measured and rejected.
4. **The price ceiling is not a feature.** Measured in two encodings, both rejected.
5. **`PRICE_MAPE_FLOOR = 100.0` TRY/MWh, and sMAPE is the scale-free metric quoted.**
   0.91% of hours settle at exactly zero. The load runner's floor of 1.0 rescues MAPE
   arithmetically while still letting an hour priced at 3 dominate the average; any floor
   is arbitrary, so the floored MAPE is kept for continuity and sMAPE is what gets quoted.

## Alternatives considered

**Use the load plan as a price feature.** Demand is half of the price formation story and
the plan is already ingested. Not rejected — untested, and deferred deliberately: like
`include_load_plan` on the load side it converts the task into correcting someone else's
forecast, and its publication time relative to the 12:30 deadline is still unverified
(ADR 0004).

**Model zero-priced hours separately.** The zeros are not scattered. They peak at 6.92% of
hours at 12:00 and are absent after 19:00, and in May 2026 they took 68.8% of the midday
block. That is a solar mechanism, not noise, and it may deserve its own treatment.
Deferred rather than rejected: the natural first move is a supply-side feature, not a
second model.

**Log-transform the target.** Prices are bounded below at zero and right-skewed, which
usually argues for it. Not attempted this round. Autocorrelation gets *weaker* under
`log1p` at every seasonal lag measured (168h: 0.729 → 0.689), which is weak evidence
against, but not a measurement of forecast error and not recorded as one.

**Tune LightGBM.** Untuned here as on the load side, and for the same reason: tuning
before the feature set is settled optimises the wrong thing, and it must happen in an
inner split when it happens.

## Consequences

The price model has a smaller headline than the load model — 17.7% over a naive rather
than 34% over a professional forecast — and that gap is real rather than a deficiency in
the modelling. Stating it plainly is the point.

Two of the three constraints written down before this work began turned out to be wrong
or unnecessary. "A sliding window will be tried" assumed the answer; the measurement says
it does not matter. The ceiling, which was not in the constraint list at all, is the most
striking structural fact in the series and is *still* not worth encoding. Both are now
measured, and the negative results are the durable part.

The largest known gap is supply. Price is set where supply meets demand, and the model
currently sees only demand.
