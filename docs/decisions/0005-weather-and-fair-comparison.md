# 0005 — Archived weather forecasts, and correcting our own bias

**Status:** Accepted
**Date:** 2026-08-06

## Context

Two separate problems arrived together, and both are about fairness of measurement
rather than about modelling.

**Which temperature.** Electricity demand is driven by temperature, and the published
plan almost certainly uses it. Adding temperature was the obvious next step, but there
are two very different quantities available under that name:

* *Realised* temperature, from a reanalysis product such as ERA5 — what the weather
  actually did.
* *Forecast* temperature — what was being predicted for that hour beforehand.

Only the second existed when a bid was submitted. Realised temperature is the single
largest leak available in this project: it would improve every backtest metric and
degrade the moment the model met a real forecast, because it would have learned to rely
on precision that does not exist at prediction time.

**Whose bias gets corrected.** The published plan runs several hundred MWh low, and a
one-line constant correction improves it more than Ridge does. That correction was
already being applied to the plan so that a learned model had to beat something
non-trivial. It was *not* being applied to our own model — whose bias was larger — and
RMSE penalises exactly that.

## Decision

**Use Open-Meteo's archived day-ahead forecast** (`temperature_2m_previous_day1`): for
each hour of delivery day D, the temperature that was being forecast for it one day
earlier. It is noisier than the truth, and that noise is part of the problem rather than
a defect to remove.

Aggregate eight metropolitan areas weighted by population, and derive heating and
cooling degree-days at an 18 °C base, because demand responds to temperature in a V that
a single linear term cannot express.

**Apply the bias correction symmetrically** — to our best model as well as to the plan.

## Alternatives considered

**ERA5 reanalysis.** Longer history (back to 1940), higher quality, free. Rejected
outright: it is the leak described above. The longer history is not worth a metric that
cannot be reproduced in production.

**Realised temperature, with a caveat in the README.** Rejected for the same reason
`lag(24)` was rejected in ADR 0004 — a caveat in prose does not stop the number being
quoted, and the number would be wrong.

**`temperature_2m_previous_day2`.** More conservative: the forecast issued two days
before delivery, unambiguously available at the 12:30 deadline. Rejected as the default
because a bidder genuinely did have the D-1 run, and using D-2 would understate what a
real forecaster could achieve. Worth revisiting if evidence emerges that the D-1 run
lands after the deadline.

**Per-city features instead of an aggregate.** Would let a model learn regional
structure. Deferred rather than rejected: it multiplies the feature count by eight on a
dataset where the aggregate has not yet been shown to be the limiting factor.

**Correcting only the plan's bias.** The status quo, and the comfortable choice, since
it made our model look worse. Rejected once noticed: a comparison rigged against
yourself is still rigged, and the asymmetry was not deliberate.

## Consequences

* The archive begins in 2022 while the electricity series begins in 2021, so weather
  features are absent for the first year. The backtest drops incomplete rows inside each
  fold, so training shrinks while the evaluation window is unchanged — the comparison
  against weather-free models stays like for like.
* Roughly 500 hours have no temperature at all. Those hours leave the evaluation set for
  *every* model, which is why the headline table covers 21,873 hours rather than 22,320.
* Weather roughly doubles the model's genuine advantage over the corrected plan, from 29
  to 58 MWh — measured, not assumed.
* The symmetric correction moved MAE from 1,155 to 1,122 and RMSE from 1,766 to 1,748.
  The model now wins clearly on MAE and is level on RMSE, meaning its advantage is in
  ordinary hours rather than extreme ones. That limitation is stated in the README.
* Both corrections estimate their offset over the whole evaluation period, so both are
  optimistic in absolute terms. The comparison between them is fair; a deployed system
  would have to estimate the offset from history, and doing so is outstanding work.
* The 18 °C base and the population weights are conventions, not measurements. Both
  should be re-checked against this series rather than inherited indefinitely.
