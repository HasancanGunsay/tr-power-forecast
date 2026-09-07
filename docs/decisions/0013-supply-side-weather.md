# 0013 — Supply-side weather, and the generation schedule that could not be used

**Status:** Accepted
**Date:** 2026-08-25

## Context

ADR 0012 closed with the largest known gap in the price model: price is set where
supply meets demand, and the model saw only demand. May 2026 made the cost concrete —
68.8% of midday hours settled at exactly zero while the evening still cleared near 673
TRY/MWh. That is a supply event, and nothing in the feature set could see it.

### The obvious supply feature is inadmissible

**KGÜP** — the finalised daily generation programme — is published, day-ahead, and
broken down by source. It is also submitted between 14:00 and 15:30 on D-1, *after* the
day-ahead market closes at 12:30, because it is a consequence of the market clearing
rather than an input to it. **EAK** (available capacity) follows the same timetable.

Using either to forecast the day-ahead price would feed the model something derived
from the answer. Both would look excellent in backtest and be unreproducible in
production, which is the exact failure mode ADR 0004 exists to prevent.

This was checked before any code was written, and it disqualified the step this
project had recorded as its own next move.

### Weather is admissible, and reaches the same mechanism

A weather forecast for delivery day D exists on D-1 and is on the bidder's desk before
the deadline. ADR 0005 already makes this argument for temperature, and Open-Meteo
serves irradiance and wind through the same archived-forecast endpoint
(`shortwave_radiation_previous_day1`, `wind_speed_100m_previous_day1`), so the same
guarantee carries over unchanged.

### The trade-off that had to be measured

Those archives begin **2024-02-16**, two years later than the temperature archive.
Turning the features on costs history — the same shape as the sliding-window
experiment in ADR 0012, where a better-conditioned model lost to a worse-conditioned
one that had more data. Assuming the outcome either way would have repeated the
mistake this project has now made three times.

Measured across 18 folds on 12,960 identical evaluation hours, with test windows held
fixed and only the training rows varying:

| model | MAE | RMSE | sMAPE |
|---|---|---|---|
| seasonal naive 168h | 521.5 | 823.6 | 38.6% |
| demand only, 2022+ *(4.5y — the prior model)* | 421.3 | 608.4 | 34.7% |
| demand only, 2024-02+ *(2.5y)* | 432.1 | 619.6 | 35.0% |
| **+ supply weather, 2024-02+** | **406.8** | **585.6** | **34.0%** |

Decomposed:

| effect | MAE | change |
|---|---|---|
| cost of losing two years of history | 421.3 → 432.1 | +2.6% |
| value of the supply signal | 432.1 → 406.8 | −5.8% |
| **net against the prior model** | **421.3 → 406.8** | **−3.4%** |

The signal is worth more than twice the history it costs.

The improvement is consistent rather than driven by the dislocation: 17 of 19 months
improve, with only April 2026 worse (772 → 804) and June flat. By hour block it helps
everywhere — 11:00–16:00 most (506 → 466, −7.9%), but 21:00–23:00 by the same margin,
where irradiance is zero. Two indices are carrying two different signals rather than
one being a shadow of the other.

## Decision

**Add solar and wind conditions at the generating regions, sourced from archived
day-ahead weather forecasts, and switch them on for the price target.**

1. `data/supply_weather.py` defines the sites and the two aggregates; the dataset is
   stored separately from `weather` because the site lists differ and a change to one
   must not rewrite the other's files.
2. **Sites are weighted by generating capacity, not population.** The existing eight
   cities are a demand proxy; population-weighted irradiance would give İstanbul's
   cloud more weight than Konya's sun. Solar sites sit on the Konya-Karaman plateau and
   in the southeast, wind on the Çanakkale-Balıkesir-İzmir corridor. The weights are
   approximate and ordinal, and are not reported as a result anywhere.
3. **`wind_index` cubes the speed, and cubes it per site before averaging.** Power in
   moving air goes as v³, so a mean of wind *speed* would compress exactly the variation
   that matters. Cubing after the average is arithmetically tidier and wrong: a turbine
   responds to the wind at its own site, so the national signal is the sum of site
   powers, not the power of the average site. Both properties are pinned by tests
   because neither would ever raise.
4. **`FeatureSpec.include_supply_weather` defaults to `False`.** These are supply
   signals and the load target is a demand quantity; the load model must not inherit
   columns with no mechanism behind them.

## Alternatives considered

**KGÜP or EAK as features.** Rejected on availability, above. Not a matter of degree —
they are downstream of the price being forecast.

**Reuse the existing eight cities and just add radiation there.** Cheapest option, and
it would have avoided a second site list. Rejected on mechanism rather than on a
measurement: population weighting is the wrong aggregate for generation, and a null
result from a knowingly wrong proxy would not have ruled the idea out. The honest
comparison would have been a third arm, and the site list was cheap enough that
building it directly was better than measuring a strawman.

**Model the turbine power curve.** Real output saturates at rated speed and cuts out in
a gale, so an uncapped cube overstates the high tail. Rejected for now: the rated and
cut-out speeds are not measured anywhere in this project, and a tree can learn a
saturating response from an uncapped input. Revisit if the residuals concentrate in
high-wind hours.

**Temperature derating for PV.** Panels lose efficiency when hot, and the effect is
real. Rejected: panel temperature is not observable, and an unvalidated correction
would be inventing precision.

## Consequences

The price model's headline improves and its training window shortens. Both are true and
both belong in any description of it.

The cost term shrinks every month the archive lengthens, so the trade can only improve
with time — but it also means **this measurement has a shelf life**. It should be rerun
when the archive is materially longer, not treated as settled.

The zero-price hours now have a plausible mechanism in the feature set rather than only
in the diagnosis. Whether they need separate treatment is still open, and is now a
better-posed question than it was.

Weather is a proxy for generation, not generation. It cannot see an outage, a
curtailment, or a maintenance window, and those are real drivers of price that remain
invisible. The admissible-information boundary makes that permanent rather than
temporary: the market's own supply data arrives after the bid deadline, so no amount of
further ingestion closes this gap.
