# 0014 — Serving two targets without forking the pipeline

**Status:** Accepted
**Date:** 2026-09-07

## Context

The price model was measured (ADR 0012, ADR 0013) but nothing operated it. Every module
below the modelling layer assumed a single target: the forecast store had a column called
`forecast_mwh`, the monitoring layer compared against the operator's load plan, the daily
job loaded one model, and the service exposed one endpoint.

Two ways of fixing that are both wrong. Copying the pipeline gives two implementations of
"forecast a delivery day" which drift while both keep returning plausible numbers — the
failure `forecasts/day.py` was written to prevent. Threading `if target == "price"`
through every function scatters the differences so widely that no reader can see them all
at once.

The differences are **data**. There are five of them and they are worth stating together,
because that list is the whole argument for a registry:

| | load | price |
|---|---|---|
| panel column | `consumption_mwh` | `price_try_mwh` |
| unit | MWh | TRY/MWh |
| published competitor | operator load plan | **none** (ADR 0012) |
| supply weather | off | on (ADR 0013) |
| MAPE floor | 1.0 | 100.0 |

## Decision

**One pipeline, one registry, and the target derived from the model wherever possible.**

### 1. `powerforecast.targets` holds the differences

A `Target` is a frozen dataclass and `TARGETS` is the registry. Adding a third target is
a row there, not an edit spread across five modules.

### 2. The forecast store partitions by target directory

`data/processed/forecasts/<target>/YYYY-MM.parquet`, mirroring `data/raw/<series>/`.

The alternative was one directory with a `target` column, which the existing
`(timestamp, model_version)` key would already have separated. Rejected because that
separation is *incidental*: `model_version` is a bare timestamp with no target in it, so
nothing structural stops two targets sharing an hour. A reader that spans the store and
forgets to filter would average megawatt-hours with lira into a number that means nothing
and raises nothing. Directories make that impossible rather than merely detectable.

Column names became unit-neutral — `forecast_value`, `bias_offset` — with the unit moved
into a column of its own. `forecast_mwh` was honest while load was the only target and
would have become a lie in half the rows. Every row still carries `unit`, so a frame
lifted out of its directory can still say what it holds. Old files are read through a
rename and a fill; the migration was 24 rows.

### 3. The target is derived from the model card, not passed alongside it

The card records the feature spec and the spec names the target column, so
`targets.for_column` turns a loaded model into its unit and its store partition.
`forecast_day` takes no target argument, and the daily job names no target when it writes.
**A caller that cannot supply the target cannot supply the wrong one.**

### 4. Drift falls back to a seasonal naive where no plan is published

ADR 0007 built drift detection on *skill against the published plan* rather than on error
alone, because error alone cannot tell a hard week from a worse model. There is no
published price forecast, so the same detector applied to price would collapse to "error
went up" — the exact failure that ADR was written to avoid.

The control falls back to `seasonal_naive` at the target's own lag. It is available at bid
time, needs no data the project does not hold, and it gets harder exactly when the market
does, which is the property that makes a control a control. It is a **weaker** competitor
than a professional forecast, so `daily_summary` now reports a `control` column naming
what skill was measured against: 0.35 against an operator forecast and 0.35 against a
naive are not the same claim.

### 5. Retraining reads its feature set off the incumbent's card

Found while wiring this up, and it had no exception attached to it. `retrain` called
`train()` with no spec, so `FeatureSpec()` applied — the load target. A scheduled retrain
of the price model would have fitted a **load** model, compared it against the price
incumbent on a holdout, and saved it under the price name. The store resolves `latest` by
timestamp, so saving is promoting: the next restart would have served demand forecasts to
callers asking for price. Three wrong things, none of which raises.

The candidate is now fitted on `incumbent.card.spec()`. That is the correct behaviour
independently of the bug — a retrain should reproduce the incumbent's feature set, not
silently switch it.

## Alternatives considered

**A separate `price` package.** Cleanest to write and worst to own. The whole reason
`forecasts/day.py` exists is that two implementations of "forecast a delivery day" drift
invisibly, and a second package would recreate exactly that.

**A `target` column in one store directory.** Cheaper migration, and the existing key
would have kept the rows apart. Rejected above: incidental separation, silent failure.

**Keep `forecast_mwh` and add `forecast_try_mwh`.** Avoids touching existing readers. Two
schemas rather than one, and every reader would have to branch on which column is present.

## Consequences

Adding a third target is a row in `targets.py`, a model, and a line in the orchestrator's
`$targets` list.

The scheduled run now does both targets and exits zero only if **every** target's forecast
reached the store. A day where price was bid and load was not is not a successful day.

**The service could not forecast tomorrow when this was written.** It reads the panel
from disk, and tomorrow's weather is not in it; the daily job fetches the live day-ahead
forecast and the service did not. Both targets failed identically on tomorrow and both
answered for a past day. Closing it meant weighing a network call in the request path
against reading what the job already stored — resolved in
[ADR 0015](0015-price-bias-and-serving-from-the-store.md) in favour of the store.

`fetch_live` for supply weather exists because running the job revealed it was missing.
Everything passed, the model was trained and saved, and the first real invocation refused
the day with `missing_features: ['solar_index', 'wind_index']`. Reading the code did not
show it. That is the third time on this project that running found what reading did not.
