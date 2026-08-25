# 0009 — Removing the linear trend, and what it had been hiding

**Status:** Accepted
**Date:** 2026-08-20

## Context

Verification of real forecasts (ADR 0007) turned up something that was not the
thing being looked for. Chasing a discrepancy between two out-of-sample windows,
one fixed July window was scored using models identical in every way except their
training cut-off:

| training cut-off | rows | MAE | bias |
|---|---|---|---|
| 31 January | 35,290 | **1,057** | +287 |
| 31 March | 36,706 | 1,136 | +108 |
| 31 May | 38,170 | **1,764** | +1,551 |
| 30 June | 38,890 | 1,134 | −169 |

More data made the model dramatically worse, non-monotonically, with a 67% swing.
That is not a tuning detail; it means any single out-of-sample number is fragile
in a way nothing in the project accounted for.

**The cause is `years_elapsed`.** It is a linear trend — years since the start of
the series — added so a model could express year-on-year demand growth. It is
also monotonically increasing, which has a consequence that is obvious in
hindsight and was not obvious at all:

```
years_elapsed in training : 1.000 .. 5.081
years_elapsed in test     : 5.082 .. 5.591
```

**Every prediction row lies above every training row.** A tree cannot
extrapolate; it can only put all of them on one side of its highest split. The
prediction therefore inherits whatever the *last stretch of training data*
happened to look like.

Usually harmless. Once not: **Kurban Bayramı 2026 fell on 27–30 May**, so a
cut-off at 31 May ends with five consecutive holiday-collapsed days as its final
observations. The top bin was a holiday, and July inherited it — under-forecasting
by 1,551 MWh all month.

Cutting eleven days earlier, before the holiday, gives 1,087. Removing the
feature gives 1,076 at *every* cut-off:

| cut-off | MAE with trend | MAE without |
|---|---|---|
| 31 Jan | 1,057 | 1,011 |
| 31 Mar | 1,136 | 1,095 |
| 20 May *(before Kurban)* | 1,087 | 1,044 |
| 26 May *(the eve)* | 1,302 | 1,034 |
| 31 May *(after)* | **1,764** | 1,076 |
| 5 Jun | 1,367 | 1,063 |
| 30 Jun | 1,134 | 1,069 |

The 67% swing collapses to 8%.

## Decision

**`include_trend` on `FeatureSpec`, default `False`.** The feature is available
for models that can extrapolate — it does help ridge, MAE 1,231 to 1,214 — and is
off for the default spec, which is what LightGBM uses.

A flag rather than a deletion, because the failure is not in the feature. It is
in the combination of a monotonic feature with a model class that cannot
extrapolate.

## Alternatives considered

**Keeping it, and scheduling retrains away from holidays.** Rejected. It makes
correctness depend on a calendar constraint that nothing enforces, and the
religious holidays move about eleven days earlier each year, so they will keep
landing near month ends. A property that must be remembered is a property that
will eventually be forgotten.

**Replacing it with a bounded trend** — years since the start, capped, or
differenced. Rejected as unnecessary: the origin-rolling means
(`consumption_mwh_mean_24h_at_origin`, `..._168h_at_origin`) already carry the
current level, in a range the trees have actually seen. That redundancy is why
removing the trend costs LightGBM nothing.

**Leaving the default as it was and documenting the hazard.** Rejected. The
measurement below shows the default was not merely risky, it was worse.

**Predicting relative to an observable level instead.** Removing the trend leaves
the underlying problem — demand grows, the tree cannot extrapolate, and an
expanding training window keeps older and lower years in view. The principled fix
is to stop asking the tree for an absolute number: fit on
`y - consumption_mwh_mean_168h_at_origin` and add the level back, so the quantity
being learned is "how far from the recent week", which does not grow.

Built and backtested on the same 31 folds. **Worse: MAE 921.6 against 890.1**,
RMSE 1,261.0 against 1,221.1. Per weekday it helps Tuesday (973 to 891) and hurts
Friday through Sunday badly (Saturday 756 to 881, Sunday 815 to 940).

The reason is visible in that split. The 168-hour mean spans weekdays and
weekends together, so subtracting it discards exactly the absolute weekend level
the model had been learning well. A single level is the wrong normaliser for a
series with two regimes inside every window. Not shipped.

## Consequences

* **The headline improves, from a diagnosis rather than from trying models.**
  Same 21,873 out-of-sample hours, same 31 folds, same baselines:

  | | before | after |
  |---|---|---|
  | best model MAE | 914 | **831** |
  | best model RMSE | 1,251 | **1,148** |
  | vs published plan (MAE) | −27% | **−34%** |
  | vs bias-corrected plan (MAE) | −25% | **−31%** |

  The plan and naive baselines are unchanged to the decimal, which is what
  confirms the comparison is like for like.

* **Corrected by [ADR 0010](0010-bias-correction-and-retraining.md).** The
  conclusion below — that bias correction is worthless here — was drawn from a
  28-day window applied to a stale model. With a 119-day window on a regularly
  retrained model the same code is worth **−6.3% of MAE**. The measurement
  below stands for the configuration it was taken in; the generalisation does
  not.

* **[ADR 0008](0008-deployable-bias-correction.md) is reversed by this.** The
  deployable rolling bias correction was measured as worth +0.7% MAE *with* the
  trend feature. Re-measured without it, every variant is now **worse than no
  correction at all** (raw 915.2; corrected 937.5–946.7).

  The reason is measurable rather than arguable. Over 181 delivery days, the
  correlation between the offset the estimator proposed and the day's realised
  median error is **−0.024**, and the correction pushed the forecast the wrong
  way on **43%** of days. The proposed offsets have a standard deviation of 263
  MWh against a realised daily median error with a standard deviation of 932.

  So the rolling correction was mostly repairing the damage the trend feature was
  doing. Remove the cause and there is nothing left to correct — the residual
  bias carries no information from one month to the next. `forecasts/bias.py`
  stays in the repository as a tested component with a recorded measurement, and
  is **not deployed**.

* The whole-period correction in the backtest still improves our model (890 to
  831), which is expected: it is estimated over the period it is scored on. It
  remains a hard competitor and is now *measurably* non-deployable rather than
  argued to be.

* Old model cards are incompatible with the new default spec, and fail loudly:
  the card lists `years_elapsed`, the rebuilt design matrix does not, and
  `assert_compatible` raises `missing ['years_elapsed']`. That is the guard from
  ADR 0006 doing exactly its job. The deployed model was retrained.

* **The "weekend concentration" was wrong, and this is the correction.** It was
  read off a single walk-forward with one fixed training cut-off. Measured on the
  backtest instead — 21,873 hours, refit every fold — the weekend is where the
  model is *strongest*:

  | day | MAE | bias | skill vs plan |
  |---|---|---|---|
  | Monday | **1,049** | −488 | 0.22 |
  | Tuesday | 977 | −490 | 0.28 |
  | Wednesday | 932 | −406 | 0.19 |
  | Thursday | 878 | −459 | 0.25 |
  | Friday | 825 | −354 | 0.31 |
  | **Saturday** | **757** | −215 | **0.35** |
  | Sunday | 815 | −218 | **0.43** |

  The real concentration is **weekday daytime**: 11:00–17:00 carries MAE
  1,120–1,310 with the forecast running low by up to 811 MWh, against 520–600
  overnight. Monday is the worst day, Saturday the best.

  Twice now a single-window reading has been generalised into a claim the
  backtest does not support. The rule that follows: **a characterisation of where
  a model fails comes from the backtest, not from one walk-forward.** One window
  is a measurement; the backtest is the distribution.

* The model **runs low almost everywhere** (bias −215 to −490 by weekday), which
  is consistent with growth the tree cannot extrapolate. The obvious fix for that
  was measured and rejected — see the level-relative entry above. Where the
  remaining error actually lives is now an open question rather than an assumed
  one.
