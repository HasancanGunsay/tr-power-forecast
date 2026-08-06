# tr-power-forecast

> Day-ahead electricity **load and price forecasting** for the Turkish (EPİAŞ) and
> European (ENTSO-E) markets — hourly, 24 steps ahead, evaluated with a rolling-origin
> backtest against a seasonal-naive baseline.

[![CI](https://github.com/HasancanGunsay/tr-power-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/HasancanGunsay/tr-power-forecast/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Status: in progress.** Pipeline, baselines, weather and the first learned models are
> done. Short version: the best model beats the published forecast by **7.5% on MAE**
> against the strongest baseline available — and is level with it on RMSE, so the gain
> is in ordinary hours, not extreme ones.

![Learned models against the published plan](reports/figures/model_comparison.png)

## Headline result

Backtested over **21,873 out-of-sample hours** (31 monthly folds, Jan 2024 – Jul 2026),
every model refitted from scratch each fold, every forecast scored on the same hours:

| Forecast | MAE (MWh) | RMSE (MWh) | MAPE | Bias |
|---|---|---|---|---|
| **LightGBM + weather + hourly bias** | **1,122** | 1,748 | **2.9%** | −1 |
| LightGBM + weather + constant bias | 1,129 | 1,752 | 2.9% | −1 |
| LightGBM + weather | 1,155 | 1,766 | 3.0% | −223 |
| LightGBM | 1,186 | 1,773 | 3.0% | −125 |
| Ridge + weather | 1,209 | 1,757 | 3.2% | +23 |
| Published plan **+ hourly bias** | 1,213 | **1,744** | 3.1% | 0 |
| Published plan **+ constant bias** | 1,221 | 1,753 | 3.1% | 0 |
| Ridge | 1,229 | 1,782 | 3.2% | +1 |
| Published plan (raw) | 1,260 | 1,786 | 3.2% | −338 |
| Seasonal naive, 168h | 2,077 | 3,340 | 5.5% | −53 |
| Seasonal naive, 48h | 3,142 | 4,409 | 8.1% | −11 |

### The comparison is against the corrected plan, not the raw one

The published plan runs **338 MWh low** on average. Adding that back is a one-line
change with no model, no training and no features, and it improves the plan more than
Ridge does. A model that beats the raw plan but not this has removed a constant anybody
could remove — so `plan + hourly bias` is the number to beat, not `plan`.

The same correction is applied to our own best model, for the same reason in reverse.
Correcting only the competitor would rig the comparison in the opposite direction to the
usual one, and our model's bias was larger than the corrected plan's.

### What the two metrics say

**On MAE the model wins clearly**: 1,122 against 1,213, a **7.5% improvement over the
strongest baseline** and 11% over the raw plan.

**On RMSE it does not**: 1,748 against 1,744 — a 0.2% difference, which is noise. Since
RMSE weights large errors more heavily, the reading is that the model is better on
typical hours and no better on the extreme ones. That is a real limitation, and it is
where the next work goes.

### Weather roughly doubled the genuine advantage

Measured against the corrected plan rather than the raw one:

| | Advantage over `plan + hourly bias` |
|---|---|
| LightGBM, no weather | 29 MWh |
| LightGBM + weather | **58 MWh** |

Temperature enters as the value that was being **forecast** for the delivery hour one
day earlier, not the temperature that was actually recorded. Realised temperature would
be the single worst leak available in this project: it would look excellent in backtest
and collapse in production, because no bidder had it. See
[`data/weather.py`](src/powerforecast/data/weather.py).

Demand responds to temperature in a V — heating below a comfort point, cooling above it
— so temperature is also split into heating and cooling degree-days, which lets even a
linear model fit each arm separately.

*(Both bias corrections estimate their offset over the whole evaluation period, so the
absolute numbers are optimistic on both sides. The comparison between them is fair; a
deployed system would have to estimate the offset from history.)*

## Baselines

![Baselines against the published plan](reports/figures/baselines.png)

> **Different period from the table above.** This section covers the **full history**
> (2021–2026, 24,408 hours), while the headline results cover only the backtest window
> (2024–2026, 21,873 hours) — models cannot be scored on data they were trained on. So
> the same forecast appears with different numbers in the two tables: the seasonal naive
> scores 1,676 here and 2,077 there, because the later period is harder, not because
> either number is wrong. Comparisons are only ever made *within* a table.

| Forecast | MAE (MWh) | RMSE (MWh) | MAPE | Bias | Coverage |
|---|---|---|---|---|---|
| **Published load plan** | **1,020** | **1,526** | **2.9%** | −270 | 100% |
| Seasonal naive, 168h | 1,676 | 2,878 | 4.9% | −45 | 99.7% |
| Seasonal naive, 24h *(leaky)* | 1,888 | 3,116 | 5.4% | −5 | **50.0%** |
| Seasonal naive, 48h | 2,744 | 4,067 | 7.9% | −8 | 99.9% |

Three things this table is worth reading for:

**The 24-hour naive covers exactly half the horizon.** Bids for delivery day D close
at 12:30 on D-1, and hourly values are stamped at the start of the hour, so the newest
complete observation is the one stamped 11:00. For any delivery hour after 11:00 the
"same hour yesterday" had not finished happening when the bid was submitted. It is
listed because it is the baseline everyone reaches for first, and because a backtest
that uses it silently reports a number it could never reproduce in production.
`seasonal_naive` returns `NaN` for those hours rather than a value.

**Last week beats yesterday.** The weekly lag wins by a wide margin, and beats even
the leaky 24-hour version. At this horizon the day of the week carries more
information than recency — which says where feature engineering should go first.

**The published plan is genuinely skilful**, roughly 40% better than the best naive
baseline. That gap is what motivated adding weather, and closing it is what the headline
table above reports.

The plan also runs about 270 MWh low over this period. A systematic bias is easier to
correct than random error, which is why the bias-corrected plan — not the raw one — is
used as the benchmark in the headline table.

### Is the published plan a fair benchmark?

It would not be, if it were a series quietly revised after delivery. Two checks say it
is not:

- **It exists before delivery.** Queried mid-morning, the plan already covers all 24
  hours of the current day, while realised consumption covers only the hours that have
  actually elapsed.
- **It never changes.** Re-fetching a full month already on disk returned 744 of 744
  hours byte-identical.

So it is a genuine ex-ante forecast, frozen once published — a legitimate bar.

> **Remaining caveat.** The exact publication time within the previous day is not yet
> established. If the plan is issued *after* the 12:30 bid deadline it sees more
> history than the baselines here are allowed to, and the comparison flatters it.
> Establishing this needs an observation of when tomorrow's plan first appears, which
> is a pending task rather than a settled fact.

## The problem

Electricity cannot be stored cheaply at grid scale, so supply must match demand
continuously. Market participants commit to positions in the day-ahead market before
knowing tomorrow's actual load, and a wrong position is settled at the imbalance price
— which is deliberately punitive. A forecast error therefore carries a direct and
asymmetric financial cost.

Two quantities are forecast here, 24 hours ahead at hourly resolution:

- **Load** — total system consumption. Driven by weather, calendar effects and economic
  activity; comparatively smooth and predictable.
- **Day-ahead price** — the market clearing price. Driven by the merit order of
  generation, renewable availability and fuel costs. Harder than load, though not for
  the reason usually assumed: a regulated ceiling truncates the distribution, so the
  Turkish series is close to symmetric (skewness ≈ 0) rather than heavy-tailed. The
  difficulty is the level shift between regimes, not the tail.

Forecasting both in one codebase is deliberate: they share features and infrastructure
but fail in different ways, which makes the comparison informative.

## Why two markets

The same pipeline runs against Türkiye (EPİAŞ Şeffaflık) and at least one European
bidding zone (ENTSO-E Transparency). The two markets differ in renewable share, price
formation and volatility, so a model that transfers between them is evidence of
generalisation rather than of fitting one market's quirks.

## Approach

Deliberately staged, cheapest first — each model earns its place only by beating what
came before it:

1. **Seasonal naive baseline** — yesterday's same hour, and last week's same hour.
   Everything is measured against this.
2. **Linear / ridge** with calendar and lag features.
3. **Gradient boosting** (LightGBM), the usual strong baseline on tabular time series.
4. **Global neural model** (N-BEATS or a compact TFT) trained across series.
5. **Probabilistic forecasts** — quantile regression for P10/P50/P90 intervals, because
   a point forecast without uncertainty is not decision-ready.

**Validation.** Rolling-origin backtest with a fixed 24-hour horizon and no shuffling.
Every feature is computed from information available at prediction time, and the
pipeline is written so that a lag shorter than the horizon cannot silently leak.

## Data

| | Türkiye | Europe |
|---|---|---|
| Source | [EPİAŞ Şeffaflık](https://seffaflik.epias.com.tr/) | [ENTSO-E Transparency](https://transparency.entsoe.eu/) |
| Auth | CAS ticket (username / password) | API security token |
| Resolution | hourly | hourly |
| Weather | to be added (temperature, HDD/CDD) | to be added |

Credentials are read from `.env` and are never committed. See `.env.example`.

## Reproduce

```bash
uv sync --all-groups
cp .env.example .env      # add EPIAS_USERNAME and EPIAS_PASSWORD
```

Verify the checks pass — no credentials or network needed, every test mocks its HTTP:

```bash
make check                # .\make.ps1 check on Windows
```

Download the data. EPİAŞ needs a free account; Open-Meteo needs nothing. The backfills
are resumable, so an interrupted run continues where it stopped:

```bash
uv run python -m powerforecast.data.backfill --start 2021-01-01
```

```bash
uv run python -m powerforecast.data.backfill_weather
```

Then reproduce the figures and the leaderboard:

```bash
uv run python -m powerforecast.analysis.figures
```

```bash
uv run python -m powerforecast.analysis.experiment
```

The experiment refits four model variants across 31 folds and takes a few minutes.

## Design decisions

Non-obvious choices are recorded in [`docs/decisions/`](docs/decisions/) — the
problem, the choice, the alternatives that lost and why, and what each choice costs.

## Project structure

```
src/powerforecast/
├── config.py       # every path and setting derives from here
├── seed.py         # reproducibility helpers
├── data/           # API clients, ingestion, schema validation, weather
├── features/       # availability rule, calendar, lags, design matrix
├── models/         # naive baselines and estimator factories
├── evaluation/     # metrics, rolling-origin backtesting, comparison
└── analysis/       # figures and the experiment runner
```

Everything that decides *what a model may see* lives in
[`features/availability.py`](src/powerforecast/features/availability.py). It is
deliberately one module: the same rule governs the baselines, the features and the
train/test boundary, and three separate copies would be three chances to disagree.

## License

MIT — see [LICENSE](LICENSE).
