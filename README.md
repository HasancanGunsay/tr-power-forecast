# tr-power-forecast

> Day-ahead electricity **load and price forecasting** for the Turkish (EPİAŞ) and
> European (ENTSO-E) markets — hourly, 24 steps ahead, evaluated with a rolling-origin
> backtest against a seasonal-naive baseline.

[![CI](https://github.com/HasancanGunsay/tr-power-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/HasancanGunsay/tr-power-forecast/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Status: in progress.** In backtest, the best model beats the published forecast by
> **27% on MAE** and **29% on RMSE** — and still wins by 25% and 27% against that
> forecast with its systematic bias already corrected, which is the harder comparison.
>
> **That is a backtest claim, and the deployed configuration does not match it.**
> The winning variant's bias correction is estimated over the whole evaluation period, so
> it is a competitor rather than a component. Walked forward honestly, the served model
> beats the plan by **19.8%** and a deployable correction adds **0.7%** — measured and
> reported: see [Running unattended, and checking afterwards](#running-unattended-and-checking-afterwards).

![Learned models against the published plan](reports/figures/model_comparison.png)

## Headline result

Backtested over **21,873 out-of-sample hours** (31 monthly folds, Jan 2024 – Jul 2026),
every model refitted from scratch each fold, every forecast scored on the same hours:

| Forecast | MAE (MWh) | RMSE (MWh) | MAPE | Bias |
|---|---|---|---|---|
| **LightGBM + weather + hourly bias** | **914** | **1,251** | **2.3%** | −1 |
| LightGBM + weather + constant bias | 921 | 1,258 | 2.4% | −1 |
| LightGBM + weather | 931 | 1,270 | 2.4% | −177 |
| LightGBM | 977 | 1,320 | 2.5% | −103 |
| Ridge + weather | 1,122 | 1,555 | 2.9% | −22 |
| Ridge | 1,160 | 1,613 | 3.0% | −18 |
| Published plan **+ hourly bias** | 1,213 | 1,744 | 3.1% | 0 |
| Published plan **+ constant bias** | 1,221 | 1,753 | 3.1% | 0 |
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

### How the holiday feature was found, and what it was worth

The interesting part of this result is not the number — it is that the number came from
a diagnosis rather than from trying models until one worked.

An earlier version won on MAE and was **exactly level on RMSE** (1,748 against 1,744).
Since RMSE weights large misses, that combination said the remaining gap lived in the
tail, so the tail was where the investigation went.

**The error was concentrated.** The worst 1% of hours carried 29.5% of all squared
error. RMSE was being decided by a few hundred hours, and improving the average hour
could not have moved it.

**The two forecasts failed on different days.** Scoring each on the *other's* hardest
hours produced near mirror images — the published plan was 3.6× worse on the model's
easy-but-plan-hard hours, and the model was 4.9× worse on its own. Two separate
weaknesses, not one shared one.

**The model's catastrophic days, read chronologically, were these:**

```
2024-04-08/09/10     2025-03-29/30/31     2026-03-19/20/23
2024-06-15/17        2025-06-05/06/10     2026-05-25/26/27
```

Two clusters, each sliding about **11 days earlier every year**, roughly 70 days apart —
the signature of the Hijri calendar. Ramazan and Kurban Bayramı, plus fixed-date national
holidays. Bias on those days was **+7,702 MWh**: an ordinary day forecast while industry
shut and demand collapsed.

[`features/holidays.py`](src/powerforecast/features/holidays.py) closes that gap, and
emits more than a single flag because the diagnosis showed demand falling *before* the
eve and still depressed *after* the last day — the eve, the position within a multi-day
holiday, and a signed ±3-day distance all carry signal.

| | MAE | RMSE | worst 1% share of squared error |
|---|---|---|---|
| before holiday features | 1,122 | 1,748 | 29.5% |
| after | **914** | **1,273** | **21.0%** |

RMSE fell 27%, which is what aiming at the tail is supposed to look like.

*Computed Hijri dates can differ from the official Turkish ones by a day, and a day's
error would mark the wrong days while looking correct. The converter's output is
therefore checked against the days on which demand was observed to collapse — all six
match, and that check is a test rather than a note.*

### What still fails, and why it is being left alone

Two refinements followed from the diagnosis and were made: the 2024 administrative
extension now overrides its start date as well as its length (it ran from 8 April, not
10), and statutory half-days are separated from ordinary holiday eves, since only the
religious arife and 28 October are half working days by law.

The result is worth reporting precisely because it is mixed:

| | MAE | RMSE | worst 1% share |
|---|---|---|---|
| before refinement | 914 | 1,273 | 21.0% |
| after | 914 | **1,250** | **17.5%** |

RMSE and concentration improved; MAE did not move at all. Looking at the targeted days
explains why — **one error was traded for another**. 15 April 2024 went from −6,981 to
−1,192, the single largest error in the set. But 8 April flipped from +4,187 to −3,453:
an administrative extension closes the public sector while industry partly works, so
treating it as a full holiday makes the model expect a deeper collapse than happens.

The remaining worst days are Kurban's later days, New Year, and the administrative
extension. Each could be given its own feature, and each would be fitted on **two to
five examples** — one, in the case of the extension. That is fitting noise, and this
project has spent a lot of effort not doing that elsewhere.

So the honest position is that holiday modelling has reached the limit of what five
years of data supports. The next real gain is more history or a different target, not a
sixth holiday column.

*Aside:* the disjoint failure sets mean a combination of the two forecasts would beat
either. That is a legitimate result but a different task — see the note on using the
plan as a feature above — so it is recorded rather than claimed.

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

```bash
uv run python -m powerforecast.analysis.bias_correction
```

The experiment refits four model variants across 31 folds and takes a few minutes.

## Serving

The backtest measures an approach; serving ships one model. Train a single model on all
available history and store it with a card describing what it is:

```bash
uv run python -m powerforecast.models.train
```

That writes `models/load-lightgbm/<version>/`, holding the fitted estimator and a
`card.json` recording its feature columns *in order*, the feature spec, the training
window, the seed and the library versions it was fitted under. Nothing loads a bare
pickle — see [ADR 0006](docs/decisions/0006-model-store.md).

Run the service:

```bash
uv run uvicorn powerforecast.serving.app:app --reload
```

`GET /health` reports which model version is answering, and how stale the cached data
is. `GET /forecast?date=2026-08-01` returns all 24 hours of that delivery day, together
with the forecast origin and the model version that produced them. Interactive
documentation is generated from the code at `/docs`.

Three refusals are deliberate and are the reason the service is worth more than a
notebook:

- **A partial day is an error, not a short answer.** If any hour of the requested day
  lacks a complete feature set the request fails with 422, naming the missing columns
  and how far the data actually reaches. Twenty-three values where the market needs
  twenty-four is a position nobody bid.
- **Feature mismatch cannot be overridden.** The stored card fixes the columns and their
  order; a design matrix that differs raises rather than predicting. Reordering is the
  dangerous case, because the shapes still match and nothing else objects.
- **The process refuses to start** if the model is missing or was fitted under different
  library versions. A service that starts and then errors looks healthy to everything
  watching it.

### In a container

```bash
docker build -t tr-power-forecast .
```

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/models:/app/models:ro" \
  tr-power-forecast
```

The image is 245 MB and holds the code and its dependencies — most of it the virtual
environment, in a single 670 MB layer before compression. Data and models are mounted at
run time rather than baked in, read-only: both change on schedules of their own, and an
image containing one model could only ever serve that model. Rolling back to yesterday's
model is a different mount, not a rebuild.

Two things the image needs that are easy to miss locally, both recorded in the
Dockerfile:

- `README.md` has to be copied, because hatchling validates the `readme` field when the
  project is installed. It sits with the source rather than with the manifest, so a
  documentation edit does not invalidate the dependency layer.
- `libgomp1` has to be installed. LightGBM links against OpenMP; the Windows wheel
  bundles it and the Linux wheel expects the system to provide it, which `slim` does
  not. A correct lockfile says nothing about this — **uv resolves Python packages, not
  system libraries**, and that is where the reproducibility guarantee stops.

## Running unattended, and checking afterwards

Produce and store tomorrow's forecast before the bid deadline:

```bash
uv run python -m powerforecast.jobs.daily_forecast
```

It refreshes the panel, fetches the delivery day's temperature from the live forecast
run — the same quantity training used, read from the other side of the day — predicts,
and merges 24 rows into `data/processed/forecasts/`. Re-running the same day updates
rather than duplicates; schedulers retry, and a job that is not idempotent corrupts the
record on its second run. Failures exit with distinct codes (missing model, weather
unavailable, incomplete day) so an alert can say which stage broke without anyone
opening a log.

Then check the forecasts against what happened:

```bash
uv run python -m powerforecast.monitoring.verify
```

Two things make this more than a scoreboard, both recorded in
[ADR 0007](docs/decisions/0007-forecast-store-and-drift.md):

- **Drift is judged on skill, not on error.** Error rises both when a model decays and
  when a period is simply harder, and a recent-versus-baseline comparison cannot tell
  those apart. The operator's published forecast is a free control, so the verdict is a
  difference-in-differences: `DEGRADED` (we lost ground the plan did not),
  `HARDER_PERIOD` (everyone's error rose), or `OK`.
- **In-sample hours are excluded and the exclusions are printed.** A delivery hour
  inside the producing model's training window measures the fit, not the forecast. This
  is not a hypothetical: the first pipeline run reported MAE 285 against a backtest MAE
  of 914, entirely because every day it scored lay before the model's `train_end`.

### What the out-of-sample check actually says

A model fitted to 2026-01-31 and walked forward over every delivery day to 3 August —
4,439 hours nobody chose, with no refit:

| forecast | MAE | RMSE |
|---|---|---|
| + hourly offset, **median** | **907.8** | 1,223.4 |
| + constant offset, median | 909.9 | **1,217.7** |
| raw, no correction | 914.5 | 1,233.6 |
| + hourly offset, mean | 918.1 | 1,230.4 |
| + constant offset, mean | 923.8 | 1,230.9 |
| published plan | 1,140.5 | 1,530.8 |

**The raw model beats the plan by 19.8% out of sample**, and a deployable bias
correction — rolling window, availability cutoff enforced in code — is worth a further
**0.7%**. Not 27%. The backtest's corrected variant looks stronger because its
correction is estimated over the whole period and therefore sees the future; that is
what makes it a good competitor and a bad component.

Two results worth more than the 0.7%:

- **The mean makes MAE worse and RMSE better.** Shifting by the mean error minimises
  squared error; shifting by the *median* minimises absolute error. Correcting with the
  mean optimises the metric this project does not headline.
- **Finer grouping loses.** Hour-by-weekday offsets score 979 against 912 for no
  correction at all: a 28-day window gives 28 samples per hour but four per
  hour-and-weekday cell. Measured, then not shipped — see
  [ADR 0008](docs/decisions/0008-deployable-bias-correction.md).

**Correction to an earlier claim.** A first check on a single 30-day window in July put
the deployed model *behind* the plan and was reported that way. Over the longer window
it does not hold — July is hard for this model and easy for the plan. One month was not
a verdict.

What does still hold: the error concentrates on weekends, and the correction does not
touch it (Saturday −3.9%, Sunday −7.2%, against +5.7% to +10.5% Wednesday to Friday).
That is a modelling problem, and it is where the next real gain is.

Training cut-off turns out to matter more than any of this. On one fixed July window,
MAE by cut-off: 31 Jan **1,057**, 31 Mar 1,136, 31 May **1,764**, 30 Jun 1,134 —
non-monotonic, a 67% swing, and not yet explained. Retraining cadence is a first-order
decision here, not housekeeping.

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
├── models/         # baselines, estimator factories, training, the model store
├── evaluation/     # metrics, rolling-origin backtesting, comparison
├── forecasts/      # producing one delivery day, and where produced forecasts live
├── serving/        # the HTTP forecast service
├── jobs/           # the unattended daily run
├── monitoring/     # verification against outcomes, and drift
└── analysis/       # figures and the experiment runner
```

Everything that decides *what a model may see* lives in
[`features/availability.py`](src/powerforecast/features/availability.py). It is
deliberately one module: the same rule governs the baselines, the features and the
train/test boundary, and three separate copies would be three chances to disagree.

## License

MIT — see [LICENSE](LICENSE).
