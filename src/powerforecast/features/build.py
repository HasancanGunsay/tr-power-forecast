"""Assemble the design matrix.

One place decides what a model is allowed to see. Scattering that across
training scripts is how two experiments end up incomparable, and how a leaky
feature survives review — it is much easier to check one function than to check
every notebook that ever built features.

Every column here comes from `calendar_features` (safe by construction, since a
calendar is known years ahead) or from `features.lags` (safe by enforcement,
since anything unobservable is `NaN`). No column is assembled by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from powerforecast.features.availability import LAST_OBSERVED_HOUR, LOCAL_TZ
from powerforecast.features.calendar import calendar_features
from powerforecast.features.holidays import holiday_features
from powerforecast.features.lags import origin_lag, origin_rolling, target_lag

# Target-relative lags, all at least 36 hours so they are observable for every
# delivery hour. 48 is yesterday-like, 168 and 336 are the same hour one and two
# weeks back — and the baseline comparison already showed the weekly rhythm
# carries more signal at this horizon than recency does.
DEFAULT_TARGET_LAGS = (48, 72, 168, 336)

# Windows ending at the forecast origin. These describe the level and volatility
# the forecaster could see when bidding, and are constant across a delivery day.
DEFAULT_ORIGIN_WINDOWS = (24, 168)


@dataclass(frozen=True)
class FeatureSpec:
    """What goes into the design matrix.

    Kept as data so an experiment can be described by a value rather than by a
    diff, and so two runs can be compared by comparing their specs.
    """

    target: str = "consumption_mwh"
    target_lags: tuple[int, ...] = DEFAULT_TARGET_LAGS
    origin_windows: tuple[int, ...] = DEFAULT_ORIGIN_WINDOWS
    origin_offsets: tuple[int, ...] = (0, 24)
    include_trend: bool = False
    include_weather: bool = True
    include_supply_weather: bool = False
    include_holidays: bool = True
    include_load_plan: bool = False
    last_observed_hour: int = LAST_OBSERVED_HOUR
    tz: str = LOCAL_TZ
    extra: tuple[str, ...] = field(default_factory=tuple)


def build_design_matrix(
    panel: pd.DataFrame,
    spec: FeatureSpec | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build features and target from a panel of raw series.

    The published load plan is available as a feature but **off by default**.
    It is a strong predictor — it is a professional forecast of exactly the
    quantity being modelled — and using it turns the problem from "forecast
    demand" into "correct someone else's forecast", which is a legitimate but
    different task. It also depends on the plan being published before the bid
    deadline, which is still unverified (ADR 0004). Turning it on without saying
    so would make a headline number that beats the plan meaningless.

    Returns:
        `(features, target)` on the same index, with no rows dropped. Removing
        incomplete rows is left to the caller, because which rows are usable
        depends on the model and on the evaluation window.
    """
    spec = spec or FeatureSpec()

    if spec.target not in panel.columns:
        raise KeyError(f"target {spec.target!r} is not in the panel: {list(panel.columns)}")

    target = panel[spec.target]
    index = pd.DatetimeIndex(panel.index)

    columns: list[pd.Series] = []

    for hours in spec.target_lags:
        columns.append(
            target_lag(target, hours, last_observed_hour=spec.last_observed_hour, tz=spec.tz)
        )

    for offset in spec.origin_offsets:
        columns.append(
            origin_lag(
                target,
                hours_before_origin=offset,
                last_observed_hour=spec.last_observed_hour,
                tz=spec.tz,
            )
        )

    for window in spec.origin_windows:
        for statistic in ("mean", "std"):
            columns.append(
                origin_rolling(
                    target,
                    window,
                    statistic=statistic,
                    last_observed_hour=spec.last_observed_hour,
                    tz=spec.tz,
                )
            )

    if spec.include_weather:
        # Weather needs no lag and no availability mask, which is worth being
        # explicit about. These columns hold the temperature that was being
        # *forecast* for the delivery hour one day earlier, so they were already
        # on the bidder's desk. Using realised temperature here would be the
        # single worst leak available in this project — it would look excellent
        # in backtest and collapse in production.
        for column in ("temperature_c", "hdd", "cdd"):
            if column in panel.columns:
                columns.append(panel[column])

    if spec.include_supply_weather:
        # Irradiance and wind at the generating regions. Safe for the same
        # reason temperature is: these are the values that were being *forecast*
        # for the delivery hour on D-1, so they were on the bidder's desk.
        #
        # Off by default. They are a supply signal and the load target is a
        # demand quantity, so switching them on for load would be adding columns
        # with no mechanism behind them; the price target is where they belong.
        #
        # This is emphatically not the same as using KGÜP or EAK, which are
        # submitted at 14:00-15:30 on D-1 — after the 12:30 deadline, and
        # downstream of the market clearing. See `data.supply_weather`.
        for column in ("solar_index", "wind_index"):
            if column in panel.columns:
                columns.append(panel[column])

    if spec.include_load_plan:
        if "load_plan_mwh" not in panel.columns:
            raise KeyError("include_load_plan is set but the panel has no 'load_plan_mwh' column")
        columns.append(panel["load_plan_mwh"].rename("load_plan_mwh"))

    for name in spec.extra:
        columns.append(panel[name])

    features = calendar_features(index, tz=spec.tz, include_trend=spec.include_trend).join(
        pd.concat(columns, axis=1)
    )

    if spec.include_holidays:
        # Also safe by construction: a calendar for next year is known this year.
        # Error diagnosis showed these are the model's most expensive blind spot —
        # every one of its worst delivery days was a holiday, with a bias of
        # +7,700 MWh from forecasting an ordinary day while demand collapsed.
        features = features.join(holiday_features(index, tz=spec.tz))

    return features, target


def usable_rows(features: pd.DataFrame, target: pd.Series) -> pd.Index:
    """Rows where every feature and the target are present.

    Reported rather than applied silently, so a caller can see how much history
    the longest lag costs: with a 336-hour lag the first two weeks of any series
    are unusable, and a model trained without noticing that is training on less
    data than its author believes.
    """
    complete = features.notna().all(axis=1) & target.notna()
    return features.index[complete]
