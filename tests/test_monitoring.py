"""Tests for forecast verification and drift detection.

The interesting assertions are the ones about *not* raising an alarm. Any
comparison of two windows will eventually report a difference; what makes a
monitor useful is that it stays quiet when the difference has an innocent
explanation. So the central case here is `HARDER_PERIOD`: error rose, and the
verdict is still "do nothing", because the plan's error rose with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from powerforecast.evaluation.metrics import bias as metrics_bias
from powerforecast.monitoring.verify import (
    daily_summary,
    detect_drift,
    overall,
    verify,
)


def hours(start: str, days: int) -> pd.DatetimeIndex:
    begin = pd.Timestamp(start, tz="Europe/Istanbul")
    end = begin + pd.DateOffset(days=days)
    return pd.DatetimeIndex(pd.date_range(begin, end, freq="h", inclusive="left")).tz_convert("UTC")


def scenario(
    days: int = 40,
    *,
    our_error: float = 500.0,
    plan_error: float = 1_000.0,
    recent_days: int = 14,
    recent_our_error: float | None = None,
    recent_plan_error: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a forecast store and a panel with errors we choose.

    Errors are deterministic and alternate in sign so that MAE is exactly the
    magnitude asked for and bias is zero. A test that has to reason about noise
    is a test whose failures nobody can interpret.
    """
    index = hours("2026-01-01", days)
    actual = pd.Series(40_000.0, index=index)

    split = index[-recent_days * 24] if days > recent_days else index[len(index) // 2]
    is_recent = pd.Series(index >= split, index=index)
    sign = pd.Series(np.where(np.arange(len(index)) % 2 == 0, 1.0, -1.0), index=index)

    ours = pd.Series(np.where(is_recent, recent_our_error or our_error, our_error), index=index)
    theirs = pd.Series(
        np.where(is_recent, recent_plan_error or plan_error, plan_error), index=index
    )

    panel = pd.DataFrame(
        {"consumption_mwh": actual, "load_plan_mwh": actual - sign * theirs},
        index=index,
    )
    forecasts = pd.DataFrame(
        {
            "forecast_mwh": actual - sign * ours,
            "model_name": "load-lightgbm",
            "model_version": "v1",
            "forecast_origin": index[0],
            "generated_at": pd.Timestamp("2026-01-01", tz="UTC"),
        },
        index=index,
    )
    return forecasts, panel


def test_only_hours_with_an_actual_are_verified():
    """An hour still in the future is not a miss."""
    forecasts, panel = scenario(days=4)
    panel.loc[panel.index[-24:], "consumption_mwh"] = np.nan

    verified = verify(forecasts, panel)

    assert len(verified) == 72
    assert verified["actual_mwh"].notna().all()


def test_bias_has_one_sign_convention_across_the_repository():
    """Two conventions under one word produced two wrong sentences once already.

    `residual = actual - forecast` (positive = forecast came in low).
    `bias` = `metrics.bias` (positive = forecast runs high). Opposites, and the
    test pins them to each other so a future edit cannot quietly flip one.
    """
    # Forecast deliberately below actual: residual positive, bias negative.
    forecasts, panel = scenario(days=4)
    forecasts["forecast_mwh"] = panel["consumption_mwh"] - 500.0

    verified = verify(forecasts, panel)
    daily = daily_summary(verified)

    assert (verified["residual"] == 500.0).all()
    assert (daily["bias"] == -500.0).all()
    assert daily["bias"].iloc[0] == metrics_bias(verified["actual_mwh"], verified["forecast_mwh"])


def test_verification_carries_the_plan_as_a_control():
    forecasts, panel = scenario(days=4)
    verified = verify(forecasts, panel)

    assert {"plan_mwh", "plan_abs_error"} <= set(verified.columns)
    assert verified["abs_error"].mean() == 500.0
    assert verified["plan_abs_error"].mean() == 1_000.0


def test_daily_summary_reports_skill_against_the_plan():
    forecasts, panel = scenario(days=4)
    daily = daily_summary(verify(forecasts, panel))

    assert len(daily) == 4
    assert (daily["hours"] == 24).all()
    # We halve the plan's error, so skill is 0.5.
    assert daily["skill_vs_plan"].round(3).eq(0.5).all()


def test_a_stable_model_reports_ok():
    forecasts, panel = scenario(days=40)
    verdict = detect_drift(verify(forecasts, panel))

    assert verdict.status == "OK"
    assert verdict.recent_hours >= 168 and verdict.baseline_hours >= 168


def test_error_rising_for_everyone_is_not_degradation():
    """The case that separates a useful monitor from a noisy one.

    Our error doubles — but the plan's error doubles too, so skill is unchanged.
    A recent-vs-baseline comparison without a control would raise an alarm here,
    and the alarm would be wrong: the period was harder, the model is fine.
    """
    forecasts, panel = scenario(
        days=40,
        our_error=500.0,
        plan_error=1_000.0,
        recent_our_error=1_000.0,
        recent_plan_error=2_000.0,
    )

    verdict = detect_drift(verify(forecasts, panel))

    assert verdict.status == "HARDER_PERIOD"
    assert round(verdict.baseline_skill, 3) == round(verdict.recent_skill, 3) == 0.5
    assert "chasing conditions" in verdict.detail


def test_losing_ground_the_plan_did_not_lose_is_degradation():
    """Our error rises while the plan's holds — that is the model, not the weather."""
    forecasts, panel = scenario(
        days=40,
        our_error=500.0,
        plan_error=1_000.0,
        recent_our_error=950.0,
        recent_plan_error=1_000.0,
    )

    verdict = detect_drift(verify(forecasts, panel))

    assert verdict.status == "DEGRADED"
    assert verdict.baseline_skill > verdict.recent_skill
    assert "retrain" in verdict.detail


def test_too_little_data_produces_no_verdict_at_all():
    """A monitor that judges on a handful of days teaches people to ignore it."""
    forecasts, panel = scenario(days=6, recent_days=2)
    verdict = detect_drift(verify(forecasts, panel), recent_days=2)

    assert verdict.status == "INSUFFICIENT"
    assert "168 hours per window" in verdict.detail


def test_an_empty_store_is_handled_rather_than_crashed():
    empty = pd.DataFrame(
        columns=["forecast_mwh", "model_name", "model_version", "forecast_origin", "generated_at"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    _, panel = scenario(days=2)

    verified = verify(empty, panel)

    assert verified.empty
    assert overall(verified) is None
    assert detect_drift(verified).status == "INSUFFICIENT"


def test_only_the_newest_version_is_scored_for_an_hour():
    """Two versions on one hour must not make that hour count twice."""
    forecasts, panel = scenario(days=4)
    older = forecasts.copy()
    older["model_version"] = "v0"
    older["forecast_mwh"] = older["forecast_mwh"] - 5_000.0
    older["generated_at"] = pd.Timestamp("2025-12-01", tz="UTC")

    verified = verify(pd.concat([older, forecasts]).sort_index(), panel)

    assert len(verified) == 96
    assert set(verified["model_version"]) == {"v1"}
