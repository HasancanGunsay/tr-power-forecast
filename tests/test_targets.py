"""Tests for the target registry and the multi-target forecast store.

The failure this file exists to prevent has no exception attached to it: a price
forecast filed under load, or a load model retrained on the price target. Every
one of those keeps returning plausible numbers.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from powerforecast.features.build import FeatureSpec
from powerforecast.forecasts.day import DayForecast
from powerforecast.forecasts.store import (
    forecasts_root,
    month_path,
    read_forecasts,
    write_forecasts,
)
from powerforecast.models.train import spec_for
from powerforecast.targets import LOAD, PRICE, TARGETS, for_column, resolve


def _day(target, *, base: float, version: str = "v1") -> pd.DataFrame:
    start = pd.Timestamp("2026-09-08", tz="Europe/Istanbul")
    hours = pd.DatetimeIndex(
        pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left")
    ).tz_convert("UTC")
    return DayForecast(
        delivery_date=date(2026, 9, 8),
        forecast_origin=hours[0] - pd.Timedelta(hours=13),
        model_name=f"{target.name}-lightgbm",
        model_version=version,
        generated_at=datetime(2026, 9, 7, 9, 0, tzinfo=UTC),
        values=pd.Series(base, index=hours, dtype="float64"),
        bias_offset=pd.Series(0.0, index=hours, dtype="float64"),
        target=target,
    ).to_frame()


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def test_resolve_accepts_a_name_an_object_or_nothing() -> None:
    assert resolve("price") is PRICE
    assert resolve(PRICE) is PRICE
    assert resolve(None) is LOAD, "load stays the default so old callers are unchanged"


def test_an_unknown_target_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="load, price"):
        resolve("elektrik")


def test_for_column_maps_a_panel_column_to_its_target() -> None:
    """This is what lets a model card decide where its forecasts are stored."""
    assert for_column("consumption_mwh") is LOAD
    assert for_column("price_try_mwh") is PRICE


def test_for_column_refuses_a_column_nothing_forecasts() -> None:
    with pytest.raises(KeyError, match="no target forecasts"):
        for_column("temperature_c")


def test_only_price_has_no_published_competitor() -> None:
    """The platform publishes a load plan and no price plan (ADR 0012).

    Pinned because it is the reason price drift falls back to a naive control,
    and a stray plan column would silently change what skill means.
    """
    assert LOAD.plan_column == "load_plan_mwh"
    assert PRICE.plan_column is None


def test_units_differ_between_targets() -> None:
    assert LOAD.unit != PRICE.unit


def test_every_target_has_a_distinct_column_and_name() -> None:
    columns = [target.column for target in TARGETS.values()]
    assert len(columns) == len(set(columns)), "for_column would become ambiguous"


# --------------------------------------------------------------------------- #
# Feature presets
# --------------------------------------------------------------------------- #


def test_supply_weather_is_on_for_price_and_off_for_load() -> None:
    """Measured, not stylistic: worth -3.4% net for price, meaningless for load."""
    assert spec_for(PRICE).include_supply_weather is True
    assert spec_for(LOAD).include_supply_weather is False


def test_spec_for_targets_the_right_column() -> None:
    assert spec_for(PRICE).target == "price_try_mwh"
    assert spec_for(LOAD).target == "consumption_mwh"


def test_a_specs_target_round_trips_through_for_column() -> None:
    """The loop the store relies on: spec -> column -> target -> directory."""
    for target in TARGETS.values():
        assert for_column(spec_for(target).target) is target


# --------------------------------------------------------------------------- #
# The store partition
# --------------------------------------------------------------------------- #


def test_targets_are_stored_in_separate_directories(tmp_path) -> None:
    assert forecasts_root(tmp_path, target=LOAD) != forecasts_root(tmp_path, target=PRICE)
    assert month_path("2026-09", root=tmp_path, target=PRICE).parent.name == "price"


def test_a_price_forecast_does_not_land_in_the_load_store(tmp_path) -> None:
    """The whole point of the partition.

    MWh and TRY/MWh in one column would be averaged by any reader that forgot to
    filter, producing a number that means nothing and raises nothing.
    """
    write_forecasts(_day(LOAD, base=40_000.0), root=tmp_path, target=LOAD)
    write_forecasts(_day(PRICE, base=2_500.0), root=tmp_path, target=PRICE)

    load = read_forecasts(root=tmp_path, target=LOAD)
    price = read_forecasts(root=tmp_path, target=PRICE)

    assert len(load) == 24
    assert len(price) == 24
    assert load["forecast_value"].max() == pytest.approx(40_000.0)
    assert price["forecast_value"].max() == pytest.approx(2_500.0)


def test_every_stored_row_carries_its_unit(tmp_path) -> None:
    """So a frame lifted out of its directory can still say what it holds."""
    write_forecasts(_day(PRICE, base=2_500.0), root=tmp_path, target=PRICE)
    stored = read_forecasts(root=tmp_path, target=PRICE)

    assert set(stored["unit"]) == {"TRY/MWh"}


def test_reading_a_target_that_was_never_written_is_empty_not_an_error(tmp_path) -> None:
    write_forecasts(_day(LOAD, base=40_000.0), root=tmp_path, target=LOAD)

    assert read_forecasts(root=tmp_path, target=PRICE).empty


def test_legacy_rows_are_read_as_load(tmp_path) -> None:
    """The 24 rows written before the store was partitioned.

    They were load, because load was the only target that existed. Filling the
    unit in on read is what keeps one old file from forcing every reader to
    handle two shapes.
    """
    directory = forecasts_root(tmp_path, target=LOAD)
    directory.mkdir(parents=True)

    legacy = _day(LOAD, base=40_000.0).rename(
        columns={"forecast_value": "forecast_mwh", "bias_offset": "bias_offset_mwh"}
    )
    legacy.drop(columns=["unit"]).to_parquet(directory / "2026-09.parquet", index=True)

    stored = read_forecasts(root=tmp_path, target=LOAD)

    assert len(stored) == 24
    assert set(stored["unit"]) == {"MWh"}
    assert stored["forecast_value"].max() == pytest.approx(40_000.0)


def test_the_frame_records_the_target_it_came_from() -> None:
    frame = _day(PRICE, base=2_500.0)

    assert set(frame["unit"]) == {"TRY/MWh"}


def test_feature_spec_defaults_stay_on_load() -> None:
    """A caller that names no target must keep getting exactly what it used to."""
    assert FeatureSpec().target == LOAD.column


# --------------------------------------------------------------------------- #
# The silent bug this refactor found
# --------------------------------------------------------------------------- #


def test_retrain_takes_its_feature_set_from_the_incumbent_card(tmp_path) -> None:
    """A scheduled retrain must reproduce the incumbent's target, not a default.

    Before this was fixed, `retrain` called `train()` with no spec, so
    `FeatureSpec()` applied — the *load* target. Retraining the price model
    would have fitted a load model, saved it under the price name, and compared
    it against the price incumbent on a holdout. The store resolves `latest` by
    timestamp, so saving is promoting: the next service restart would have
    served demand forecasts to callers asking for price. Nothing raises at any
    step.
    """
    from powerforecast.models.persistence import save_model
    from powerforecast.models.train import train

    hours = pd.date_range("2024-03-01", periods=24 * 400, freq="h", tz="UTC")
    ramp = pd.Series(range(len(hours)), index=hours, dtype="float64")
    panel = pd.DataFrame(
        {
            "consumption_mwh": 30_000 + ramp % 5_000,
            "price_try_mwh": 2_000 + ramp % 900,
            "load_plan_mwh": 30_000 + ramp % 5_000,
            "temperature_c": 15.0,
            "hdd": 0.0,
            "cdd": 0.0,
            "solar_index": 300.0,
            "wind_index": 200.0,
        },
        index=hours,
    )

    incumbent = train(panel=panel, spec=spec_for(PRICE), name="price-lightgbm", directory=tmp_path)
    assert incumbent.card.spec().target == "price_try_mwh"

    # What `retrain` now does: read the spec off the card rather than defaulting.
    derived = incumbent.card.spec()

    assert derived.target == "price_try_mwh", "a retrain would have switched target"
    assert derived.include_supply_weather is True, "and would have dropped the supply signal"
    assert save_model is not None  # keeps the import meaningful to a reader
