"""What this project forecasts, and what differs between the two.

Serving a second target is where a codebase usually goes wrong in one of two
ways: it grows a parallel copy of every module, or it grows an `if target ==`
inside every function. Both are the same mistake — the differences between load
and price are *data*, and scattering data through control flow is what makes it
impossible to see them all at once.

So they live here, in one table, and everything downstream takes a `Target`.
Reading this file should be enough to know what changes when the target does.

The asymmetry worth understanding before reading the fields:

**Load has a published competitor and price does not.** The system operator
publishes a load plan, so a load forecast can be judged on *skill against the
forecast the market actually runs on* — which is what ADR 0007 built drift
detection around, because error alone cannot tell a hard week from a worse
model. No such forecast exists for price. Without a control the same detector
would degrade to "error went up", the exact failure ADR 0007 rejected.

The substitute is a seasonal naive. It is available at bid time, needs no data
the project does not already hold, and it gets harder exactly when the market
does — which is the property that makes a control a control. It is a weaker
competitor than a professional forecast, and that weakness belongs in any
statement of price drift, but it is a real one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    """One forecastable quantity and everything that varies with it."""

    name: str
    """Short identifier. Partitions the forecast store, so changing it orphans
    stored history — treat it as part of the on-disk schema."""

    column: str
    """The panel column holding the realised value."""

    unit: str
    """Written into every stored row. Forecast values are unit-neutral in the
    schema, so this is what stops a reader that spans both targets from
    averaging megawatt-hours with lira."""

    plan_column: str | None
    """Panel column holding a published competing forecast, if one exists.
    `None` means drift must fall back to the naive control below."""

    naive_season_hours: int
    """Lag used as the fallback control, and as the baseline in reporting."""

    correct_bias: bool
    """Whether the deployed forecast applies the rolling bias correction.

    Measured per target, never inherited. On load it is worth -6.3% of MAE on a
    freshly retrained model (ADR 0010). On price the same machinery is worth
    -0.4% at best and **+2.5% worse** in the configuration load settled on, so
    it is off (ADR 0015). The condition is not merely weaker on price, it is
    inverted: the correction pays only on a stale model, and a stale price model
    should be retrained rather than patched.
    """

    include_supply_weather: bool
    """Whether the deployed model gets irradiance and wind at the generators.

    On for price and off for load, and the asymmetry is measured rather than
    stylistic: these are supply signals, and load is a demand quantity. For
    price the feature is worth -3.4% of MAE *net of* the two years of history it
    costs, because the archive starts later than the temperature one (ADR 0013).
    """

    mape_floor: float
    """Denominator floor for MAPE. A target that reaches zero needs a floor
    large enough to matter — see `analysis.price_experiment.PRICE_MAPE_FLOOR`
    for why price uses 100 and why sMAPE is the metric quoted there anyway."""


LOAD = Target(
    name="load",
    column="consumption_mwh",
    unit="MWh",
    plan_column="load_plan_mwh",
    naive_season_hours=168,
    correct_bias=True,
    include_supply_weather=False,
    mape_floor=1.0,
)

PRICE = Target(
    name="price",
    column="price_try_mwh",
    unit="TRY/MWh",
    # No published price forecast exists on the platform (ADR 0012). This is a
    # fact about the market, not a gap waiting to be filled by more ingestion.
    plan_column=None,
    naive_season_hours=168,
    correct_bias=False,
    include_supply_weather=True,
    mape_floor=100.0,
)

TARGETS: dict[str, Target] = {target.name: target for target in (LOAD, PRICE)}

DEFAULT_TARGET = LOAD
"""Load remains the default so that every existing caller keeps its behaviour.
Price is opted into explicitly, which is the same reasoning that keeps
`FeatureSpec.include_supply_weather` off by default."""


def resolve(target: Target | str | None) -> Target:
    """Accept a `Target`, its name, or nothing, and return a `Target`.

    Callers reach this from a CLI flag, an HTTP query parameter and Python code,
    and forcing each of them to import the registry and index it would spread
    the same three lines everywhere — including the error message, which is the
    part worth having in one place.
    """
    if target is None:
        return DEFAULT_TARGET
    if isinstance(target, Target):
        return target
    try:
        return TARGETS[target]
    except KeyError:
        known = ", ".join(sorted(TARGETS))
        raise KeyError(f"unknown target {target!r}; known targets are {known}") from None


def for_column(column: str) -> Target:
    """Find the target that owns a panel column.

    This is what lets a model card decide where its forecasts are stored. The
    card already records the feature spec, and the spec already names the target
    column — so the store partition and the unit are *derivable* rather than
    passed in. A caller that cannot supply them cannot supply them wrongly, and
    writing a price forecast into the load directory stops being possible.
    """
    for target in TARGETS.values():
        if target.column == column:
            return target
    known = ", ".join(sorted(target.column for target in TARGETS.values()))
    raise KeyError(f"no target forecasts column {column!r}; known columns are {known}")
