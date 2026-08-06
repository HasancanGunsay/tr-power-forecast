"""Tests for the Turkish holiday calendar.

The most valuable tests here are the ones that check computed religious holiday
dates against days on which demand was *observed* to collapse. A converter that
is off by one day would produce a feature that marks the wrong days while
looking entirely correct — the exact failure this module was written to avoid.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pandas as pd
import pytest
from hijridate import Gregorian, Hijri

from powerforecast.features.holidays import (
    KURBAN,
    RAMAZAN,
    Holiday,
    holiday_features,
    holidays_in,
    national_holidays,
    religious_holidays,
)


def _index(start: str, days: int) -> pd.DatetimeIndex:
    local = pd.date_range(start, periods=days * 24, freq="h", tz="Europe/Istanbul")
    return pd.DatetimeIndex(local.tz_convert("UTC"))


def _first(name: str, year: int) -> date:
    return next(h.first_day for h in religious_holidays(year) if h.name == name)


def _computed(name: str, year: int) -> date:
    """The Hijri conversion alone, before any administrative override.

    Kept separate from `_first` so the converter can be validated independently
    of policy: an extension moving a start date must not be able to hide a
    conversion that drifted.
    """
    month, day, _ = RAMAZAN if name == "ramazan" else KURBAN
    hijri_year = Gregorian(year, 6, 15).to_hijri().datetuple()[0]

    for offset in (-1, 0, 1):
        candidate = Hijri(hijri_year + offset, month, day).to_gregorian()
        if candidate.year == year:
            return date(candidate.year, candidate.month, candidate.day)
    raise AssertionError(f"no {name} computed for {year}")


# --------------------------------------------------------------------------- #
# Dates, cross-checked against observed demand collapses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2024, date(2024, 4, 10)),
        (2025, date(2025, 3, 30)),
        (2026, date(2026, 3, 20)),
    ],
)
def test_ramazan_matches_the_observed_collapse(year: int, expected: date) -> None:
    # Error diagnosis found the worst delivery days clustered on 2024-04-08/09/10,
    # 2025-03-29/30/31 and 2026-03-19/20/23 — eve, holiday, and the return.
    assert _computed("ramazan", year) == expected


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2024, date(2024, 6, 16)),
        (2025, date(2025, 6, 6)),
        (2026, date(2026, 5, 27)),
    ],
)
def test_kurban_matches_the_observed_collapse(year: int, expected: date) -> None:
    assert _computed("kurban", year) == expected


def test_religious_holidays_slide_earlier_each_year() -> None:
    # Compare position *within* the year: the raw difference between two dates in
    # consecutive years is about 355 days and says nothing about the slide.
    day_of_year = [_computed("ramazan", y).timetuple().tm_yday for y in (2023, 2024, 2025, 2026)]
    shifts = [earlier - later for earlier, later in pairwise(day_of_year)]

    # The Hijri year is about eleven days shorter, which is precisely why these
    # dates cannot be derived from a Gregorian timestamp.
    assert all(10 <= shift <= 12 for shift in shifts), shifts


def test_the_two_feasts_keep_a_constant_gap() -> None:
    gaps = [(_computed("kurban", y) - _computed("ramazan", y)).days for y in (2024, 2025, 2026)]

    # 1 Shawwal to 10 Dhul-Hijja spans two Hijri months, so the gap is fixed at
    # 67-68 days and varies only with month length.
    assert all(66 <= gap <= 70 for gap in gaps), gaps


def test_national_holidays_are_fixed() -> None:
    days = {h.first_day for h in national_holidays(2026)}

    assert date(2026, 1, 1) in days
    assert date(2026, 10, 29) in days
    assert len(days) == 7


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_kurban_runs_longer_than_ramazan() -> None:
    ramazan = next(h for h in religious_holidays(2025) if h.name == "ramazan")
    kurban = next(h for h in religious_holidays(2025) if h.name == "kurban")

    assert ramazan.length_days == 3
    assert kurban.length_days == 4


def test_an_administrative_extension_moves_both_ends() -> None:
    ramazan = next(h for h in religious_holidays(2024) if h.name == "ramazan")

    # 2024 was extended by government decision to a continuous 8-14 April break.
    # Encoding only the length would have kept the 10 April start and pushed the
    # end to the 15th — which is exactly the error diagnosis found: 8 April
    # over-forecast by 4,187 MWh, 15 April under-forecast by 6,981.
    assert ramazan.first_day == date(2024, 4, 8)
    assert ramazan.last_day == date(2024, 4, 14)
    assert ramazan.length_days == 7


def test_an_unextended_holiday_keeps_its_computed_dates() -> None:
    ramazan = next(h for h in religious_holidays(2025) if h.name == "ramazan")

    assert ramazan.first_day == date(2025, 3, 30)
    assert ramazan.length_days == 3


def test_eve_is_the_day_before_the_first_day() -> None:
    holiday = Holiday("test", date(2026, 3, 20), 3, religious=True)

    assert holiday.eve == date(2026, 3, 19)
    assert holiday.last_day == date(2026, 3, 22)
    assert len(holiday.days()) == 3


def test_holidays_in_covers_every_year_in_range() -> None:
    found = holidays_in(2024, 2026)
    years = {h.first_day.year for h in found}

    assert years == {2024, 2025, 2026}
    assert all(a.first_day <= b.first_day for a, b in pairwise(found))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def test_holiday_and_eve_are_flagged_on_the_right_days() -> None:
    features = holiday_features(_index("2026-03-17", 8))
    local = pd.DatetimeIndex(features.index).tz_convert("Europe/Istanbul")
    by_date = features.groupby(local.date).max()

    assert by_date.loc[date(2026, 3, 19), "is_eve"] == 1
    assert by_date.loc[date(2026, 3, 20), "is_holiday"] == 1
    assert by_date.loc[date(2026, 3, 22), "is_holiday"] == 1
    assert by_date.loc[date(2026, 3, 23), "is_holiday"] == 0


def test_position_counts_through_a_multi_day_holiday() -> None:
    features = holiday_features(_index("2026-03-19", 4))
    local = pd.DatetimeIndex(features.index).tz_convert("Europe/Istanbul")
    by_date = features.groupby(local.date)["holiday_position"].max()

    # The third day of a holiday does not look like the first, and a single
    # boolean would tell the model they are identical.
    assert list(by_date) == [0.0, 1.0, 2.0, 3.0]


def test_only_statutory_half_days_are_flagged_as_such() -> None:
    def half_day(day: str) -> int:
        return int(holiday_features(_index(day, 1))["is_half_day"].max())

    def eve(day: str) -> int:
        return int(holiday_features(_index(day, 1))["is_eve"].max())

    # Religious arife and 28 October are half working days by law.
    assert half_day("2026-03-19") == 1  # Ramazan arife
    assert half_day("2026-10-28") == 1  # Cumhuriyet arife

    # The day before 23 April is an ordinary working day. It is still an eve —
    # travel and behaviour shift — but conflating the two taught the model that
    # 28 October looked normal.
    assert eve("2026-04-22") == 1
    assert half_day("2026-04-22") == 0


def test_religious_and_national_holidays_are_distinguished() -> None:
    religious = holiday_features(_index("2026-03-20", 1))["is_religious"].max()
    national = holiday_features(_index("2026-10-29", 1))["is_religious"].max()

    assert religious == 1
    assert national == 0


def test_distance_is_negative_before_and_positive_after() -> None:
    features = holiday_features(_index("2026-03-16", 12))
    local = pd.DatetimeIndex(features.index).tz_convert("Europe/Istanbul")
    by_date = features.groupby(local.date)["days_to_holiday"].max()

    assert by_date[date(2026, 3, 17)] == -2  # two days before the eve
    assert by_date[date(2026, 3, 20)] == 0  # inside
    assert by_date[date(2026, 3, 24)] == 2  # two days after the last day


def test_distance_is_clipped_far_from_any_holiday() -> None:
    features = holiday_features(_index("2026-02-01", 3))

    # A day three weeks out is just an ordinary day; letting the number grow
    # would invite the model to read a trend into it.
    assert features["days_to_holiday"].abs().max() <= 3


def test_no_column_is_ever_null() -> None:
    # The backtest drops rows with a missing feature, and ~95% of days are
    # ordinary — a NaN for "not a holiday" would delete the dataset.
    features = holiday_features(_index("2026-02-01", 60))

    assert not features.isna().any().any()

    # The eve is not an official holiday but does have a position (0), so the
    # sentinel applies only to days that are neither.
    ordinary = (features["is_holiday"] == 0) & (features["is_eve"] == 0)
    assert features.loc[ordinary, "holiday_position"].eq(-1).all()
