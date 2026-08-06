"""Turkish public holidays, and the demand collapse around them.

Error diagnosis put a number on why this module exists. The model's worst
delivery days were, without exception, holidays — and its bias on them was
**+7,700 MWh**: it forecast an ordinary day while industry shut and demand
collapsed. Nothing in a timestamp reveals that.

Two families, and they behave differently.

**National holidays** are fixed dates. Mostly single days, mostly a normal
weekend-shaped dip.

**Religious holidays** follow the Hijri calendar and slide about eleven days
earlier each Gregorian year, so they cannot be derived from a date. They also
last several days and trigger mass travel, which makes their effect much larger
and much wider than the official days alone — the diagnosis found demand already
falling the day *before* the eve, and still depressed the day after the holiday
ended.

That is why this module emits more than `is_holiday`. Proximity and position
within a multi-day holiday carry real signal, and a single boolean would throw
it away.

**Availability:** none of this is a leak. A calendar for next year is known this
year, so every column here is legitimately available at bid time.

**Known limitation:** the Turkish government sometimes *extends* a religious
holiday by administrative decision — 2024's Ramazan Bayramı was stretched to
nine days. Those extensions are political decisions announced weeks ahead and
cannot be computed from any calendar. `EXTENSIONS` records the ones known to
affect this dataset; the list is necessarily incomplete for future dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from hijridate import Gregorian, Hijri

from powerforecast.features.availability import LOCAL_TZ

# Fixed-date national holidays: (month, day, name, eve is a legal half day).
#
# Only Cumhuriyet Bayramı has a statutory half-day eve; the day before 23 April
# or 30 August is ordinary working time. That distinction matters because error
# diagnosis found 28 October among the worst days — the model had learned from
# the many ordinary national eves that an eve looks normal.
NATIONAL_HOLIDAYS: tuple[tuple[int, int, str, bool], ...] = (
    (1, 1, "yilbasi", False),
    (4, 23, "ulusal_egemenlik", False),
    (5, 1, "emek_ve_dayanisma", False),
    (5, 19, "genclik_ve_spor", False),
    (7, 15, "demokrasi", False),
    (8, 30, "zafer", False),
    (10, 29, "cumhuriyet", True),
)

# Religious holidays, as (Hijri month, Hijri day of the first full day, length).
# Ramazan Bayramı starts on 1 Shawwal and runs three days; Kurban Bayramı starts
# on 10 Dhul-Hijja and runs four. Both are preceded by a half-day eve (arife).
RAMAZAN = (10, 1, 3)
KURBAN = (12, 10, 4)

# Administrative extensions, which no calendar can compute.
#
# Maps the *computed* first day of a holiday to the period actually granted, as
# (first non-working day, total days). Both ends have to be overridable: an
# extension usually starts earlier as well as ending later, and encoding only the
# length shifts the whole window forward.
#
# The 2024 entry is what error diagnosis found. Encoded as length-only, the model
# missed 8 April (demand had already collapsed, +4,187 MWh over-forecast) and
# wrongly treated 15 April as a holiday (demand had recovered, −6,981 under).
EXTENSIONS: dict[date, tuple[date, int]] = {
    date(2024, 4, 10): (date(2024, 4, 8), 7),  # a continuous 8-14 April break
}


@dataclass(frozen=True)
class Holiday:
    """One holiday period, from its eve to its last day."""

    name: str
    first_day: date
    length_days: int
    religious: bool
    # Whether the eve is a statutory half working day. True for every religious
    # arife and for 28 October; false for the other national holidays, whose eves
    # are ordinary working days.
    half_day_eve: bool = False

    @property
    def eve(self) -> date:
        return self.first_day - timedelta(days=1)

    @property
    def last_day(self) -> date:
        return self.first_day + timedelta(days=self.length_days - 1)

    def days(self) -> list[date]:
        return [self.first_day + timedelta(days=i) for i in range(self.length_days)]


def religious_holidays(year: int) -> list[Holiday]:
    """Both religious holidays falling in a Gregorian year.

    A Gregorian year can contain two occurrences of the same Hijri feast, or
    none, because the Hijri year is shorter. Scanning the Hijri years that
    overlap the Gregorian one and filtering afterwards handles that without
    special cases.
    """
    holidays: list[Holiday] = []

    # The Hijri year is ~354 days, so at most three can touch one Gregorian year.
    hijri_year = Gregorian(year, 6, 15).to_hijri().datetuple()[0]

    for offset in (-1, 0, 1):
        for name, (month, day, length) in (("ramazan", RAMAZAN), ("kurban", KURBAN)):
            try:
                first = Hijri(hijri_year + offset, month, day).to_gregorian()
            except (ValueError, OverflowError):
                continue

            computed = date(first.year, first.month, first.day)
            if computed.year != year:
                continue

            first_day, length_days = EXTENSIONS.get(computed, (computed, length))
            holidays.append(
                Holiday(
                    name=name,
                    first_day=first_day,
                    length_days=length_days,
                    religious=True,
                    half_day_eve=True,
                )
            )

    return sorted(holidays, key=lambda h: h.first_day)


def national_holidays(year: int) -> list[Holiday]:
    return [
        Holiday(
            name=name,
            first_day=date(year, month, day),
            length_days=1,
            religious=False,
            half_day_eve=half_day,
        )
        for month, day, name, half_day in NATIONAL_HOLIDAYS
    ]


def holidays_in(start_year: int, end_year: int) -> list[Holiday]:
    """Every holiday between two years, inclusive."""
    found: list[Holiday] = []
    for year in range(start_year, end_year + 1):
        found.extend(national_holidays(year))
        found.extend(religious_holidays(year))
    return sorted(found, key=lambda h: h.first_day)


def holiday_features(index: pd.DatetimeIndex, *, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Holiday columns for the given target hours.

    Columns:
        `is_holiday`         an official non-working day
        `is_eve`             the day before a holiday — where travel begins
        `is_half_day`        a statutory half working day: every religious arife, plus
                             28 October. Separate from `is_eve` because most national
                             eves are ordinary working days, and conflating them taught
                             the model that 28 October looked normal
        `is_religious`       inside a religious holiday, which behaves unlike a national one
        `holiday_position`   0 on the eve, 1..n through the holiday, -1 on ordinary days
        `days_to_holiday`    signed distance to the nearest holiday period, clipped to ±3

    `holiday_position` uses -1 rather than NaN for ordinary days. NaN would be the
    more natural encoding of "not applicable", but the backtest drops any row with
    a missing feature, and roughly 95% of days are ordinary — the whole dataset
    would quietly disappear. A sentinel outside the valid range keeps the column
    ordered (-1 < 0 < 1 < 2 …) and costs a tree one extra split.

    `days_to_holiday` is the column the diagnosis actually asked for: demand was
    already falling the day before the eve and had not recovered the day after
    the last day, so the effect is a window rather than a flag.
    """
    local = pd.DatetimeIndex(index).tz_convert(tz)
    dates = pd.Series(local.date, index=index, name="local_date")

    years = sorted({d.year for d in dates})
    periods = holidays_in(min(years) - 1, max(years) + 1)

    is_holiday: dict[date, bool] = {}
    is_eve: dict[date, bool] = {}
    is_half_day: dict[date, bool] = {}
    is_religious: dict[date, bool] = {}
    position: dict[date, int] = {}

    for holiday in periods:
        is_eve[holiday.eve] = True
        if holiday.half_day_eve:
            is_half_day[holiday.eve] = True
        is_religious.setdefault(holiday.eve, holiday.religious)
        position.setdefault(holiday.eve, 0)
        for offset, day in enumerate(holiday.days(), start=1):
            is_holiday[day] = True
            is_religious[day] = holiday.religious
            position[day] = offset

    frame = pd.DataFrame(index=index)
    frame["is_holiday"] = dates.map(is_holiday).fillna(False).astype("int8")
    frame["is_eve"] = dates.map(is_eve).fillna(False).astype("int8")
    frame["is_half_day"] = dates.map(is_half_day).fillna(False).astype("int8")
    frame["is_religious"] = dates.map(is_religious).fillna(False).astype("int8")
    frame["holiday_position"] = dates.map(position).fillna(-1).astype("float64")
    frame["days_to_holiday"] = _signed_distance(dates, periods)
    return frame


def _signed_distance(dates: pd.Series, periods: list[Holiday], *, clip: int = 3) -> pd.Series:
    """Days to the nearest holiday: negative before, positive after, 0 inside.

    Clipped, because a day three weeks from a holiday is simply an ordinary day
    and letting the number grow would invite a model to read a trend into it.
    """
    spans = [(h.eve, h.last_day) for h in periods]
    if not spans:
        return pd.Series(float(clip), index=dates.index)

    def distance(day: date) -> float:
        days_until = [(start - day).days for start, _ in spans]
        days_since = [(day - end).days for _, end in spans]

        # Inside a period means the day is at or after its start *and* at or
        # before its end — both distances non-positive. Chaining these as
        # `until <= 0 <= since` reads naturally and means something else
        # entirely ("after the start and after the end"), which every past
        # holiday satisfies, so every day looked like a holiday.
        inside = any(
            until <= 0 and since <= 0 for until, since in zip(days_until, days_since, strict=True)
        )
        if inside:
            return 0.0

        upcoming = min((d for d in days_until if d > 0), default=clip + 1)
        passed = min((d for d in days_since if d > 0), default=clip + 1)

        return -float(min(upcoming, clip)) if upcoming <= passed else float(min(passed, clip))

    unique = {d: distance(d) for d in dates.unique()}
    return dates.map(unique).astype("float64")
