"""
Tests for CalendarService — three-layer holiday lookup.

Categories:
    A  Calendar constants and basic structure
    B  Layer 1 — Standard library holidays
    C  Layer 2 — RBI overrides (IN_RBI_FX)
    D  Layer 2 — UAE lunar overrides (AE_BANKING)
    E  Layer 3 — Runtime overrides
    F  Business day logic
    G  Convenience predicates
    H  upcoming_holidays
    I  today() and timezone conversion
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from services.calendar_service import (
    AE_BANKING,
    ALL_CALENDARS,
    CALENDAR_TIMEZONE,
    IN_RBI_FX,
    UK_CHAPS,
    US_FEDWIRE,
    CalendarService,
    HolidayDecision,
    HolidayInfo,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def svc() -> CalendarService:
    """Fresh CalendarService for each test."""
    return CalendarService()


def utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Create a timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Calendar constants and basic structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalendarConstants:
    def test_all_four_constants_exist(self) -> None:
        assert US_FEDWIRE == "US_FEDWIRE"
        assert UK_CHAPS == "UK_CHAPS"
        assert IN_RBI_FX == "IN_RBI_FX"
        assert AE_BANKING == "AE_BANKING"

    def test_constants_are_distinct(self) -> None:
        constants = [US_FEDWIRE, UK_CHAPS, IN_RBI_FX, AE_BANKING]
        assert len(set(constants)) == 4

    def test_all_calendars_list(self) -> None:
        assert set(ALL_CALENDARS) == {US_FEDWIRE, UK_CHAPS, IN_RBI_FX, AE_BANKING}

    def test_timezone_map_has_all_calendars(self) -> None:
        for cal in ALL_CALENDARS:
            assert cal in CALENDAR_TIMEZONE

    def test_timezone_strings_are_valid_iana(self) -> None:
        for cal, tz_name in CALENDAR_TIMEZONE.items():
            # ZoneInfo raises if the timezone is invalid
            zi = ZoneInfo(tz_name)
            assert zi is not None

    def test_timezone_values(self) -> None:
        assert CALENDAR_TIMEZONE[US_FEDWIRE] == "America/New_York"
        assert CALENDAR_TIMEZONE[UK_CHAPS] == "Europe/London"
        assert CALENDAR_TIMEZONE[IN_RBI_FX] == "Asia/Kolkata"
        assert CALENDAR_TIMEZONE[AE_BANKING] == "Asia/Dubai"

    def test_unknown_calendar_raises_valueerror(self, svc: CalendarService) -> None:
        with pytest.raises(ValueError, match="Unknown calendar"):
            svc.is_holiday(date(2025, 1, 1), "XY_UNKNOWN")

    def test_unknown_calendar_in_is_business_day(self, svc: CalendarService) -> None:
        with pytest.raises(ValueError, match="Unknown calendar"):
            svc.is_business_day(date(2025, 1, 1), "BOGUS")

    def test_unknown_calendar_in_explain_holiday(self, svc: CalendarService) -> None:
        with pytest.raises(ValueError, match="Unknown calendar"):
            svc.explain_holiday(date(2025, 1, 1), "IN")  # old jurisdiction code


# ═══════════════════════════════════════════════════════════════════════════════
# B. Layer 1 — Standard library holidays
# ═══════════════════════════════════════════════════════════════════════════════


class TestLayer1Library:
    # US_FEDWIRE
    def test_us_new_year(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 1, 1), US_FEDWIRE) is True

    def test_us_independence_day(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 7, 4), US_FEDWIRE) is True

    def test_us_christmas(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 12, 25), US_FEDWIRE) is True

    def test_us_thanksgiving(self, svc: CalendarService) -> None:
        # 4th Thursday of November 2025 = Nov 27
        assert svc.is_holiday(date(2025, 11, 27), US_FEDWIRE) is True

    def test_us_normal_weekday_is_not_holiday(self, svc: CalendarService) -> None:
        # Jan 13 2025 (Mon) — no holiday
        assert svc.is_holiday(date(2025, 1, 13), US_FEDWIRE) is False

    # UK_CHAPS
    def test_uk_new_year(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 1, 1), UK_CHAPS) is True

    def test_uk_good_friday(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 4, 18), UK_CHAPS) is True

    def test_uk_easter_monday(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 4, 21), UK_CHAPS) is True

    def test_uk_christmas(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 12, 25), UK_CHAPS) is True

    def test_uk_boxing_day(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 12, 26), UK_CHAPS) is True

    def test_uk_normal_weekday_is_not_holiday(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 2, 3), UK_CHAPS) is False

    # IN_RBI_FX — Library passthrough (date not in RBI_ADDITIONS)
    def test_in_milad_un_nabi_layer1(self, svc: CalendarService) -> None:
        # Sep 5 2025 Milad-un-Nabi — in India library, not in RBI_ADDITIONS
        assert svc.is_holiday(date(2025, 9, 5), IN_RBI_FX) is True

    def test_in_explain_milad_is_layer1(self, svc: CalendarService) -> None:
        decision = svc.explain_holiday(date(2025, 9, 5), IN_RBI_FX)
        assert decision.is_holiday is True
        assert decision.info is not None
        assert decision.info.source_layer == 1

    # AE_BANKING — Fixed national holiday (not lunar)
    def test_ae_national_day_dec2(self, svc: CalendarService) -> None:
        # UAE National Day is December 2 per the holidays library
        assert svc.is_holiday(date(2025, 12, 2), AE_BANKING) is True

    def test_ae_national_day_dec3(self, svc: CalendarService) -> None:
        # December 3 is also a UAE National Day holiday
        assert svc.is_holiday(date(2025, 12, 3), AE_BANKING) is True

    def test_ae_new_year(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 1, 1), AE_BANKING) is True

    def test_ae_explain_national_day_layer1(self, svc: CalendarService) -> None:
        # Dec 2 is not in UAE_LUNAR_OVERRIDES → Layer 1 applies
        decision = svc.explain_holiday(date(2025, 12, 2), AE_BANKING)
        assert decision.is_holiday is True
        assert decision.info is not None
        assert decision.info.source_layer == 1
        assert decision.info.confirmed is True


# ═══════════════════════════════════════════════════════════════════════════════
# C. Layer 2 — RBI overrides (IN_RBI_FX)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLayer2RBI:
    def test_diwali_laxmi_puja_confirmed(self, svc: CalendarService) -> None:
        # 2025-10-20 in RBI_ADDITIONS with confirmed=True
        assert svc.is_holiday(date(2025, 10, 20), IN_RBI_FX) is True

    def test_diwali_explain_layer2(self, svc: CalendarService) -> None:
        decision = svc.explain_holiday(date(2025, 10, 20), IN_RBI_FX)
        assert decision.is_holiday is True
        assert decision.info is not None
        assert decision.info.source_layer == 2
        assert decision.info.name == "Diwali Laxmi Puja"
        assert decision.info.confirmed is True

    def test_holi_2026_unconfirmed_strict_false(self, svc: CalendarService) -> None:
        # 2026-03-04 Holi* — confirmed=False — strict=False (default)
        assert svc.is_holiday(date(2026, 3, 4), IN_RBI_FX, strict=False) is False

    def test_holi_2026_unconfirmed_strict_true(self, svc: CalendarService) -> None:
        # 2026-03-04 Holi* — confirmed=False — strict=True
        assert svc.is_holiday(date(2026, 3, 4), IN_RBI_FX, strict=True) is True

    def test_holi_2026_explain_strict_false(self, svc: CalendarService) -> None:
        decision = svc.explain_holiday(date(2026, 3, 4), IN_RBI_FX, strict=False)
        assert decision.is_holiday is False
        assert decision.info is not None
        assert decision.info.source_layer == 2
        assert decision.info.confirmed is False

    def test_id_ul_fitr_confirmed(self, svc: CalendarService) -> None:
        # 2025-03-31 Id-ul-Fitr — confirmed=True in RBI_ADDITIONS
        assert svc.is_holiday(date(2025, 3, 31), IN_RBI_FX) is True

    def test_id_ul_fitr_explain_layer2(self, svc: CalendarService) -> None:
        decision = svc.explain_holiday(date(2025, 3, 31), IN_RBI_FX)
        assert decision.is_holiday is True
        assert decision.info is not None
        assert decision.info.source_layer == 2
        assert decision.info.name == "Id-ul-Fitr"

    def test_rbi_removals_overrides_layer1(self, svc: CalendarService) -> None:
        # 2025-09-05 Milad-un-Nabi is in India library (Layer 1)
        # If we add it to _rbi_removals, Layer 2 should suppress it
        assert svc.is_holiday(date(2025, 9, 5), IN_RBI_FX) is True  # baseline
        svc._rbi_removals.add(date(2025, 9, 5))
        assert svc.is_holiday(date(2025, 9, 5), IN_RBI_FX) is False

    def test_rbi_removals_explain(self, svc: CalendarService) -> None:
        svc._rbi_removals.add(date(2025, 9, 5))
        decision = svc.explain_holiday(date(2025, 9, 5), IN_RBI_FX)
        assert decision.is_holiday is False
        assert "removal" in decision.reason.lower()

    def test_makar_sankranti_2025_confirmed(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 1, 14), IN_RBI_FX) is True

    def test_makar_sankranti_2026_unconfirmed_strict_false(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2026, 1, 14), IN_RBI_FX, strict=False) is False

    def test_makar_sankranti_2026_unconfirmed_strict_true(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2026, 1, 14), IN_RBI_FX, strict=True) is True


# ═══════════════════════════════════════════════════════════════════════════════
# D. Layer 2 — UAE lunar overrides (AE_BANKING)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLayer2UAE:
    def test_eid_al_fitr_strict_false(self, svc: CalendarService) -> None:
        # 2025-03-30 in UAE_LUNAR_OVERRIDES with confirmed=False
        # Layer 2 wins over Layer 1; strict=False → not a holiday
        assert svc.is_holiday(date(2025, 3, 30), AE_BANKING, strict=False) is False

    def test_eid_al_fitr_strict_true(self, svc: CalendarService) -> None:
        # strict=True → is_holiday=True
        assert svc.is_holiday(date(2025, 3, 30), AE_BANKING, strict=True) is True

    def test_eid_al_fitr_day2_strict_false(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 3, 31), AE_BANKING, strict=False) is False

    def test_eid_al_fitr_day3_strict_false(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 4, 1), AE_BANKING, strict=False) is False

    def test_eid_al_adha_strict_false(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 6, 7), AE_BANKING, strict=False) is False

    def test_eid_al_adha_strict_true(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 6, 7), AE_BANKING, strict=True) is True

    def test_islamic_new_year_strict_false(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 7, 27), AE_BANKING, strict=False) is False

    def test_prophets_birthday_strict_true(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2025, 9, 5), AE_BANKING, strict=True) is True

    def test_explain_eid_layer2_unconfirmed(self, svc: CalendarService) -> None:
        decision = svc.explain_holiday(date(2025, 3, 30), AE_BANKING, strict=False)
        assert decision.is_holiday is False
        assert decision.info is not None
        assert decision.info.source_layer == 2
        assert decision.info.confirmed is False
        assert decision.info.name == "Eid Al Fitr Day 1"

    def test_explain_eid_layer2_strict_true(self, svc: CalendarService) -> None:
        decision = svc.explain_holiday(date(2025, 3, 30), AE_BANKING, strict=True)
        assert decision.is_holiday is True
        assert decision.info is not None
        assert decision.info.source_layer == 2

    def test_ae_national_day_not_in_layer2(self, svc: CalendarService) -> None:
        # Dec 2 is NOT in UAE_LUNAR_OVERRIDES → falls to Layer 1 → confirmed
        decision = svc.explain_holiday(date(2025, 12, 2), AE_BANKING, strict=False)
        assert decision.is_holiday is True
        assert decision.info is not None
        assert decision.info.source_layer == 1

    def test_eid_2026_unconfirmed_strict_false(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2026, 3, 19), AE_BANKING, strict=False) is False

    def test_eid_2026_unconfirmed_strict_true(self, svc: CalendarService) -> None:
        assert svc.is_holiday(date(2026, 3, 19), AE_BANKING, strict=True) is True


# ═══════════════════════════════════════════════════════════════════════════════
# E. Layer 3 — Runtime overrides
# ═══════════════════════════════════════════════════════════════════════════════


class TestLayer3Runtime:
    def test_add_holiday_makes_date_holiday(self, svc: CalendarService) -> None:
        d = date(2025, 3, 3)  # Normal Monday — not a holiday
        assert svc.is_holiday(d, US_FEDWIRE) is False
        svc.add_holiday(d, US_FEDWIRE, "Test Bank Holiday")
        assert svc.is_holiday(d, US_FEDWIRE) is True

    def test_add_holiday_explain_shows_layer3(self, svc: CalendarService) -> None:
        d = date(2025, 3, 3)
        svc.add_holiday(d, US_FEDWIRE, "Test Bank Holiday", kind="banking")
        decision = svc.explain_holiday(d, US_FEDWIRE)
        assert decision.is_holiday is True
        assert decision.info is not None
        assert decision.info.source_layer == 3
        assert decision.info.name == "Test Bank Holiday"
        assert decision.info.kind == "banking"
        assert decision.info.confirmed is True

    def test_remove_holiday_overrides_layer1(self, svc: CalendarService) -> None:
        # Jan 1 is a Layer 1 holiday for US_FEDWIRE
        d = date(2025, 1, 1)
        assert svc.is_holiday(d, US_FEDWIRE) is True
        svc.remove_holiday(d, US_FEDWIRE, user="ops-team")
        assert svc.is_holiday(d, US_FEDWIRE) is False

    def test_remove_holiday_overrides_layer2(self, svc: CalendarService) -> None:
        # 2025-10-20 Diwali is a Layer 2 holiday for IN_RBI_FX
        d = date(2025, 10, 20)
        assert svc.is_holiday(d, IN_RBI_FX) is True
        svc.remove_holiday(d, IN_RBI_FX)
        assert svc.is_holiday(d, IN_RBI_FX) is False

    def test_remove_explain_reason(self, svc: CalendarService) -> None:
        d = date(2025, 1, 1)
        svc.remove_holiday(d, US_FEDWIRE)
        decision = svc.explain_holiday(d, US_FEDWIRE)
        assert decision.is_holiday is False
        assert "Layer 3" in decision.reason

    def test_list_overrides_returns_only_layer3_additions(self, svc: CalendarService) -> None:
        overrides = svc.list_overrides()
        assert overrides == []
        svc.add_holiday(date(2025, 5, 1), US_FEDWIRE, "Custom Holiday")
        overrides = svc.list_overrides()
        assert len(overrides) == 1
        assert overrides[0].source_layer == 3

    def test_list_overrides_filters_by_calendar(self, svc: CalendarService) -> None:
        svc.add_holiday(date(2025, 5, 1), US_FEDWIRE, "US Custom")
        svc.add_holiday(date(2025, 5, 2), UK_CHAPS, "UK Custom")
        us_overrides = svc.list_overrides(US_FEDWIRE)
        assert len(us_overrides) == 1
        assert us_overrides[0].calendar == US_FEDWIRE

    def test_list_overrides_none_returns_all(self, svc: CalendarService) -> None:
        svc.add_holiday(date(2025, 5, 1), US_FEDWIRE, "US Custom")
        svc.add_holiday(date(2025, 5, 2), UK_CHAPS, "UK Custom")
        all_overrides = svc.list_overrides(None)
        assert len(all_overrides) == 2

    def test_audit_log_add(self, svc: CalendarService) -> None:
        svc.add_holiday(date(2025, 5, 1), US_FEDWIRE, "Labour Day", user="alice")
        log = svc.audit_log()
        assert len(log) == 1
        assert log[0].action == "add"
        assert log[0].d == date(2025, 5, 1)
        assert log[0].calendar == US_FEDWIRE
        assert log[0].name == "Labour Day"
        assert log[0].user == "alice"

    def test_audit_log_remove(self, svc: CalendarService) -> None:
        svc.remove_holiday(date(2025, 1, 1), US_FEDWIRE, user="bob")
        log = svc.audit_log()
        assert len(log) == 1
        assert log[0].action == "remove"
        assert log[0].user == "bob"
        assert log[0].name is None

    def test_audit_log_preserves_order(self, svc: CalendarService) -> None:
        svc.add_holiday(date(2025, 5, 1), US_FEDWIRE, "A")
        svc.remove_holiday(date(2025, 1, 1), US_FEDWIRE)
        svc.add_holiday(date(2025, 6, 1), UK_CHAPS, "B")
        log = svc.audit_log()
        assert len(log) == 3
        assert log[0].action == "add"
        assert log[1].action == "remove"
        assert log[2].action == "add"

    def test_layer3_wins_over_layer2_name(self, svc: CalendarService) -> None:
        # 2025-10-20 Diwali Laxmi Puja is Layer 2; add Layer 3 → Layer 3 name wins
        d = date(2025, 10, 20)
        svc.add_holiday(d, IN_RBI_FX, "Emergency Bank Closure")
        decision = svc.explain_holiday(d, IN_RBI_FX)
        assert decision.info is not None
        assert decision.info.source_layer == 3
        assert decision.info.name == "Emergency Bank Closure"

    def test_add_then_remove_clears_addition(self, svc: CalendarService) -> None:
        d = date(2025, 3, 3)
        svc.add_holiday(d, US_FEDWIRE, "Test")
        svc.remove_holiday(d, US_FEDWIRE)
        # remove_holiday removes the L3 add entry → falls back to L1/L2
        assert svc.is_holiday(d, US_FEDWIRE) is False
        # Verify it's not in list_overrides
        assert svc.list_overrides(US_FEDWIRE) == []

    def test_remove_then_add_clears_removal(self, svc: CalendarService) -> None:
        d = date(2025, 1, 1)
        svc.remove_holiday(d, US_FEDWIRE)
        assert svc.is_holiday(d, US_FEDWIRE) is False
        svc.add_holiday(d, US_FEDWIRE, "Override New Year")
        # add_holiday should cancel the removal
        assert svc.is_holiday(d, US_FEDWIRE) is True

    def test_unknown_calendar_add_raises(self, svc: CalendarService) -> None:
        with pytest.raises(ValueError, match="Unknown calendar"):
            svc.add_holiday(date(2025, 1, 1), "XX_BOGUS", "Test")

    def test_unknown_calendar_remove_raises(self, svc: CalendarService) -> None:
        with pytest.raises(ValueError, match="Unknown calendar"):
            svc.remove_holiday(date(2025, 1, 1), "XX_BOGUS")


# ═══════════════════════════════════════════════════════════════════════════════
# F. Business day logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestBusinessDayLogic:
    # Weekend detection — US/UK/IN (Sat/Sun)
    def test_us_saturday_is_not_business_day(self, svc: CalendarService) -> None:
        sat = date(2025, 1, 4)  # Saturday
        assert svc.is_business_day(sat, US_FEDWIRE) is False

    def test_us_sunday_is_not_business_day(self, svc: CalendarService) -> None:
        sun = date(2025, 1, 5)  # Sunday
        assert svc.is_business_day(sun, US_FEDWIRE) is False

    def test_uk_saturday_is_not_business_day(self, svc: CalendarService) -> None:
        assert svc.is_business_day(date(2025, 1, 4), UK_CHAPS) is False

    def test_in_saturday_is_not_business_day(self, svc: CalendarService) -> None:
        assert svc.is_business_day(date(2025, 1, 4), IN_RBI_FX) is False

    # UAE — Fri/Sat weekend, Sun is business day
    def test_ae_friday_is_not_business_day(self, svc: CalendarService) -> None:
        fri = date(2025, 1, 3)  # Friday (weekday=4)
        assert fri.weekday() == 4
        assert svc.is_business_day(fri, AE_BANKING) is False

    def test_ae_saturday_is_not_business_day(self, svc: CalendarService) -> None:
        sat = date(2025, 1, 4)
        assert svc.is_business_day(sat, AE_BANKING) is False

    def test_ae_sunday_is_business_day(self, svc: CalendarService) -> None:
        # UAE works Sun-Thu; Jan 5 2025 is Sunday
        sun = date(2025, 1, 5)
        assert sun.weekday() == 6
        # Jan 5 is not a holiday for AE_BANKING
        assert svc.is_business_day(sun, AE_BANKING) is True

    def test_ae_thursday_is_business_day(self, svc: CalendarService) -> None:
        thu = date(2025, 1, 2)  # Thursday
        assert thu.weekday() == 3
        assert svc.is_business_day(thu, AE_BANKING) is True

    def test_holiday_is_not_business_day(self, svc: CalendarService) -> None:
        # Jul 4 Independence Day — US_FEDWIRE
        assert svc.is_business_day(date(2025, 7, 4), US_FEDWIRE) is False

    def test_normal_weekday_is_business_day(self, svc: CalendarService) -> None:
        # Jan 13 2025 (Mon) — no US holiday
        assert svc.is_business_day(date(2025, 1, 13), US_FEDWIRE) is True

    # next_business_day
    def test_next_business_day_skips_weekend(self, svc: CalendarService) -> None:
        # Fri Jan 3 2025 → next business day = Mon Jan 6
        assert svc.next_business_day(date(2025, 1, 3), US_FEDWIRE) == date(2025, 1, 6)

    def test_next_business_day_skips_holiday(self, svc: CalendarService) -> None:
        # Dec 24 (Wed) → Dec 25 Christmas (holiday) → Dec 26 Fri (US business day)
        result = svc.next_business_day(date(2025, 12, 24), US_FEDWIRE)
        assert result == date(2025, 12, 26)

    def test_next_business_day_skips_holiday_and_weekend(self, svc: CalendarService) -> None:
        # Dec 31 2025 (Wed) → Jan 1 2026 (Thu, New Year) → Jan 2 2026 (Fri)
        result = svc.next_business_day(date(2025, 12, 31), US_FEDWIRE)
        assert result == date(2026, 1, 2)

    # previous_business_day
    def test_previous_business_day_skips_weekend(self, svc: CalendarService) -> None:
        # Mon Jan 6 → prev is Fri Jan 3 (no US holiday on Jan 3)
        result = svc.previous_business_day(date(2025, 1, 6), US_FEDWIRE)
        assert result == date(2025, 1, 3)

    def test_previous_business_day_skips_holiday(self, svc: CalendarService) -> None:
        # Jan 2 2025 (Thu) → Jan 1 (New Year, holiday) → Dec 31 2024 (Tue)
        result = svc.previous_business_day(date(2025, 1, 2), US_FEDWIRE)
        assert result == date(2024, 12, 31)

    # business_days_between
    def test_business_days_between_normal_week(self, svc: CalendarService) -> None:
        # Mon Jan 6 to Fri Jan 10 (exclusive) = Mon, Tue, Wed, Thu = 4 business days
        count = svc.business_days_between(date(2025, 1, 6), date(2025, 1, 10), US_FEDWIRE)
        assert count == 4

    def test_business_days_between_same_date(self, svc: CalendarService) -> None:
        # [d, d) = empty interval → 0
        assert svc.business_days_between(date(2025, 1, 6), date(2025, 1, 6), US_FEDWIRE) == 0

    def test_business_days_between_includes_start_excludes_end(self, svc: CalendarService) -> None:
        # Mon Jan 6 to Tue Jan 7 (exclusive) → just Monday = 1
        count = svc.business_days_between(date(2025, 1, 6), date(2025, 1, 7), US_FEDWIRE)
        assert count == 1

    # consecutive_non_business_days
    def test_consecutive_non_business_days_mlk_weekend(self, svc: CalendarService) -> None:
        # Jan 18 (Sat), Jan 19 (Sun), Jan 20 (MLK Day Mon) → 3 consecutive non-business
        count = svc.consecutive_non_business_days(date(2025, 1, 18), US_FEDWIRE)
        assert count == 3

    def test_consecutive_non_business_days_regular_weekend(self, svc: CalendarService) -> None:
        # Sat Jan 11, Sun Jan 12 → 2 consecutive
        count = svc.consecutive_non_business_days(date(2025, 1, 11), US_FEDWIRE)
        assert count == 2

    def test_consecutive_non_business_days_on_business_day(self, svc: CalendarService) -> None:
        # Monday Jan 6 is a business day → returns 0
        count = svc.consecutive_non_business_days(date(2025, 1, 6), US_FEDWIRE)
        assert count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# G. Convenience predicates
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvenientPredicates:
    # is_day_before_holiday
    def test_is_day_before_good_friday_uk(self, svc: CalendarService) -> None:
        # Thu Apr 17 → next day is Good Friday Apr 18 (UK holiday)
        assert svc.is_day_before_holiday(date(2025, 4, 17), UK_CHAPS) is True

    def test_is_day_before_holiday_normal_tue(self, svc: CalendarService) -> None:
        # Tue Jan 21 → next day Wed Jan 22 is a business day
        assert svc.is_day_before_holiday(date(2025, 1, 21), UK_CHAPS) is False

    def test_is_day_before_holiday_before_weekend(self, svc: CalendarService) -> None:
        # Fri Jan 3 → next day Sat Jan 4 is non-business → True
        assert svc.is_day_before_holiday(date(2025, 1, 3), US_FEDWIRE) is True

    def test_is_day_before_holiday_normal_monday(self, svc: CalendarService) -> None:
        # Mon Jan 6 → next day Tue Jan 7 is business → False
        assert svc.is_day_before_holiday(date(2025, 1, 6), US_FEDWIRE) is False

    # is_day_after_holiday
    def test_is_day_after_christmas_us(self, svc: CalendarService) -> None:
        # Dec 26 (Fri, US business day) → yesterday was Christmas → True
        assert svc.is_day_after_holiday(date(2025, 12, 26), US_FEDWIRE) is True

    def test_is_day_after_holiday_normal_wed(self, svc: CalendarService) -> None:
        # Wed Jan 22 → yesterday Tue Jan 21 is business → False
        assert svc.is_day_after_holiday(date(2025, 1, 22), US_FEDWIRE) is False

    def test_is_day_after_holiday_monday_after_weekend(self, svc: CalendarService) -> None:
        # Mon Jan 6 → yesterday Sun Jan 5 is non-business → True
        assert svc.is_day_after_holiday(date(2025, 1, 6), US_FEDWIRE) is True

    def test_is_day_after_boxing_day_uk(self, svc: CalendarService) -> None:
        # Dec 27 (Sat) after Boxing Day... let's use Dec 29 Mon (after Xmas/Boxing Day)
        # Dec 27 is Sat but let's verify Dec 29 (Mon): yesterday Dec 28 (Sun) non-business → True
        assert svc.is_day_after_holiday(date(2025, 12, 29), UK_CHAPS) is True


# ═══════════════════════════════════════════════════════════════════════════════
# H. upcoming_holidays
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpcomingHolidays:
    def test_returns_sorted_list(self, svc: CalendarService) -> None:
        holidays = svc.upcoming_holidays(date(2025, 12, 1), US_FEDWIRE, days=31)
        dates = [h.date for h in holidays]
        assert dates == sorted(dates)

    def test_does_not_include_dates_past_window(self, svc: CalendarService) -> None:
        # 30-day window starting Jan 1 should not include Feb 1
        holidays = svc.upcoming_holidays(date(2025, 1, 1), US_FEDWIRE, days=30)
        for h in holidays:
            assert h.date < date(2025, 1, 31)

    def test_includes_start_date(self, svc: CalendarService) -> None:
        # Jan 1 is a US holiday; starting on Jan 1 should include it
        holidays = svc.upcoming_holidays(date(2025, 1, 1), US_FEDWIRE, days=5)
        assert any(h.date == date(2025, 1, 1) for h in holidays)

    def test_excludes_unconfirmed_strict_false(self, svc: CalendarService) -> None:
        # UAE_LUNAR_OVERRIDES for Mar 30 has confirmed=False → excluded by default
        holidays = svc.upcoming_holidays(date(2025, 3, 28), AE_BANKING, days=5, strict=False)
        dates = [h.date for h in holidays]
        assert date(2025, 3, 30) not in dates

    def test_includes_unconfirmed_strict_true(self, svc: CalendarService) -> None:
        holidays = svc.upcoming_holidays(date(2025, 3, 28), AE_BANKING, days=5, strict=True)
        dates = [h.date for h in holidays]
        assert date(2025, 3, 30) in dates

    def test_layer3_name_wins_over_layer2(self, svc: CalendarService) -> None:
        # Add Layer 3 override on top of Layer 2 Diwali
        d = date(2025, 10, 20)
        svc.add_holiday(d, IN_RBI_FX, "Emergency Closure")
        holidays = svc.upcoming_holidays(d, IN_RBI_FX, days=1)
        assert len(holidays) == 1
        assert holidays[0].name == "Emergency Closure"
        assert holidays[0].source_layer == 3

    def test_deduplication_same_date(self, svc: CalendarService) -> None:
        # Layer 1 and Layer 2 both cover 2025-01-26 Republic Day (IN)
        # Should appear exactly once
        holidays = svc.upcoming_holidays(date(2025, 1, 26), IN_RBI_FX, days=1)
        republic_day_entries = [h for h in holidays if h.date == date(2025, 1, 26)]
        assert len(republic_day_entries) == 1

    def test_empty_window(self, svc: CalendarService) -> None:
        holidays = svc.upcoming_holidays(date(2025, 1, 13), US_FEDWIRE, days=0)
        assert holidays == []

    def test_uk_good_friday_in_window(self, svc: CalendarService) -> None:
        holidays = svc.upcoming_holidays(date(2025, 4, 14), UK_CHAPS, days=10)
        dates = [h.date for h in holidays]
        assert date(2025, 4, 18) in dates

    def test_rbi_diwali_window(self, svc: CalendarService) -> None:
        holidays = svc.upcoming_holidays(date(2025, 10, 18), IN_RBI_FX, days=5)
        dates = [h.date for h in holidays]
        assert date(2025, 10, 20) in dates


# ═══════════════════════════════════════════════════════════════════════════════
# I. today() and timezone conversion
# ═══════════════════════════════════════════════════════════════════════════════


class TestTodayTimezone:
    def test_ae_banking_crosses_midnight(self, svc: CalendarService) -> None:
        # UTC 23:00 on Jan 1 = Jan 2 in AE (UTC+4)
        now_utc = utc(2025, 1, 1, 23, 0)
        assert svc.today(AE_BANKING, now_utc) == date(2025, 1, 2)

    def test_us_fedwire_same_calendar_date(self, svc: CalendarService) -> None:
        # UTC 23:00 on Jan 1 = 18:00 EST = Jan 1 in US (UTC-5)
        now_utc = utc(2025, 1, 1, 23, 0)
        assert svc.today(US_FEDWIRE, now_utc) == date(2025, 1, 1)

    def test_in_rbi_fx_early_morning(self, svc: CalendarService) -> None:
        # UTC 00:30 on Jan 2 = 06:00 IST = Jan 2 in India (UTC+5:30)
        now_utc = utc(2025, 1, 2, 0, 30)
        assert svc.today(IN_RBI_FX, now_utc) == date(2025, 1, 2)

    def test_uk_chaps_just_before_midnight(self, svc: CalendarService) -> None:
        # UTC 23:59 on Jan 1 = 23:59 GMT (winter) = Jan 1 in UK
        now_utc = utc(2025, 1, 1, 23, 59)
        assert svc.today(UK_CHAPS, now_utc) == date(2025, 1, 1)

    def test_ae_banking_before_midnight_stays_same(self, svc: CalendarService) -> None:
        # UTC 19:00 on Jan 1 = 23:00 Dubai (UTC+4) = Jan 1 in AE
        now_utc = utc(2025, 1, 1, 19, 0)
        assert svc.today(AE_BANKING, now_utc) == date(2025, 1, 1)

    def test_today_unknown_calendar_raises(self, svc: CalendarService) -> None:
        now_utc = utc(2025, 1, 1, 12, 0)
        with pytest.raises(ValueError, match="Unknown calendar"):
            svc.today("BOGUS", now_utc)

    def test_today_uses_local_timezone(self, svc: CalendarService) -> None:
        # Verify different calendars give different dates at UTC 20:00 Jan 1
        # AE (UTC+4): Jan 2 00:00 → next day
        # US (UTC-5): Jan 1 15:00 → same day
        now_utc = utc(2025, 1, 1, 20, 0)
        ae_date = svc.today(AE_BANKING, now_utc)
        us_date = svc.today(US_FEDWIRE, now_utc)
        assert ae_date == date(2025, 1, 2)
        assert us_date == date(2025, 1, 1)
        assert ae_date != us_date


# ── J. DST transition detection ───────────────────────────────────────────────

class TestDSTTransitions:

    def test_dst_spring_forward_us_2026(self, svc: CalendarService) -> None:
        """US springs forward overnight Mar 7→8 2026.
        The detector records Mar 7 — the last day with the old (EST) offset."""
        transitions = svc.get_dst_transitions(date(2026, 3, 1), date(2026, 3, 15))
        us = [t for t in transitions if t["jurisdiction"] == "US"]
        assert len(us) == 1
        assert us[0]["date"]          == "2026-03-07"
        assert us[0]["direction"]     == "spring_forward"
        assert us[0]["shift_minutes"] == 60
        assert us[0]["is_holiday"]    is False

    def test_dst_fall_back_us_2026(self, svc: CalendarService) -> None:
        """US falls back on Nov 1 2026 (first Sunday of November)."""
        transitions = svc.get_dst_transitions(date(2026, 10, 25), date(2026, 11, 5))
        us = [t for t in transitions if t["jurisdiction"] == "US"]
        assert len(us) == 1
        assert us[0]["direction"]     == "fall_back"
        assert us[0]["shift_minutes"] == 60

    def test_no_dst_for_india_or_uae(self, svc: CalendarService) -> None:
        """IN and AE don't observe DST — must never appear in transitions."""
        transitions = svc.get_dst_transitions(date(2026, 1, 1), date(2026, 12, 31))
        jurisdictions = {t["jurisdiction"] for t in transitions}
        assert "IN" not in jurisdictions
        assert "AE" not in jurisdictions

    def test_empty_range_returns_no_transitions(self, svc: CalendarService) -> None:
        """A range with no DST events returns an empty list."""
        # Mid-summer: no transitions for any jurisdiction
        transitions = svc.get_dst_transitions(date(2026, 7, 1), date(2026, 7, 31))
        assert transitions == []
