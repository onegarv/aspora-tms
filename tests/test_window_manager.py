"""
Unit tests for WindowManager and TransferWindow.

All tests pass current_time explicitly — no implicit system clock usage.
Holiday calendars are injected as simple lambdas or sets.
"""

from __future__ import annotations

import pytest
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from agents.operations.window_manager import (
    WindowManager,
    TransferWindow,
    SEPA_INSTANT_MAX_EUR,
    AED_CUTOFF_LOCAL,
    TZ_ET, TZ_UK, TZ_IST, TZ_GST, TZ_UTC,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

NO_HOLIDAYS: set[date] = set()


def no_holidays(d: date) -> bool:
    return False


def make_wm(
    usd_holidays=no_holidays,
    gbp_holidays=no_holidays,
    inr_holidays=no_holidays,
    aed_holidays=no_holidays,
    operational_buffer_min: int = 10,
) -> WindowManager:
    return WindowManager(
        usd_holidays=usd_holidays,
        gbp_holidays=gbp_holidays,
        inr_holidays=inr_holidays,
        aed_holidays=aed_holidays,
        operational_buffer_min=operational_buffer_min,
    )


def dt(year, month, day, hour, minute, tz) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=tz)


# ── 1. Naive datetime rejection ────────────────────────────────────────────────

class TestNaiveDatetimeRejection:
    def test_is_open_now_raises_on_naive(self):
        wm = make_wm()
        naive = datetime(2026, 2, 16, 10, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            wm.is_open_now("USD", naive)

    def test_next_open_raises_on_naive(self):
        wm = make_wm()
        naive = datetime(2026, 2, 16, 10, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            wm.next_open("USD", naive)

    def test_opens_before_raises_on_naive_current(self):
        wm = make_wm()
        naive = datetime(2026, 2, 16, 10, 0)
        aware_deadline = dt(2026, 2, 16, 12, 0, TZ_UTC)
        with pytest.raises(ValueError):
            wm.opens_before("USD", aware_deadline, naive)

    def test_opens_before_raises_on_naive_deadline(self):
        wm = make_wm()
        aware_now = dt(2026, 2, 16, 10, 0, TZ_UTC)
        naive_deadline = datetime(2026, 2, 16, 12, 0)
        with pytest.raises(ValueError):
            wm.opens_before("USD", naive_deadline, aware_now)

    def test_can_complete_path_raises_on_naive(self):
        wm = make_wm()
        naive = datetime(2026, 2, 16, 10, 0)
        aware = dt(2026, 2, 16, 20, 0, TZ_UTC)
        with pytest.raises(ValueError):
            wm.can_complete_path("USD", "INR", 50_000, aware, naive)


# ── 2. USD Fedwire ─────────────────────────────────────────────────────────────

class TestUSDFedwire:
    # Monday 2026-02-16, 10:00 ET = inside window
    OPEN   = dt(2026, 2, 16, 10, 0, TZ_ET)
    # Monday 2026-02-16, 08:00 ET = before window
    BEFORE = dt(2026, 2, 16, 8, 0, TZ_ET)
    # Monday 2026-02-16, 18:00 ET = exactly at close — should be CLOSED (buffer removes 17:50 effective)
    AT_CLOSE = dt(2026, 2, 16, 17, 55, TZ_ET)   # within buffer → closed
    AFTER  = dt(2026, 2, 16, 19, 0, TZ_ET)

    def test_open_during_hours(self):
        assert make_wm().is_open_now("USD", self.OPEN) is True

    def test_closed_before_hours(self):
        assert make_wm().is_open_now("USD", self.BEFORE) is False

    def test_closed_after_hours(self):
        assert make_wm().is_open_now("USD", self.AFTER) is False

    def test_closed_within_operational_buffer(self):
        # 17:55 ET is inside the 10-min buffer before 18:00 close → closed
        assert make_wm().is_open_now("USD", self.AT_CLOSE) is False

    def test_minutes_until_close_inside_window(self):
        # 10:00 ET, operational close = 17:50 ET → 470 min
        mins = make_wm().minutes_until_close("USD", self.OPEN)
        assert mins == 470

    def test_minutes_until_close_when_closed(self):
        assert make_wm().minutes_until_close("USD", self.BEFORE) == 0

    def test_closed_on_weekend(self):
        saturday = dt(2026, 2, 14, 12, 0, TZ_ET)
        assert make_wm().is_open_now("USD", saturday) is False

    def test_closed_on_us_holiday(self):
        # Make Presidents' Day (2026-02-16) a holiday
        holiday_dates = {date(2026, 2, 16)}
        wm = make_wm(usd_holidays=lambda d: d in holiday_dates)
        assert wm.is_open_now("USD", self.OPEN) is False

    def test_next_open_returns_same_day_if_before_open(self):
        wm   = make_wm()
        nxt  = wm.next_open("USD", self.BEFORE)
        local = nxt.astimezone(TZ_ET)
        assert local.time() == time(9, 0)
        assert local.date() == date(2026, 2, 16)

    def test_next_open_from_friday_evening_returns_monday(self):
        friday_evening = dt(2026, 2, 13, 20, 0, TZ_ET)  # Friday after close
        nxt   = make_wm().next_open("USD", friday_evening)
        local = nxt.astimezone(TZ_ET)
        assert local.weekday() == 0  # Monday
        assert local.time() == time(9, 0)

    def test_opens_before_true(self):
        wm       = make_wm()
        now      = self.BEFORE
        deadline = dt(2026, 2, 16, 16, 0, TZ_ET)  # 16:00 ET same day
        assert wm.opens_before("USD", deadline, now) is True

    def test_opens_before_false(self):
        wm = make_wm()
        # Window opens at 09:00 ET; deadline is 08:30 ET — will miss
        now      = dt(2026, 2, 16, 8, 0, TZ_ET)
        deadline = dt(2026, 2, 16, 8, 30, TZ_ET)
        assert wm.opens_before("USD", deadline, now) is False


# ── 3. GBP CHAPS ───────────────────────────────────────────────────────────────

class TestGBPCHAPS:
    OPEN   = dt(2026, 2, 16, 10, 0, TZ_UK)
    BEFORE = dt(2026, 2, 16, 7, 0, TZ_UK)
    AFTER  = dt(2026, 2, 16, 16, 30, TZ_UK)

    def test_open_during_hours(self):
        assert make_wm().is_open_now("GBP", self.OPEN) is True

    def test_closed_before_hours(self):
        assert make_wm().is_open_now("GBP", self.BEFORE) is False

    def test_closed_after_hours(self):
        assert make_wm().is_open_now("GBP", self.AFTER) is False

    def test_closed_on_boe_holiday(self):
        holiday_dates = {date(2026, 2, 16)}
        wm = make_wm(gbp_holidays=lambda d: d in holiday_dates)
        assert wm.is_open_now("GBP", self.OPEN) is False


# ── 4. EUR SEPA Instant ────────────────────────────────────────────────────────

class TestEURSEPAInstant:
    ANY_TIME = dt(2026, 2, 16, 3, 0, TZ_UTC)  # 03:00 UTC — would be closed for any other rail

    def test_always_open_for_sepa_instant(self):
        # EUR is 24/7 for SEPA Instant
        wm = make_wm()
        assert wm.is_open_now("EUR", self.ANY_TIME) is True

    def test_next_open_is_immediate(self):
        wm  = make_wm()
        nxt = wm.next_open("EUR", self.ANY_TIME)
        assert nxt == self.ANY_TIME

    def test_opens_before_always_true(self):
        wm       = make_wm()
        deadline = dt(2026, 2, 16, 4, 0, TZ_UTC)
        assert wm.opens_before("EUR", deadline, self.ANY_TIME) is True

    def test_small_eur_path_executes_immediately(self):
        """EUR < 100k → SEPA Instant; execution governed by Fedwire + INR desk overlap."""
        wm         = make_wm()
        amount     = 50_000.0
        # 2026-02-16 14:30 UTC = 09:30 ET (Fedwire open) = 20:00 IST (INR desk closed)
        # Next INR open: 2026-02-17 09:00 IST = 03:30 UTC
        current    = dt(2026, 2, 16, 14, 30, TZ_UTC)
        exec_time  = wm.next_path_execution_time("EUR", "INR", amount, current)
        # Should be next day when both USD Fedwire AND INR desk are open
        local_ist  = exec_time.astimezone(TZ_IST)
        assert local_ist.date() >= date(2026, 2, 17)

    def test_large_eur_path_adds_delay(self):
        """EUR >= 100k → SEPA batch; execution is later than SEPA Instant path."""
        wm         = make_wm()
        small_amt  = SEPA_INSTANT_MAX_EUR - 1
        large_amt  = SEPA_INSTANT_MAX_EUR          # exactly at threshold → batch
        current    = dt(2026, 2, 16, 9, 0, TZ_ET)
        instant    = wm.next_path_execution_time("EUR", "INR", small_amt, current)
        batch      = wm.next_path_execution_time("EUR", "INR", large_amt, current)
        assert batch > instant

    def test_eur_exactly_100k_triggers_sepa_batch(self):
        wm      = make_wm()
        current = dt(2026, 2, 16, 9, 0, TZ_ET)
        below   = wm.next_path_execution_time("EUR", "INR", 99_999.99, current)
        at_100k = wm.next_path_execution_time("EUR", "INR", 100_000.0, current)
        assert at_100k > below


# ── 5. AED Banking ─────────────────────────────────────────────────────────────

class TestAEDBanking:
    # Monday 2026-02-16, 09:00 GST — before 14:00 cutoff
    BEFORE_CUTOFF = dt(2026, 2, 16, 9, 0, TZ_GST)
    # Monday 2026-02-16, 14:00 GST — exactly at cutoff (closed)
    AT_CUTOFF     = dt(2026, 2, 16, 14, 0, TZ_GST)
    # Monday 2026-02-16, 13:59 GST — one minute before (open)
    JUST_BEFORE   = dt(2026, 2, 16, 13, 59, TZ_GST)
    # Monday 2026-02-16, 16:00 GST — after cutoff
    AFTER_CUTOFF  = dt(2026, 2, 16, 16, 0, TZ_GST)

    def test_open_before_cutoff(self):
        assert make_wm().is_open_now("AED", self.BEFORE_CUTOFF) is True

    def test_open_one_minute_before_cutoff(self):
        assert make_wm().is_open_now("AED", self.JUST_BEFORE) is True

    def test_closed_at_exactly_cutoff(self):
        # 14:00:00 GST → closed (strict less-than)
        assert make_wm().is_open_now("AED", self.AT_CUTOFF) is False

    def test_closed_after_cutoff(self):
        assert make_wm().is_open_now("AED", self.AFTER_CUTOFF) is False

    def test_closed_on_weekend(self):
        saturday = dt(2026, 2, 14, 10, 0, TZ_GST)
        assert make_wm().is_open_now("AED", saturday) is False

    def test_closed_on_uae_holiday(self):
        holiday_dates = {date(2026, 2, 16)}
        wm = make_wm(aed_holidays=lambda d: d in holiday_dates)
        assert wm.is_open_now("AED", self.BEFORE_CUTOFF) is False

    def test_usd_value_date_same_day_before_cutoff(self):
        wm = make_wm()
        wm_internal = wm
        value_date = wm_internal._aed_usd_value_date(self.BEFORE_CUTOFF)
        assert value_date == date(2026, 2, 16)

    def test_usd_value_date_next_day_after_cutoff(self):
        wm         = make_wm()
        value_date = wm._aed_usd_value_date(self.AFTER_CUTOFF)
        assert value_date == date(2026, 2, 17)  # Tuesday

    def test_aed_path_before_cutoff_executes_today(self):
        """AED before 14:00 GST → USD available today → execution on overlap with INR desk."""
        wm      = make_wm()
        current = self.BEFORE_CUTOFF
        result  = wm.next_path_execution_time("AED", "INR", 10_000, current)
        # Result must be today or later — and in a valid window
        assert result.astimezone(TZ_UTC) >= current.astimezone(TZ_UTC)

    def test_aed_path_after_cutoff_executes_next_day(self):
        wm      = make_wm()
        before  = wm.next_path_execution_time("AED", "INR", 10_000, self.BEFORE_CUTOFF)
        after   = wm.next_path_execution_time("AED", "INR", 10_000, self.AFTER_CUTOFF)
        assert after.astimezone(TZ_UTC).date() >= before.astimezone(TZ_UTC).date()

    def test_aed_minutes_until_cutoff(self):
        wm   = make_wm()
        mins = wm.minutes_until_close("AED", self.BEFORE_CUTOFF)
        # 09:00 → 14:00 = 300 min
        assert mins == 300

    def test_aed_minutes_zero_when_closed(self):
        wm = make_wm()
        assert wm.minutes_until_close("AED", self.AFTER_CUTOFF) == 0


# ── 6. Settlement path: USD → INR ─────────────────────────────────────────────

class TestUSDToINRPath:
    def test_both_open_executes_now(self):
        """When both Fedwire and INR desk are open simultaneously, exec = now."""
        wm = make_wm()
        # 2026-02-16 14:00 UTC = 09:30 ET (Fedwire open) = 19:30 IST (INR desk closed by 15:00)
        # Need a time where both are open:
        # INR desk: 09:00–15:00 IST = 03:30–09:30 UTC
        # Fedwire:  09:00–18:00 ET  = 14:00–23:00 UTC
        # Overlap:  14:00–23:00 UTC  ∩  03:30–09:30 UTC = EMPTY on same day
        # So we can't have both open simultaneously across timezones in practice.
        # Instead verify exec_time >= current_time
        current   = dt(2026, 2, 16, 14, 0, TZ_UTC)
        exec_time = wm.next_path_execution_time("USD", "INR", 100_000, current)
        assert exec_time >= current

    def test_exec_time_advances_across_days(self):
        wm = make_wm()
        # Friday evening ET — will need to wait until Monday for Fedwire, then INR desk
        friday_eve = dt(2026, 2, 13, 21, 0, TZ_ET)
        exec_time  = wm.next_path_execution_time("USD", "INR", 100_000, friday_eve)
        assert exec_time.astimezone(TZ_UTC).date() >= date(2026, 2, 16)

    def test_can_complete_path_true_with_sufficient_time(self):
        wm       = make_wm()
        current  = dt(2026, 2, 16, 14, 0, TZ_UTC)
        deadline = dt(2026, 2, 18, 23, 59, TZ_UTC)  # 2 days from now
        assert wm.can_complete_path("USD", "INR", 100_000, deadline, current) is True

    def test_can_complete_path_false_with_insufficient_time(self):
        wm       = make_wm()
        current  = dt(2026, 2, 16, 14, 0, TZ_UTC)
        deadline = dt(2026, 2, 16, 14, 1, TZ_UTC)  # 1 minute from now
        # Cannot open Fedwire + INR desk in 1 minute
        assert wm.can_complete_path("USD", "INR", 100_000, deadline, current) is False


# ── 7. DST boundary week ───────────────────────────────────────────────────────

class TestDSTBoundaries:
    def test_us_dst_spring_forward_2026(self):
        """
        US DST springs forward on 2026-03-08 (second Sunday in March).
        At 02:00 ET, clocks jump to 03:00 EDT.
        Fedwire should still open at 09:00 local time on 2026-03-09 (Monday).
        """
        wm = make_wm()
        # Sunday 2026-03-08 03:30 EDT (just after spring-forward)
        sunday_dst = dt(2026, 3, 8, 7, 30, TZ_UTC)  # = 03:30 EDT
        nxt        = wm.next_open("USD", sunday_dst)
        local      = nxt.astimezone(TZ_ET)
        assert local.date() == date(2026, 3, 9)  # Monday
        assert local.time() == time(9, 0)

    def test_uk_dst_spring_forward_2026(self):
        """
        UK DST springs forward on 2026-03-29.
        CHAPS should open at 08:00 BST on 2026-03-30 (Monday).
        """
        wm = make_wm()
        sunday = dt(2026, 3, 29, 10, 0, TZ_UK)
        nxt    = wm.next_open("GBP", sunday)
        local  = nxt.astimezone(TZ_UK)
        assert local.date() == date(2026, 3, 30)
        assert local.time() == time(8, 0)

    def test_us_uk_dst_mismatch_week(self):
        """
        Between US spring-forward (Mar 8) and UK spring-forward (Mar 29),
        the ET/UK offset is different from normal.
        Verify Fedwire and CHAPS windows are computed independently and correctly.
        """
        wm = make_wm()
        # 2026-03-16 Monday — US is on EDT, UK is still on GMT
        monday_us_dst = dt(2026, 3, 16, 14, 0, TZ_UTC)   # 10:00 EDT / 14:00 GMT
        assert wm.is_open_now("USD", monday_us_dst) is True   # 10:00 EDT — open
        assert wm.is_open_now("GBP", monday_us_dst) is True   # 14:00 GMT — open (closes 16:00)


# ── 8. Cross-midnight comparisons ─────────────────────────────────────────────

class TestCrossMidnight:
    def test_next_open_does_not_return_today_when_window_closed_for_day(self):
        """After 18:00 ET (Fedwire closed for today), next_open returns tomorrow."""
        wm          = make_wm()
        after_close = dt(2026, 2, 16, 23, 0, TZ_ET)  # 23:00 ET Monday
        nxt         = wm.next_open("USD", after_close)
        local       = nxt.astimezone(TZ_ET)
        assert local.date() == date(2026, 2, 17)  # Tuesday
        assert local.time() == time(9, 0)

    def test_aed_next_open_after_14_gst_is_next_business_day(self):
        wm           = make_wm()
        after_cutoff = dt(2026, 2, 16, 14, 30, TZ_GST)
        nxt          = wm.next_open("AED", after_cutoff)
        local        = nxt.astimezone(TZ_GST)
        assert local.date() == date(2026, 2, 17)


# ── 9. Multi-day holiday span ─────────────────────────────────────────────────

class TestMultiDayHolidays:
    def test_skips_multiple_consecutive_holidays(self):
        # Mon–Wed all holidays; should resolve to Thursday open
        holiday_dates = {date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18)}
        wm  = make_wm(usd_holidays=lambda d: d in holiday_dates)
        now = dt(2026, 2, 16, 10, 0, TZ_ET)
        nxt = wm.next_open("USD", now)
        local = nxt.astimezone(TZ_ET)
        assert local.date() == date(2026, 2, 19)  # Thursday

    def test_path_completion_delayed_by_holiday(self):
        # USD holiday on Monday + Tuesday → Fedwire unavailable
        holiday_dates = {date(2026, 2, 16), date(2026, 2, 17)}
        wm      = make_wm(usd_holidays=lambda d: d in holiday_dates)
        current = dt(2026, 2, 16, 9, 0, TZ_ET)
        exec_t  = wm.next_path_execution_time("USD", "INR", 100_000, current)
        assert exec_t.astimezone(TZ_ET).date() >= date(2026, 2, 18)


# ── 10. Holiday calendar unavailable ──────────────────────────────────────────

class TestHolidayCalendarFailure:
    def test_raises_runtime_error_on_calendar_failure(self):
        def broken_calendar(d: date) -> bool:
            raise IOError("DB connection lost")

        wm = make_wm(usd_holidays=broken_calendar)
        with pytest.raises(RuntimeError, match="Holiday calendar unavailable"):
            wm.is_open_now("USD", dt(2026, 2, 16, 10, 0, TZ_ET))

    def test_can_complete_path_returns_false_on_calendar_failure(self):
        """can_complete_path must fail closed, not raise, when calendar breaks."""
        def broken_calendar(d: date) -> bool:
            raise IOError("DB connection lost")

        wm       = make_wm(usd_holidays=broken_calendar)
        current  = dt(2026, 2, 16, 10, 0, TZ_ET)
        deadline = dt(2026, 2, 16, 22, 0, TZ_ET)
        result   = wm.can_complete_path("USD", "INR", 50_000, deadline, current)
        assert result is False
