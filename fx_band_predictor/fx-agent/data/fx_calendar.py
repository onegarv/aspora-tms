"""
Calendar System — Single Source of Truth for All Calendar Events

Indian public holidays, US public holidays, RBI MPC meetings, Fed FOMC meetings.
Provides functions to determine trading days, calendar context for predictions,
and upcoming risk events.

Used by:
    - agents/fx_prediction_agent.py (weekly forecast)
    - api/main.py (/predict/weekly endpoint)
"""

from datetime import date, timedelta
from typing import Optional

# ===========================================================================
# INDIAN PUBLIC HOLIDAYS (NSE closed → no FX trading)
# ===========================================================================

INDIA_HOLIDAYS_2025 = {
    date(2025, 1, 26): "Republic Day",
    date(2025, 2, 26): "Maha Shivaratri",
    date(2025, 3, 14): "Holi",
    date(2025, 3, 31): "Eid-ul-Fitr",
    date(2025, 4, 10): "Shri Mahavir Jayanti",
    date(2025, 4, 14): "Dr Ambedkar Jayanti",
    date(2025, 4, 18): "Good Friday",
    date(2025, 5, 1): "Maharashtra Day / Labour Day",
    date(2025, 8, 15): "Independence Day",
    date(2025, 8, 16): "Janmashtami",
    date(2025, 10, 2): "Gandhi Jayanti / Dussehra",
    date(2025, 10, 20): "Diwali (Laxmi Puja)",
    date(2025, 11, 5): "Guru Nanak Jayanti",
    date(2025, 12, 25): "Christmas",
}

INDIA_HOLIDAYS_2026 = {
    date(2026, 1, 26): "Republic Day",
    date(2026, 3, 17): "Holi",
    date(2026, 4, 2): "Ram Navami",
    date(2026, 4, 14): "Dr Ambedkar Jayanti",
    date(2026, 4, 15): "Good Friday",
    date(2026, 5, 1): "Maharashtra Day / Labour Day",
    date(2026, 8, 15): "Independence Day",
    date(2026, 8, 27): "Janmashtami",
    date(2026, 10, 2): "Gandhi Jayanti",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 1): "Diwali (Laxmi Puja)",
    date(2026, 11, 5): "Guru Nanak Jayanti",
    date(2026, 12, 25): "Christmas",
}

INDIA_HOLIDAYS = {**INDIA_HOLIDAYS_2025, **INDIA_HOLIDAYS_2026}

# ===========================================================================
# US PUBLIC HOLIDAYS (thin liquidity → skip predictions)
# ===========================================================================

US_HOLIDAYS_2025 = {
    date(2025, 1, 1): "New Year's Day",
    date(2025, 1, 20): "MLK Day",
    date(2025, 2, 17): "Presidents' Day",
    date(2025, 5, 26): "Memorial Day",
    date(2025, 7, 4): "Independence Day",
    date(2025, 9, 1): "Labor Day",
    date(2025, 11, 27): "Thanksgiving",
    date(2025, 12, 25): "Christmas",
}

US_HOLIDAYS_2026 = {
    date(2026, 1, 1): "New Year's Day",
    date(2026, 1, 19): "MLK Day",
    date(2026, 2, 16): "Presidents' Day",
    date(2026, 5, 25): "Memorial Day",
    date(2026, 7, 4): "Independence Day (observed Jul 3)",
    date(2026, 9, 7): "Labor Day",
    date(2026, 11, 26): "Thanksgiving",
    date(2026, 12, 25): "Christmas",
}

US_HOLIDAYS = {**US_HOLIDAYS_2025, **US_HOLIDAYS_2026}

# ===========================================================================
# RBI MONETARY POLICY COMMITTEE MEETINGS
# Dates are the full meeting range [start, end].
# Announcement (rate decision) is on the LAST day.
# ===========================================================================

RBI_MEETINGS = [
    # 2025
    (date(2025, 2, 5), date(2025, 2, 7)),
    (date(2025, 4, 7), date(2025, 4, 9)),
    (date(2025, 6, 4), date(2025, 6, 6)),
    (date(2025, 8, 6), date(2025, 8, 8)),
    (date(2025, 10, 6), date(2025, 10, 8)),
    (date(2025, 12, 3), date(2025, 12, 5)),
    # 2026
    (date(2026, 2, 5), date(2026, 2, 7)),
    (date(2026, 4, 7), date(2026, 4, 9)),
    (date(2026, 6, 9), date(2026, 6, 11)),
    (date(2026, 8, 4), date(2026, 8, 6)),
    (date(2026, 10, 6), date(2026, 10, 8)),
    (date(2026, 12, 8), date(2026, 12, 10)),
]

# Announcement days (last day of each meeting)
RBI_ANNOUNCEMENT_DATES = [end for _, end in RBI_MEETINGS]

# ===========================================================================
# US FEDERAL RESERVE FOMC MEETINGS
# Dates are the full meeting range [start, end].
# Announcement is on the LAST day.
# ===========================================================================

FED_MEETINGS = [
    # 2025
    (date(2025, 1, 28), date(2025, 1, 29)),
    (date(2025, 3, 18), date(2025, 3, 19)),
    (date(2025, 5, 6), date(2025, 5, 7)),
    (date(2025, 6, 17), date(2025, 6, 18)),
    (date(2025, 7, 29), date(2025, 7, 30)),
    (date(2025, 9, 16), date(2025, 9, 17)),
    (date(2025, 10, 28), date(2025, 10, 29)),
    (date(2025, 12, 16), date(2025, 12, 17)),
    # 2026
    (date(2026, 1, 28), date(2026, 1, 29)),
    (date(2026, 3, 18), date(2026, 3, 19)),
    (date(2026, 5, 6), date(2026, 5, 7)),
    (date(2026, 6, 17), date(2026, 6, 18)),
    (date(2026, 7, 29), date(2026, 7, 30)),
    (date(2026, 9, 16), date(2026, 9, 17)),
    (date(2026, 10, 28), date(2026, 10, 29)),
    (date(2026, 12, 9), date(2026, 12, 10)),
]

# Announcement days (last day of each meeting)
FED_ANNOUNCEMENT_DATES = [end for _, end in FED_MEETINGS]


# ===========================================================================
# HELPER: days to next event
# ===========================================================================

def _days_to_next(current: date, event_dates: list[date]) -> int:
    """Days until the next event date. Returns 999 if no upcoming event known."""
    future = [d for d in event_dates if d >= current]
    if future:
        return (future[0] - current).days
    return 999


def _is_within_meeting(current: date, meetings: list[tuple[date, date]]) -> bool:
    """Check if current date falls within any meeting range."""
    return any(start <= current <= end for start, end in meetings)


# ===========================================================================
# PUBLIC API
# ===========================================================================

def is_trading_day(d: date) -> bool:
    """
    Returns True if d is a trading day for USD/INR.
    False if: weekend OR Indian public holiday OR US public holiday.
    """
    # Weekend
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # Indian holiday (NSE closed = no FX trading)
    if d in INDIA_HOLIDAYS:
        return False
    # US holiday (thin liquidity)
    if d in US_HOLIDAYS:
        return False
    return True


def get_calendar_context(d: date) -> dict:
    """
    Returns full calendar context for a given date.
    Used by the weekly forecast to provide day-level detail.
    """
    is_weekend = d.weekday() >= 5
    india_holiday = INDIA_HOLIDAYS.get(d)
    us_holiday = US_HOLIDAYS.get(d)
    trading = is_trading_day(d)

    # Build skip reason
    skip_reason = None
    if not trading:
        reasons = []
        if is_weekend:
            reasons.append(f"Weekend — {d.strftime('%A')}")
        if india_holiday:
            reasons.append(f"Indian holiday: {india_holiday}")
        if us_holiday:
            reasons.append(f"US holiday: {us_holiday}")
        skip_reason = " | ".join(reasons)

    # RBI context
    days_to_rbi = _days_to_next(d, RBI_ANNOUNCEMENT_DATES)
    is_rbi_week = days_to_rbi <= 3
    is_rbi_day = d in RBI_ANNOUNCEMENT_DATES
    is_rbi_meeting = _is_within_meeting(d, RBI_MEETINGS)

    # Fed context
    days_to_fed = _days_to_next(d, FED_ANNOUNCEMENT_DATES)
    is_fed_week = days_to_fed <= 3
    is_fed_day = d in FED_ANNOUNCEMENT_DATES
    is_fed_meeting = _is_within_meeting(d, FED_MEETINGS)

    # Month-end / month-start
    # Days to month end
    if d.month == 12:
        next_month_first = date(d.year + 1, 1, 1)
    else:
        next_month_first = date(d.year, d.month + 1, 1)
    days_to_month_end = (next_month_first - d).days
    is_month_end = days_to_month_end <= 3
    is_month_start = d.day <= 3

    # Build special_note — human-readable summary of what makes this day interesting
    special_note = _build_special_note(
        d, india_holiday, us_holiday, is_rbi_day, is_rbi_week, is_rbi_meeting,
        is_fed_day, is_fed_week, is_fed_meeting, days_to_rbi, days_to_fed,
        is_month_end, is_month_start,
    )

    return {
        "is_trading_day": trading,
        "skip_reason": skip_reason,
        "is_india_holiday": india_holiday is not None,
        "india_holiday_name": india_holiday,
        "is_us_holiday": us_holiday is not None,
        "us_holiday_name": us_holiday,
        "is_weekend": is_weekend,
        "days_to_next_rbi": days_to_rbi,
        "days_to_next_fed": days_to_fed,
        "is_rbi_week": is_rbi_week,
        "is_rbi_day": is_rbi_day,
        "is_fed_week": is_fed_week,
        "is_fed_day": is_fed_day,
        "days_to_month_end": days_to_month_end,
        "is_month_end": is_month_end,
        "is_month_start": is_month_start,
        "day_of_week": d.weekday(),
        "day_name": d.strftime("%A"),
        "special_note": special_note,
    }


def _build_special_note(
    d: date, india_holiday, us_holiday,
    is_rbi_day, is_rbi_week, is_rbi_meeting,
    is_fed_day, is_fed_week, is_fed_meeting,
    days_to_rbi, days_to_fed,
    is_month_end, is_month_start,
) -> Optional[str]:
    """Build a human-readable special note for a date."""
    notes = []

    if india_holiday:
        notes.append(f"{india_holiday} — Indian markets closed")
    if us_holiday:
        notes.append(f"{us_holiday} — US markets closed, thin FX liquidity")

    if is_rbi_day:
        notes.append("RBI rate decision day — extreme volatility expected")
    elif is_rbi_meeting:
        notes.append("RBI MPC meeting in progress — market on standby")
    elif is_rbi_week and days_to_rbi > 0:
        notes.append(f"RBI meeting in {days_to_rbi} day{'s' if days_to_rbi > 1 else ''} — elevated uncertainty")

    if is_fed_day:
        notes.append("Fed rate decision day — dollar volatility likely to spike")
    elif is_fed_meeting:
        notes.append("FOMC meeting in progress — dollar on edge")
    elif is_fed_week and days_to_fed > 0:
        notes.append(f"Fed meeting in {days_to_fed} day{'s' if days_to_fed > 1 else ''} — dollar volatility likely")

    if is_month_end:
        notes.append("Month-end: expect elevated dollar demand from Indian importers")
    if is_month_start:
        notes.append("Month-start: post month-end — reduced importer demand")

    if d.weekday() == 4:  # Friday
        notes.append("Friday — end-of-week positioning and profit-taking")
    if d.weekday() == 0:  # Monday
        if not notes:  # don't pile on if already busy
            notes.append("Monday — potential weekend gap")

    if not notes:
        # Check if anything interesting is approaching
        if 4 <= days_to_rbi <= 7:
            notes.append(f"RBI meeting approaching in {days_to_rbi} days")
        elif 4 <= days_to_fed <= 7:
            notes.append(f"Fed meeting approaching in {days_to_fed} days")
        else:
            notes.append("Clean day: no major events, technical signals most reliable")

    return " | ".join(notes)


def get_next_n_trading_days(from_date: date, n: int = 7) -> list[date]:
    """
    Returns the next N trading days starting from from_date (exclusive).
    Skips weekends and all holidays.
    """
    result = []
    current = from_date
    # Safety limit to prevent infinite loop
    max_iter = n * 5
    iterations = 0
    while len(result) < n and iterations < max_iter:
        current = current + timedelta(days=1)
        iterations += 1
        if is_trading_day(current):
            result.append(current)
    return result


def get_next_7_calendar_days(from_date: date) -> list[date]:
    """
    Returns the next 7 calendar days starting from from_date (inclusive of next day).
    Includes weekends and holidays — the frontend needs the full calendar.
    """
    return [from_date + timedelta(days=i) for i in range(1, 8)]


def get_week_risk_events(from_date: date, days: int = 7) -> list[dict]:
    """
    Returns any major risk events in the next `days` calendar days.
    Events: RBI meetings, Fed meetings, Indian holidays, US holidays.
    """
    events = []
    for i in range(1, days + 1):
        d = from_date + timedelta(days=i)

        if d in RBI_ANNOUNCEMENT_DATES:
            events.append({"date": d.isoformat(), "event": "RBI Rate Decision",
                           "impact": "HIGH"})
        elif _is_within_meeting(d, RBI_MEETINGS):
            events.append({"date": d.isoformat(), "event": "RBI MPC Meeting",
                           "impact": "MEDIUM"})

        if d in FED_ANNOUNCEMENT_DATES:
            events.append({"date": d.isoformat(), "event": "Fed Rate Decision",
                           "impact": "HIGH"})
        elif _is_within_meeting(d, FED_MEETINGS):
            events.append({"date": d.isoformat(), "event": "FOMC Meeting",
                           "impact": "MEDIUM"})

        india_h = INDIA_HOLIDAYS.get(d)
        if india_h:
            events.append({"date": d.isoformat(), "event": f"Indian Holiday: {india_h}",
                           "impact": "MEDIUM"})

        us_h = US_HOLIDAYS.get(d)
        if us_h:
            events.append({"date": d.isoformat(), "event": f"US Holiday: {us_h}",
                           "impact": "LOW"})

    return events


# ===========================================================================
# Entry point — standalone test
# ===========================================================================

if __name__ == "__main__":
    from datetime import datetime

    today = datetime.today().date()
    print(f"Calendar System Test — Today: {today} ({today.strftime('%A')})")
    print("=" * 70)

    # Test: is_trading_day for the next 14 days
    print("\nNext 14 calendar days:")
    print(f"  {'Date':<12s} {'Day':<10s} {'Trading?':<10s} {'Note'}")
    print("  " + "-" * 60)
    for i in range(14):
        d = today + timedelta(days=i)
        ctx = get_calendar_context(d)
        trading = "YES" if ctx["is_trading_day"] else "NO"
        note = ctx["skip_reason"] or ctx["special_note"] or ""
        print(f"  {d.isoformat():<12s} {ctx['day_name']:<10s} {trading:<10s} {note}")

    # Test: next 7 trading days
    print(f"\nNext 7 trading days from {today}:")
    trading_days = get_next_n_trading_days(today, 7)
    for i, d in enumerate(trading_days, 1):
        ctx = get_calendar_context(d)
        print(f"  Day {i}: {d.isoformat()} ({ctx['day_name']}) — {ctx['special_note']}")

    # Test: risk events
    print(f"\nRisk events in next 14 days:")
    events = get_week_risk_events(today, 14)
    if events:
        for e in events:
            print(f"  {e['date']} — {e['event']} [{e['impact']}]")
    else:
        print("  No major events")

    # Test: calendar context for specific dates
    print("\nCalendar context for specific dates:")
    test_dates = [
        date(2026, 1, 26),   # Republic Day
        date(2026, 3, 19),   # Fed meeting day
        date(2026, 4, 9),    # RBI meeting day
        date(2026, 4, 15),   # Good Friday
        date(2026, 11, 1),   # Diwali
        date(2026, 2, 28),   # Month end (Saturday)
        date(2026, 3, 1),    # Month start (Sunday)
        date(2026, 3, 2),    # Month start (Monday, trading day)
    ]
    for d in test_dates:
        ctx = get_calendar_context(d)
        print(f"\n  {d.isoformat()} ({ctx['day_name']}):")
        print(f"    Trading: {ctx['is_trading_day']}")
        if ctx['skip_reason']:
            print(f"    Skip: {ctx['skip_reason']}")
        print(f"    India holiday: {ctx['india_holiday_name']}")
        print(f"    US holiday: {ctx['us_holiday_name']}")
        print(f"    RBI: day={ctx['is_rbi_day']} week={ctx['is_rbi_week']} days_to={ctx['days_to_next_rbi']}")
        print(f"    Fed: day={ctx['is_fed_day']} week={ctx['is_fed_week']} days_to={ctx['days_to_next_fed']}")
        print(f"    Month: end={ctx['is_month_end']} start={ctx['is_month_start']} days_to_end={ctx['days_to_month_end']}")
        print(f"    Note: {ctx['special_note']}")

    # Test: holiday count
    print(f"\n2025 India holidays: {len(INDIA_HOLIDAYS_2025)}")
    print(f"2026 India holidays: {len(INDIA_HOLIDAYS_2026)}")
    print(f"2025 US holidays: {len(US_HOLIDAYS_2025)}")
    print(f"2026 US holidays: {len(US_HOLIDAYS_2026)}")
    print(f"RBI meetings: {len(RBI_MEETINGS)}")
    print(f"Fed meetings: {len(FED_MEETINGS)}")

    print("\n--- Calendar test complete ---")
