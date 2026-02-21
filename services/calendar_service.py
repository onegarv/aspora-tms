"""
CalendarService — three-layer, in-memory, synchronous holiday calendar.

Supported banking calendars:
    US_FEDWIRE  — US Federal Reserve / Fedwire
    UK_CHAPS    — UK CHAPS / Bank of England
    IN_RBI_FX   — India RBI FX desk
    AE_BANKING  — UAE banking (Central Bank of UAE)

Layer architecture:
    Layer 3 (Runtime)  — add_holiday() / remove_holiday() with audit trail
    Layer 2 (Overrides)— RBI_ADDITIONS/RBI_REMOVALS, UAE_LUNAR_OVERRIDES
    Layer 1 (Library)  — `holidays` Python library

Layer 3 wins over Layer 2, Layer 2 wins over Layer 1 for date collisions.
Strict mode: when strict=False (default), unconfirmed lunar holidays return
is_holiday=False. When strict=True, they return is_holiday=True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import holidays as hlib

logger = logging.getLogger("tms.services.calendar")

# ── Calendar type constants ────────────────────────────────────────────────────

US_FEDWIRE = "US_FEDWIRE"
UK_CHAPS   = "UK_CHAPS"
IN_RBI_FX  = "IN_RBI_FX"
AE_BANKING = "AE_BANKING"

ALL_CALENDARS = [US_FEDWIRE, UK_CHAPS, IN_RBI_FX, AE_BANKING]

# ── Timezone map ───────────────────────────────────────────────────────────────

CALENDAR_TIMEZONE: dict[str, str] = {
    US_FEDWIRE: "America/New_York",
    UK_CHAPS:   "Europe/London",
    IN_RBI_FX:  "Asia/Kolkata",
    AE_BANKING: "Asia/Dubai",
}

# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HolidayInfo:
    date: date
    name: str
    calendar: str
    kind: str           # "public", "lunar", "special", "banking"
    confirmed: bool
    source_layer: int   # 1, 2, or 3


@dataclass(frozen=True)
class HolidayDecision:
    is_holiday: bool
    info: HolidayInfo | None
    reason: str


# ── Audit entry (mutable, not frozen) ─────────────────────────────────────────

@dataclass
class _AuditEntry:
    action: str          # "add" or "remove"
    d: date
    calendar: str
    name: str | None
    user: str
    timestamp: datetime


# ── Layer 2: RBI override data ─────────────────────────────────────────────────

# Tuples: (date, name, kind, confirmed)
RBI_ADDITIONS: list[tuple[date, str, str, bool]] = [
    # 2025 — confirmed
    (date(2025,  1, 14), "Makar Sankranti",              "public", True),
    (date(2025,  1, 26), "Republic Day",                 "public", True),
    (date(2025,  2, 26), "Maha Shivaratri",              "public", True),
    (date(2025,  3, 14), "Holi",                         "public", True),
    (date(2025,  3, 31), "Id-ul-Fitr",                   "lunar",  True),
    (date(2025,  4, 10), "Ram Navami",                   "public", True),
    (date(2025,  4, 14), "Dr. Ambedkar Jayanti",         "public", True),
    (date(2025,  4, 18), "Good Friday",                  "public", True),
    (date(2025,  5, 12), "Buddha Purnima",               "public", True),
    (date(2025,  6,  7), "Id-ul-Zuha (Bakrid)",          "lunar",  True),
    (date(2025,  7,  6), "Muharram",                     "lunar",  True),
    (date(2025,  8, 15), "Independence Day",             "public", True),
    (date(2025,  8, 16), "Parsi New Year",               "public", True),
    (date(2025,  8, 27), "Ganesh Chaturthi",             "public", True),
    (date(2025, 10,  2), "Gandhi Jayanti / Mahatma Gandhi", "public", True),
    (date(2025, 10,  2), "Dussehra",                     "public", True),
    (date(2025, 10, 20), "Diwali Laxmi Puja",            "public", True),
    (date(2025, 10, 21), "Diwali Balipratipada",         "public", True),
    (date(2025, 11,  5), "Guru Nanak Jayanti",           "public", True),
    (date(2025, 12, 25), "Christmas Day",                "public", True),
    # 2026 — unconfirmed lunar
    (date(2026,  1, 14), "Makar Sankranti*",             "lunar",  False),
    (date(2026,  2, 15), "Maha Shivaratri*",             "lunar",  False),
    (date(2026,  3,  4), "Holi*",                        "lunar",  False),
]

# Dates that RBI removes from the generic India library calendar.
# Structure: set of date objects. Empty for now, but used in layer logic.
RBI_REMOVALS: set[date] = set()

# ── Layer 2: UAE lunar override data ──────────────────────────────────────────

# Tuples: (date, name, kind, confirmed)
UAE_LUNAR_OVERRIDES: list[tuple[date, str, str, bool]] = [
    # 2025 — observed (not yet officially confirmed, lunar-dependent)
    (date(2025,  3, 30), "Eid Al Fitr Day 1",   "lunar", False),
    (date(2025,  3, 31), "Eid Al Fitr Day 2",   "lunar", False),
    (date(2025,  4,  1), "Eid Al Fitr Day 3",   "lunar", False),
    (date(2025,  6,  6), "Arafat Day",           "lunar", False),
    (date(2025,  6,  7), "Eid Al Adha Day 1",   "lunar", False),
    (date(2025,  6,  8), "Eid Al Adha Day 2",   "lunar", False),
    (date(2025,  6,  9), "Eid Al Adha Day 3",   "lunar", False),
    (date(2025,  7, 27), "Islamic New Year",     "lunar", False),
    (date(2025,  9,  5), "Prophet's Birthday",   "lunar", False),
    # 2026 — unconfirmed
    (date(2026,  3, 19), "Eid Al Fitr Day 1*",  "lunar", False),
    (date(2026,  3, 20), "Eid Al Fitr Day 2*",  "lunar", False),
    (date(2026,  3, 21), "Eid Al Fitr Day 3*",  "lunar", False),
]


# ── Helper: build Layer 1 holiday sets ────────────────────────────────────────

def _build_layer1(calendar: str, year: int) -> dict[date, str]:
    """Return {date: name} from the `holidays` library for the given calendar and year."""
    if calendar == US_FEDWIRE:
        return dict(hlib.UnitedStates(years=year, observed=True))
    elif calendar == UK_CHAPS:
        return dict(hlib.UnitedKingdom(years=year, subdiv="England"))
    elif calendar == IN_RBI_FX:
        return dict(hlib.India(years=year))
    elif calendar == AE_BANKING:
        return dict(hlib.AE(years=year))
    else:
        raise ValueError(f"Unknown calendar: {calendar!r}")


# ── CalendarService ────────────────────────────────────────────────────────────

class CalendarService:
    """
    Synchronous, three-layer, in-memory banking calendar service.

    Thread-safety: not guaranteed for concurrent add_holiday/remove_holiday.
    Single-threaded ops-agent usage is the intended deployment model.
    """

    def __init__(self) -> None:
        # Layer 1 cache: (calendar, year) -> {date: name}
        self._l1_cache: dict[tuple[str, int], dict[date, str]] = {}

        # Layer 2 indexes (built once at init)
        self._rbi_additions: dict[date, tuple[str, str, bool]] = {}
        self._rbi_removals: set[date] = set(RBI_REMOVALS)
        self._uae_lunar: dict[date, tuple[str, str, bool]] = {}

        # Layer 3: runtime overrides
        # Additions: (date, calendar) -> HolidayInfo
        # Removals:  set of (date, calendar)
        self._l3_add: dict[tuple[date, str], HolidayInfo] = {}
        self._l3_remove: set[tuple[date, str]] = set()

        # Audit trail
        self._audit: list[_AuditEntry] = []

        self._init_layer2()

    # ── Layer 2 init ──────────────────────────────────────────────────────────

    def _init_layer2(self) -> None:
        for d, name, kind, confirmed in RBI_ADDITIONS:
            # Last write wins for same date (Dussehra and Gandhi Jayanti both on 2025-10-02)
            self._rbi_additions[d] = (name, kind, confirmed)

        for d, name, kind, confirmed in UAE_LUNAR_OVERRIDES:
            self._uae_lunar[d] = (name, kind, confirmed)

    # ── Layer 1 ───────────────────────────────────────────────────────────────

    def _layer1(self, d: date, calendar: str) -> HolidayInfo | None:
        key = (calendar, d.year)
        if key not in self._l1_cache:
            self._l1_cache[key] = _build_layer1(calendar, d.year)
        holidays_for_year = self._l1_cache[key]
        name = holidays_for_year.get(d)
        if name is None:
            return None
        return HolidayInfo(
            date=d,
            name=name,
            calendar=calendar,
            kind="public",
            confirmed=True,
            source_layer=1,
        )

    # ── Layer 2 ───────────────────────────────────────────────────────────────

    def _layer2(self, d: date, calendar: str) -> HolidayInfo | None:
        """
        Returns a HolidayInfo if Layer 2 has an entry, or a sentinel meaning
        'remove from Layer 1'.  Removals are signaled via kind="removal".
        """
        if calendar == IN_RBI_FX:
            if d in self._rbi_removals:
                # Signal that Layer 1 entry should be suppressed
                return HolidayInfo(
                    date=d,
                    name="__removed__",
                    calendar=calendar,
                    kind="removal",
                    confirmed=False,
                    source_layer=2,
                )
            if d in self._rbi_additions:
                name, kind, confirmed = self._rbi_additions[d]
                return HolidayInfo(
                    date=d, name=name, calendar=calendar,
                    kind=kind, confirmed=confirmed, source_layer=2,
                )
        elif calendar == AE_BANKING:
            if d in self._uae_lunar:
                name, kind, confirmed = self._uae_lunar[d]
                return HolidayInfo(
                    date=d, name=name, calendar=calendar,
                    kind=kind, confirmed=confirmed, source_layer=2,
                )
        return None

    # ── Weekend check ─────────────────────────────────────────────────────────

    def _is_weekend(self, d: date, calendar: str) -> bool:
        """
        Returns True if `d` falls on a non-business weekend day per calendar.
        AE_BANKING: Fri (4) and Sat (5) are weekend.
        All others: Sat (5) and Sun (6) are weekend.
        """
        wd = d.weekday()  # Mon=0 … Sun=6
        if calendar == AE_BANKING:
            return wd in {4, 5}
        return wd >= 5

    # ── Core lookup ───────────────────────────────────────────────────────────

    def explain_holiday(self, d: date, calendar: str, *, strict: bool = False) -> HolidayDecision:
        """
        Full three-layer lookup returning a HolidayDecision with audit trail.
        """
        if calendar not in ALL_CALENDARS:
            raise ValueError(f"Unknown calendar: {calendar!r}. Valid: {ALL_CALENDARS}")

        # ── Layer 3 check ────────────────────────────────────────────────────
        l3_key = (d, calendar)
        if l3_key in self._l3_remove:
            return HolidayDecision(
                is_holiday=False,
                info=None,
                reason="Layer 3 runtime removal",
            )
        if l3_key in self._l3_add:
            info = self._l3_add[l3_key]
            return HolidayDecision(
                is_holiday=True,
                info=info,
                reason="Layer 3 runtime addition",
            )

        # ── Layer 2 check ────────────────────────────────────────────────────
        l2 = self._layer2(d, calendar)
        if l2 is not None:
            if l2.kind == "removal":
                # Layer 2 removes Layer 1; not a holiday
                return HolidayDecision(
                    is_holiday=False,
                    info=None,
                    reason="Layer 2 RBI removal overrides Layer 1",
                )
            # Layer 2 has an override entry
            if not l2.confirmed and not strict:
                return HolidayDecision(
                    is_holiday=False,
                    info=l2,
                    reason="Layer 2 unconfirmed lunar — strict=False",
                )
            return HolidayDecision(
                is_holiday=True,
                info=l2,
                reason="Layer 2 override" if l2.confirmed else "Layer 2 unconfirmed lunar — strict=True",
            )

        # ── Layer 1 check ────────────────────────────────────────────────────
        l1 = self._layer1(d, calendar)
        if l1 is not None:
            return HolidayDecision(
                is_holiday=True,
                info=l1,
                reason="Layer 1 library holiday",
            )

        return HolidayDecision(
            is_holiday=False,
            info=None,
            reason="No holiday found in any layer",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_holiday(self, d: date, calendar: str, *, strict: bool = False) -> bool:
        return self.explain_holiday(d, calendar, strict=strict).is_holiday

    def is_business_day(self, d: date, calendar: str, *, strict: bool = False) -> bool:
        if calendar not in ALL_CALENDARS:
            raise ValueError(f"Unknown calendar: {calendar!r}. Valid: {ALL_CALENDARS}")
        if self._is_weekend(d, calendar):
            return False
        return not self.is_holiday(d, calendar, strict=strict)

    def next_business_day(self, d: date, calendar: str, *, strict: bool = False) -> date:
        candidate = d + timedelta(days=1)
        while not self.is_business_day(candidate, calendar, strict=strict):
            candidate += timedelta(days=1)
        return candidate

    def previous_business_day(self, d: date, calendar: str, *, strict: bool = False) -> date:
        candidate = d - timedelta(days=1)
        while not self.is_business_day(candidate, calendar, strict=strict):
            candidate -= timedelta(days=1)
        return candidate

    def consecutive_non_business_days(self, d: date, calendar: str, *, strict: bool = False) -> int:
        """
        Count how many consecutive non-business days start at (and include) `d`.
        If `d` is itself a business day, returns 0.
        """
        if self.is_business_day(d, calendar, strict=strict):
            return 0
        count = 0
        candidate = d
        while not self.is_business_day(candidate, calendar, strict=strict):
            count += 1
            candidate += timedelta(days=1)
        return count

    def is_day_before_holiday(self, d: date, calendar: str, *, strict: bool = False) -> bool:
        """
        True if tomorrow (d+1) is a non-business day (holiday or weekend).
        """
        tomorrow = d + timedelta(days=1)
        return not self.is_business_day(tomorrow, calendar, strict=strict)

    def is_day_after_holiday(self, d: date, calendar: str, *, strict: bool = False) -> bool:
        """
        True if yesterday (d-1) was a non-business day (holiday or weekend).
        """
        yesterday = d - timedelta(days=1)
        return not self.is_business_day(yesterday, calendar, strict=strict)

    def business_days_between(
        self, start: date, end: date, calendar: str, *, strict: bool = False
    ) -> int:
        """
        Count business days in the half-open interval [start, end).
        i.e. start is included, end is excluded.
        """
        if calendar not in ALL_CALENDARS:
            raise ValueError(f"Unknown calendar: {calendar!r}. Valid: {ALL_CALENDARS}")
        count = 0
        current = start
        while current < end:
            if self.is_business_day(current, calendar, strict=strict):
                count += 1
            current += timedelta(days=1)
        return count

    def upcoming_holidays(
        self,
        from_date: date,
        calendar: str,
        days: int = 30,
        *,
        strict: bool = False,
    ) -> list[HolidayInfo]:
        """
        Return sorted list of HolidayInfo for holidays in [from_date, from_date+days).
        Unconfirmed entries are excluded when strict=False.
        """
        if calendar not in ALL_CALENDARS:
            raise ValueError(f"Unknown calendar: {calendar!r}. Valid: {ALL_CALENDARS}")

        result: dict[date, HolidayInfo] = {}
        end = from_date + timedelta(days=days)
        current = from_date
        while current < end:
            decision = self.explain_holiday(current, calendar, strict=strict)
            if decision.is_holiday and decision.info is not None:
                # Deduplicate by date: explain_holiday already returns winning layer
                result[current] = decision.info
            current += timedelta(days=1)
        return sorted(result.values(), key=lambda h: h.date)

    def today(self, calendar: str, now_utc: datetime) -> date:
        """
        Convert a UTC datetime to the calendar-local date.
        `now_utc` must be timezone-aware (UTC).
        """
        if calendar not in ALL_CALENDARS:
            raise ValueError(f"Unknown calendar: {calendar!r}. Valid: {ALL_CALENDARS}")
        tz_name = CALENDAR_TIMEZONE[calendar]
        tz = ZoneInfo(tz_name)
        local_dt = now_utc.astimezone(tz)
        return local_dt.date()

    # ── Layer 3 management ────────────────────────────────────────────────────

    def add_holiday(
        self,
        d: date,
        calendar: str,
        name: str,
        kind: str = "special",
        *,
        user: str = "system",
    ) -> None:
        """
        Add a runtime holiday override (Layer 3).
        Removes any existing Layer 3 removal for the same (date, calendar).
        """
        if calendar not in ALL_CALENDARS:
            raise ValueError(f"Unknown calendar: {calendar!r}. Valid: {ALL_CALENDARS}")
        key = (d, calendar)
        info = HolidayInfo(
            date=d, name=name, calendar=calendar,
            kind=kind, confirmed=True, source_layer=3,
        )
        self._l3_add[key] = info
        self._l3_remove.discard(key)
        self._audit.append(_AuditEntry(
            action="add", d=d, calendar=calendar, name=name,
            user=user, timestamp=datetime.now(timezone.utc),
        ))
        logger.info("Layer 3 add_holiday: %s %s %r (user=%s)", d, calendar, name, user)

    def remove_holiday(
        self,
        d: date,
        calendar: str,
        *,
        user: str = "system",
    ) -> None:
        """
        Add a runtime removal override (Layer 3).
        Forces is_holiday=False for this (date, calendar), overriding Layers 1 & 2.
        """
        if calendar not in ALL_CALENDARS:
            raise ValueError(f"Unknown calendar: {calendar!r}. Valid: {ALL_CALENDARS}")
        key = (d, calendar)
        self._l3_remove.add(key)
        self._l3_add.pop(key, None)
        self._audit.append(_AuditEntry(
            action="remove", d=d, calendar=calendar, name=None,
            user=user, timestamp=datetime.now(timezone.utc),
        ))
        logger.info("Layer 3 remove_holiday: %s %s (user=%s)", d, calendar, user)

    def list_overrides(self, calendar: str | None = None) -> list[HolidayInfo]:
        """
        Return all Layer 3 additions currently in effect.
        Removals are not represented as HolidayInfo (they have no holiday metadata).
        Pass calendar=None to list all calendars.
        """
        result = []
        for (d, cal), info in self._l3_add.items():
            if calendar is None or cal == calendar:
                result.append(info)
        return sorted(result, key=lambda h: (h.calendar, h.date))

    def audit_log(self) -> list[_AuditEntry]:
        """Return the full audit trail of Layer 3 operations."""
        return list(self._audit)

    # ── DST transition detection ───────────────────────────────────────────────

    # Jurisdictions that observe DST and their IANA timezone identifiers.
    # IN and AE intentionally omitted — neither observes DST.
    DST_JURISDICTIONS: dict[str, str] = {
        "US": "America/New_York",
        "UK": "Europe/London",
        "EU": "Europe/Berlin",
    }

    def get_dst_transitions(self, start: date, end: date) -> list[dict]:
        """
        Return all DST transitions across tracked jurisdictions between
        `start` and `end` dates (inclusive).

        These are NOT bank holidays — payment rails are open. But the
        IST-equivalent window open/close times shift by the transition
        amount, which is an operational risk for anyone who has memorised
        those times.

        Returns a list of dicts with keys:
            date, jurisdiction, timezone, direction ("spring_forward" |
            "fall_back"), shift_minutes, new_utc_offset_hours,
            is_holiday (always False), operational_note
        """
        transitions: list[dict] = []

        for jurisdiction, tz_name in self.DST_JURISDICTIONS.items():
            tz = ZoneInfo(tz_name)
            current = start

            while current <= end:
                dt_noon      = datetime(current.year, current.month, current.day, 12, 0, tzinfo=tz)
                dt_noon_next = dt_noon + timedelta(days=1)

                offset_today = dt_noon.utcoffset()
                offset_next  = dt_noon_next.utcoffset()

                if offset_today != offset_next:
                    shift_min = int((offset_next - offset_today).total_seconds() / 60)  # type: ignore[operator]
                    direction = "spring_forward" if shift_min > 0 else "fall_back"
                    transitions.append({
                        "date":                  current.isoformat(),
                        "jurisdiction":          jurisdiction,
                        "timezone":              tz_name,
                        "direction":             direction,
                        "shift_minutes":         abs(shift_min),
                        "new_utc_offset_hours":  offset_next.total_seconds() / 3600,  # type: ignore[union-attr]
                        "type":                  "dst_transition",
                        "is_holiday":            False,
                        "operational_note": (
                            f"{jurisdiction} clocks "
                            f"{'advance' if shift_min > 0 else 'go back'} "
                            f"{abs(shift_min)} min. "
                            f"Transfer window IST equivalents shift."
                        ),
                    })

                current += timedelta(days=1)

        return transitions
