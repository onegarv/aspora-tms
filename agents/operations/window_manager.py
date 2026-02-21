"""
WindowManager — timezone-aware, deterministic banking window evaluation engine.

Design principles (from spec):
  - All datetimes MUST be timezone-aware; naive datetimes raise ValueError.
  - All comparisons are performed in UTC internally.
  - No implicit system time calls — current_time is always passed explicitly.
  - Fail closed: any missing dependency (holiday calendar, timezone) raises an exception;
    the system never silently assumes a window is open.
  - Deterministic: given the same inputs, always produces the same output.
  - No side effects.

Supported rails:
  USD  Fedwire        09:00–18:00 ET  (Mon–Fri, US Fed holidays)
  GBP  CHAPS          08:00–16:00 UK  (Mon–Fri, BoE holidays)
  EUR  SEPA Instant   24/7 if amount < 100_000 EUR; else SEPA batch (next TARGET2 day)
       All EUR is ultimately converted to USD — Fedwire window governs RDA usability.
  AED  UAE Banking    Cutoff 14:00 GST (Mon–Fri excl UAE holidays).
       After 14:00 → value date is next UAE business day.
       AED is converted to USD before funding — Fedwire window applies to USD leg.
  INR  Treasury Desk  09:00–15:00 IST (Mon–Fri, RBI holidays)

Settlement path rules:
  USD → INR  :  Fedwire open  AND  INR desk open
  GBP → INR  :  CHAPS open    AND  INR desk open
  EUR → INR  :  if amount < 100k → immediate; else add 1 business day
               then USD leg governed by Fedwire, INR governed by INR desk
  AED → INR  :  if before 14:00 GST + business day → same-day USD value
               else next UAE business day USD value
               then USD governed by Fedwire, INR governed by INR desk
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Optional, Set
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger("tms.ops.window_manager")

# ── Timezone constants ─────────────────────────────────────────────────────────
TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_ET  = ZoneInfo("America/New_York")
TZ_UK  = ZoneInfo("Europe/London")
TZ_CET = ZoneInfo("Europe/Berlin")
TZ_GST = ZoneInfo("Asia/Dubai")
TZ_UTC = ZoneInfo("UTC")

# EUR SEPA Instant threshold (spec: exclusive upper bound — at 100k switches to batch)
SEPA_INSTANT_MAX_EUR = 100_000.0

# AED banking cutoff in local GST
AED_CUTOFF_LOCAL = time(14, 0)

# Default SEPA batch delay in business days
SEPA_BATCH_DELAY_DAYS = 1

# Operational buffer subtracted from official close time (minutes)
DEFAULT_OPERATIONAL_BUFFER_MIN = 10


# ── Holiday calendar type ──────────────────────────────────────────────────────
# A callable that returns True if the given date is a holiday.
# Injected at construction time so tests can supply fakes.
HolidayCalendar = Callable[[date], bool]


def _raise_if_naive(dt: datetime, name: str = "datetime") -> None:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got naive datetime: {dt!r}")


# ── TransferWindow ─────────────────────────────────────────────────────────────

@dataclass
class TransferWindow:
    """
    Represents a single payment rail.

    official_open_time / official_close_time are in the rail's local timezone.
    Operational close = official_close - operational_buffer_minutes.
    """
    currency: str
    rail_name: str
    timezone: ZoneInfo
    official_open_time: time
    official_close_time: time
    holiday_calendar: HolidayCalendar
    weekend_days: Set[int] = field(default_factory=lambda: {5, 6})  # Mon=0 … Sun=6
    operational_buffer_minutes: int = DEFAULT_OPERATIONAL_BUFFER_MIN

    # ── Derived property ──────────────────────────────────────────────────

    @property
    def operational_close_time(self) -> time:
        """Official close minus the operational buffer."""
        dummy = datetime.combine(date.today(), self.official_close_time, tzinfo=TZ_UTC)
        adjusted = dummy - timedelta(minutes=self.operational_buffer_minutes)
        return adjusted.time()

    # ── Core methods ──────────────────────────────────────────────────────

    def is_open_now(self, current_time: datetime) -> bool:
        """
        Returns True if the rail is currently open for transfers.

        Raises:
            ValueError   — current_time is naive
            RuntimeError — holiday calendar not available
        """
        _raise_if_naive(current_time, "current_time")
        local = current_time.astimezone(self.timezone)

        if local.weekday() in self.weekend_days:
            return False

        try:
            if self.holiday_calendar(local.date()):
                return False
        except Exception as exc:
            raise RuntimeError(
                f"Holiday calendar unavailable for {self.currency}: {exc}"
            ) from exc

        return self.official_open_time <= local.time() < self.operational_close_time

    def minutes_until_close(self, current_time: datetime) -> int:
        """
        Returns minutes until operational close.
        Returns 0 if already closed or not open today.

        Raises:
            ValueError — current_time is naive
        """
        _raise_if_naive(current_time, "current_time")
        if not self.is_open_now(current_time):
            return 0

        local     = current_time.astimezone(self.timezone)
        close_dt  = datetime.combine(
            local.date(), self.operational_close_time, tzinfo=self.timezone
        )
        delta_sec = (close_dt - local).total_seconds()
        return max(0, int(delta_sec / 60))

    # ── IST conversion helpers ─────────────────────────────────────────────
    # Always compute dynamically — never hardcode IST equivalents, as they
    # shift by 1 hour twice a year with US/UK/EU DST transitions.

    def close_time_ist(self, for_date: date | None = None) -> time:
        """
        Return this rail's official close time converted to IST for the given date.
        Accounts for DST in the source timezone automatically.
        """
        d = for_date or date.today()
        close_dt = datetime.combine(d, self.official_close_time, tzinfo=self.timezone)
        return close_dt.astimezone(TZ_IST).time()

    def open_time_ist(self, for_date: date | None = None) -> time:
        """
        Return this rail's official open time converted to IST for the given date.
        Accounts for DST in the source timezone automatically.
        """
        d = for_date or date.today()
        open_dt = datetime.combine(d, self.official_open_time, tzinfo=self.timezone)
        return open_dt.astimezone(TZ_IST).time()

    def window_ist_summary(self, for_date: date | None = None) -> dict:
        """
        Full summary of this window in both local and IST times for the given date.
        Use this for dashboard display and alert messages — never hardcode IST equivalents.
        """
        d = for_date or date.today()
        return {
            "currency":    self.currency,
            "rail":        self.rail_name,
            "local_open":  self.official_open_time.isoformat(),
            "local_close": self.official_close_time.isoformat(),
            "local_tz":    str(self.timezone),
            "ist_open":    self.open_time_ist(d).isoformat(),
            "ist_close":   self.close_time_ist(d).isoformat(),
            "date":        d.isoformat(),
        }

    def next_open(self, current_time: datetime) -> datetime:
        """
        Returns the next datetime when this window will open (in the rail's tz).

        Raises:
            ValueError   — current_time is naive
            RuntimeError — holiday calendar unavailable
        """
        _raise_if_naive(current_time, "current_time")
        local = current_time.astimezone(self.timezone)

        # Search up to 14 days ahead to handle multi-day holiday spans
        for day_offset in range(14):
            candidate_date = local.date() + timedelta(days=day_offset)
            candidate_open = datetime.combine(
                candidate_date, self.official_open_time, tzinfo=self.timezone
            )

            if candidate_date.weekday() in self.weekend_days:
                continue

            try:
                if self.holiday_calendar(candidate_date):
                    continue
            except Exception as exc:
                raise RuntimeError(
                    f"Holiday calendar unavailable for {self.currency}: {exc}"
                ) from exc

            if candidate_open > local:
                return candidate_open

        raise RuntimeError(
            f"Could not find next open window for {self.currency} within 14 days"
        )

    def opens_before(self, deadline: datetime, current_time: datetime) -> bool:
        """
        Returns True if this window will open before `deadline`.

        Raises:
            ValueError   — either datetime is naive
            RuntimeError — holiday calendar unavailable
        """
        _raise_if_naive(current_time, "current_time")
        _raise_if_naive(deadline, "deadline")

        if self.is_open_now(current_time):
            return True

        try:
            nxt = self.next_open(current_time)
        except RuntimeError:
            return False  # fail closed

        # Compare in UTC
        return nxt.astimezone(TZ_UTC) < deadline.astimezone(TZ_UTC)


# ── WindowManager (public facade) ─────────────────────────────────────────────

class WindowManager:
    """
    Manages transfer windows for all TMS currencies.

    Exposes a high-level interface for:
      - Simple open/close queries
      - Settlement path feasibility
      - Next execution time computation
    """

    def __init__(
        self,
        usd_holidays: HolidayCalendar,
        gbp_holidays: HolidayCalendar,
        inr_holidays: HolidayCalendar,
        aed_holidays: HolidayCalendar,
        sepa_batch_delay_days: int = SEPA_BATCH_DELAY_DAYS,
        operational_buffer_min: int = DEFAULT_OPERATIONAL_BUFFER_MIN,
    ) -> None:
        self._sepa_batch_delay = sepa_batch_delay_days

        self._windows: dict[str, TransferWindow] = {
            "USD": TransferWindow(
                currency="USD", rail_name="fedwire",
                timezone=TZ_ET,
                official_open_time=time(9, 0),
                official_close_time=time(18, 0),
                holiday_calendar=usd_holidays,
                operational_buffer_minutes=operational_buffer_min,
            ),
            "GBP": TransferWindow(
                currency="GBP", rail_name="chaps",
                timezone=TZ_UK,
                official_open_time=time(8, 0),
                official_close_time=time(16, 0),
                holiday_calendar=gbp_holidays,
                operational_buffer_minutes=operational_buffer_min,
            ),
            "INR": TransferWindow(
                currency="INR", rail_name="bank_desk",
                timezone=TZ_IST,
                official_open_time=time(9, 0),
                official_close_time=time(15, 0),
                holiday_calendar=inr_holidays,
                operational_buffer_minutes=operational_buffer_min,
            ),
        }
        self._usd_holidays = usd_holidays
        self._aed_holidays = aed_holidays

    # ── Simple window queries ─────────────────────────────────────────────

    def is_open_now(self, currency: str, current_time: datetime) -> bool:
        """
        Is the rail for this currency currently open?

        EUR: always True for SEPA Instant — caller must check amount threshold.
        AED: True if before 14:00 GST on a UAE business day.

        Raises ValueError on naive current_time.
        Raises RuntimeError if holiday calendar unavailable.
        """
        _raise_if_naive(current_time, "current_time")

        if currency == "EUR":
            # SEPA Instant is 24/7
            return True

        if currency == "AED":
            return self._aed_is_open(current_time)

        window = self._get_window(currency)
        return window.is_open_now(current_time)

    def next_open(self, currency: str, current_time: datetime) -> datetime:
        """
        Returns next datetime when this currency's rail opens.

        Raises ValueError on naive current_time.
        """
        _raise_if_naive(current_time, "current_time")

        if currency == "EUR":
            # SEPA Instant never closes
            return current_time

        if currency == "AED":
            return self._aed_next_open(current_time)

        return self._get_window(currency).next_open(current_time)

    def opens_before(
        self, currency: str, deadline: datetime, current_time: datetime
    ) -> bool:
        """Will this currency's window open before `deadline`?"""
        _raise_if_naive(current_time, "current_time")
        _raise_if_naive(deadline, "deadline")

        if currency == "EUR":
            return True  # SEPA Instant always available

        if currency == "AED":
            nxt = self._aed_next_open(current_time)
            return nxt.astimezone(TZ_UTC) < deadline.astimezone(TZ_UTC)

        return self._get_window(currency).opens_before(deadline, current_time)

    # ── Settlement path evaluation ────────────────────────────────────────

    def can_complete_path(
        self,
        source_currency: str,
        target_currency: str,
        amount: float,
        deadline: datetime,
        current_time: datetime,
    ) -> bool:
        """
        Can we complete the full settlement path before `deadline`?

        Fail closed: any ambiguity returns False.

        Raises:
            ValueError   — naive datetimes
            RuntimeError — holiday calendar unavailable
        """
        _raise_if_naive(current_time, "current_time")
        _raise_if_naive(deadline, "deadline")

        try:
            execution_time = self.next_path_execution_time(
                source_currency, target_currency, amount, current_time
            )
            return execution_time.astimezone(TZ_UTC) < deadline.astimezone(TZ_UTC)
        except Exception as exc:
            logger.warning(
                "can_complete_path defaulting to False due to error",
                extra={"error": str(exc), "source": source_currency},
            )
            return False  # fail closed

    def next_path_execution_time(
        self,
        source_currency: str,
        target_currency: str,
        amount: float,
        current_time: datetime,
    ) -> datetime:
        """
        Returns the earliest datetime at which the full settlement path can execute.

        Path rules (all EUR → USD before INR; AED → USD → INR):
          USD → INR : max(fedwire_open, inr_desk_open)
          GBP → INR : max(chaps_open, inr_desk_open)
          EUR → INR : if amount < 100k → immediate source leg
                      else add sepa_batch_delay business days
                      then apply Fedwire + INR desk
          AED → INR : USD value date = today if before 14:00 GST + biz day, else next UAE biz day
                      then apply Fedwire + INR desk

        Raises:
            ValueError   — naive current_time
            NotImplementedError — unsupported currency pair
        """
        _raise_if_naive(current_time, "current_time")

        src = source_currency.upper()
        tgt = target_currency.upper()

        if tgt != "INR":
            raise NotImplementedError(
                f"Settlement path to {tgt} not implemented (only INR supported)"
            )

        if src == "USD":
            return self._usd_to_inr_execution(current_time)

        if src == "GBP":
            return self._gbp_to_inr_execution(current_time)

        if src == "EUR":
            return self._eur_to_inr_execution(amount, current_time)

        if src == "AED":
            return self._aed_to_inr_execution(current_time)

        raise NotImplementedError(f"Unsupported source currency: {src}")

    # ── Convenience ───────────────────────────────────────────────────────

    def get_window(self, currency: str) -> TransferWindow:
        """Return the TransferWindow for a currency. AED → USD window."""
        key = "USD" if currency.upper() == "AED" else currency.upper()
        return self._get_window(key)

    def get_rail(self, currency: str) -> str:
        return self.get_window(currency).rail_name

    def minutes_until_close(self, currency: str, current_time: datetime) -> int:
        """Minutes until operational close. 0 if closed or EUR (Instant)."""
        _raise_if_naive(current_time, "current_time")
        if currency in ("EUR",):
            return 0  # Never closes
        if currency == "AED":
            return self._aed_minutes_until_cutoff(current_time)
        return self._get_window(currency).minutes_until_close(current_time)

    # ── AED helpers ───────────────────────────────────────────────────────

    def _aed_is_open(self, current_time: datetime) -> bool:
        local = current_time.astimezone(TZ_GST)
        if local.weekday() in {5, 6}:   # UAE weekend = Sat/Sun
            return False
        try:
            if self._aed_holidays(local.date()):
                return False
        except Exception as exc:
            raise RuntimeError(f"AED holiday calendar unavailable: {exc}") from exc
        return local.time() < AED_CUTOFF_LOCAL

    def _aed_next_open(self, current_time: datetime) -> datetime:
        local = current_time.astimezone(TZ_GST)
        for offset in range(14):
            candidate_date = local.date() + timedelta(days=offset)
            candidate_open = datetime.combine(
                candidate_date, time(0, 0), tzinfo=TZ_GST
            )
            if candidate_date.weekday() in {5, 6}:
                continue
            try:
                if self._aed_holidays(candidate_date):
                    continue
            except Exception as exc:
                raise RuntimeError(f"AED holiday calendar unavailable: {exc}") from exc
            # The "open" period starts at 00:00 GST and closes at 14:00 GST
            cutoff_dt = datetime.combine(
                candidate_date, AED_CUTOFF_LOCAL, tzinfo=TZ_GST
            )
            if cutoff_dt > local:
                return candidate_open
        raise RuntimeError("Could not find next AED open window within 14 days")

    def _aed_minutes_until_cutoff(self, current_time: datetime) -> int:
        if not self._aed_is_open(current_time):
            return 0
        local    = current_time.astimezone(TZ_GST)
        cutoff   = datetime.combine(local.date(), AED_CUTOFF_LOCAL, tzinfo=TZ_GST)
        return max(0, int((cutoff - local).total_seconds() / 60))

    def _aed_usd_value_date(self, current_time: datetime) -> date:
        """
        The date on which USD funds will be available after AED conversion.
        If before 14:00 GST on a business day → today.
        Otherwise → next UAE business day.
        """
        local = current_time.astimezone(TZ_GST)
        if self._aed_is_open(current_time):
            return local.date()
        # Find next UAE business day
        candidate = local.date() + timedelta(days=1)
        for _ in range(14):
            if candidate.weekday() not in {5, 6} and not self._aed_holidays(candidate):
                return candidate
            candidate += timedelta(days=1)
        raise RuntimeError("Could not find next UAE business day within 14 days")

    # ── Settlement path helpers ───────────────────────────────────────────

    def _usd_to_inr_execution(self, current_time: datetime) -> datetime:
        usd_window = self._get_window("USD")
        inr_window = self._get_window("INR")
        usd_open   = usd_window.next_open(current_time) if not usd_window.is_open_now(current_time) else current_time
        inr_open   = inr_window.next_open(current_time) if not inr_window.is_open_now(current_time) else current_time
        # Both windows must be open simultaneously
        return max(
            usd_open.astimezone(TZ_UTC),
            inr_open.astimezone(TZ_UTC),
        )

    def _gbp_to_inr_execution(self, current_time: datetime) -> datetime:
        gbp_window = self._get_window("GBP")
        inr_window = self._get_window("INR")
        gbp_open   = gbp_window.next_open(current_time) if not gbp_window.is_open_now(current_time) else current_time
        inr_open   = inr_window.next_open(current_time) if not inr_window.is_open_now(current_time) else current_time
        return max(
            gbp_open.astimezone(TZ_UTC),
            inr_open.astimezone(TZ_UTC),
        )

    def _eur_to_inr_execution(self, amount: float, current_time: datetime) -> datetime:
        """
        EUR is converted to USD, then Fedwire + INR desk governs.

        SEPA Instant (amount < 100k): source leg is immediate.
        SEPA batch   (amount >= 100k): add sepa_batch_delay business days.
        """
        if amount < SEPA_INSTANT_MAX_EUR:
            # Immediate EUR → USD conversion; then standard USD → INR path
            usd_available_time = current_time
        else:
            # Batch: USD value date is today + sepa_batch_delay business days
            usd_available_time = self._add_usd_business_days(
                current_time, self._sepa_batch_delay
            )

        return self._usd_to_inr_execution(usd_available_time)

    def _aed_to_inr_execution(self, current_time: datetime) -> datetime:
        """AED → USD (value date depends on GST 14:00 cutoff) → INR."""
        usd_value_date = self._aed_usd_value_date(current_time)
        # USD funds available at start of Fedwire window on usd_value_date
        usd_available = datetime.combine(
            usd_value_date, time(9, 0), tzinfo=TZ_ET
        )
        # Ensure we don't go backwards in time
        usd_available = max(
            usd_available.astimezone(TZ_UTC),
            current_time.astimezone(TZ_UTC),
        )
        return self._usd_to_inr_execution(usd_available)

    def _add_usd_business_days(self, current_time: datetime, days: int) -> datetime:
        """Advance current_time by `days` US business days."""
        result = current_time
        added  = 0
        while added < days:
            result = result + timedelta(days=1)
            local  = result.astimezone(TZ_ET)
            if local.weekday() in {5, 6}:
                continue
            if self._usd_holidays(local.date()):
                continue
            added += 1
        return result

    # ── Live convenience methods ───────────────────────────────────────────
    # These call the core methods with datetime.now(timezone.utc) so callers
    # don't need to manage the clock themselves.  The core methods remain pure
    # and fully testable; only use these wrappers in production runtime code.

    def is_open_right_now(self, currency: str) -> bool:
        """Is this currency's window open right now?"""
        from datetime import timezone
        return self.is_open_now(currency, datetime.now(timezone.utc))

    def minutes_until_close_now(self, currency: str) -> int:
        """Minutes until operational close, measured from right now."""
        from datetime import timezone
        return self.minutes_until_close(currency, datetime.now(timezone.utc))

    def next_open_now(self, currency: str) -> datetime:
        """Next datetime this currency's window will open, from right now."""
        from datetime import timezone
        return self.next_open(currency, datetime.now(timezone.utc))

    def opens_before_now(self, currency: str, deadline: datetime) -> bool:
        """Will this window open before `deadline`, checking from right now?"""
        from datetime import timezone
        return self.opens_before(currency, deadline, datetime.now(timezone.utc))

    def can_complete_path_now(
        self,
        source_currency: str,
        target_currency: str,
        amount: float,
        deadline: datetime,
    ) -> bool:
        """Can the full settlement path complete before `deadline`, from right now?"""
        from datetime import timezone
        return self.can_complete_path(
            source_currency, target_currency, amount, deadline,
            datetime.now(timezone.utc),
        )

    def next_path_execution_time_now(
        self,
        source_currency: str,
        target_currency: str,
        amount: float,
    ) -> datetime:
        """Earliest datetime the settlement path can execute, from right now."""
        from datetime import timezone
        return self.next_path_execution_time(
            source_currency, target_currency, amount,
            datetime.now(timezone.utc),
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_window(self, currency: str) -> TransferWindow:
        window = self._windows.get(currency.upper())
        if window is None:
            raise ValueError(f"No transfer window configured for currency: {currency}")
        return window
