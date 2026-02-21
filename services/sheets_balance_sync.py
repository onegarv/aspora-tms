"""
sheets_balance_sync — pull live nostro balances from a publicly-shareable Google Sheet.

The finance team maintains daily nostro balances in a Google Sheet; the four live
balance rows (USD, GBP, EUR, AED) live at rows 185-188.  Because the sheet is shared
with "Anyone with link can view", we can download a CSV export without any OAuth flow.

CSV export URL pattern:
    https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}

Row layout (each row = one currency):
  Column B (index 1) — currency code, e.g. "USD"
  Rightmost non-empty column — the most-recent balance, e.g. "1.03M"

Value format: abbreviated millions — "1.03M" → 1_030_000, ".5M" → 500_000.
Plain numeric strings (no M suffix) are also accepted.
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from decimal import Decimal
from typing import Sequence

log = logging.getLogger("tms.sheets_balance_sync")

# ── Fallback balances used when the sheet is unreachable ──────────────────────

_FALLBACK_BALANCES: dict[str, Decimal] = {
    "USD": Decimal("10_000_000"),
    "GBP": Decimal("5_000_000"),
    "EUR": Decimal("2_000_000"),
    "AED": Decimal("2_000_000"),
}

_SHEET_BASE_URL = (
    "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
)


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_nostro_balances(
    sheet_id: str,
    gid: str,
    row_numbers: Sequence[int],
) -> dict[str, Decimal]:
    """
    Download CSV from Google Sheets and return {currency: Decimal balance}.

    Parameters
    ----------
    sheet_id:    Google Sheets document ID
    gid:         Sheet (tab) GID
    row_numbers: 1-indexed row numbers to read, e.g. [185, 186, 187, 188]

    Returns the parsed balances, or the hardcoded fallback dict if the sheet
    is unreachable or the data cannot be parsed.
    """
    url = _SHEET_BASE_URL.format(sheet_id=sheet_id, gid=gid)
    try:
        rows = _download_csv(url)
    except Exception as exc:
        log.warning(
            "sheets_balance_sync: could not download CSV, using fallback balances",
            extra={"url": url, "error": str(exc)},
        )
        return dict(_FALLBACK_BALANCES)

    try:
        balances = _parse_rows(rows, row_numbers)
    except Exception as exc:
        log.warning(
            "sheets_balance_sync: could not parse rows, using fallback balances",
            extra={"error": str(exc)},
        )
        return dict(_FALLBACK_BALANCES)

    if not balances:
        log.warning(
            "sheets_balance_sync: no balances parsed from sheet, using fallback"
        )
        return dict(_FALLBACK_BALANCES)

    log.info(
        "sheets_balance_sync: loaded balances",
        extra={"currencies": list(balances.keys())},
    )
    return balances


# ── Internals ─────────────────────────────────────────────────────────────────


def _download_csv(url: str) -> list[list[str]]:
    """Fetch URL and parse as CSV.  Raises on HTTP / network errors."""
    req = urllib.request.Request(url, headers={"User-Agent": "aspora-tms/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return list(csv.reader(io.StringIO(raw)))


def _parse_rows(
    all_rows: list[list[str]],
    row_numbers: Sequence[int],
) -> dict[str, Decimal]:
    """
    Extract balances from the specified 1-indexed row numbers.

    For each row:
      - currency = row[1] (column B), stripped and uppercased
      - balance  = rightmost non-empty cell, parsed via _parse_value()
    """
    result: dict[str, Decimal] = {}
    for row_num in row_numbers:
        idx = row_num - 1          # convert to 0-indexed
        if idx < 0 or idx >= len(all_rows):
            log.warning(
                "sheets_balance_sync: row %d out of range (sheet has %d rows)",
                row_num, len(all_rows),
            )
            continue

        row = all_rows[idx]
        if len(row) < 2:
            log.warning("sheets_balance_sync: row %d too short (%d cols)", row_num, len(row))
            continue

        currency = row[1].strip().upper()
        if not currency:
            log.warning("sheets_balance_sync: row %d has empty currency column", row_num)
            continue

        # Find rightmost non-empty cell
        raw_value: str | None = None
        for cell in reversed(row):
            stripped = cell.strip()
            if stripped:
                raw_value = stripped
                break

        if raw_value is None:
            log.warning("sheets_balance_sync: row %d has no non-empty values", row_num)
            continue

        try:
            result[currency] = _parse_value(raw_value)
        except (ValueError, ArithmeticError) as exc:
            log.warning(
                "sheets_balance_sync: could not parse '%s' for %s: %s",
                raw_value, currency, exc,
            )

    return result


def _parse_value(raw: str) -> Decimal:
    """
    Parse an abbreviated balance string into a Decimal.

    Examples:
        "1.03M"  → Decimal("1030000")
        ".5M"    → Decimal("500000")
        "-4.2M"  → Decimal("-4200000")
        "500000" → Decimal("500000")
        "1,500,000" → Decimal("1500000")   (commas stripped)
    """
    # Strip whitespace, commas (thousands separators), and currency symbols
    cleaned = raw.strip().replace(",", "").replace("$", "").replace("£", "").replace("€", "")

    if cleaned.upper().endswith("M"):
        numeric_part = cleaned[:-1]
        return Decimal(numeric_part) * Decimal("1000000")

    if cleaned.upper().endswith("K"):
        numeric_part = cleaned[:-1]
        return Decimal(numeric_part) * Decimal("1000")

    return Decimal(cleaned)
