"""
Metabase REST client for Aspora TMS.

Fetches live FX rates and real TPV volumes at agent startup.
Requires Cloudflare WARP to be active (internal URL).

Data sources:
  - analytics_orders_master_data (Redshift, db=3) — historical TPV by corridor
  - XE USDINR Rates (question 9974)              — live USD/INR tick every 10 min
  - analytics_orders_master_data avg rate        — GBP/EUR/AED rates from recent trades
"""
from __future__ import annotations

import http.client
import json
import logging
import ssl
from datetime import datetime
from typing import Optional

logger = logging.getLogger("tms.data.metabase")

METABASE_HOST    = "metabase.internal.vance.tech"
METABASE_API_KEY = "mb_3M+ug7TVS+d/ihpNvXniQruXtr03M8hM0hvP7Lk7GwA="
REDSHIFT_DB_ID   = 3

Q_XE_USDINR = 9974   # XE live USD/INR rate, updated every 10 minutes

# Redshift queries are slow (~45 s) — use a generous read timeout
_TIMEOUT_FAST = 30    # card questions (cached by Metabase)
_TIMEOUT_SLOW = 120   # raw SQL on Redshift

_HEADERS = {
    "x-api-key": METABASE_API_KEY,
    "Content-Type": "application/json",
}


# ─── Low-level helpers ────────────────────────────────────────────────────────

def _parse(result: dict) -> list[dict]:
    """Convert Metabase response {data: {cols, rows}} → list of row-dicts."""
    data = result.get("data", {})
    cols = [c["name"] for c in data.get("cols", [])]
    return [dict(zip(cols, row)) for row in data.get("rows", [])]


def _post(path: str, payload: dict, timeout: int) -> dict:
    """POST to Metabase and return the parsed JSON body."""
    body = json.dumps(payload).encode()
    # Internal host uses a corp CA not in the system trust store — skip verify.
    ctx  = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    conn = http.client.HTTPSConnection(METABASE_HOST, timeout=timeout, context=ctx)
    conn.request("POST", path, body=body, headers=_HEADERS)
    resp = conn.getresponse()
    return json.loads(resp.read())


def _run_question(question_id: int, max_results: int = 2000) -> list[dict]:
    """Execute a saved Metabase question; returns list of row-dicts."""
    result = _post(
        f"/api/card/{question_id}/query",
        {"parameters": [], "constraints": {"max-results": max_results}},
        timeout=_TIMEOUT_FAST,
    )
    return _parse(result)


def _run_sql(sql: str, max_results: int = 2000) -> list[dict]:
    """Execute raw SQL on Redshift via Metabase; returns list of row-dicts."""
    result = _post(
        "/api/dataset",
        {
            "database": REDSHIFT_DB_ID,
            "type": "native",
            "native": {"query": sql},
            "constraints": {"max-results": max_results},
        },
        timeout=_TIMEOUT_SLOW,
    )
    return _parse(result)


# ─── Public API ───────────────────────────────────────────────────────────────

def fetch_live_rates() -> Optional[dict[str, float]]:
    """
    Fetch current FX rates: {USD_INR, GBP_INR, EUR_INR, AED_INR}.

    USD/INR  → live XE tick (question 9974, most recent row)
    Others   → 3-day average transfer_rate from completed orders

    Returns None on any network/API failure so callers can use fallback rates.
    """
    try:
        xe_rows = _run_question(Q_XE_USDINR, max_results=1)
        usd_inr = float(xe_rows[0]["fxrate"]) if xe_rows else None

        sql = """
        SELECT
            currency_from,
            AVG(transfer_rate::float) AS avg_rate
        FROM public.analytics_orders_master_data
        WHERE order_status = 'COMPLETED'
          AND created_at >= CURRENT_DATE - 3
          AND currency_from IN ('AED', 'GBP', 'USD', 'EUR')
        GROUP BY currency_from
        """
        rows     = _run_sql(sql)
        rate_map = {r["currency_from"]: float(r["avg_rate"]) for r in rows}

        rates = {
            "USD_INR": round(rate_map.get("USD", usd_inr or 84.0), 4),
            "GBP_INR": round(rate_map.get("GBP", 106.0), 4),
            "EUR_INR": round(rate_map.get("EUR",  91.0), 4),
            "AED_INR": round(rate_map.get("AED",  22.9), 4),
        }
        logger.info(
            "Live rates fetched: USD/INR=%.4f  GBP/INR=%.4f  EUR/INR=%.4f  AED/INR=%.4f",
            rates["USD_INR"], rates["GBP_INR"], rates["EUR_INR"], rates["AED_INR"],
        )
        return rates

    except Exception as exc:
        logger.warning(
            "fetch_live_rates failed (%s: %s) — caller should use fallback rates",
            type(exc).__name__, exc,
        )
        return None


def fetch_corridor_volumes(lookback_days: int = 84) -> Optional[dict[str, list[dict]]]:
    """
    Fetch real daily TPV by corridor from analytics_orders_master_data.

    Returns:
        {
          "AED_INR": [{"date": "2025-11-29", "volume_usd": 4_500_000.0, "dow": 5}, ...],
          "GBP_INR": [...],
          "USD_INR": [...],
          "EUR_INR": [...],
        }
    Returns None if Metabase is unreachable.
    """
    sql = f"""
    SELECT
        currency_from,
        DATE_TRUNC('day', created_at)::date AS txn_day,
        SUM(send_amount_usd)                AS total_usd
    FROM public.analytics_orders_master_data
    WHERE order_status = 'COMPLETED'
      AND created_at >= CURRENT_DATE - {lookback_days}
      AND currency_from IN ('AED', 'GBP', 'USD', 'EUR')
    GROUP BY 1, 2
    ORDER BY 2 ASC, 1
    """
    try:
        rows = _run_sql(sql, max_results=2000)
    except Exception as exc:
        logger.warning(
            "fetch_corridor_volumes failed (%s: %s) — caller should use fallback",
            type(exc).__name__, exc,
        )
        return None

    _CORRIDOR_MAP = {"AED": "AED_INR", "GBP": "GBP_INR", "USD": "USD_INR", "EUR": "EUR_INR"}
    volumes: dict[str, list[dict]] = {"AED_INR": [], "GBP_INR": [], "USD_INR": [], "EUR_INR": []}

    for r in rows:
        corridor = _CORRIDOR_MAP.get(r.get("currency_from", ""))
        if not corridor:
            continue
        day_str = str(r.get("txn_day", ""))[:10]
        if not day_str:
            continue
        try:
            dt = datetime.fromisoformat(day_str).date()
        except ValueError:
            continue
        volumes[corridor].append({
            "date":       day_str,
            "volume_usd": float(r.get("total_usd") or 0.0),
            "dow":        dt.weekday(),
        })

    if not any(volumes.values()):
        logger.warning("fetch_corridor_volumes returned empty data")
        return None

    total_rows = sum(len(v) for v in volumes.values())
    logger.info(
        "Corridor volumes loaded: %d days total "
        "(AED=%d GBP=%d USD=%d EUR=%d)",
        total_rows,
        len(volumes["AED_INR"]), len(volumes["GBP_INR"]),
        len(volumes["USD_INR"]), len(volumes["EUR_INR"]),
    )
    return volumes
