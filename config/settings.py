"""
Global configuration for Aspora TMS.

All values are read from environment variables (prefixed ASPORA_).
Defaults are safe for local development; override in production via .env or secrets manager.
"""

from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict


class Currency(str, Enum):
    USD = "USD"
    GBP = "GBP"
    EUR = "EUR"
    AED = "AED"
    INR = "INR"


class DealType(str, Enum):
    SPOT = "spot"   # T+2 settlement
    TOM  = "tom"    # T+1 settlement
    CASH = "cash"   # T+0 same-day settlement


class TransferRail(str, Enum):
    FEDWIRE = "fedwire"
    CHAPS   = "chaps"
    TARGET2 = "target2"
    SWIFT   = "swift"       # SWIFT gpi fallback
    BANK_DESK = "bank_desk" # INR treasury desk


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASPORA_", env_file=".env")

    # ── Database ──────────────────────────────────────────────────────────
    postgres_url: str = "postgresql+asyncpg://localhost:5432/aspora_tms"
    timescale_url: str = "postgresql+asyncpg://localhost:5433/aspora_ts"

    # ── Message bus ───────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    bus_type: str = "redis"           # "redis" | "kafka"
    kafka_brokers: str = "localhost:9092"

    # ── FX risk limits ────────────────────────────────────────────────────
    max_single_deal_usd: float = 5_000_000        # Max per FX deal (USD equiv)
    max_open_exposure_pct: float = 0.30           # Max unhedged exposure after 2 PM
    stop_loss_paise: int = 5                      # Halt if blended rate drifts > 5 paise vs RBI ref
    flash_crash_threshold_pct: float = 0.005      # 0.5% intra-day move triggers freeze

    # ── Operations / fund movement ────────────────────────────────────────
    prefunding_buffer_pct: float = 0.10           # 10% safety buffer on top of shortfall
    maker_checker_timeout_min: int = 30           # Auto-escalate if not approved within 30 min
    dual_checker_threshold_usd: float = 10_000_000  # Transfers above this need 2 approvers
    duplicate_window_hours: int = 2               # Reject duplicate (same idempotency key) within 2 h

    # ── Confirmation SLAs ─────────────────────────────────────────────────
    fedwire_sla_min: int = 30
    chaps_sla_min: int = 120
    sepa_sla_min: int = 240
    swift_sla_min: int = 480

    # ── Banking window alert ──────────────────────────────────────────────
    window_closing_alert_min: int = 30            # Alert this many minutes before cut-off

    # ── Forecast ──────────────────────────────────────────────────────────
    forecast_lookback_weeks: int = 8
    max_total_multiplier: float = 2.5

    # ── FX Band integration ────────────────────────────────────────────
    fx_band_rate_strategy: str = "worst_case"       # "worst_case" | "midpoint" | "spot"
    holiday_prefund_lookahead_days: int = 3          # days ahead to scan for non-business-day stretches

    # ── Alerting ──────────────────────────────────────────────────────────
    alert_email: str = "treasury@aspora.com"
    slack_webhook_url: str = ""

    # ── Auth / RBAC ───────────────────────────────────────────────────────
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"


settings = Settings()
