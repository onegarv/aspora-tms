"""Configuration management for ECM Pi Manager."""
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SlackConfig:
    bot_token: str
    channel_id: str
    channel_name: str

@dataclass
class SheetConfig:
    spreadsheet_id: str
    assignments_tab: str = "Assignments"
    agents_tab: str = "Agents"
    dashboard_tab: str = "ECM Dashboard"

@dataclass
class TriageConfig:
    high_value_threshold: int = 5000
    time_range_days: int = 30
    min_age_hours: int = 12
    max_orders: int = 500

    # Validation thresholds
    expected_count_min: int = 200
    expected_count_max: int = 600
    fail_count_threshold: int = 2000

    # Excluded agents (test accounts)
    excluded_agents: tuple = ("dinesh", "snita", "aakash@aspora.com")

@dataclass
class SLAConfig:
    """SLA hours by diagnosis type."""
    payment_failed: int = 2
    refund_triggered: int = 2
    manual_review: float = 0.5
    status_sync_issue: int = 1
    brn_pending: int = 4
    rfi_pending: int = 24
    default: int = 8

@dataclass
class Config:
    slack: SlackConfig
    sheet: SheetConfig
    triage: TriageConfig
    sla: SLAConfig

    @classmethod
    def from_env(cls, env_file: str = None) -> "Config":
        """Load config from environment variables."""
        if env_file and Path(env_file).exists():
            _load_dotenv(env_file)

        return cls(
            slack=SlackConfig(
                bot_token=os.environ.get("SLACK_BOT_TOKEN", ""),
                channel_id=os.environ.get("SLACK_CHANNEL_ID", "C0AD6C36LVC"),
                channel_name=os.environ.get("SLACK_CHANNEL", "wg-asap-agent-pilot"),
            ),
            sheet=SheetConfig(
                spreadsheet_id=os.environ.get(
                    "SPREADSHEET_ID",
                    "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks"
                ),
            ),
            triage=TriageConfig(),
            sla=SLAConfig(),
        )

def _load_dotenv(env_file: str):
    """Simple .env loader without external dependencies."""
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
