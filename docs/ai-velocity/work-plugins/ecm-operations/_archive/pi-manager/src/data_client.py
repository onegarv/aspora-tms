"""Data clients for Redshift and Google Sheets via MCP or direct API.

Supports two modes:
1. MCP mode (Claude Code / Pi.dev) - uses MCP tools
2. Direct mode (K8s / VPS) - uses direct DB connections
"""
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

from .models import Order, Agent, Assignment, AssignmentStatus, Priority


class RedshiftClient(ABC):
    """Abstract Redshift client."""

    @abstractmethod
    def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL and return results as list of dicts."""
        pass


class MCPRedshiftClient(RedshiftClient):
    """Redshift client via MCP (for Pi.dev / Claude Code)."""

    def __init__(self, mcp_executor):
        """
        Args:
            mcp_executor: Callable that executes MCP tool calls.
                         In Claude Code: the MCP tool function
                         In Pi.dev: the Pi MCP bridge
        """
        self.executor = mcp_executor

    def execute(self, sql: str) -> List[Dict[str, Any]]:
        result = self.executor("mcp__ecm-gateway__redshift_execute_sql_tool", {"sql": sql})
        return self._parse_result(result)

    def _parse_result(self, result: str) -> List[Dict[str, Any]]:
        """Parse MCP result table format into list of dicts."""
        lines = result.strip().split("\n")
        if len(lines) < 2:
            return []

        # Find header row (contains column names)
        header_line = None
        data_start = 0
        for i, line in enumerate(lines):
            if "|" in line and "---" not in line:
                header_line = line
                data_start = i + 1
                break

        if not header_line:
            return []

        # Parse headers
        headers = [h.strip() for h in header_line.split("|") if h.strip()]

        # Parse data rows
        results = []
        for line in lines[data_start:]:
            if "---" in line or not line.strip():
                continue
            values = [v.strip() for v in line.split("|") if v.strip() or v == ""]
            if len(values) >= len(headers):
                row = {}
                for j, header in enumerate(headers):
                    val = values[j] if j < len(values) else None
                    # Convert numeric strings
                    if val and val.replace(".", "").replace("-", "").isdigit():
                        try:
                            row[header] = float(val) if "." in val else int(val)
                        except ValueError:
                            row[header] = val
                    else:
                        row[header] = val
                results.append(row)

        return results


class DirectRedshiftClient(RedshiftClient):
    """Direct Redshift client for K8s/VPS deployment."""

    def __init__(self):
        """Initialize from environment variables."""
        self.host = os.environ.get("REDSHIFT_HOST")
        self.port = int(os.environ.get("REDSHIFT_PORT", "5439"))
        self.database = os.environ.get("REDSHIFT_DATABASE")
        self.user = os.environ.get("REDSHIFT_USER")
        self.password = os.environ.get("REDSHIFT_PASSWORD")
        self._conn = None

    def _get_connection(self):
        """Get or create Redshift connection."""
        if self._conn is None:
            try:
                import redshift_connector
                self._conn = redshift_connector.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password
                )
            except ImportError:
                raise RuntimeError("redshift_connector not installed. Run: pip install redshift-connector")
        return self._conn

    def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL and return results as list of dicts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Fetch all rows
        rows = cursor.fetchall()
        cursor.close()

        # Convert to list of dicts
        return [dict(zip(columns, row)) for row in rows]

    def close(self):
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class SheetsClient(ABC):
    """Abstract Google Sheets client."""

    @abstractmethod
    def get_data(self, spreadsheet_id: str, sheet: str, range_: str = None) -> List[List[Any]]:
        pass

    @abstractmethod
    def update_cells(self, spreadsheet_id: str, sheet: str, range_: str, data: List[List[Any]]) -> bool:
        pass


class MCPSheetsClient(SheetsClient):
    """Google Sheets client via MCP."""

    def __init__(self, mcp_executor):
        self.executor = mcp_executor

    def get_data(self, spreadsheet_id: str, sheet: str, range_: str = None) -> List[List[Any]]:
        params = {"spreadsheet_id": spreadsheet_id, "sheet": sheet}
        if range_:
            params["range"] = range_
        result = self.executor("mcp__ecm-gateway__sheets_get_sheet_data", params)
        return self._parse_result(result)

    def update_cells(self, spreadsheet_id: str, sheet: str, range_: str, data: List[List[Any]]) -> bool:
        params = {
            "spreadsheet_id": spreadsheet_id,
            "sheet": sheet,
            "range": range_,
            "data": data
        }
        self.executor("mcp__ecm-gateway__sheets_update_cells", params)
        return True

    def _parse_result(self, result) -> List[List[Any]]:
        """Parse MCP sheets result."""
        if isinstance(result, dict):
            ranges = result.get("valueRanges", [])
            if ranges:
                return ranges[0].get("values", [])
        return []


class DirectSheetsClient(SheetsClient):
    """Direct Google Sheets client for K8s/VPS deployment."""

    def __init__(self):
        """Initialize from environment variables."""
        self._service = None
        self._creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

    def _get_service(self):
        """Get or create Sheets API service."""
        if self._service is None:
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build

                if self._creds_json:
                    creds_dict = json.loads(self._creds_json)
                    creds = service_account.Credentials.from_service_account_info(
                        creds_dict,
                        scopes=["https://www.googleapis.com/auth/spreadsheets"]
                    )
                else:
                    # Try default credentials (for GKE workload identity)
                    import google.auth
                    creds, _ = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/spreadsheets"]
                    )

                self._service = build("sheets", "v4", credentials=creds)
            except ImportError:
                raise RuntimeError("Google API libraries not installed. Run: pip install google-auth google-api-python-client")
        return self._service

    def get_data(self, spreadsheet_id: str, sheet: str, range_: str = None) -> List[List[Any]]:
        """Get data from sheet."""
        service = self._get_service()
        range_str = f"{sheet}!{range_}" if range_ else sheet

        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_str
        ).execute()

        return result.get("values", [])

    def update_cells(self, spreadsheet_id: str, sheet: str, range_: str, data: List[List[Any]]) -> bool:
        """Update cells in sheet."""
        service = self._get_service()
        range_str = f"{sheet}!{range_}"

        body = {"values": data}
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_str,
            valueInputOption="RAW",
            body=body
        ).execute()

        return True


class SlackClient:
    """Slack client using Bot API."""

    def __init__(self, bot_token: str):
        self.token = bot_token
        self.base_url = "https://slack.com/api"

    def post_message(self, channel: str, text: str, thread_ts: str = None) -> Optional[str]:
        """Post message to Slack. Returns message timestamp."""
        payload = {
            "channel": channel,
            "text": text,
            "unfurl_links": False
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts

        result = self._api_call("chat.postMessage", payload)
        if result.get("ok"):
            return result.get("ts")
        return None

    def _api_call(self, method: str, payload: Dict) -> Dict:
        """Make Slack API call."""
        url = f"{self.base_url}/{method}"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
        )

        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))


class DataService:
    """Unified data access service."""

    def __init__(self, redshift: RedshiftClient, sheets: SheetsClient, config):
        self.redshift = redshift
        self.sheets = sheets
        self.config = config
        self._queries_path = Path(__file__).parent.parent / "queries"

    def get_pending_orders(self) -> List[Order]:
        """Get actionable stuck orders from Redshift."""
        sql = self._load_query("ecm-pending-list.sql")
        rows = self.redshift.execute(sql)

        return [
            Order(
                order_id=row.get("order_id", ""),
                order_date=str(row.get("order_date", "")),
                status=row.get("order_status", ""),
                sub_state=row.get("sub_state", ""),
                currency=row.get("currency_from", ""),
                send_amount=float(row.get("send_amount", 0) or 0),
                receive_amount=float(row.get("receive_amount", 0) or 0),
                hours_stuck=float(row.get("hours_diff", 0) or 0),
                category=row.get("category", ""),
            )
            for row in rows
        ]

    def get_order_details(self, order_ids: List[str]) -> Dict[str, Dict]:
        """Get detailed info for orders (enrichment)."""
        if not order_ids:
            return {}

        ids_str = ", ".join(f"'{oid}'" for oid in order_ids[:10])  # Batch of 10
        sql = self._load_query("ecm-triage-fast.sql")
        sql = sql.replace("'{order_id}'", ids_str)

        rows = self.redshift.execute(sql)
        return {row.get("order_id"): row for row in rows}

    def get_current_assignments(self) -> List[Assignment]:
        """Get current assignments from Google Sheet."""
        data = self.sheets.get_data(
            self.config.sheet.spreadsheet_id,
            self.config.sheet.assignments_tab
        )

        if len(data) < 2:
            return []

        assignments = []
        for row in data[1:]:  # Skip header
            if len(row) >= 10:
                try:
                    assignments.append(Assignment(
                        order_id=row[0],
                        agent_email=row[1],
                        assigned_at=row[2],
                        status=AssignmentStatus(row[3]) if row[3] else AssignmentStatus.OPEN,
                        priority=Priority(row[4]) if row[4] else Priority.P3,
                        currency=row[5],
                        amount=float(row[6]) if row[6] else 0,
                        hours_stuck=float(row[7]) if row[7] else 0,
                        diagnosis=row[8] if len(row) > 8 else "",
                        notes=row[9] if len(row) > 9 else "",
                    ))
                except (ValueError, IndexError):
                    continue

        return assignments

    def get_active_agents(self) -> List[Agent]:
        """Get active agents from Google Sheet."""
        data = self.sheets.get_data(
            self.config.sheet.spreadsheet_id,
            self.config.sheet.agents_tab
        )

        if len(data) < 2:
            return []

        agents = []
        for row in data[1:]:  # Skip header
            if len(row) >= 6:
                email = row[0]
                active = str(row[4]).upper() == "TRUE"

                # Skip excluded agents
                if any(exc in email.lower() for exc in self.config.triage.excluded_agents):
                    continue

                if active:
                    agents.append(Agent(
                        email=email,
                        name=row[1],
                        team=row[2],
                        slack_handle=row[3],
                        active=active,
                        max_tickets=int(row[5]) if row[5] else 100,
                    ))

        return agents

    def write_assignments(self, assignments: List[Assignment], start_row: int) -> bool:
        """Write assignments to Google Sheet."""
        data = []
        for a in assignments:
            data.append([
                a.order_id,
                a.agent_email,
                a.assigned_at,
                a.status.value,
                a.priority.value,
                a.currency,
                str(a.amount),
                str(a.hours_stuck),
                a.diagnosis,
                a.notes,
            ])

        range_ = f"A{start_row}:J{start_row + len(data) - 1}"
        return self.sheets.update_cells(
            self.config.sheet.spreadsheet_id,
            self.config.sheet.assignments_tab,
            range_,
            data
        )

    def get_resolved_in_redshift(self, order_ids: List[str]) -> List[str]:
        """Check which orders are now COMPLETED in Redshift."""
        if not order_ids:
            return []

        ids_str = ", ".join(f"'{oid}'" for oid in order_ids)
        sql = f"""
        SELECT order_id
        FROM orders_goms
        WHERE order_id IN ({ids_str})
          AND status = 'COMPLETED'
        """
        rows = self.redshift.execute(sql)
        return [row.get("order_id") for row in rows if row.get("order_id")]

    def _load_query(self, filename: str) -> str:
        """Load SQL query from file."""
        # Try symlinked queries first, then parent
        for path in [self._queries_path / filename,
                     self._queries_path.parent.parent / "queries" / filename]:
            if path.exists():
                return path.read_text()
        raise FileNotFoundError(f"Query file not found: {filename}")
