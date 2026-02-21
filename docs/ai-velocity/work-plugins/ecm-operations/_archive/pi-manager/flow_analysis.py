#!/usr/bin/env python3
"""
ECM Flow Analysis - Analyst View

Shows backlog vs yesterday vs today order inflow.
Produces an ASCII chart and summary table.

Usage:
    python flow_analysis.py

Requires: REDSHIFT_HOST, REDSHIFT_USER, REDSHIFT_PASSWORD, REDSHIFT_DATABASE env vars
"""
import os
import sys
from pathlib import Path

def load_env():
    """Load .env file if present."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


class DirectRedshiftClient:
    """Direct Redshift client - minimal standalone version."""

    def __init__(self):
        self.host = os.environ.get("REDSHIFT_HOST")
        self.port = int(os.environ.get("REDSHIFT_PORT", "5439"))
        self.database = os.environ.get("REDSHIFT_DATABASE")
        self.user = os.environ.get("REDSHIFT_USER")
        self.password = os.environ.get("REDSHIFT_PASSWORD")
        self._conn = None

    def _get_connection(self):
        if self._conn is None:
            import redshift_connector
            self._conn = redshift_connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
        return self._conn

    def execute(self, sql):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        return [dict(zip(columns, row)) for row in rows]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


def run_flow_analysis():
    """Run the flow analysis query and display results."""

    # Load query
    query_path = Path(__file__).parent.parent / "queries" / "ecm-flow-analysis.sql"
    if not query_path.exists():
        print(f"ERROR: Query file not found: {query_path}")
        return 1

    sql = query_path.read_text()

    # Execute
    print("Connecting to Redshift...")
    try:
        client = DirectRedshiftClient()
        print("Running flow analysis query...")
        rows = client.execute(sql)
        client.close()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    if not rows:
        print("No data returned")
        return 0

    # Aggregate totals by segment
    totals = {}
    for row in rows:
        seg = row.get("inflow_segment", "unknown")
        if seg not in totals:
            totals[seg] = {"orders": 0, "amount": 0, "currencies": {}}
        totals[seg]["orders"] += row.get("order_count", 0)
        totals[seg]["amount"] += float(row.get("total_amount", 0) or 0)
        curr = row.get("currency", "?")
        totals[seg]["currencies"][curr] = row.get("order_count", 0)

    # Display summary
    print("\n" + "=" * 60)
    print("ECM FLOW ANALYSIS - Backlog vs Yesterday vs Today")
    print("=" * 60)

    segments = ["backlog", "yesterday", "today"]
    max_orders = max(totals.get(s, {}).get("orders", 0) for s in segments) or 1

    for seg in segments:
        data = totals.get(seg, {"orders": 0, "amount": 0, "currencies": {}})
        orders = data["orders"]
        amount = data["amount"]
        currencies = data["currencies"]

        # ASCII bar
        bar_len = int((orders / max_orders) * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)

        label = {"backlog": "📦 Backlog  ", "yesterday": "📥 Yesterday", "today": "📌 Today    "}
        print(f"\n{label.get(seg, seg)}: {orders:4d} orders | {amount:,.0f} total")
        print(f"  {bar}")

        # Currency breakdown
        for curr, cnt in sorted(currencies.items()):
            print(f"    {curr}: {cnt}")

    # Summary table
    print("\n" + "-" * 60)
    print("DETAILED BREAKDOWN")
    print("-" * 60)
    print(f"{'Segment':<12} {'Currency':<8} {'Count':>8} {'Amount':>15} {'Avg Hrs':>10}")
    print("-" * 60)

    for row in rows:
        seg = row.get("inflow_segment", "?")
        curr = row.get("currency", "?")
        cnt = row.get("order_count", 0)
        amt = float(row.get("total_amount", 0) or 0)
        avg_hrs = float(row.get("avg_hours_stuck", 0) or 0)
        print(f"{seg:<12} {curr:<8} {cnt:>8} {amt:>15,.2f} {avg_hrs:>10.1f}")

    print("-" * 60)

    # Total
    total_orders = sum(totals.get(s, {}).get("orders", 0) for s in segments)
    total_amount = sum(totals.get(s, {}).get("amount", 0) for s in segments)
    print(f"{'TOTAL':<12} {'':<8} {total_orders:>8} {total_amount:>15,.2f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    load_env()
    sys.exit(run_flow_analysis())
