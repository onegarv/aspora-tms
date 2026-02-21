#!/bin/bash
set -e

# ECM Manager Agent Entrypoint
# Usage: ./entrypoint.sh [command]
# Commands: daily, backlog, triage, patterns, health, test

COMMAND=${1:-daily}
SLACK_CHANNEL="${SLACK_CHANNEL_ID:-C0AD6C36LVC}"
SHEET_ID="${SPREADSHEET_ID:-1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks}"

echo "ECM Manager Agent - $COMMAND"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Portable timeout: Linux has `timeout`, macOS needs `gtimeout` from coreutils
if command -v timeout &>/dev/null; then
    TIMEOUT_CMD="timeout"
elif command -v gtimeout &>/dev/null; then
    TIMEOUT_CMD="gtimeout"
else
    TIMEOUT_CMD=""
fi

# Validate required env vars
check_env() {
    local missing=()
    [ -z "$ANTHROPIC_AUTH_TOKEN" ] && missing+=("ANTHROPIC_AUTH_TOKEN")
    [ -z "$SLACK_BOT_TOKEN" ] && missing+=("SLACK_BOT_TOKEN")

    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: Missing required environment variables: ${missing[*]}"
        exit 1
    fi

    # OpenRouter requires ANTHROPIC_API_KEY to be empty (auth via ANTHROPIC_AUTH_TOKEN)
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        echo "WARNING: ANTHROPIC_API_KEY is set — clearing it (OpenRouter uses ANTHROPIC_AUTH_TOKEN)"
        export ANTHROPIC_API_KEY=""
    fi

    echo "Environment validated (provider: ${ANTHROPIC_BASE_URL:-default})"
}

# MCP health check — verify ecm-gateway is reachable
check_mcp() {
    local mcp_url
    mcp_url=$(cat .mcp.json 2>/dev/null | jq -r '.mcpServers["ecm-gateway"].url // empty')

    if [ -z "$mcp_url" ]; then
        echo "WARNING: No MCP URL found in .mcp.json — skipping health check"
        return 0
    fi

    # Extract host:port from URL
    local host_port
    host_port=$(echo "$mcp_url" | sed 's|http://||' | sed 's|/.*||')

    if curl -sf --max-time 10 "http://${host_port}/health" > /dev/null 2>&1; then
        echo "MCP gateway reachable: $host_port"
    else
        echo "WARNING: MCP gateway health check failed ($host_port) — proceeding anyway"
    fi
}

# Post error to Slack
notify_error() {
    local msg="$1"
    if [ -n "$SLACK_BOT_TOKEN" ]; then
        curl -sf -X POST "https://slack.com/api/chat.postMessage" \
            -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{\"channel\": \"$SLACK_CHANNEL\", \"text\": \"🚨 ECM Manager Error\\n\`\`\`$msg\`\`\`\"}" \
            > /dev/null 2>&1 || true
    fi
}

# Run claude --print with timeout and error handling
run_claude() {
    local label="$1"
    local prompt="$2"

    echo "[$label] Starting..."

    # Prevent nested-session error
    unset CLAUDECODE 2>/dev/null || true

    if [ -n "$TIMEOUT_CMD" ]; then
        $TIMEOUT_CMD 600 claude --print "$prompt" 2>&1
    else
        claude --print "$prompt" 2>&1
    fi

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$label] Complete"
        return 0
    else
        echo "[$label] FAILED (exit code: $exit_code)"
        notify_error "$label failed with exit code $exit_code"
        return $exit_code
    fi
}

# Run backlog analysis via Claude Code (waterfall, 7-day trend, currency split, sub-state breakdown)
run_backlog() {
    run_claude "Backlog" "
You are the ECM Manager agent. Execute the daily backlog flow analysis:

1. Read manager/CLAUDE.md for your persona and skill routing
2. Read shared/guardrails.md for ECM guardrails
3. Read shared/config/slack-formatting.md for Slack message formatting rules
4. Read manager/skills/ecm-daily-flow.md for the full daily flow analysis workflow
5. Execute ALL queries from the skill: Phase 1 (backlog segments), Phase 2 (7-day inflow trend), Phase 3 (sub-state breakdown)
6. Build the analyst report with waterfall, severity, 7-day trend, currency split, sub-state breakdown
7. Post to Slack channel $SLACK_CHANNEL using curl and SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN
   CRITICAL — Follow shared/config/slack-formatting.md exactly:
   - Title MUST be exactly: :bar_chart: *ECM Backlog* — {date} {time} IST
   - Main message < 2000 chars: backlog count + direction, severity split, value at risk, top 3 actions
   - Thread reply 1: waterfall + severity + 7-day inflow chart (code blocks, NOT markdown tables)
   - Thread reply 2: currency split + sub-state breakdown (code blocks)
   - Use *bold* not **bold**, bullets not tables, numbers lead sentences

All timestamps in IST (UTC+5:30). Execute now — no confirmation needed.
"
}

# Run triage via Claude Code
run_triage() {
    run_claude "Triage" "
You are the ECM Manager agent. Execute the full triage workflow:

1. Read manager/CLAUDE.md for your persona and skill routing
2. Read shared/guardrails.md for ECM guardrails
3. Read shared/config/slack-formatting.md for Slack message formatting rules
4. Read manager/skills/triage-and-assign.md for the full triage workflow
5. Query Redshift via mcp__ecm-gateway__redshift_execute_sql_tool for stuck orders
6. Get already-assigned orders from Google Sheet (spreadsheet $SHEET_ID)
7. Score and prioritize unassigned orders
8. Auto-assign to available agents — skip confirmation (batch mode)
9. Write assignments to Google Sheet Assignments tab
10. Post to Slack channel $SLACK_CHANNEL using curl and SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN
    CRITICAL — Follow shared/config/slack-formatting.md exactly:
    - Title MUST be exactly: :dart: *ECM Triage* — {date} {time} IST
    - Main message < 2000 chars: orders assigned, total value, agent capacity, top 3 critical assignments
    - Thread reply 1: full assignment list as code block (NOT markdown table)
    - Thread reply per agent: their specific orders + next actions
    - Use *bold* not **bold**, bullets not tables, numbers lead sentences

All timestamps in IST (UTC+5:30). Execute now — no confirmation needed.
"
}

# Run pattern intelligence via Claude Code
run_patterns() {
    run_claude "Patterns" "
You are the ECM Manager agent. Execute pattern intelligence:

1. Read manager/CLAUDE.md for your persona and skill routing
2. Read shared/guardrails.md for ECM guardrails
3. Read shared/config/slack-formatting.md for Slack message formatting rules
4. Read manager/skills/pattern-intelligence.md for the full pattern intelligence workflow
5. Read and execute shared/queries/ecm-pattern-clusters.sql via mcp__ecm-gateway__redshift_execute_sql_tool
6. Classify each cluster to a stuck_reason using shared/stuck-reasons.yaml
7. Compute impact scores and rank patterns
8. Read previous run data from Google Sheet Pattern Intelligence tab (spreadsheet $SHEET_ID)
9. Compute trends (UP/DOWN/NEW/GONE) vs previous baseline
10. Write results to Google Sheet Pattern Intelligence tab
11. Post to Slack channel $SLACK_CHANNEL as separate message
    using curl and SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN
    CRITICAL — Follow shared/config/slack-formatting.md exactly:
    - Title MUST be exactly: :microscope: *ECM Patterns* — {date} {time} IST
    - Main message < 2000 chars: total orders, top 3 patterns (name + count + amount + trend), novel alert if any
    - Thread reply 1: top 10 pattern table as code block (NOT markdown table)
    - Thread reply 2: novel patterns detail + recommended actions (if any)
    - Use *bold* not **bold**, bullets not tables, numbers lead sentences

All timestamps in IST (UTC+5:30). Execute now — no confirmation needed.
"
}

# Daily flow: backlog → triage → patterns (the full morning flow)
run_daily() {
    echo "=== Daily Flow: Backlog + Triage + Patterns ==="

    local failed=0

    run_backlog || failed=1

    if [ $failed -eq 0 ]; then
        echo ""
        echo "--- Backlog complete, starting triage ---"
        echo ""
        run_triage || failed=1
    else
        echo "Backlog failed — continuing with triage"
        failed=0
        run_triage || failed=1
    fi

    if [ $failed -eq 0 ]; then
        echo ""
        echo "--- Triage complete, starting patterns ---"
        echo ""
        run_patterns || failed=1
    else
        echo "Triage failed — skipping patterns"
        notify_error "Daily flow: triage failed, patterns skipped"
    fi

    if [ $failed -eq 0 ]; then
        echo "=== Daily Flow Complete ==="
    else
        echo "=== Daily Flow FAILED ==="
        exit 1
    fi
}

# Health check endpoint
run_health() {
    echo '{"status": "healthy", "agent": "ecm-manager", "version": "2.0.0"}'
}

# Test mode — validate connections
run_test() {
    echo "Testing connections..."

    # Test Slack
    SLACK_TEST=$(curl -sf -X POST "https://slack.com/api/auth.test" \
        -H "Authorization: Bearer $SLACK_BOT_TOKEN" | jq -r '.ok')

    if [ "$SLACK_TEST" = "true" ]; then
        echo "Slack: OK"
    else
        echo "Slack: FAILED"
        exit 1
    fi

    # Test Claude auth
    unset CLAUDECODE 2>/dev/null || true
    if [ -n "$TIMEOUT_CMD" ]; then
        CLAUDE_TEST=$($TIMEOUT_CMD 30 claude --print "Reply with exactly: AUTH_OK" 2>&1)
    else
        CLAUDE_TEST=$(claude --print "Reply with exactly: AUTH_OK" 2>&1)
    fi
    if echo "$CLAUDE_TEST" | grep -q "AUTH_OK"; then
        echo "Claude auth: OK"
    else
        echo "Claude auth: FAILED"
        echo "$CLAUDE_TEST"
        exit 1
    fi

    echo "All tests passed"
}

# Main
case $COMMAND in
    daily)
        check_env
        check_mcp
        run_daily
        ;;
    backlog)
        check_env
        check_mcp
        run_backlog
        ;;
    triage)
        check_env
        check_mcp
        run_triage
        ;;
    patterns)
        check_env
        check_mcp
        run_patterns
        ;;
    health)
        run_health
        ;;
    test)
        check_env
        check_mcp
        run_test
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: entrypoint.sh [daily|backlog|triage|patterns|health|test]"
        exit 1
        ;;
esac
