#!/bin/bash
set -e

# ECM Pattern Intelligence — Recurring scheduled runner
# Designed for crontab: runs claude --print from the repo root
# Cron: 30 1 * * * /path/to/ecm-operations/manager/run-patterns.sh

MANAGER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$MANAGER_DIR/.." && pwd)"
LOG_DIR="$MANAGER_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/patterns_${TIMESTAMP}.log"
SLACK_CHANNEL="${SLACK_CHANNEL_ID:-C0AD6C36LVC}"
MAX_RETRIES=1

mkdir -p "$LOG_DIR"

# Log rotation — keep last 7 days
find "$LOG_DIR" -name "*.log" -mtime +7 -delete 2>/dev/null || true

log() {
    echo "[$(date +%Y%m%d_%H%M%S)] $1" | tee -a "$LOG_FILE"
}

notify_error() {
    local msg="$1"
    if [ -n "$SLACK_BOT_TOKEN" ]; then
        curl -sf -X POST "https://slack.com/api/chat.postMessage" \
            -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{\"channel\": \"$SLACK_CHANNEL\", \"text\": \"🚨 ECM Pattern Intelligence Error\\n\`\`\`$msg\`\`\`\"}" \
            > /dev/null 2>&1 || true
    fi
}

log "Pattern Intelligence starting..."

# Run from repo root so both manager/ and shared/ are accessible
cd "$REPO_ROOT"

# Load OpenRouter credentials
if [ -f "$MANAGER_DIR/.env" ]; then
    set -a
    source "$MANAGER_DIR/.env"
    set +a
    log "Loaded .env (API provider: ${ANTHROPIC_BASE_URL:-default})"
else
    log "ERROR: $MANAGER_DIR/.env not found — cannot authenticate"
    exit 1
fi

# Validate auth
if [ -z "$ANTHROPIC_AUTH_TOKEN" ]; then
    log "ERROR: ANTHROPIC_AUTH_TOKEN not set"
    exit 1
fi

# Clear ANTHROPIC_API_KEY (OpenRouter uses ANTHROPIC_AUTH_TOKEN)
export ANTHROPIC_API_KEY=""

# MCP health check
MCP_URL=$(cat .mcp.json 2>/dev/null | jq -r '.mcpServers["ecm-gateway"].url // empty')
if [ -n "$MCP_URL" ]; then
    MCP_HOST=$(echo "$MCP_URL" | sed 's|http://||' | sed 's|/.*||')
    if curl -sf --max-time 10 "http://${MCP_HOST}/health" > /dev/null 2>&1; then
        log "MCP gateway reachable: $MCP_HOST"
    else
        log "WARNING: MCP gateway health check failed ($MCP_HOST) — proceeding anyway"
    fi
fi

# Prevent nested-session error when CLAUDECODE is set
unset CLAUDECODE 2>/dev/null || true

# Run with retry
run_patterns() {
    timeout 300 claude --print "
You are the ECM Manager agent. Execute pattern intelligence:

1. Read manager/CLAUDE.md for your persona and skill routing
2. Read shared/guardrails.md for ECM guardrails
3. Read manager/skills/pattern-intelligence.md for the full pattern intelligence workflow
4. Read and execute shared/queries/ecm-pattern-clusters.sql via mcp__ecm-gateway__redshift_execute_sql_tool
5. Classify each cluster to a stuck_reason using shared/stuck-reasons.yaml
6. Compute impact scores and rank patterns
7. Read previous run data from Google Sheet Pattern Intelligence tab (spreadsheet ${SPREADSHEET_ID:-1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks})
8. Compute trends (UP/DOWN/NEW/GONE) vs previous baseline
9. Write results to Google Sheet Pattern Intelligence tab
10. Post pattern intelligence report to Slack channel $SLACK_CHANNEL as separate message
    using curl and SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN

All timestamps in IST (UTC+5:30). Execute now - no confirmation needed.
" >> "$LOG_FILE" 2>&1
}

attempt=0
while [ $attempt -le $MAX_RETRIES ]; do
    if [ $attempt -gt 0 ]; then
        log "Retry $attempt/$MAX_RETRIES — waiting 30s..."
        sleep 30
    fi

    if run_patterns; then
        log "Pattern Intelligence complete"
        exit 0
    fi

    attempt=$((attempt + 1))
done

log "Pattern Intelligence FAILED after $((MAX_RETRIES + 1)) attempts"
notify_error "Pattern Intelligence failed after $((MAX_RETRIES + 1)) attempts. Check logs: $LOG_FILE"
exit 1
