#!/bin/bash
set -e

# ECM Triage Skill Entrypoint
# Usage: ./entrypoint.sh [command]
# Commands: triage, health, test

COMMAND=${1:-triage}

echo "🚀 ECM Triage Skill - $COMMAND"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Validate required env vars
check_env() {
    local missing=()
    [ -z "$ANTHROPIC_API_KEY" ] && missing+=("ANTHROPIC_API_KEY")
    [ -z "$SLACK_BOT_TOKEN" ] && missing+=("SLACK_BOT_TOKEN")

    if [ ${#missing[@]} -gt 0 ]; then
        echo "❌ Missing required environment variables: ${missing[*]}"
        exit 1
    fi
    echo "✅ Environment validated"
}

# Run triage via Claude Code
run_triage() {
    echo "📊 Starting ECM triage..."

    # Run Claude Code with the skill
    claude --print "
You are the ECM Triage agent. Execute the following steps:

1. Query Redshift for stuck orders (last 30 days, 12h+ old)
2. Filter to actionable sub_states only
3. Get already-assigned orders from Google Sheet
4. Score and prioritize unassigned orders
5. Distribute high-value orders (>5K) round-robin across agents
6. Distribute remaining orders round-robin
7. Write assignments to Google Sheet
8. Post summary to Slack with:
   - Main message (motivational, stats)
   - Thread per agent with their order IDs
   - Getting started guide

Use these environment variables:
- SLACK_BOT_TOKEN: $SLACK_BOT_TOKEN
- SLACK_CHANNEL_ID: ${SLACK_CHANNEL_ID:-C0AD6C36LVC}
- SPREADSHEET_ID: $SPREADSHEET_ID

Read SKILL.md for full workflow details.
Execute now - no confirmation needed.
"

    echo "✅ Triage complete"
}

# Health check endpoint
run_health() {
    echo '{"status": "healthy", "skill": "ecm-triage", "version": "1.0.0"}'
}

# Test mode - validate connections
run_test() {
    echo "🧪 Testing connections..."

    # Test Slack
    SLACK_TEST=$(curl -s -X POST "https://slack.com/api/auth.test" \
        -H "Authorization: Bearer $SLACK_BOT_TOKEN" | jq -r '.ok')

    if [ "$SLACK_TEST" = "true" ]; then
        echo "✅ Slack connection OK"
    else
        echo "❌ Slack connection failed"
        exit 1
    fi

    echo "✅ All tests passed"
}

# Main
case $COMMAND in
    triage)
        check_env
        run_triage
        ;;
    health)
        run_health
        ;;
    test)
        check_env
        run_test
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: entrypoint.sh [triage|health|test]"
        exit 1
        ;;
esac
