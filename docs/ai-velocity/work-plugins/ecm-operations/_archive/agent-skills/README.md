# ECM Agent Skills

Skills for ECM operations agents running Claude Code locally.

## Quick Start

```bash
# 1. Install Claude Code
# Visit: https://claude.ai/download

# 2. Clone or copy this folder to your machine
git clone <repo> && cd agent-skills

# 3. Configure MCP (one-time)
# Add ecm-gateway to your ~/.mcp.json

# 4. Run Claude Code
claude

# 5. Use commands
/my-tickets          # See your queue
/order AE12Y0K4BU00  # Get diagnosis
/resolve AE12Y0K4BU00 "Fixed via AlphaDesk"
/escalate AE12Y0K4BU00 "Lulu timeout"
```

## Commands

| Command | Description |
|---------|-------------|
| `/my-tickets` | View your assigned orders with priority and SLA |
| `/order {id}` | Get full diagnosis + runbook steps |
| `/resolve {id} "{notes}"` | Mark order as resolved |
| `/escalate {id} "{reason}"` | Escalate to another team |

## Files

```
agent-skills/
├── CLAUDE.md           # Claude Code reads this for context
├── README.md           # You're here
├── skills/
│   ├── guardrails.md   # Rules for all skills
│   ├── my-tickets.md   # /my-tickets command
│   ├── order-details.md # /order command
│   ├── resolve-ticket.md # /resolve command
│   └── escalate-ticket.md # /escalate command
├── queries/            # SQL queries (symlinked from parent)
├── runbooks/           # Resolution playbooks (symlinked from parent)
└── config/             # Stuck reasons, SLAs (symlinked from parent)
```

## MCP Setup

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "ecm-gateway": {
      "command": "npx",
      "args": ["mcporter", "start", "ecm-gateway"]
    }
  }
}
```

## Data Sources

| Source | Purpose |
|--------|---------|
| Google Sheets | Your assignments (Assignments tab) |
| Redshift | Live order data (read-only) |
| Local runbooks | Resolution steps |

**Spreadsheet ID:** `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`

## Workflow

1. **Morning**: Check `/my-tickets` for your queue
2. **Work**: Run `/order {id}` for each ticket, follow runbook
3. **Resolve**: Run `/resolve {id} "what you did"` when done
4. **Stuck?**: Run `/escalate {id} "why"` and contact the team

## Rules

- Only work on tickets assigned to YOU
- Follow runbook steps exactly
- RFI < 24h: Do NOT nudge customer
- High-value (>50K): Get supervisor approval
- If stuck: Escalate, don't guess

## Help

- Slack: #wg-asap-agent-pilot
- Dashboard: [ECM Dashboard](https://docs.google.com/spreadsheets/d/1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks/edit)
