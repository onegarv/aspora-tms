# ECM Agent Skills

You are an ECM (Exception Case Management) operations agent for Aspora Remittance.
These skills help you work on your assigned tickets.

## Setup

1. Clone this repo or copy the `agent-skills/` folder to your machine
2. Run Claude Code in this directory
3. Use the commands below

## Commands

| Command | What it does |
|---------|--------------|
| `/my-tickets` | See your assigned orders |
| `/order {id}` | Get diagnosis + runbook for an order |
| `/resolve {id} "{notes}"` | Mark an order as resolved |
| `/escalate {id} "{reason}"` | Escalate an order |

## Quick Start

```bash
# See your queue
/my-tickets

# Get details on an order
/order AE12Y0K4BU00

# Mark as resolved
/resolve AE12Y0K4BU00 "Refund processed via Checkout"

# Escalate if stuck
/escalate AE12Y0K4BU00 "Lulu timeout >48h, need Binoy"
```

## Data Sources

- **Assignments**: Google Sheet (read via MCP)
- **Order Details**: Redshift queries (read-only via MCP)
- **Runbooks**: Local markdown files in `runbooks/`

## MCP Configuration

These skills require the `ecm-gateway` MCP server. Your `.mcp.json` should include:

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

## Guardrails

- Only work on tickets assigned to YOU
- Follow runbook steps exactly — don't improvise
- RFI < 24h: Do NOT nudge the customer
- High-value (>50K AED): Requires supervisor approval before resolution
- If something is unclear, escalate — don't guess

## Links

- [ECM Dashboard](https://docs.google.com/spreadsheets/d/1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks/edit)
- [Slack Channel](https://aspora.slack.com/archives/C0AD6C36LVC)
