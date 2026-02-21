# ECM Field Agent

Interactive ECM agent for Aspora operations staff. Work your ticket queue, diagnose orders, resolve or escalate — all through Claude Code.

## Quick Start (3 commands)

```bash
git clone <repo> && cd ecm-operations/field
claude    # starts Claude Code with field CLAUDE.md
> my tickets
```

## Commands

| Command | What it does |
|---------|-------------|
| `my tickets` | Your assigned queue with actionables |
| `order <id>` | Diagnose a specific order — stuck reason + runbook steps |
| `resolve <id> <notes>` | Mark a ticket resolved |
| `escalate <id> <reason>` | Escalate to supervisor/team |

## Slash Commands

These are available as `/command` in Claude Code:

| Slash Command | Same as |
|---------------|---------|
| `/my-tickets` | `my tickets` |
| `/order-details <id>` | `order <id>` |
| `/resolve-ticket <id> <notes>` | `resolve <id> <notes>` |
| `/escalate-ticket <id> <reason>` | `escalate <id> <reason>` |
| `/ecm-dashboard` | Read-only dashboard view |

## How It Works

1. Manager agent runs triage 3x daily (7AM, 2PM, 8PM UAE)
2. Your tickets appear in Google Sheets and via `my tickets`
3. For each ticket: diagnose with `order <id>`, follow the runbook steps
4. When done: `resolve <id> "what you did"`
5. If stuck: `escalate <id> "why you're stuck"`

## Prerequisites

- VPN access (for ecm-gateway MCP)
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)
- Or: Cursor with Claude configured

## Structure

```
field/
├── CLAUDE.md               # Field persona + skill routing
├── README.md               # This file
├── AGENTS.md               # Command quick reference
├── .mcp.json               # MCP gateway config
├── plugin.yaml             # Trigger definitions
├── skills/
│   ├── my-tickets.md
│   ├── order-details.md
│   ├── resolve-ticket.md
│   └── escalate-ticket.md
└── .claude/
    └── commands/           # Slash commands for Claude Code CLI
        ├── my-tickets.md
        ├── order-details.md
        ├── resolve-ticket.md
        ├── escalate-ticket.md
        └── ecm-dashboard.md
```

## Shared Resources

Queries, runbooks, and config live in `../shared/` — not duplicated here.
