'use strict';

const fs = require('fs');
const path = require('path');

const SHEET_ID = process.env.SPREADSHEET_ID || '1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks';
const REPO_ROOT = path.resolve(__dirname, '..', '..');

// Pre-load SQL at startup — file remains source of truth (DEC-001)
const load = (rel) => fs.readFileSync(path.join(REPO_ROOT, rel), 'utf8');
const TRIAGE_SQL = load('shared/queries/ecm-triage-fast.sql');

// Compact action map from shared/stuck-reasons.yaml (loaded, not hardcoded)
const STUCK_REASONS_RAW = load('shared/stuck-reasons.yaml');

const SLACK_FMT = 'Format as Slack mrkdwn: *bold* NOT **bold**, NO ## headers, ``` code blocks NOT | tables |, numbers lead sentences, bullets with *bold lead*. Under 3000 chars.';

const prompts = {
  'order-details': ({ order_id }) => {
    const sql = TRIAGE_SQL.replace(/\{order_id\}/g, order_id);
    return `You are the ECM Field agent. Diagnose order ${order_id}.

Execute this SQL via mcp__ecm-gateway__redshift_execute_sql_tool:
${sql}

Then use this stuck_reason reference to explain what's wrong and what to do:
${STUCK_REASONS_RAW}

Present as:
*{order_id}* | {priority P1-P4} | {team}
*What's Wrong* — plain English explanation
*What To Do* — numbered steps from the action field
Order facts as code block
*SLA:* {hours}h | *Escalation:* {contact}

${SLACK_FMT}
Execute now — no file reads needed, just run the SQL.`;
  },

  'my-tickets': ({ agent_email }) => `You are the ECM Field agent. Show ticket queue for: ${agent_email}

1. Read Assignments tab from Sheet ${SHEET_ID} via mcp__ecm-gateway__sheets_get_sheet_data
2. Filter: column B matches agent, column D is OPEN or IN_PROGRESS
3. For each order, run: SELECT order_id, status, sub_state, ROUND(EXTRACT(EPOCH FROM (GETDATE() - created_at)) / 3600, 1) AS hours_diff FROM orders_goms WHERE order_id IN ({ids}) via mcp__ecm-gateway__redshift_execute_sql_tool
4. Sort by priority (col E), show queue with actionables per ticket

${SLACK_FMT}
Execute now.`,

  'resolve-ticket': ({ order_id, notes, agent_email }) => `You are the ECM Field agent. Resolve ticket ${order_id}.
Agent: ${agent_email} | Notes: ${notes} | Sheet: ${SHEET_ID}

1. Read Assignments tab via mcp__ecm-gateway__sheets_get_sheet_data — find row for ${order_id}
2. Query Redshift for order status via mcp__ecm-gateway__redshift_execute_sql_tool
3. Append to Resolutions tab via mcp__ecm-gateway__sheets_update_cells: Timestamp | ${order_id} | ${agent_email} | ${notes} | AssignedAt | Time(min) | SLA | SLAStatus | StuckReason | Amount | Currency | CORRECT | YES | AGENT_RESOLVED
4. Update Assignments Status to RESOLVED via mcp__ecm-gateway__sheets_update_cells

${SLACK_FMT}
Show: resolved confirmation, resolution time, SLA met/missed, remaining queue count.
Execute now.`,

  'escalate-ticket': ({ order_id, reason, agent_email }) => `You are the ECM Field agent. Escalate ticket ${order_id}.
Agent: ${agent_email} | Reason: ${reason} | Sheet: ${SHEET_ID}

1. Read Assignments tab via mcp__ecm-gateway__sheets_get_sheet_data — find row for ${order_id}
2. Query Redshift for order details via mcp__ecm-gateway__redshift_execute_sql_tool
3. Append to Escalations tab via mcp__ecm-gateway__sheets_update_cells
4. Update Assignments Status to ESCALATED via mcp__ecm-gateway__sheets_update_cells

${SLACK_FMT}
Show: escalation confirmation, order summary, reason.
Execute now.`,
};

function buildPrompt(skill, params) {
  const builder = prompts[skill];
  if (!builder) throw new Error(`Unknown skill: ${skill}`);
  return builder(params);
}

module.exports = { buildPrompt };
