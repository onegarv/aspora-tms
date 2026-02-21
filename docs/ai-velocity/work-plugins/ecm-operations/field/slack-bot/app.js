'use strict';

require('dotenv').config();

const { App } = require('@slack/bolt');
const { spawn } = require('child_process');
const path = require('path');
const { buildPrompt } = require('./prompts');
const { formatForSlack, truncate, splitForThread } = require('./formatter');

// Repo root — claude --print runs from here so field/ and shared/ paths resolve
const REPO_ROOT = path.resolve(__dirname, '..', '..');

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  appToken: process.env.SLACK_APP_TOKEN,
  socketMode: true,
});

// In-memory cache: Slack user ID → email
const emailCache = new Map();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function lookupEmail(client, userId) {
  if (emailCache.has(userId)) return emailCache.get(userId);
  try {
    const res = await client.users.info({ user: userId });
    const email = res.user?.profile?.email;
    if (email) emailCache.set(userId, email);
    return email || null;
  } catch {
    return null;
  }
}

const ALLOWED_TOOLS = [
  'mcp__ecm-gateway__redshift_execute_sql_tool',
  'mcp__ecm-gateway__sheets_get_sheet_data',
  'mcp__ecm-gateway__sheets_update_cells',
  'mcp__ecm-gateway__sheets_list_sheets',
  'Read', 'Glob', 'Grep',
].join(',');

function runClaude(skill, params) {
  const prompt = buildPrompt(skill, params);
  const env = { ...process.env, CLAUDECODE: '' };

  return new Promise((resolve, reject) => {
    const proc = spawn('claude', ['--print', '--allowedTools', ALLOWED_TOOLS, '--'], {
      cwd: REPO_ROOT,
      env,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (d) => { stdout += d; });
    proc.stderr.on('data', (d) => { stderr += d; });

    proc.on('close', (code) => {
      if (code === 0) return resolve(stdout);
      const errMsg = stderr.trim() || `claude --print exited with code ${code}`;
      reject(new Error(errMsg));
    });

    const timeout = setTimeout(() => {
      proc.kill('SIGTERM');
      reject(new Error('claude --print timed out after 300s'));
    }, 300_000);

    proc.on('close', () => clearTimeout(timeout));

    // Pipe prompt via stdin to avoid shell escaping issues
    proc.stdin.write(prompt);
    proc.stdin.end();
  });
}

async function replyInThread(say, threadTs, text) {
  const chunks = splitForThread(formatForSlack(text));
  for (const chunk of chunks) {
    await say({ text: truncate(chunk), thread_ts: threadTs });
  }
}

// ---------------------------------------------------------------------------
// Intent: order {id} / lookup {id}
// ---------------------------------------------------------------------------

app.message(/^(?:order|lookup)\s+(\S+)/i, async ({ message, say, client, context }) => {
  const orderId = context.matches[1];
  const ack = await say({ text: `:hourglass_flowing_sand: Looking up order \`${orderId}\`...`, thread_ts: message.ts });

  try {
    const output = await runClaude('order-details', { order_id: orderId });
    await replyInThread(say, message.ts, output);
  } catch (err) {
    await say({ text: `:x: Order lookup failed: \`${err.message}\``, thread_ts: message.ts });
  }
});

// ---------------------------------------------------------------------------
// Intent: my tickets / my queue / tickets for {name}
// ---------------------------------------------------------------------------

app.message(/^(?:my tickets|my queue|show my tickets|what should I work on|next task)$/i, async ({ message, say, client }) => {
  const ack = await say({ text: ':hourglass_flowing_sand: Loading your ticket queue...', thread_ts: message.ts });

  try {
    const email = await lookupEmail(client, message.user);
    if (!email) {
      await say({ text: ':warning: Could not resolve your Slack email. Make sure your Slack profile has an email set.', thread_ts: message.ts });
      return;
    }
    const output = await runClaude('my-tickets', { agent_email: email });
    await replyInThread(say, message.ts, output);
  } catch (err) {
    await say({ text: `:x: Tickets lookup failed: \`${err.message}\``, thread_ts: message.ts });
  }
});

app.message(/^tickets?\s+for\s+(.+)$/i, async ({ message, say, context }) => {
  const agentName = context.matches[1].trim();
  const ack = await say({ text: `:hourglass_flowing_sand: Loading tickets for *${agentName}*...`, thread_ts: message.ts });

  try {
    // Pass the name directly — the prompt will look it up in the Agents tab
    const output = await runClaude('my-tickets', { agent_email: agentName });
    await replyInThread(say, message.ts, output);
  } catch (err) {
    await say({ text: `:x: Tickets lookup failed: \`${err.message}\``, thread_ts: message.ts });
  }
});

// ---------------------------------------------------------------------------
// Intent: resolve {id} "{notes}" / done {id} "{notes}"
// ---------------------------------------------------------------------------

app.message(/^(?:resolve|fixed|done|close)\s+(\S+)\s+"?(.+?)"?\s*$/i, async ({ message, say, client, context }) => {
  const orderId = context.matches[1];
  const notes = context.matches[2];
  const ack = await say({ text: `:hourglass_flowing_sand: Resolving \`${orderId}\`...`, thread_ts: message.ts });

  try {
    const email = await lookupEmail(client, message.user);
    const output = await runClaude('resolve-ticket', {
      order_id: orderId,
      notes,
      agent_email: email || 'unknown',
    });
    await replyInThread(say, message.ts, output);
  } catch (err) {
    await say({ text: `:x: Resolve failed: \`${err.message}\``, thread_ts: message.ts });
  }
});

// Resolve without notes — error
app.message(/^(?:resolve|fixed|done|close)\s+(\S+)\s*$/i, async ({ message, say, context }) => {
  const orderId = context.matches[1];
  await say({
    text: `:x: Resolution notes are required.\n\nExample: \`resolve ${orderId} "Replayed webhook, confirmed at Lulu"\``,
    thread_ts: message.ts,
  });
});

// ---------------------------------------------------------------------------
// Intent: escalate {id} "{reason}" / stuck {id} "{reason}"
// ---------------------------------------------------------------------------

app.message(/^(?:escalate|stuck|can't fix)\s+(\S+)\s+"?(.+?)"?\s*$/i, async ({ message, say, client, context }) => {
  const orderId = context.matches[1];
  const reason = context.matches[2];
  const ack = await say({ text: `:hourglass_flowing_sand: Escalating \`${orderId}\`...`, thread_ts: message.ts });

  try {
    const email = await lookupEmail(client, message.user);
    const output = await runClaude('escalate-ticket', {
      order_id: orderId,
      reason,
      agent_email: email || 'unknown',
    });
    await replyInThread(say, message.ts, output);
  } catch (err) {
    await say({ text: `:x: Escalation failed: \`${err.message}\``, thread_ts: message.ts });
  }
});

// Escalate without reason — error
app.message(/^(?:escalate|stuck|can't fix)\s+(\S+)\s*$/i, async ({ message, say, context }) => {
  const orderId = context.matches[1];
  await say({
    text: `:x: Escalation reason is required.\n\nExample: \`escalate ${orderId} "Lulu not responding after 3 attempts"\``,
    thread_ts: message.ts,
  });
});

// ---------------------------------------------------------------------------
// Intent: help
// ---------------------------------------------------------------------------

app.message(/^help$/i, async ({ message, say }) => {
  await say({
    text: [
      ':robot_face: *ECM Field Bot — Commands*',
      '',
      '`order {id}` — Full order diagnosis with runbook steps',
      '`my tickets` — Your assigned ticket queue with actionables',
      '`tickets for {name}` — Another agent\'s queue',
      '`resolve {id} "notes"` — Mark ticket resolved (notes required)',
      '`escalate {id} "reason"` — Escalate ticket (reason required)',
      '`help` — This message',
    ].join('\n'),
    thread_ts: message.ts,
  });
});

// ---------------------------------------------------------------------------
// App mention fallback — show help
// ---------------------------------------------------------------------------

app.event('app_mention', async ({ event, say }) => {
  const text = event.text.replace(/<@[^>]+>\s*/g, '').trim();

  // If mention includes a command, let the message handlers above handle it
  if (/^(order|lookup|my tickets|tickets for|resolve|done|escalate|stuck|help)/i.test(text)) {
    return;
  }

  await say({
    text: ':wave: I\'m the ECM Field Bot. Type `help` to see available commands.',
    thread_ts: event.ts,
  });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

(async () => {
  await app.start();
  console.log('ECM Field Bot running (Socket Mode)');
})();
