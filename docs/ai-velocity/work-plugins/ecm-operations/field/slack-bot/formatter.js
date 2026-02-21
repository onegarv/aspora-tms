'use strict';

const SLACK_CHAR_LIMIT = 3000;

/**
 * Convert claude --print output (standard markdown) to Slack mrkdwn.
 * Follows shared/config/slack-formatting.md rules.
 */
function formatForSlack(text) {
  if (!text) return '_No output from Claude._';

  let out = text;

  // **bold** → *bold* (Slack mrkdwn) — do this before header conversion
  out = out.replace(/\*\*(.+?)\*\*/g, '*$1*');

  // Markdown headers → bold text (Slack has no headers)
  // ### H3 → *text*
  // ## H2 → *text*
  // # H1 → *text*
  out = out.replace(/^#{1,3}\s+(.+)$/gm, '*$1*');

  // Horizontal rules → Slack separator
  out = out.replace(/^---+$/gm, '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  // Convert markdown tables to code blocks
  out = convertTablesToCodeBlocks(out);

  // Markdown links [text](url) → <url|text>
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<$2|$1>');

  // Trim trailing whitespace per line
  out = out.replace(/[ \t]+$/gm, '');

  // Collapse 3+ consecutive blank lines into 2
  out = out.replace(/\n{3,}/g, '\n\n');

  return out;
}

/**
 * Detect markdown tables (lines with | delimiters) and wrap them in code blocks.
 * Strips the separator row (|---|---|).
 */
function convertTablesToCodeBlocks(text) {
  const lines = text.split('\n');
  const result = [];
  let tableLines = [];
  let inTable = false;

  for (const line of lines) {
    const isTableRow = /^\s*\|/.test(line);

    if (isTableRow) {
      if (!inTable) inTable = true;
      // Skip separator rows like |---|---|
      if (/^\s*\|[\s\-:|]+\|\s*$/.test(line)) continue;
      // Strip leading/trailing pipes and trim cells
      const cells = line
        .replace(/^\s*\|\s*/, '')
        .replace(/\s*\|\s*$/, '')
        .split(/\s*\|\s*/);
      tableLines.push(cells);
    } else {
      if (inTable && tableLines.length > 0) {
        result.push(formatTableAsCodeBlock(tableLines));
        tableLines = [];
        inTable = false;
      }
      result.push(line);
    }
  }

  // Flush any remaining table
  if (inTable && tableLines.length > 0) {
    result.push(formatTableAsCodeBlock(tableLines));
  }

  return result.join('\n');
}

/**
 * Render table rows as a fixed-width code block with aligned columns.
 */
function formatTableAsCodeBlock(rows) {
  if (rows.length === 0) return '';

  // Calculate max width per column
  const colCount = Math.max(...rows.map((r) => r.length));
  const widths = Array(colCount).fill(0);
  for (const row of rows) {
    for (let i = 0; i < row.length; i++) {
      widths[i] = Math.max(widths[i], (row[i] || '').length);
    }
  }

  const formatted = rows.map((row) =>
    row.map((cell, i) => (cell || '').padEnd(widths[i])).join('  ')
  );

  return '```\n' + formatted.join('\n') + '\n```';
}

/**
 * Truncate text to Slack's limit, appending an ellipsis marker.
 */
function truncate(text, limit = SLACK_CHAR_LIMIT) {
  if (text.length <= limit) return text;
  const suffix = '\n\n_... truncated (use `order {id}` for full details)_';
  const budget = limit - suffix.length;
  const cutoff = text.lastIndexOf('\n', budget);
  const end = cutoff > budget * 0.5 ? cutoff : budget;
  return text.slice(0, end) + suffix;
}

/**
 * Split a long response into [main, ...threadReplies].
 * Main gets the first section (up to limit), rest goes into thread chunks.
 */
function splitForThread(text, limit = SLACK_CHAR_LIMIT) {
  if (text.length <= limit) return [text];

  // Split on double newlines (section breaks)
  const sections = text.split(/\n{2,}/);
  const chunks = [];
  let current = '';

  for (const section of sections) {
    if (current.length + section.length + 2 > limit && current.length > 0) {
      chunks.push(current.trim());
      current = section;
    } else {
      current += (current ? '\n\n' : '') + section;
    }
  }
  if (current.trim()) chunks.push(current.trim());

  return chunks;
}

module.exports = { formatForSlack, truncate, splitForThread };
