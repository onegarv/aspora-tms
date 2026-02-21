/**
 * Skill Runner — Minimal trigger that lets LLM execute skills via MCP
 *
 * Philosophy: "In skill-based frameworks, the deliverable is a SKILL, not code"
 *
 * This file is ONLY a trigger. All execution logic lives in skills/*.md
 * The LLM reads the skill and executes it using MCP tools.
 */
import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

interface SkillRunnerConfig {
  openrouterApiKey: string;
  slackBotToken: string;
  slackChannelId: string;
  mcpGatewayUrl: string;
  model?: string;
}

interface MCPTool {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

/**
 * Load skill file content
 */
function loadSkill(skillName: string): string {
  const skillsDir = resolve(__dirname, "../../skills");
  const path = resolve(skillsDir, `${skillName}.md`);
  return readFileSync(path, "utf-8");
}

/**
 * Get available MCP tools from ecm-gateway
 */
async function getMCPTools(gatewayUrl: string): Promise<MCPTool[]> {
  const response = await fetch(gatewayUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json, text/event-stream",
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "tools/list",
      id: 1,
    }),
  });

  const text = await response.text();
  const jsonText = text.startsWith("data: ")
    ? text.split("\n").find(l => l.startsWith("data: "))?.replace("data: ", "") || "{}"
    : text;

  const result = JSON.parse(jsonText);
  return result.result?.tools || [];
}

/**
 * Call an MCP tool
 */
async function callMCPTool(
  gatewayUrl: string,
  name: string,
  args: Record<string, unknown>
): Promise<string> {
  const response = await fetch(gatewayUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json, text/event-stream",
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "tools/call",
      params: { name, arguments: args },
      id: Date.now(),
    }),
  });

  const text = await response.text();
  const jsonText = text.startsWith("data: ")
    ? text.split("\n").filter(l => l.startsWith("data: ")).pop()?.replace("data: ", "") || "{}"
    : text;

  const result = JSON.parse(jsonText);
  if (result.error) {
    throw new Error(`MCP Error: ${result.error.message}`);
  }

  const content = result.result?.content?.find((c: { type: string }) => c.type === "text");
  return content?.text || JSON.stringify(result.result);
}

/**
 * Convert MCP tools to OpenAI function format
 */
function toOpenAITools(mcpTools: MCPTool[]) {
  return mcpTools.map(tool => ({
    type: "function" as const,
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    },
  }));
}

/**
 * Run a skill with LLM + MCP tools
 */
export async function runSkill(
  skillName: string,
  config: SkillRunnerConfig
): Promise<string> {
  // Load skill and guardrails
  const guardrails = loadSkill("guardrails");
  const skill = loadSkill(skillName);

  // Get MCP tools
  const mcpTools = await getMCPTools(config.mcpGatewayUrl);
  const tools = toOpenAITools(mcpTools);

  // System prompt: guardrails + skill + strict instructions
  const systemPrompt = `${guardrails}

---

${skill}

---

# EXECUTION RULES (CRITICAL)

1. **DO NOT EXPLORE** — Do not call list_databases, list_schemas, list_tables. The skill above has the exact SQL queries you need.

2. **EXECUTE DIRECTLY** — Run the SQL queries exactly as written in the skill. Do not modify them.

3. **STEPS:**
   - Step 1: Run the stuck orders SQL query via redshift_execute_sql_tool
   - Step 2: Get Assignments and Agents tabs from Google Sheet via sheets_get_sheet_data
   - Step 3: Run a TRENDING query to show backlog vs inflow:
     \`\`\`sql
     SELECT
       CASE
         WHEN created_at::date < CURRENT_DATE - 1 THEN 'backlog'
         WHEN created_at::date = CURRENT_DATE - 1 THEN 'yesterday'
         ELSE 'today'
       END as period,
       COUNT(*) as order_count,
       SUM(meta_postscript_pricing_info_send_amount) as total_amount
     FROM orders_goms
     WHERE status IN ('PROCESSING_DEAL_IN', 'PENDING', 'FAILED')
       AND meta_postscript_pricing_info_send_currency IN ('AED', 'GBP', 'EUR')
       AND created_at >= CURRENT_DATE - 30
       AND sub_state IN ('FULFILLMENT_PENDING', 'REFUND_TRIGGERED', 'TRIGGER_REFUND', 'MANUAL_REVIEW', 'AWAIT_EXTERNAL_ACTION', 'AWAIT_RETRY_INTENT')
     GROUP BY 1 ORDER BY 1;
     \`\`\`
   - Step 4: Generate the briefing with a TREND CHART

4. **AGENT CAPACITY OVERRIDE (CRITICAL):**
   - Agents have UNLIMITED capacity — completely ignore "Max Tickets" column
   - Do NOT show "OVERLOADED", "Near Capacity", or any capacity warnings
   - Do NOT show "Max" or "Available" columns in agent tables
   - Show ONLY: Agent | Team | Current | Status (where Status is just "Active")
   - Do NOT mention capacity limits or workload redistribution
   - Do NOT block or warn about assignments due to capacity

5. **OUTPUT FORMAT** — Your response must contain TWO separate Slack messages separated by "---SPLIT---":

   **MESSAGE 1 (Triage Briefing):**
   - Start with 🎯 **ECM Triage Report**
   - Overview numbers (stuck, assigned, critical)
   - Top priority unassigned orders table
   - Agent workloads (current count only)
   - Dashboard link

   **MESSAGE 2 (Trend Analysis):**
   - Start with 📈 **ECM Trend Analysis**
   - ASCII bar chart showing:
     Backlog (>2d): ████████ X orders (Y amount)
     Yesterday:     ████ X orders (Y amount)
     Today:         ██ X orders (Y amount)
   - Trend insight (increasing/decreasing, % change)
   - Breakdown by currency and diagnosis type

   Format your response as:
   [Message 1 content]
   ---SPLIT---
   [Message 2 content]

   - NO preamble, NO explanations
   - NO code blocks, NO SQL

6. **STOP AFTER OUTPUT** — Once you generate both messages, you are DONE.`;

  // Call LLM with tool loop
  const model = config.model || "anthropic/claude-sonnet-4";
  let messages: Array<{ role: string; content: string; tool_calls?: unknown[]; tool_call_id?: string; name?: string }> = [
    { role: "system", content: systemPrompt },
    { role: "user", content: `Execute the ${skillName} skill now. Today is ${new Date().toISOString().split("T")[0]}.` },
  ];

  const maxIterations = 20;
  let iteration = 0;

  while (iteration < maxIterations) {
    iteration++;
    console.log(`  [Iteration ${iteration}/${maxIterations}]`);

    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${config.openrouterApiKey}`,
      },
      body: JSON.stringify({
        model,
        messages,
        tools,
        tool_choice: "auto",
        max_tokens: 4096,
      }),
    });

    const result = await response.json();
    const choice = result.choices?.[0];

    if (!choice) {
      throw new Error(`LLM error: ${JSON.stringify(result)}`);
    }

    const assistantMessage = choice.message;
    messages.push(assistantMessage);

    // Check if done (no tool calls)
    if (!assistantMessage.tool_calls || assistantMessage.tool_calls.length === 0) {
      // Clean up any preamble - find first emoji or markdown header
      let content = assistantMessage.content || "";
      const emojiStart = content.search(/[🎯📊⚡💼🚨📈🔄✅❌⚠️🔥💰👥📅🎫]/);
      if (emojiStart > 0) {
        content = content.slice(emojiStart);
      }
      return content;
    }

    // Execute tool calls
    for (const toolCall of assistantMessage.tool_calls) {
      const toolName = toolCall.function.name;
      const toolArgs = JSON.parse(toolCall.function.arguments || "{}");

      console.log(`  [Tool] ${toolName}`);

      try {
        const toolResult = await callMCPTool(config.mcpGatewayUrl, toolName, toolArgs);
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          name: toolName,
          content: toolResult,
        });
      } catch (e) {
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          name: toolName,
          content: `Error: ${e instanceof Error ? e.message : String(e)}`,
        });
      }
    }
  }

  throw new Error("Max iterations reached without completion");
}

/**
 * Post message to Slack
 */
export async function postToSlack(
  message: string,
  config: { slackBotToken: string; slackChannelId: string }
): Promise<{ ok: boolean; error?: string }> {
  const response = await fetch("https://slack.com/api/chat.postMessage", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${config.slackBotToken}`,
    },
    body: JSON.stringify({
      channel: config.slackChannelId,
      text: message,
      unfurl_links: false,
    }),
  });

  return response.json();
}
