/**
 * ECM Skill Trigger — Minimal entry point for K8s CronJob
 *
 * Usage:
 *   npx tsx src/run.ts triage           # Run triage-and-assign skill
 *   npx tsx src/run.ts triage --dry-run # Run without posting to Slack
 *   npx tsx src/run.ts assign-tickets   # Run assign-tickets skill
 *
 * All logic lives in skills/*.md — this file is ONLY the trigger.
 */
import { config as loadEnv } from "dotenv";
loadEnv();

import { runSkill, postToSlack } from "./skill-runner.js";

const DEFAULT_MCP_URL =
  "http://internal-ecm-gateway-svc-464678860.eu-west-2.elb.amazonaws.com:80/mcp";

// Parse args
const args = process.argv.slice(2);
const skillName = args[0] || "triage-and-assign";
const dryRun = args.includes("--dry-run");

// Validate env
const required = ["OPENROUTER_API_KEY", "SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID"];
const missing = required.filter((k) => !process.env[k]);
if (missing.length > 0) {
  console.error("Missing env vars:", missing.join(", "));
  process.exit(1);
}

async function main() {
  console.log(`Running skill: ${skillName}`);
  if (dryRun) console.log("(dry run — no Slack post)\n");

  const config = {
    openrouterApiKey: process.env.OPENROUTER_API_KEY!,
    slackBotToken: process.env.SLACK_BOT_TOKEN!,
    slackChannelId: process.env.SLACK_CHANNEL_ID!,
    mcpGatewayUrl: process.env.ECM_GATEWAY_URL || DEFAULT_MCP_URL,
    model: process.env.LLM_MODEL || "anthropic/claude-sonnet-4",
  };

  try {
    const output = await runSkill(skillName, config);

    // Split output into multiple messages if ---SPLIT--- is present
    const messages = output.split("---SPLIT---").map((m) => m.trim()).filter(Boolean);

    for (let i = 0; i < messages.length; i++) {
      console.log(`\n=== MESSAGE ${i + 1}/${messages.length} ===\n`);
      console.log(messages[i]);
    }
    console.log("\n=== END ===\n");

    if (!dryRun) {
      console.log(`Posting ${messages.length} message(s) to Slack...`);
      for (let i = 0; i < messages.length; i++) {
        const result = await postToSlack(messages[i], config);
        console.log(`  Message ${i + 1}: ${result.ok ? "Posted." : `Error: ${result.error}`}`);
        // Small delay between messages
        if (i < messages.length - 1) {
          await new Promise((r) => setTimeout(r, 1000));
        }
      }
    }
  } catch (e) {
    console.error("Failed:", e);
    process.exit(1);
  }
}

main();
