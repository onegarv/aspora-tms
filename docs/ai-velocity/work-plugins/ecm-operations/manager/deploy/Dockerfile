# ECM Manager Agent - Containerized
# Mirrors repo layout so ../shared/ resolves from manager/
FROM node:20-slim

RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app

# Mirror repo structure: manager/ and shared/ as siblings
# This ensures ../shared/ works from manager/skills/*.md
COPY shared/ ./shared/
COPY manager/ ./manager/

# Root-level files that skills reference via ../
COPY CLAUDE.md ./CLAUDE.md
COPY DECISIONS.md ./DECISIONS.md
COPY ESCALATION.md ./ESCALATION.md

# MCP server config (ecm-gateway HTTP endpoint — DEC-002)
COPY .mcp.json ./.mcp.json

# Claude settings — pre-approve MCP tools + Bash(curl) for headless mode
COPY .claude/ ./.claude/

# Entrypoint runs from repo root so relative paths work
COPY manager/deploy/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

# OpenRouter config (override at runtime via env or .env file)
ENV ANTHROPIC_BASE_URL="https://openrouter.ai/api"
ENV ANTHROPIC_AUTH_TOKEN=""
ENV ANTHROPIC_API_KEY=""
ENV SLACK_BOT_TOKEN=""
ENV SLACK_CHANNEL_ID="C0AD6C36LVC"
ENV SPREADSHEET_ID="1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["daily"]
