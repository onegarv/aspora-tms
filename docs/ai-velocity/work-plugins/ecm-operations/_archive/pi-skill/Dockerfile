# ECM Triage Skill - Containerized
FROM node:20-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Create app directory
WORKDIR /app

# Copy skill files (queries are copied by deploy.sh before build)
COPY SKILL.md ./
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY queries/ ./queries/
COPY entrypoint.sh ./

RUN chmod +x entrypoint.sh

# Environment variables (override at runtime)
ENV ANTHROPIC_API_KEY=""
ENV SLACK_BOT_TOKEN=""
ENV SLACK_CHANNEL_ID=""
ENV SPREADSHEET_ID="1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["triage"]
