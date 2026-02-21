# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for asyncpg + cryptography
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install project + all deps (including dev extras for test stage)
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

# ── Stage 2: api ──────────────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage 3: agents ───────────────────────────────────────────────────────────
FROM base AS agents
CMD ["python", "main.py"]
