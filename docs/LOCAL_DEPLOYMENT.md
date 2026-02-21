# Local Deployment Guide — Aspora TMS

## Prerequisites

- **Docker Desktop** 4.x+ (for `make up`)
- **Python 3.11+** (for local dev without Docker)
- **Git** (to check out branches)

---

## Quick Start (Docker)

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd aspora-tms

# 2. Start the full stack (redis + postgres + api + agents)
make up

# 3. Confirm services are healthy
make ps

# 4. Dashboard API is live at:
#    http://localhost:8000/docs
```

Tear down (removes volumes):

```bash
make down
```

---

## Running Without Docker

```bash
# Install project + dev deps
make install

# Run all tests
make test

# Start just the API (no Redis required — uses InMemoryBus)
make run-api
# → http://localhost:8000/docs

# Start agents (InMemoryBus by default)
make run-agents

# Start agents with Redis bus
ASPORA_BUS_TYPE=redis ASPORA_REDIS_URL=redis://localhost:6379/0 make run-agents
```

---

## Environment Variables

The `.env` file is used by docker-compose. For local runs outside Docker, export these variables or copy `.env.example` to `.env` and adjust `ASPORA_REDIS_URL` to `redis://localhost:6379/0`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ASPORA_BUS_TYPE` | `memory` | `memory` or `redis` |
| `ASPORA_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `ASPORA_POSTGRES_URL` | — | PostgreSQL async URL |
| `ASPORA_JWT_SECRET` | — | JWT signing secret (**change in prod**) |
| `ASPORA_PREFUNDING_BUFFER_PCT` | `0.10` | Safety buffer for fund transfers |
| `ASPORA_FORECAST_LOOKBACK_WEEKS` | `8` | EMA lookback for volume forecasts |
| `ASPORA_METABASE_URL` | _(empty)_ | Leave empty to use fallback rates |

See `.env.example` for the full list.

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make install` | `pip install -e ".[dev]"` |
| `make test` | Run all 246 tests |
| `make test-cov` | Tests with coverage report |
| `make lint` | `ruff check .` |
| `make typecheck` | `mypy .` |
| `make run-api` | Uvicorn dev server on :8000 |
| `make run-agents` | Start ops + liquidity agents |
| `make up` | `docker compose up --build -d` |
| `make down` | `docker compose down -v` |
| `make logs` | Tail all container logs |
| `make ps` | Show container health |

---

## Smoke Test: Event Bus

After `make up`, confirm the API is connected to the bus:

```bash
# Should return 200 with an event list (empty is fine)
curl -s http://localhost:8000/api/v1/events | python3 -m json.tool

# JWT-protected endpoints return 403 without a token (expected)
curl -s http://localhost:8000/api/v1/proposals
```

---

## Full-Merge Deployment Checklist

When `integration/fx` and remaining branches are merged into `master`:

### 1. Add FX Agent service

```yaml
# docker-compose.yml
fx-agent:
  build:
    context: .
    target: agents
  command: python -m agents.fx.runner
  env_file: .env
  depends_on:
    redis:
      condition: service_healthy
```

### 2. Run Alembic migrations

```bash
alembic upgrade head
```

### 3. Activate TimescaleDB extension

```sql
-- Connect to aspora_tms and run:
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

### 4. Add monitoring stack (optional)

```yaml
# docker-compose.yml
prometheus:
  image: prom/prometheus:latest
  volumes:
    - ./infra/prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"
  depends_on:
    - prometheus
```

### 5. Lock down CORS

Update `api/app.py`:

```python
allow_origins=["https://your-frontend-domain.com"]
```

### 6. Set strong JWT secret

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
# Copy output into ASPORA_JWT_SECRET in your production .env
```
