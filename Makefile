.PHONY: install test test-cov lint typecheck run run-api run-agents up down logs ps

# ── Local dev ─────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=. --cov-report=term-missing

lint:
	ruff check .

typecheck:
	mypy .

run:
	python3 run.py

run-api:
	uvicorn api.app:app --reload --port 3001

run-agents:
	python3 main.py

# ── Docker ────────────────────────────────────────────────────────────────────

up:
	docker compose up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f

ps:
	docker compose ps
