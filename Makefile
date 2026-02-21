.PHONY: install test test-cov lint typecheck run-api run-agents up down logs ps

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

run-api:
	uvicorn api.app:app --reload --port 8000

run-agents:
	python main.py

# ── Docker ────────────────────────────────────────────────────────────────────

up:
	docker compose up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f

ps:
	docker compose ps
