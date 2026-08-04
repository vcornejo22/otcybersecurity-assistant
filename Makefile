.PHONY: build run test lint format clean build-image up down rebuild reset ingest logs

# ── Local development ────────────────────────────────────────────
build:
	uv sync

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf .venv __pycache__ .pytest_cache app/**/__pycache__

# ── Docker workflows ──────────────────────────────────────────────
build-image:
	docker build -t otcybersecurity-assistant .

up:
	docker compose up --build -d

down:
	docker compose down

rebuild:
	docker compose down && docker compose build --no-cache && docker compose up -d

reset:
	docker compose down -v && docker compose build --no-cache && docker compose up -d

logs:
	docker compose logs -f api

# ── Data ingestion ───────────────────────────────────────────────
ingest:
	python scripts/ingest.py
