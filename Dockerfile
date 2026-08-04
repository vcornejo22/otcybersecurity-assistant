# syntax=docker/dockerfile:1
FROM python:3.14-slim

WORKDIR /app

# Install curl for the compose healthcheck and uv for fast dependency management.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy manifest files first for dependency caching.
COPY pyproject.toml uv.lock README.md ./

# Sync dependencies using the lock file.
RUN uv sync --frozen

# Copy the application code.
COPY app/ ./app/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
