# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11-slim

# ---------- Base (deps only, cached) ----------
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1         PYTHONPATH=/app/src         PYTHONUNBUFFERED=1         PIP_DISABLE_PIP_VERSION_CHECK=1         STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Create non-root user early (stable UID helps on some hosts)
RUN useradd -m -u 10001 appuser

# System libs needed by some wheels (e.g., xgboost -> libgomp)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked         python -m pip install -U pip &&         python -m pip install -r requirements.txt

# ---------- Test stage ----------
FROM base AS test
COPY requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked         python -m pip install -r requirements-dev.txt
COPY . .
RUN python -m ruff check . &&         python -m ruff format --check . &&         python -m pytest -q

# ---------- Runtime: API ----------
FROM base AS api
COPY --chown=appuser:appuser . /app
RUN install -d -o appuser -g appuser /app/data
USER appuser
EXPOSE 8000
CMD ["python","-m","uvicorn","fraud_dashboard.api.main:app","--host","0.0.0.0","--port","8000"]

# ---------- Runtime: UI (default) ----------
FROM base AS ui
COPY --chown=appuser:appuser . /app
USER appuser
EXPOSE 8501
CMD ["python","-m","streamlit","run","app.py","--server.address","0.0.0.0","--server.port","8501"]
