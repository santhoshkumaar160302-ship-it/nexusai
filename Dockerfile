# syntax=docker/dockerfile:1
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY nexusai ./nexusai
COPY scripts ./scripts
COPY data ./data
COPY wsgi.py .

# Build metadata: CI passes the git commit so /health can prove which version is live.
ARG GIT_SHA=dev
ENV NEXUS_BUILD_SHA=$GIT_SHA
LABEL org.opencontainers.image.title="nexusai" \
      org.opencontainers.image.description="Fault-tolerant, cost-aware LLM gateway with RAG" \
      org.opencontainers.image.revision=$GIT_SHA

RUN useradd --create-home --uid 10001 nexus
USER nexus

# PORT is set by most hosting platforms (Render, Cloud Run, Railway); 8000 locally.
# Keep one worker with the in-memory store (state is per process); raise
# NEXUS_WORKERS only when DATABASE_URL and REDIS_URL are set.
ENV PORT=8000 NEXUS_WORKERS=1 NEXUS_THREADS=8
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

CMD exec gunicorn --bind "0.0.0.0:${PORT}" --workers "${NEXUS_WORKERS}" --threads "${NEXUS_THREADS}" \
    --timeout 120 --access-logfile - wsgi:app
