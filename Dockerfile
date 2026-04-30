# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build deps separately so source changes don't bust the dep layer.
COPY pyproject.toml README.md ./
COPY llmrouter ./llmrouter
RUN pip install --no-cache-dir .

# Default config ships at /etc/llmrouter/config.yaml so a bare `docker run`
# works out-of-the-box. Mount your own at the same path to override.
RUN mkdir -p /etc/llmrouter /data \
    && cp /app/llmrouter/config.default.yaml /etc/llmrouter/config.yaml

ENV LLMROUTER_CONFIG=/etc/llmrouter/config.yaml \
    LLMROUTER_DB_PATH=/data/llmrouter.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["llmrouter"]
