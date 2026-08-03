# syntax=docker/dockerfile:1.7
# Образ Celery-воркера: то же приложение + ffmpeg + рантайм OpenCV.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/models

# ffmpeg — из пакетов Debian, а не сборкой из исходников: версия в bookworm
# свежая, а экономия 20 минут на каждой сборке важнее экзотических кодеков.
# libgl1/libglib2.0-0 — рантайм-библиотеки, без которых `import cv2` падает
# (Этап 5); ставим сразу, чтобы образ не менялся посреди разработки.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------- deps
FROM base AS deps

COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install "hatchling>=1.25" && \
    pip install ".[worker]"

# ------------------------------------------------------------- runtime
FROM deps AS runtime

COPY src/ /app/src/
COPY docker/entrypoint.sh /app/docker/entrypoint.sh

RUN chmod +x /app/docker/entrypoint.sh && \
    useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /media /models /tmp/work && \
    chown -R appuser:appuser /app /media /models /tmp/work
USER appuser

# Имя ноды строится по той же формуле, что в entrypoint.sh: <первая очередь>@<хост>.
# Расхождение между этими двумя местами = контейнер, который вечно unhealthy
# при живом воркере, — проверено на практике.
HEALTHCHECK --interval=30s --timeout=15s --start-period=40s --retries=3 \
    CMD Q="${CELERY_QUEUES:-cpu}"; celery -A workers.app inspect ping -d "${Q%%,*}@$HOSTNAME" || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["worker"]
