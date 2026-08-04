# syntax=docker/dockerfile:1.7
# Образ GPU-воркера: то же приложение, но поверх CUDA-рантайма.
#
# Отдельный Dockerfile, а не флаг в общем: разница не в паре пакетов.
# CUDA-рантайм и cuDNN — это три с лишним гигабайта, которые CPU-воркеру
# не нужны никогда, а держать один образ на обе роли значит платить эти
# гигабайты на каждой машине без GPU.
#
# База — `runtime`, а не `devel`: компилировать ядра нам не нужно,
# CTranslate2 приезжает готовым колесом. Разница между ними — ещё 4 ГБ.

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/models \
    TORCH_HOME=/models \
    DEBIAN_FRONTEND=noninteractive

# Ubuntu 24.04 несёт python3.12 в базовых репозиториях — ровно ту версию,
# что требует проект. Отдельный PPA не нужен.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Виртуальное окружение вместо установки в системный python: в Ubuntu 24.04
# системный python помечен как externally-managed, и pip в него откажется
# ставить пакеты (PEP 668). Venv — прямой и предсказуемый обход.
RUN python3.12 -m venv /venv
ENV PATH="/venv/bin:${PATH}"

WORKDIR /app

# ---------------------------------------------------------------- deps
FROM base AS deps

COPY pyproject.toml README.md ./
# Здесь, в отличие от CPU-образа, torch ставится обычным путём — с PyPI.
# Колесо по умолчанию как раз CUDA-сборка, и на этой машине она нужна:
# лишние два гигабайта покупают ускорение SER и AudioSet-классификатора
# на порядок. Отдельного шага с индексом не требуется.
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

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD Q="${CELERY_QUEUES:-gpu}"; celery -A workers.app inspect ping -d "${Q%%,*}@$HOSTNAME" || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["worker"]
