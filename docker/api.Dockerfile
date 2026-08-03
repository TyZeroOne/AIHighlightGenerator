# syntax=docker/dockerfile:1.7
# Образ API.
#
# На Этапе 1 предполагалось, что API обойдётся без ffmpeg. Этап 2 это
# опроверг: DoD требует отвечать 415 на переименованный .txt **сразу**,
# в ответе на POST /upload. Единственный надёжный способ отличить видео
# от не-видео — прочитать файл тем же инструментом, который потом будет
# его обрабатывать. Значит, ffprobe нужен здесь.
#
# Альтернатива — проверять асинхронно в воркере и отдавать 202 — сделала бы
# невозможным синхронный отказ и заставила бы клиента опрашивать статус
# ради ответа на вопрос «это вообще видео?».
#
# Из пакета ffmpeg фактически используется только ffprobe; отдельного
# пакета для него в Debian нет.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------- deps
FROM base AS deps

# Зависимости ставятся ДО копирования кода. Слой с зависимостями кешируется
# и переиспользуется при каждом изменении исходников — пересборка занимает
# секунды вместо минут.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install "hatchling>=1.25" && \
    pip install ".[api]"

# ------------------------------------------------------------- runtime
FROM deps AS runtime

COPY src/ /app/src/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/

# Непривилегированный пользователь: приложение принимает файлы извне,
# root ему не нужен ни для чего.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /media /tmp/highlight-generator && \
    chown -R appuser:appuser /app /media /tmp/highlight-generator
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
