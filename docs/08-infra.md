# 08 — Инфраструктура и развёртывание

## 8.1 Состав Docker Compose

| Сервис | Образ / Dockerfile | Назначение |
|---|---|---|
| `api` | `docker/api.Dockerfile` | FastAPI + Uvicorn, порт 8000 |
| `worker-cpu` | `docker/worker.Dockerfile` | Celery, очереди `cpu,io` |
| `worker-gpu` | `docker/worker-gpu.Dockerfile` (поверх `nvidia/cuda`) | Celery, очередь `gpu`, профиль `gpu` |
| `worker-net` | `docker/worker.Dockerfile` | Celery, очередь `net` (LLM), низкий конкурент |
| `beat` | тот же образ | Celery Beat — периодические задачи |
| `flower` | `mher/flower` | Мониторинг очередей, порт 5555 |
| `postgres` | `postgres:16-alpine` | БД |
| `redis` | `redis:7-alpine` | Брокер + result backend + rate limit |
| `minio` | `minio/minio` | S3-совместимое хранилище, профиль `s3` |

Профили Compose: `default` (api + cpu-воркер + postgres + redis),
`gpu` (добавляет `worker-gpu`), `s3` (добавляет `minio`), `observability`
(добавляет Prometheus + Grafana).

## 8.2 Образы

Многостадийная сборка, общий базовый слой для api и воркеров.

```dockerfile
# docker/worker.Dockerfile (структура)
FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# ffmpeg — обязателен; libgl/libglib — рантайм OpenCV

FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra worker

FROM deps AS runtime
COPY src/ /app/src/
USER 10001:10001                      # не root
ENTRYPOINT ["/app/docker/entrypoint.sh"]
```

Решения:

- **`ffmpeg` из пакетов Debian, не сборка из исходников.** Версия в bookworm свежая,
  экономия времени сборки важнее экзотических кодеков. Если понадобится NVENC —
  отдельный образ на базе `nvidia/cuda` с `ffmpeg` от `jrottenberg/ffmpeg`.
- **`uv` вместо pip.** Резолв и установка на порядок быстрее, `uv.lock` фиксирует всё дерево.
- **Модели не в образе.** Whisper/SER/детекторы качаются при первом старте
  в volume `models:/models` (`HF_HOME=/models`). Иначе образ распухает до 8+ ГБ
  и пересобирается на каждое изменение кода. Для air-gapped-развёртывания —
  `scripts/prefetch_models.py` и монтирование готового каталога.
- **Непривилегированный пользователь.** Обработка недоверенных медиафайлов —
  это парсинг сложных форматов; root тут не нужен.

> **Кто слушает очередь `gpu`**
>
> Распознавание речи (и позже — детекция лиц) маршрутизируется в очередь
> `gpu`, а сервис `worker-gpu` живёт за профилем `gpu`. Значит, в сборке
> по умолчанию эту очередь слушать было бы некому, и задача молча висела бы
> в брокере вечно — джоб в статусе «выполняется» без единой ошибки в логах.
>
> Поэтому CPU-воркер по умолчанию берёт и `gpu` тоже
> (`WORKER_CPU_QUEUES=cpu,io,gpu`), а при старте предупреждает, что CUDA
> не найдена и распознавание пойдёт на процессоре.
>
> Подняли профиль `gpu` — **сузьте** список до `cpu,io`. Иначе оба воркера
> слушают одну очередь, и задачи достаются им вперемешку: половина
> транскрипций уйдёт на процессор и будет идти впятеро дольше.

## 8.3 Тома

| Том | Монтируется в | Содержимое |
|---|---|---|
| `pgdata` | postgres | БД |
| `media` | api, все воркеры | загруженные видео и артефакты (при `STORAGE_BACKEND=local`) |
| `models` | воркеры | веса ML-моделей |
| `tmp` | воркеры | промежуточные файлы, чистится по TTL |

`media` обязан быть общим для api и воркеров — иначе воркер не увидит загруженный файл.
Именно поэтому в проде рекомендуется S3/MinIO: он снимает требование общей ФС
и открывает путь к горизонтальному масштабированию воркеров по разным хостам.

## 8.4 Переменные окружения (`.env.example`)

```bash
# --- Приложение ---
APP_ENV=dev                       # dev | staging | prod
LOG_LEVEL=INFO
LOG_FORMAT=json                   # json | console
SECRET_KEY=change-me

# --- База данных ---
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=highlights
POSTGRES_USER=hg
POSTGRES_PASSWORD=change-me
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# --- Redis / Celery ---
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
CELERY_TASK_TIME_LIMIT=3600

# --- Хранилище ---
STORAGE_BACKEND=local             # local | s3
STORAGE_LOCAL_ROOT=/media
S3_ENDPOINT_URL=http://minio:9000
S3_BUCKET=highlights
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_SIGNED_URL_TTL=900

# --- Медиа ---
FFMPEG_BIN=ffmpeg
FFPROBE_BIN=ffprobe
MAX_UPLOAD_BYTES=21474836480      # 20 GiB
MAX_DURATION_SEC=21600            # 6 ч
ALLOWED_CONTAINERS=mp4,mkv,mov

# --- Модели ---
DEVICE=auto                       # auto | cpu | cuda
HF_HOME=/models
WHISPER_MODEL=medium              # дефолт под CPU; с GPU ставьте large-v3
WHISPER_COMPUTE_TYPE=auto         # auto | float16 (GPU) | int8 (CPU)
WHISPER_BEAM_SIZE=5
WHISPER_VAD=true
WHISPER_NO_SPEECH_THRESHOLD=0.6   # выше — сегмент считается не речью
WHISPER_LANGUAGE=                 # пусто = автоопределение
WHISPER_CHUNK_SEC=600             # 0 = не резать файл на куски
WHISPER_CHUNK_OVERLAP_SEC=2
WHISPER_BATCH_SIZE=0              # 0 = авто (16 на GPU, 1 на CPU)
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
# Модели анализаторов признаков — ниже, в своём разделе.

# --- Детекция сцен ---
SCENE_DETECTOR=pyscenedetect      # pyscenedetect | ffmpeg
SCENE_THRESHOLD=27.0              # шкала PySceneDetect 0..255; для scdet пересчитывается
SCENE_MIN_LEN_SEC=1.5             # короче — сливается со следующей
SCENE_MAX_LEN_SEC=90.0            # длиннее — дробится на равные части
SCENE_DOWNSCALE=0                 # 0 = авто (делитель по ширине кадра)
SCENE_KEYFRAMES_ENABLED=true
SCENE_KEYFRAME_WIDTH=640
SCENE_KEYFRAME_WORKERS=8          # параллельных вызовов ffmpeg

# --- Скоринг ---
DEFAULT_PROFILE=default
SCORE_ALPHA=0.4                   # вес эвристики
SCORE_BETA=0.5                    # вес LLM
SCORE_GAMMA=0.1                   # бонус за согласие

# --- Анализаторы признаков ---
ANALYZERS_DISABLED=               # ключи через запятую: audio.loudness, vision.motion, …
VISION_SAMPLE_FPS=3.0             # кадров в секунду на видеоанализ
VISION_MAX_SIDE=320               # длинная сторона кадра при анализе
VISION_DETECT_FRAMES=5            # кадров на сцену для детекторов лиц и людей
AUDIO_EVENT_MODEL=MIT/ast-finetuned-audioset-10-10-0.4593
AUDIO_EVENT_WINDOW_SEC=10.0
AUDIO_EVENT_MAX_WINDOWS=3         # окон классификации на сцену
AUDIO_EVENT_THRESHOLD=0.1         # ниже — событие не пишется
SER_MODEL=superb/wav2vec2-base-superb-er
SER_MAX_SPEECH_SEC=5.0            # сколько речи на сцену подавать SER
FACE_DETECTOR=yunet               # yunet | haar
FACE_SCORE_THRESHOLD=0.7
PERSON_DETECTOR=ssdlite320_mobilenet_v3_large   # пусто = не считать людей
PERSON_SCORE_THRESHOLD=0.5

# --- LLM ---
ANTHROPIC_API_KEY=
LLM_MODEL=claude-opus-5
LLM_EFFORT=high                   # low | medium | high | xhigh | max
LLM_MAX_TOKENS=8000
LLM_BATCH_SIZE=40                 # сцен в одном запросе
LLM_USE_BATCH_API=true
LLM_ENABLED=true

# --- Рендер ---
RENDER_CRF=20
RENDER_PRESET=medium
RENDER_AUDIO_BITRATE=192k
VERTICAL_ENABLED=true
VERTICAL_WIDTH=1080
VERTICAL_HEIGHT=1920
BURN_SUBTITLES=false

# --- Лимиты ---
MAX_CONCURRENT_JOBS=2
RATE_LIMIT_RPM=60

# --- Retention (часы) ---
RETENTION_SOURCE_H=720
RETENTION_AUDIO_H=24
RETENTION_FRAMES_H=168
RETENTION_CLIPS_H=2160
```

Конфиг читается через `pydantic-settings` в `core/config.py`, единственный
`Settings`-объект, валидируется на старте — процесс не поднимется с некорректным конфигом.
Секреты в проде — из Docker secrets / внешнего vault, не из `.env`.

## 8.5 GPU

У GPU-воркера **отдельный образ**: `docker/worker-gpu.Dockerfile` поверх
`nvidia/cuda:12.6.2-cudnn-runtime`. Разница с CPU-образом — три гигабайта
CUDA-рантайма, которые машинам без видеокарты не нужны никогда.

```yaml
worker-gpu:
  profiles: ["gpu"]
  build:
    dockerfile: docker/worker-gpu.Dockerfile
  deploy:
    resources:
      reservations:
        devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }]
  environment:
    DEVICE: cuda
    CELERY_QUEUES: gpu
```

Замеры на RTX 5070 (Blackwell, sm_120) и 20-ядерном CPU. Столбца два
не для полноты: VAD не показывает модели тишину, поэтому время зависит
не от длины файла, а от того, **сколько в нём речи**.

| Конфигурация | Час фильма (речь ~⅓) | Час сплошной речи |
|---|---|---|
| `large-v3` / GPU / float16 / батч 16 | 1.54 мин | 6.20 мин |
| `large-v3` / GPU / float16 / без батчей | 6.33 мин | — |
| `medium` / CPU / int8 | 7.03 мин | ~17 мин |
| `small` / CPU / int8 | — | ~6.4 мин |

Практическое правило: `large-v3` на этой карте идёт со скоростью ×10
от времени речи. Подкаст на час GPU обработает за шесть минут, фильм
на два часа — примерно за столько же.

Батчи (`WHISPER_BATCH_SIZE`, по умолчанию авто) на GPU дают ×4: одно
тридцатисекундное окно не загружает карту и наполовину. На CPU выигрыша нет —
CTranslate2 и так занимает все ядра одним окном.

Без GPU система полностью работоспособна: `DEVICE=cpu`, `WHISPER_MODEL=small`
или `medium`, `WHISPER_COMPUTE_TYPE=auto` (даст int8). Это осознанный компромисс
качества ради доступности.

Один воркер держит **одну** копию каждой модели в памяти
(`worker_process_init` + `--concurrency=1..2` на GPU-очереди) — иначе VRAM кончится.

## 8.6 Масштабирование

| Узкое место | Что делать |
|---|---|
| Много джобов в очереди | Увеличить реплики `worker-cpu` |
| Медленная транскрипция | Реплики `worker-gpu` (по одному GPU на реплику) |
| Диск переполнен | Перейти на S3, ужесточить retention |
| БД тормозит на `scenes` | Партиционирование `scenes` по `job_id`, архивация старых джобов |
| Api упирается в загрузку | Presigned-URL: байты идут в S3 мимо приложения |

Приложение stateless — все реплики `api` равнозначны за любым балансировщиком.
Состояние — только в Postgres, Redis и storage.

## 8.7 Prod-развёртывание

- Обратный прокси (Caddy/Nginx/Traefik): TLS, `client_max_body_size`,
  таймауты ≥ 600 с на upload-эндпоинты, буферизация выключена для SSE.
- Health-чеки: `api` → `/health/ready`, воркеры → `celery inspect ping`.
- Миграции — отдельным one-shot контейнером `alembic upgrade head` **до** старта api;
  никогда не в `entrypoint` приложения (иначе гонка при нескольких репликах).
- Логи — stdout в JSON, сборщик снаружи (Loki/ELK).
- Бэкапы: `pg_dump` по расписанию + версионирование бакета S3.
- Graceful shutdown: `stop_grace_period: 120s` — воркер должен успеть доделать
  или корректно вернуть задачу в очередь (`acks_late` это гарантирует).

## 8.8 Разработка без Docker

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install uv && uv sync --extra dev --extra worker
# нужны локально: ffmpeg в PATH, postgres, redis
alembic upgrade head
uvicorn src.api.main:app --reload
celery -A src.workers.app worker -Q cpu,io,gpu,net -l info --pool=solo   # Windows: --pool=solo
```

На Windows Celery работает только с `--pool=solo` или `--pool=threads`;
для полноценной многопроцессной отладки используется Docker или WSL2.
Это учтено в `docs/09-testing.md` — тесты не зависят от пула.
