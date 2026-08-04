# AI Highlight Generator

Автоматический поиск самых интересных моментов в длинных видео и генерация готовых клипов
(горизонтальных и вертикальных 1080×1920) с субтитрами, заголовком, описанием и хештегами.

**Статус:** Этап 5 завершён, Этап 6 работает на бесплатных моделях.
Видео загружается, режется на сцены, распознаётся речью с пословными
таймингами, каждая сцена описывается двадцатью с лишним признаками
(громкость, паузы, темп и эмоция речи, звуковые события, движение, монтаж,
лица) — и получает оценку интересности от языковой модели.

509 тестов зелёные локально (в образе воркера, с настоящими весами
моделей, — 542). Замер на двухчасовом видео (CPU, без видеокарты):
1572 сцены, 53 468 признаков за 37.5 минуты при пике памяти 2.5 ГБ.
WER на живой русской речи 0.13 (`large-v3`).
Разработка ведётся строго по этапам, см. [docs/10-roadmap.md](docs/10-roadmap.md).

К каждому этапу пишется **обучающий разбор** — что сделано, зачем именно так,
какие были альтернативы и чем они хуже: [docs/stages/](docs/stages/).

---

## Что делает система

```
видео → сцены → аудио → речь → CV-признаки → скоринг → LLM → отбор → нарезка → экспорт
```

1. Принимает локальное видео (`mp4` / `mkv` / `mov`) — в будущем YouTube / Twitch / Kick без изменения ядра.
2. Режет на сцены, извлекает аудио, распознаёт речь (`faster-whisper`).
3. Считает признаки: эмоции речи, громкость, паузы, темп речи, смены сцен, движение камеры,
   лица, количество людей, динамику кадра.
4. Считает эвристическую оценку интересности каждой сцены.
5. Отдаёт описание сцен LLM (Claude), получает строгий JSON со `score` / `category` / `reason`.
6. Сливает оценки, выбирает лучшие моменты с учётом длительности и разнообразия.
7. Нарезает клипы через FFmpeg, делает вертикальную версию со smart-reframe.
8. Генерирует субтитры (SRT/ASS), заголовок, описание, хештеги.

---

## Документация

| Документ | О чём |
|---|---|
| [01 — Обзор и границы](docs/01-vision-scope.md) | Цели, не-цели, сценарии, критерии успеха |
| [02 — Архитектура](docs/02-architecture.md) | Clean Architecture, модули, порты, `SourceProvider`, плагины анализаторов |
| [03 — Модель данных](docs/03-data-model.md) | `User`, `Video`, `Job`, `Scene`, `Clip` + служебные таблицы |
| [04 — Конвейер и очереди](docs/04-pipeline.md) | Celery, граф этапов, идемпотентность, ретраи |
| [05 — Признаки и скоринг](docs/05-scoring.md) | Полный список фич, нормализация, формула, отбор моментов |
| [06 — LLM-слой](docs/06-llm.md) | Контракт, JSON-схема, промпты, кеширование, Batch API, цена |
| [07 — REST API](docs/07-api.md) | Эндпоинты, схемы, коды ошибок, загрузка больших файлов |
| [08 — Инфраструктура](docs/08-infra.md) | Docker Compose, сервисы, переменные окружения, GPU |
| [09 — Тестирование](docs/09-testing.md) | Пирамида тестов, фикстуры, golden-файлы, CI |
| [10 — Дорожная карта](docs/10-roadmap.md) | Этапы 1–10 с Definition of Done |

### Обучающие разборы этапов

| Этап | Разбор |
|---|---|
| 1 — Каркас архитектуры | [01_Stage1_Architecture.md](docs/stages/01_Stage1_Architecture.md) |
| 2 — Загрузка видео | [02_Stage2_VideoUpload.md](docs/stages/02_Stage2_VideoUpload.md) |
| 3 — Разбиение на сцены | [03_Stage3_SceneDetection.md](docs/stages/03_Stage3_SceneDetection.md) |
| 4 — Распознавание речи | [04_Stage4_Speech.md](docs/stages/04_Stage4_Speech.md) |
| 5 — Признаки сцен | [05_Stage5_Features.md](docs/stages/05_Stage5_Features.md) |
| 6 — LLM-анализ | [06_Stage6_LLM.md](docs/stages/06_Stage6_LLM.md) |

---

## Технологический стек

| Слой | Технология |
|---|---|
| Язык | Python 3.12 |
| API | FastAPI + Uvicorn, Pydantic v2 |
| Очередь | Celery 5 + Redis (broker + result backend) |
| БД | PostgreSQL 16 + SQLAlchemy 2 (async) + Alembic |
| Видео | FFmpeg 7, OpenCV 4, PySceneDetect |
| Речь | faster-whisper (CTranslate2) |
| ML | PyTorch 2 + `transformers`: AudioSet-классификатор (звуковые события), wav2vec2-SER (эмоция), SSDLite (люди); лица — YuNet прямо из OpenCV |
| LLM | Claude (`claude-opus-5`) через официальный SDK `anthropic` |
| Хранилище | Локальная ФС (dev) / S3-совместимое MinIO (prod) — за портом `ObjectStorage` |
| Контейнеризация | Docker + Docker Compose |

---

## Быстрый старт

### В Docker

```bash
cp .env.example .env
docker compose up -d --build
```

- API: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health/ready>
- Flower (профиль `tools`): <http://localhost:5555>

Проверка воркера:

```bash
docker compose exec worker-cpu python -c "from workers.tasks.system import ping; print(ping.delay().get(timeout=10))"
```

Создать пользователя и получить API-ключ:

```bash
docker compose exec api python -m manage create-user --email you@example.com
```

Загрузить видео:

```bash
curl -X POST http://localhost:8000/api/v1/upload -H "X-API-Key: hg_..." -F "file=@movie.mp4"
```

Разбить загруженное видео на сцены (`POST /analyze` появится на Этапе 10):

```bash
docker compose exec worker-cpu python -m manage detect-scenes --video-id <uuid> --enqueue
```

Без `--enqueue` этап выполняется на месте и печатает найденные сцены;
с ним — уходит в очередь `cpu` и выполняется на воркере.

Распознать речь (сцены → аудио → транскрипт одной цепочкой):

```bash
docker compose exec worker-cpu python -m manage transcribe --video-id <uuid> --enqueue
```

```bash
curl "http://localhost:8000/api/v1/jobs/<job_id>/transcript?format=srt" -H "X-API-Key: hg_..."
```

Посчитать признаки сцен — громкость, паузы, темп речи, движение, склейки,
динамику кадра:

```bash
docker compose exec worker-cpu python -m manage analyze --job-id <uuid> --groups audio,speech,vision
```

Группы независимы и могут считаться на разных воркерах. При первом запуске
качаются веса — в том `models`, рядом с моделью распознавания речи:
AudioSet-классификатор 331 МБ, SER 722 МБ, детектор людей 14 МБ,
YuNet 256 КБ. Итого чуть больше гигабайта.

Лишний анализатор отключается через `ANALYZERS_DISABLED` — например,
`ANALYZERS_DISABLED=audio.events,vision.faces` убирает самые дорогие
и возвращает этап к арифметике без нейросетей.

По умолчанию всё крутится на процессоре: CPU-воркер слушает и очередь `gpu`,
модель — `medium`. Час речи распознаётся примерно за 17 минут.

На видеокарту уходят и распознавание речи, и все три группы признаков:
в каждой есть нейросеть. Нужен NVIDIA Container Toolkit; распознавание
ускоряется вчетверо и позволяет `large-v3`:

```bash
docker compose --profile gpu up -d --build worker-gpu
```

В этом случае в `.env` нужны две строки, иначе половина задач всё равно
достанется процессору:

```bash
WORKER_CPU_QUEUES=cpu,io
WHISPER_MODEL=large-v3
```

Модель качается при первом запуске в том `models`: `large-v3` — это 3 ГБ.

Хранилище S3 вместо локальной ФС:

```bash
docker compose --profile s3 up -d minio
```

Затем выставить `STORAGE_BACKEND=s3` в `.env` и перезапустить `api`.
Консоль MinIO: <http://localhost:9001> (minioadmin / minioadmin).

### Локально, без Docker

Нужны Python 3.12, а также PostgreSQL и Redis (или поднятые
`docker compose up -d postgres redis`).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[api,dev]"
```

```bash
python -m pytest -q
```

Extra `worker` здесь намеренно не ставится: он тянет PyTorch (~2 ГБ),
который нужен трём анализаторам из девяти. Тесты этих анализаторов
работают на фейковых моделях, а те, которым нужны настоящие веса,
помечены `integration` и пропускаются, когда библиотек нет. Полный прогон
со всеми моделями делается в образе воркера.

```bash
python -m uvicorn api.main:app --reload --port 8000
```

На Windows Celery работает только с `--pool=solo`:

```bash
celery -A workers.app worker -Q cpu,io -l info --pool=solo
```

---

## Принципы

- **SOLID** — каждый модуль реализует один порт из `core/ports`, зависимости инвертированы.
- **Clean Architecture** — домен и сценарии не знают ни про FastAPI, ни про Celery, ни про SQLAlchemy.
- **DRY** — общие утилиты (тайм-коды, нормализация, конфиг, логирование) живут в `core`.
- **KISS** — никакого микросервисного дробления, пока монолит с воркерами справляется.
- **Расширяемость без правки ядра** — новые источники через `SourceProvider`,
  новые признаки через плагин-реестр анализаторов.

## Требования к окружению разработчика

- Python 3.12 (`py -3.12` на Windows)
- Docker Desktop (проверено: 29.6.2)
- FFmpeg — **на хосте не обязателен**, поставляется внутри образов;
  для локального запуска без Docker поставить FFmpeg 7+ и добавить в `PATH`
- NVIDIA GPU + CUDA 12 — опционально, ускоряет Whisper и CV-модели в 5–20 раз
