# AI Highlight Generator

Автоматический поиск самых интересных моментов в длинных видео и генерация готовых клипов
(горизонтальных и вертикальных 1080×1920) с субтитрами, заголовком, описанием и хештегами.

**Статус:** Этап 9 завершён — конвейер проходит от видео до готового
результата целиком. Видео загружается, режется на сцены, распознаётся
речью с пословными таймингами, каждая сцена описывается двадцатью
с лишним признаками (громкость, паузы, темп и эмоция речи, звуковые
события, движение, монтаж, лица), получает оценку интересности
от языковой модели, превращается в список отобранных моментов
с границами по паузам речи и режется в клипы: горизонтальный,
вертикальный 1080×1920 с кадрированием по лицам, превью-кадр,
субтитры SRT и ASS с караоке-подсветкой, заголовок с описанием
и хештегами от модели — и `manifest.json`, описывающий весь результат.
Осталось довести REST API и продовую обвязку: это этап 10.

Замер на размеченном вручную трейлере: Precision@6 = 0.67 при Recall 0.67
(до настройки скоринга было 0.25 / 0.33). Это одно видео, не датасет —
подробности и оговорки в [разборе этапа 7](docs/stages/07_Stage7_Selection.md).
Бюджет нарезки: 10 клипов по 45 с из 1080p — 2.3 минуты на 8 ядрах
вместе с вертикалью.

650 тестов зелёные локально (в образе воркера, с настоящими весами
моделей, — больше). Замер на двухчасовом видео (CPU, без видеокарты):
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
5. Отдаёт описание сцен LLM (OpenRouter или Claude), получает строгий JSON со `score` / `category` / `reason`.
6. Сливает оценки, выбирает лучшие моменты с учётом длительности и разнообразия.
7. Нарезает клипы через FFmpeg, делает вертикальную версию со smart-reframe.
8. Генерирует субтитры (SRT/ASS с караоке), заголовок, описание, хештеги и `manifest.json`.

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
| 7 — Выбор лучших моментов | [07_Stage7_Selection.md](docs/stages/07_Stage7_Selection.md) |
| 8 — Нарезка клипов | [08_Stage8_Rendering.md](docs/stages/08_Stage8_Rendering.md) |
| 9 — Субтитры и метаданные | [09_Stage9_Subtitles.md](docs/stages/09_Stage9_Subtitles.md) |

---

## Технологический стек

| Слой | Технология |
|---|---|
| Язык | Python 3.12 |
| API | FastAPI + Uvicorn, Pydantic v2 |
| Очередь | Celery 5 + Redis (broker + result backend) |
| БД | PostgreSQL 16 + SQLAlchemy 2 (async) + Alembic |
| Видео | FFmpeg 7 (резка, вертикальный кроп, вжигание субтитров), OpenCV 4, PySceneDetect |
| Речь | faster-whisper (CTranslate2) |
| ML | PyTorch 2 + `transformers`: AudioSet-классификатор (звуковые события), wav2vec2-SER (эмоция), SSDLite (люди); лица — YuNet прямо из OpenCV |
| LLM | OpenRouter (по умолчанию бесплатная модель) или Claude — за портом `LLMProvider` |
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

Запустить весь конвейер одним запросом:

```bash
curl -X POST http://localhost:8000/api/v1/jobs -H "X-API-Key: hg_..." -H "Content-Type: application/json" -d '{"video_id":"<uuid>","profile":"movie","max_clips":5}'
```

В ответе `job_id` и статус `queued`. Дальше видео проходит все этапы само:
сцены → аудио → речь → признаки → скоринг → модель → отбор → нарезка →
субтитры → метаданные → манифест.

Повторный запрос с теми же параметрами вернёт уже посчитанный джоб,
а не запустит работу заново. Нужен пересчёт — добавьте `"force": true`.

Следить за прогрессом:

```bash
curl "http://localhost:8000/api/v1/jobs/<job_id>/status" -H "X-API-Key: hg_..."
```

Или потоком, без поллинга (Server-Sent Events):

```bash
curl -N "http://localhost:8000/api/v1/jobs/<job_id>/status/stream" -H "X-API-Key: hg_..."
```

Забрать результат — список клипов со ссылками на файлы:

```bash
curl "http://localhost:8000/api/v1/clips/<job_id>" -H "X-API-Key: hg_..."
```

Скачивание поддерживает `Range`, так что клип можно открыть в браузере
и перематывать, не выкачивая целиком:

```bash
curl "http://localhost:8000/api/v1/download/<asset_id>?inline=true" -H "X-API-Key: hg_..." -o clip.mp4
```

Перезапустить с этапа, ничего не пересчитывая до него:

```bash
curl -X POST "http://localhost:8000/api/v1/jobs/<job_id>/retry" -H "X-API-Key: hg_..." -H "Content-Type: application/json" -d '{"from_stage":"llm_rank"}'
```

Отменить выполняющийся:

```bash
curl -X DELETE "http://localhost:8000/api/v1/jobs/<job_id>" -H "X-API-Key: hg_..."
```

### Этапы по отдельности

Тот же конвейер можно гонять по одному этапу — это нужно при отладке
и при разборе того, что именно получилось:

```bash
docker compose exec worker-cpu python -m manage detect-scenes --video-id <uuid> --enqueue
```

```bash
docker compose exec worker-cpu python -m manage transcribe --video-id <uuid> --enqueue
```

```bash
docker compose exec worker-cpu python -m manage analyze --job-id <uuid> --groups audio,speech,vision
```

```bash
docker compose exec worker-cpu python -m manage select --job-id <uuid> --profile movie
```

```bash
docker compose exec worker-cpu python -m manage render --job-id <uuid> --enqueue
```

При первом анализе качаются веса — в том `models`, рядом с моделью
распознавания речи: AudioSet-классификатор 331 МБ, SER 722 МБ, детектор
людей 14 МБ, YuNet 256 КБ. Итого чуть больше гигабайта.

Лишний анализатор отключается через `ANALYZERS_DISABLED` — например,
`ANALYZERS_DISABLED=audio.events,vision.faces` убирает самые дорогие
и возвращает этап к арифметике без нейросетей.

Транскрипт целиком доступен отдельно:

```bash
curl "http://localhost:8000/api/v1/jobs/<job_id>/transcript?format=srt" -H "X-API-Key: hg_..."
```

### Наблюдаемость

```bash
docker compose --profile tools up -d prometheus grafana beat
```

- Метрики API: <http://localhost:8000/metrics>
- Метрики воркера: <http://localhost:9100/metrics>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> — дашборд «Highlight Generator — конвейер»
  подключается сам, логин `admin` / `admin`

Контейнер `beat` запускает уборку: раз в час удаляет артефакты с истёкшим
сроком хранения, раз в десять минут снимает джобы, зависшие в `running`.

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
