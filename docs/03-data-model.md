# 03 — Модель данных

PostgreSQL 16, SQLAlchemy 2 (async, `Mapped[...]`), миграции — Alembic.
ORM-модели живут в `core/infra/db/models.py` и **не** являются доменными сущностями:
репозитории мапят их в `core/domain/entities.py` и обратно. Это цена, которую мы платим
за независимость домена от БД, и она окупается на тестах.

## 3.1 ER-диаграмма

```
User ──1:N──> Video ──1:N──> Job
                │              │
                ├──1:N──> Scene ──N:M──> Clip   (через clip_scenes)
                ├──1:N──> TranscriptSegment
                └──1:N──> Asset                  (файлы: аудио, кадры, клипы, субтитры)
```

## 3.2 Таблицы

### `users`

| Поле | Тип | Заметки |
|---|---|---|
| `id` | `uuid` PK | |
| `email` | `citext` UNIQUE | |
| `password_hash` | `text` | argon2 |
| `api_key_hash` | `text` NULL | для машинного доступа |
| `role` | `enum(user, admin)` | |
| `quota_minutes_month` | `int` | лимит обработки |
| `is_active` | `bool` | |
| `created_at` / `updated_at` | `timestamptz` | |

### `videos`

| Поле | Тип | Заметки |
|---|---|---|
| `id` | `uuid` PK | |
| `user_id` | `uuid` FK → users | |
| `source_uri` | `text` | `upload://…`, `file://…`, позже `https://youtube.com/…` |
| `source_kind` | `enum(upload, local_file, youtube, twitch, kick)` | расширяемый enum |
| `original_filename` | `text` NULL | |
| `storage_key` | `text` | ключ в `ObjectStorage` |
| `checksum_sha256` | `char(64)` | дедупликация повторных загрузок |
| `status` | `enum(uploading, ready, failed, deleted)` | |
| `duration_sec` | `numeric(10,3)` NULL | |
| `width` / `height` | `int` NULL | |
| `fps` | `numeric(7,3)` NULL | |
| `video_codec` / `audio_codec` | `text` NULL | |
| `container` | `text` NULL | mp4/mkv/mov |
| `size_bytes` | `bigint` NULL | |
| `has_audio` | `bool` | |
| `probe` | `jsonb` | полный вывод `ffprobe` |
| `error` | `jsonb` NULL | |
| `created_at` / `updated_at` | `timestamptz` | |

Индексы: `(user_id, created_at desc)`, UNIQUE `(user_id, checksum_sha256)` — частичный, где `status <> 'deleted'`.

### `jobs`

Один джоб = один прогон конвейера над одним видео с конкретным набором настроек.

| Поле | Тип | Заметки |
|---|---|---|
| `id` | `uuid` PK | |
| `video_id` | `uuid` FK → videos | |
| `user_id` | `uuid` FK → users | денормализация ради быстрых выборок |
| `mode` | `enum(batch, streaming)` | `streaming` зарезервирован |
| `status` | `enum(queued, running, partial, succeeded, failed, cancelled)` | |
| `stage` | `enum` | текущий этап: `scenes`, `audio`, `speech`, `vision`, `scoring`, `llm`, `select`, `render`, `subtitles`, `export` |
| `progress` | `numeric(5,2)` | 0..100 |
| `stage_states` | `jsonb` | `{"scenes": {"status":"succeeded","started_at":…,"duration":…}, …}` |
| `params` | `jsonb` | профиль скоринга, число клипов, min/max длительность, флаги |
| `params_hash` | `char(64)` | для идемпотентности и кеша результатов |
| `celery_root_id` | `text` NULL | id корневого chord'а — для отмены |
| `trace_id` | `uuid` | сквозной в логах |
| `error` | `jsonb` NULL | `{code, message, stage, retriable}` |
| `cost` | `jsonb` | `{llm_input_tokens, llm_output_tokens, llm_usd, cpu_sec, gpu_sec}` |
| `started_at` / `finished_at` | `timestamptz` NULL | |
| `created_at` / `updated_at` | `timestamptz` | |

Индексы: `(video_id, created_at desc)`, `(status)` частичный где `status in ('queued','running')`,
UNIQUE `(video_id, params_hash)` частичный где `status = 'succeeded'` — повторный запрос
с теми же параметрами возвращает готовый джоб.

### `scenes`

| Поле | Тип | Заметки |
|---|---|---|
| `id` | `uuid` PK | |
| `video_id` | `uuid` FK → videos | |
| `job_id` | `uuid` FK → jobs | сцены привязаны к прогону: разные детекторы дают разные границы |
| `index` | `int` | порядковый номер |
| `start_sec` / `end_sec` | `numeric(10,3)` | |
| `duration_sec` | `numeric(10,3)` GENERATED | |
| `keyframe_key` | `text` NULL | превью-кадр в storage |
| `features` | `jsonb` | `{"audio.loudness.peak": {"v": 0.81, "c": 1.0}, …}` |
| `heuristic_score` | `numeric(6,3)` NULL | 0..100 |
| `score_breakdown` | `jsonb` NULL | вклад групп признаков |
| `llm_score` | `numeric(6,3)` NULL | 0..100 |
| `llm_category` | `text` NULL | `action`, `funny`, `emotional`, … |
| `llm_reason` | `text` NULL | |
| `final_score` | `numeric(6,3)` NULL | после слияния |
| `transcript_text` | `text` NULL | денормализованный текст сцены — экономит джойн для LLM |
| `created_at` | `timestamptz` | |

Индексы: UNIQUE `(job_id, index)`, `(job_id, final_score desc)`, GIN по `features`.

> **Почему `features` — JSONB, а не отдельная таблица.** Признаков десятки, набор меняется
> с каждым новым плагином. Отдельная таблица `scene_features(scene_id, key, value)` дала бы
> строгость, но 50 признаков × 2000 сцен = 100k строк на видео и джойн на каждый скоринг.
> JSONB + GIN покрывает все нужные запросы, а схему валидирует Pydantic на входе.

### `transcript_segments`

| Поле | Тип | Заметки |
|---|---|---|
| `id` | `bigserial` PK | |
| `video_id` | `uuid` FK | |
| `job_id` | `uuid` FK | |
| `start_sec` / `end_sec` | `numeric(10,3)` | |
| `text` | `text` | |
| `words` | `jsonb` | `[{"w":"привет","s":1.20,"e":1.55,"p":0.98}, …]` — нужно для караоке-субтитров |
| `language` | `char(5)` | |
| `avg_logprob` | `numeric(6,4)` | качество распознавания |
| `no_speech_prob` | `numeric(6,4)` | |
| `speaker` | `text` NULL | зарезервировано под диаризацию |

Индексы: `(job_id, start_sec)`, GIN `to_tsvector('russian', text)` — полнотекстовый поиск по видео.

### `clips`

| Поле | Тип | Заметки |
|---|---|---|
| `id` | `uuid` PK | |
| `job_id` | `uuid` FK → jobs | |
| `video_id` | `uuid` FK → videos | |
| `rank` | `int` | 1 = лучший |
| `start_sec` / `end_sec` | `numeric(10,3)` | финальные границы (после снапа к границам речи/сцен) |
| `score` | `numeric(6,3)` | |
| `category` | `text` | |
| `reason` | `text` | |
| `title` | `text` NULL | от LLM |
| `description` | `text` NULL | от LLM |
| `hashtags` | `text[]` NULL | от LLM |
| `status` | `enum(pending, rendering, ready, failed)` | |
| `error` | `jsonb` NULL | |
| `created_at` | `timestamptz` | |

Индексы: `(job_id, rank)`.

### `clip_scenes` (N:M)

| Поле | Тип |
|---|---|
| `clip_id` | `uuid` FK, PK часть |
| `scene_id` | `uuid` FK, PK часть |

### `assets`

Единая таблица для всех файловых артефактов — не плодим по колонке на каждый формат.

| Поле | Тип | Заметки |
|---|---|---|
| `id` | `uuid` PK | |
| `owner_type` | `enum(video, job, clip)` | |
| `owner_id` | `uuid` | полиморфная ссылка (проверяется в коде, не FK) |
| `kind` | `enum(source, audio_wav, keyframe, clip_h, clip_v, subtitle_srt, subtitle_ass, thumbnail, manifest)` | |
| `storage_key` | `text` | |
| `mime` | `text` | |
| `size_bytes` | `bigint` | |
| `meta` | `jsonb` | ширина/высота/битрейт/длительность |
| `checksum_sha256` | `char(64)` NULL | |
| `expires_at` | `timestamptz` NULL | для автоочистки временных файлов |
| `created_at` | `timestamptz` | |

Индексы: `(owner_type, owner_id, kind)`, `(expires_at)` частичный где `expires_at is not null`.

## 3.3 Жизненный цикл статусов

```
Video:  uploading ──> ready ──> deleted
             └──> failed

Job:    queued ──> running ──> succeeded
                      ├──> partial      (часть этапов упала, но клипы есть)
                      ├──> failed
                      └──> cancelled
```

`partial` — важный статус: если упал, например, анализ лиц, но аудио и речь отработали,
клипы всё равно генерируются с оговоркой в `stage_states`. Полное падение конвейера
из-за одного анализатора недопустимо (см. док 04, §4.5).

## 3.4 Политика хранения и очистки

| Артефакт | TTL по умолчанию |
|---|---|
| Исходное видео | 30 дней (настраивается) |
| Промежуточное аудио WAV | 24 часа |
| Кадры/превью | 7 дней |
| Клипы и субтитры | 90 дней |
| Записи в БД | не удаляются, только soft-delete |

Периодическая Celery-beat задача `cleanup_expired_assets` удаляет из storage всё,
где `expires_at < now()`, и помечает `assets.deleted_at`.

## 3.5 Миграции

- Одна миграция на этап дорожной карты, не «одна большая».
- Каждая миграция обратима (`downgrade` обязателен).
- Enum'ы создаются как нативные PostgreSQL типы; расширение (`source_kind` += `youtube`)
  делается через `ALTER TYPE ... ADD VALUE` в отдельной миграции.
- В CI прогоняется `alembic upgrade head && alembic downgrade base && alembic upgrade head`.
