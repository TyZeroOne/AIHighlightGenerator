# 07 — REST API

FastAPI, OpenAPI 3.1 автоматически на `/docs` и `/openapi.json`.
Базовый префикс: `/api/v1`. Все ответы — JSON, все ошибки — RFC 7807 `application/problem+json`.

## 7.1 Аутентификация

| Способ | Заголовок | Для чего |
|---|---|---|
| API-ключ | `X-API-Key: hg_live_…` | Машинная интеграция (основной) |
| JWT | `Authorization: Bearer …` | Будущий веб-UI |

Все эндпоинты, кроме `/health` и `/metrics`, требуют аутентификации.
Доступ к ресурсу проверяется по `user_id` владельца — чужой `video_id` даёт `404`
(а не `403`, чтобы не раскрывать существование объекта).

## 7.2 Эндпоинты

### `POST /api/v1/upload`

Загрузка видеофайла.

**Запрос:** `multipart/form-data`
| Поле | Тип | Обязательно |
|---|---|---|
| `file` | binary | да |
| `title` | string | нет |

**Ответ `201`:**
```json
{
  "video_id": "8f14e45f-ea1b-4b2d-9c3a-1a2b3c4d5e6f",
  "status": "ready",
  "filename": "movie.mkv",
  "size_bytes": 4823947264,
  "duration_sec": 7241.32,
  "width": 1920, "height": 1080, "fps": 23.976,
  "video_codec": "h264", "audio_codec": "aac",
  "has_audio": true,
  "created_at": "2026-08-02T10:14:03Z"
}
```

Поведение:
- Валидация контейнера через `ffprobe` **до** записи в постоянное хранилище;
  расширение файла не является доказательством формата.
- Разрешены `mp4`, `mkv`, `mov`. Иначе `415 Unsupported Media Type`.
- Файл без аудиодорожки принимается, но конвейер пойдёт только по видеопризнакам
  (в ответе `has_audio: false`, при анализе — предупреждение).
- Дедупликация по `sha256`: повторная загрузка того же файла тем же пользователем
  возвращает `200` с существующим `video_id`.
- Лимиты по умолчанию: 20 ГБ, 6 часов (`MAX_UPLOAD_BYTES`, `MAX_DURATION_SEC`).

**Большие файлы.** Обычный multipart на 20 ГБ — плохая идея (нет возобновления,
таймауты прокси). Поэтому дополнительно:

- `POST /api/v1/upload/init` → `{upload_id, chunk_size}`
- `PUT /api/v1/upload/{upload_id}/part/{n}` — чанк, `Content-Range`
- `POST /api/v1/upload/{upload_id}/complete` → тот же ответ, что у `/upload`

При S3-бэкенде вместо этого выдаются presigned-URL'ы на multipart upload,
и байты вообще не идут через API-сервер.

**Локальный файл без загрузки:**
`POST /api/v1/videos/register` с `{"source_uri": "file:///media/in/movie.mkv"}` —
для случая, когда том уже смонтирован в контейнер. Путь валидируется против
белого списка каталогов (защита от path traversal).

---

### `POST /api/v1/analyze`

Запуск конвейера.

**Запрос:**
```json
{
  "video_id": "8f14e45f-…",
  "profile": "movie",
  "max_clips": 10,
  "min_duration_sec": 15,
  "max_duration_sec": 90,
  "vertical": true,
  "burn_subtitles": false,
  "language": null,
  "use_llm": true,
  "llm_sync": false,
  "analyzers_disabled": []
}
```

Все поля кроме `video_id` необязательны, значения по умолчанию — из профиля.

**Ответ `202`:**
```json
{
  "job_id": "3c9f2b7a-…",
  "video_id": "8f14e45f-…",
  "status": "queued",
  "stage": null,
  "progress": 0,
  "params_hash": "9ab3…",
  "created_at": "2026-08-02T10:15:00Z",
  "links": { "status": "/api/v1/status/3c9f2b7a-…" }
}
```

Идемпотентность: если для этой пары `video_id` + `params_hash` уже есть успешный
джоб, возвращается `200` с ним же (перезапуск не тратит ресурсы).
Форсировать пересчёт — `"force": true`.

---

### `GET /api/v1/status/{job_id}`

**Ответ `200`:**
```json
{
  "job_id": "3c9f2b7a-…",
  "video_id": "8f14e45f-…",
  "status": "running",
  "stage": "transcribe",
  "progress": 41.5,
  "stages": {
    "ingest":        {"status": "succeeded", "duration_sec": 2.1},
    "detect_scenes": {"status": "succeeded", "duration_sec": 88.4, "scenes": 1187},
    "extract_audio": {"status": "succeeded", "duration_sec": 31.0},
    "transcribe":    {"status": "running",   "progress": 62.0},
    "analyze_vision":{"status": "running",   "progress": 35.0},
    "score_scenes":  {"status": "pending"}
  },
  "counts": {"scenes": 1187, "clips": 0},
  "cost": {"llm_usd": 0.0, "cpu_sec": 421.0},
  "error": null,
  "started_at": "2026-08-02T10:15:04Z",
  "finished_at": null,
  "eta_sec": 640
}
```

Для «живого» прогресса вместо поллинга — `GET /api/v1/status/{job_id}/stream`
(Server-Sent Events, событие на каждое изменение этапа или прогресса ≥ 1 %).

---

### `GET /api/v1/clips/{job_id}`

**Query:** `limit` (по умолчанию 50), `offset`, `category`, `min_score`.

**Ответ `200`:**
```json
{
  "job_id": "3c9f2b7a-…",
  "status": "succeeded",
  "total": 10,
  "items": [
    {
      "clip_id": "b21c…",
      "rank": 1,
      "start": "00:15:21.480",
      "end":   "00:16:05.960",
      "start_sec": 921.48,
      "end_sec": 965.96,
      "duration_sec": 44.48,
      "score": 96.2,
      "score_breakdown": {"emotion": 0.71, "action": 0.94,
                          "narrative": 0.52, "dialogue": 0.33, "visual": 0.88},
      "heuristic_score": 91.0,
      "llm_score": 96,
      "category": "action",
      "reason": "Интенсивная сцена погони.",
      "title": "Погоня, которая решает всё",
      "description": "…",
      "hashtags": ["#погоня", "#экшен", "#нарезка"],
      "scene_ids": ["…", "…"],
      "assets": {
        "clip_h":       {"asset_id": "…", "mime": "video/mp4", "size_bytes": 18234112,
                         "width": 1920, "height": 1080},
        "clip_v":       {"asset_id": "…", "mime": "video/mp4", "size_bytes": 15120384,
                         "width": 1080, "height": 1920},
        "subtitle_srt": {"asset_id": "…", "mime": "application/x-subrip"},
        "subtitle_ass": {"asset_id": "…", "mime": "text/x-ssa"},
        "thumbnail":    {"asset_id": "…", "mime": "image/jpeg"}
      }
    }
  ]
}
```

Тайм-коды отдаются **и** строкой `HH:MM:SS.mmm` (как в ТЗ), **и** числом секунд
(как удобно машине). Это дешевле, чем заставлять каждого клиента парсить строки.

Если джоб ещё не завершён — `200` с `status: "running"` и пустым `items`
(а не `404`: джоб существует, просто результата пока нет).

---

### `GET /api/v1/download/{asset_id}`

Скачивание артефакта.

- При локальном storage — `StreamingResponse` с поддержкой `Range` (перемотка в плеере)
  и `Content-Disposition: attachment`.
- При S3/MinIO — `302` на presigned-URL с TTL 15 минут (байты не идут через API).
- `?inline=true` — `Content-Disposition: inline` для воспроизведения в браузере.

Проверка владения обязательна: `asset → owner → job/video → user_id`.

---

### Вспомогательные

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/api/v1/videos` | Список видео пользователя, пагинация |
| `GET` | `/api/v1/videos/{id}` | Метаданные + probe |
| `DELETE` | `/api/v1/videos/{id}` | Soft-delete + удаление файлов |
| `GET` | `/api/v1/jobs` | Список джобов, фильтр по статусу |
| `POST` | `/api/v1/jobs/{id}/retry` | Перезапуск с этапа: `{"from_stage": "llm"}` |
| `DELETE` | `/api/v1/jobs/{id}` | Отмена выполняющегося джоба |
| `GET` | `/api/v1/jobs/{id}/scenes` | Все сцены со скорами и признаками (сценарий «аналитик») |
| `GET` | `/api/v1/jobs/{id}/transcript` | Транскрипт, `?format=json\|srt\|vtt\|txt`, `?words=false` — без пословных таймингов (реализовано на Этапе 4) |
| `GET` | `/api/v1/profiles` | Доступные профили скоринга |
| `GET` | `/health` | Liveness: процесс жив |
| `GET` | `/health/ready` | Readiness: БД, Redis, storage доступны |
| `GET` | `/metrics` | Prometheus |

## 7.3 Формат ошибок

```json
{
  "type": "https://docs.hg.local/errors/unsupported-media",
  "title": "Unsupported media type",
  "status": 415,
  "detail": "Container 'avi' is not supported. Allowed: mp4, mkv, mov.",
  "instance": "/api/v1/upload",
  "trace_id": "0f9c…",
  "code": "UNSUPPORTED_MEDIA"
}
```

| HTTP | `code` | Когда |
|---|---|---|
| 400 | `INVALID_SOURCE` | Битый файл, `ffprobe` не смог прочитать |
| 401 | `UNAUTHENTICATED` | Нет/неверный ключ |
| 404 | `NOT_FOUND` | Нет объекта или он не принадлежит пользователю |
| 409 | `JOB_ALREADY_RUNNING` | Активный джоб на это видео с теми же параметрами |
| 413 | `FILE_TOO_LARGE` | Превышен `MAX_UPLOAD_BYTES` |
| 415 | `UNSUPPORTED_MEDIA` | Не тот контейнер/кодек |
| 422 | `VALIDATION_ERROR` | Ошибка схемы запроса, поле указано в `errors[]` |
| 429 | `RATE_LIMITED` | Превышена квота; заголовок `Retry-After` |
| 500 | `INTERNAL_ERROR` | Непредвиденное; `trace_id` для поиска в логах |
| 503 | `DEPENDENCY_UNAVAILABLE` | БД/Redis/storage недоступны |

## 7.4 Ограничения и квоты

| Лимит | Значение по умолчанию | Переменная |
|---|---|---|
| Размер файла | 20 ГБ | `MAX_UPLOAD_BYTES` |
| Длительность | 6 ч | `MAX_DURATION_SEC` |
| Параллельных джобов на пользователя | 2 | `MAX_CONCURRENT_JOBS` |
| Запросов в минуту | 60 | `RATE_LIMIT_RPM` |
| Минут обработки в месяц | из `users.quota_minutes_month` | — |

Rate limiting — на Redis, алгоритм sliding window, ключ = `user_id`.
Заголовки `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

## 7.5 Версионирование

Префикс `/api/v1`. Ломающие изменения → `/api/v2` с параллельной поддержкой
старой версии минимум 6 месяцев. Неломающие (новые поля в ответе, новые
необязательные параметры) — в рамках `v1`. Клиенты обязаны игнорировать
незнакомые поля.

## 7.6 Наблюдаемость

- Каждый запрос получает `trace_id` (принимается из `X-Request-ID` или генерируется),
  прокидывается в Celery-задачи и во все логи джоба.
- Структурные JSON-логи (`structlog`), поля: `trace_id`, `user_id`, `job_id`, `stage`, `duration_ms`.
- Метрики: `http_requests_total`, `http_request_duration_seconds`,
  `pipeline_stage_duration_seconds{stage}`, `celery_queue_depth{queue}`,
  `llm_tokens_total{kind}`, `llm_cost_usd_total`, `jobs_total{status}`.
