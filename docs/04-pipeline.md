# 04 — Конвейер обработки и очереди

## 4.1 Граф этапов

```
                       ┌─ [2] extract_audio ─┬─ [3] transcribe ─┬─ speech.rate
                       │                     │                  └─ speech.emotion
  [0] ingest           │                     ├─ audio.loudness
       │               │                     ├─ audio.pauses
       v               │                     └─ audio.events (смех/крик)
  [1] detect_scenes ───┤
                       │
                       └─ [4] vision ────────┬─ vision.motion
                                             ├─ vision.cuts
                                             ├─ vision.faces
                                             └─ vision.dynamics
                       ↓ (chord: ждём все)
                  [5] score_scenes           (эвристика, без сети)
                       ↓
                  [6] llm_rank               (батчи сцен → Claude)
                       ↓
                  [7] select_highlights      (слияние, NMS, разнообразие)
                       ↓
                  [8] render_clips           (параллельно по клипам)
                       ├─ cut_horizontal
                       ├─ reframe_vertical
                       └─ build_subtitles
                       ↓
                  [9] generate_metadata      (LLM: title/description/hashtags)
                       ↓
                 [10] finalize               (манифест, статус, уборка)
```

Ветки `[2]→…` и `[4]→…` идут параллельно. Внутри каждой — Celery `group()`
по слоям, которые вернул `registry.build_plan()`.

## 4.2 Реализация на Celery

```python
# workers/flows.py
def build_analysis_flow(job_id: UUID) -> Signature:
    return chain(
        ingest.si(job_id),
        detect_scenes.si(job_id),
        group(
            chain(
                extract_audio.si(job_id),
                group(transcribe.si(job_id), analyze_audio.si(job_id)),
                analyze_speech.si(job_id),
            ),
            analyze_vision.si(job_id),
        ),
        score_scenes.si(job_id),
        llm_rank.si(job_id),
        select_highlights.si(job_id),
        render_clips.si(job_id),  # внутри порождает chord по клипам
        generate_metadata.si(job_id),
        finalize.si(job_id),
    )
```

Все задачи принимают **только `job_id`**, состояние читают и пишут в БД/storage.
Никаких больших payload'ов в Redis: брокер не должен превращаться в хранилище.

## 4.3 Очереди и маршрутизация

| Очередь | Что выполняет | Конкурентность | Ресурс |
|---|---|---|---|
| `io` | ingest, скачивание, загрузка в storage, finalize | 8 | сеть/диск |
| `cpu` | ffmpeg, детекция сцен, аудиопризнаки, скоринг, нарезка | `nproc` | CPU |
| `gpu` | whisper, SER, детекция лиц, аудиособытия | 1–2 | VRAM |
| `net` | LLM-вызовы | 4 | внешний API, rate-limit |

```python
task_routes = {
    "workers.tasks.ingest": {"queue": "io"},
    "workers.tasks.transcribe": {"queue": "gpu"},
    "workers.tasks.analyze_vision": {"queue": "gpu"},
    "workers.tasks.llm_*": {"queue": "net"},
    "workers.tasks.*": {"queue": "cpu"},
}
```

Если GPU нет — `gpu`-очередь обслуживает тот же CPU-воркер, модели грузятся
в CPU-режиме (`WHISPER_DEVICE=cpu`, `compute_type=int8`). Никакой ветвистой
логики в коде: решает конфиг и то, какие воркеры подняты в compose.

## 4.4 Базовая задача

```python
# workers/base.py
class PipelineTask(Task):
    autoretry_for = (TransientError, ExternalServiceError)
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    max_retries = 5
    acks_late = True  # переживаем убийство воркера
    reject_on_worker_lost = True
    time_limit = 3600
    soft_time_limit = 3300

    def __call__(self, job_id, *a, **kw):
        with trace(job_id), stage_tracker(job_id, self.stage):
            return super().__call__(job_id, *a, **kw)
```

`stage_tracker` — контекстный менеджер, который:
1. пишет `jobs.stage`, `stage_states[stage].status = running`, `started_at`;
2. на выходе — `succeeded` / `failed` + длительность;
3. обновляет `jobs.progress` по фиксированным весам этапов.

## 4.5 Деградация вместо падения

Ключевое требование из док 01: один сломанный анализатор не убивает джоб.

```python
async def run_analyzers(layer: Sequence[FeatureAnalyzer], ctx, scenes):
    results, failures = {}, {}
    for res in await asyncio.gather(
        *(a.analyze(ctx, scenes) for a in layer), return_exceptions=True
    ):
        ...  # исключение → в failures, признаки анализатора просто отсутствуют
    if failures:
        mark_job_partial(ctx.job_id, failures)
    return results
```

Скоринг устроен так, что отсутствующие признаки не ломают формулу: вес отсутствующей
группы перераспределяется между присутствующими (док 05, §5.4).

Не деградируют, а валят джоб только: `ingest`, `detect_scenes`, `render_clips`.

## 4.6 Идемпотентность и возобновление

- Каждая задача в начале проверяет `stage_states[stage].status == "succeeded"` и,
  если этап уже пройден в этом джобе, **выходит немедленно**.
- Артефакты адресуются детерминированно:
  `{job_id}/audio.wav`, `{job_id}/scenes.json`, `{job_id}/clips/{rank}.mp4`.
  Повторный прогон перезаписывает, а не плодит.
- `POST /analyze` с теми же `params_hash` и `video_id` возвращает существующий
  успешный джоб вместо создания нового (см. UNIQUE-индекс в док 03).
- `POST /jobs/{id}/retry` сбрасывает `stage_states` начиная с указанного этапа
  и перезапускает хвост цепочки.

## 4.7 Отмена

`DELETE /jobs/{id}` → `AsyncResult(celery_root_id).revoke(terminate=True, signal="SIGTERM")`
плюс флаг `jobs.status = cancelled`. Каждая задача в начале и между тяжёлыми шагами
проверяет флаг отмены (дешёвый `SELECT status`), чтобы не доделывать заведомо
ненужную работу. FFmpeg-процессы завершаются через group kill.

## 4.8 Прогресс

Веса этапов (сумма 100), от них считается `jobs.progress`:

| Этап | Вес |
|---|---|
| ingest | 3 |
| detect_scenes | 7 |
| extract_audio | 5 |
| transcribe | 25 |
| analyze_audio | 5 |
| analyze_speech | 5 |
| analyze_vision | 20 |
| score_scenes | 2 |
| llm_rank | 8 |
| select_highlights | 2 |
| render_clips | 15 |
| generate_metadata | 2 |
| finalize | 1 |

Внутри длинных этапов (`transcribe`, `analyze_vision`, `render_clips`) задача
дополнительно рапортует дробный прогресс через `self.update_state`.

## 4.9 Периодические задачи (Celery Beat)

| Задача | Расписание | Что делает |
|---|---|---|
| `cleanup_expired_assets` | каждый час | удаляет из storage просроченные артефакты |
| `reap_stuck_jobs` | каждые 10 мин | джобы в `running` без апдейта > 1 ч → `failed` |
| `collect_metrics` | каждую минуту | глубина очередей, среднее время этапов → Prometheus |
| `poll_llm_batches` | каждые 5 мин | проверяет статус Batch API (док 06, §6.6) |

## 4.10 Режим стрима (задел)

`JobMode.STREAMING` меняет только оркестрацию верхнего уровня:

```python
async for window in provider.windows(uri):
    build_analysis_flow(job_id, window).apply_async()
```

Этапы `select_highlights` / `render_clips` работают по скользящему окну
(«лучшее за последние N минут»), а не по всему видео. Сами анализаторы,
скоринг и LLM не меняются — они изначально принимают `MediaWindow`.
