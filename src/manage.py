"""Служебная командная строка.

Запуск::

    python -m manage create-user --email dev@example.com
    python -m manage list-users
    python -m manage detect-scenes --video-id <uuid>
    python -m manage transcribe --video-id <uuid>

Почему argparse, а не typer/click: несколько команд не стоят новой зависимости
в образе. Когда команд станет десяток — перейдём, это локальное решение.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence
from uuid import UUID, uuid4

from core.container import get_container
from core.domain.entities import Job
from core.domain.enums import JobStatus
from core.domain.timecode import format_timecode
from core.logging import configure_logging


async def _create_user(email: str) -> int:
    # Импорт внутри функции: генерация ключа живёт в api/, а образ воркера
    # ставит только extra `worker` — без FastAPI. Модульный импорт сделал бы
    # недоступными в воркере все команды разом, включая те, что к API
    # отношения не имеют.
    from api.deps import generate_api_key

    container = get_container()
    key, key_hash = generate_api_key()

    async with container.uow() as uow:
        if await uow.users.get_by_email(email) is not None:
            print(f"Пользователь {email} уже существует", file=sys.stderr)
            return 1
        user = await uow.users.create(email=email, api_key_hash=key_hash)
        await uow.commit()

    # Ключ показывается ровно один раз: в БД лежит только его хеш,
    # восстановить открытое значение невозможно — так и задумано.
    print(f"user_id : {user.id}")
    print(f"email   : {user.email}")
    print(f"api_key : {key}")
    print("\nСохраните ключ — повторно он не отобразится.")
    return 0


async def _list_users() -> int:
    from sqlalchemy import select

    from core.infra.db.models import UserORM

    container = get_container()
    async with container.uow() as uow:
        rows = (await uow.session.execute(select(UserORM))).scalars().all()  # type: ignore[attr-defined]

    if not rows:
        print("Пользователей нет. Создайте: python -m manage create-user --email you@example.com")
        return 0
    for row in rows:
        flag = "активен" if row.is_active else "отключён"
        print(f"{row.id}  {row.email:40}  {row.role.value:6}  {flag}")
    return 0


async def _create_job(video_id: UUID, params: dict[str, object]) -> Job | None:
    """Создать джоб под видео.

    Полноценный ``POST /analyze`` появится на Этапе 10 вместе с остальным API.
    До тех пор джоб нужно откуда-то создавать — и командная строка для этого
    честнее, чем недоделанный эндпоинт, который придётся переписывать.
    """
    container = get_container()
    async with container.uow() as uow:
        video = await uow.videos.get(video_id)
        if video is None:
            print(f"Видео {video_id} не найдено", file=sys.stderr)
            return None
        if not video.is_usable:
            print(f"Видео в статусе {video.status.value}, анализ невозможен", file=sys.stderr)
            return None

        job = await uow.jobs.add(
            Job(
                id=uuid4(),
                video_id=video.id,
                user_id=video.user_id,
                status=JobStatus.QUEUED,
                params=params,
                params_hash=hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
            )
        )
        await uow.commit()
    return job


async def _detect_scenes(video_id: UUID, *, enqueue: bool, show: int) -> int:
    """Создать джоб и разбить видео на сцены."""
    container = get_container()
    job = await _create_job(
        video_id,
        {"stage": "detect_scenes", "detector": container.settings.scene_detector},
    )
    if job is None:
        return 1

    print(f"job_id  : {job.id}")

    if enqueue:
        # Через брокер: так проверяется весь путь — маршрутизация по очередям,
        # сериализация, выполнение на настоящем воркере.
        from workers.tasks.pipeline import detect_scenes as task

        async_result = task.delay(str(job.id))
        print(f"task_id : {async_result.id}")
        print("Задача поставлена в очередь cpu. Статус: docker compose logs -f worker-cpu")
        return 0

    result = await container.detect_scenes.execute(job.id)
    print(f"сцен    : {result.scene_count}")
    print(f"кадров  : {result.keyframe_count}")
    print(f"время   : {result.duration_sec:.2f} с")

    if show:
        async with container.uow() as uow:
            scenes = await uow.scenes.list_for_job(job.id)
        print("\nпервые сцены:")
        for scene in scenes[:show]:
            print(
                f"  {scene.index:4d}  {format_timecode(scene.range.start)} → "
                f"{format_timecode(scene.range.end)}  "
                f"({scene.range.duration:6.2f} с)  {scene.keyframe_key or '—'}"
            )
    return 0


async def _transcribe(
    *,
    job_id: UUID | None,
    video_id: UUID | None,
    enqueue: bool,
    show: int,
) -> int:
    """Извлечь аудио и распознать речь.

    С ``--video-id`` создаётся новый джоб и прогоняются все три этапа подряд:
    сцены нужны, чтобы к ним привязался текст. С ``--job-id`` работа идёт
    по существующему джобу — так проверяется возобновление с середины.
    """
    container = get_container()

    if job_id is None:
        if video_id is None:
            print("Укажите --job-id или --video-id", file=sys.stderr)
            return 1
        job = await _create_job(
            video_id,
            {"stage": "transcribe", "model": container.settings.whisper_model},
        )
        if job is None:
            return 1
        job_id = job.id
        print(f"job_id  : {job_id}")
        if not enqueue:
            scenes = await container.detect_scenes.execute(job_id)
            print(f"сцен    : {scenes.scene_count}")

    if enqueue:
        # Цепочка, а не три отдельных вызова: распознавать нечего, пока
        # не извлечено аудио, а привязывать текст не к чему без сцен.
        from celery import chain

        from workers.tasks.pipeline import detect_scenes, extract_audio, transcribe

        signature = chain(
            detect_scenes.si(str(job_id)),
            extract_audio.si(str(job_id)),
            transcribe.si(str(job_id)),
        )
        async_result = signature.apply_async()
        print(f"task_id : {async_result.id}")
        print("Цепочка поставлена в очередь. Статус: docker compose logs -f worker-cpu")
        return 0

    audio = await container.extract_audio.execute(job_id)
    if audio.skipped:
        print("аудио   : нет дорожки, этап пропущен")
    else:
        print(f"аудио   : {audio.storage_key} ({audio.duration_sec:.2f} с работы)")

    result = await container.transcribe.execute(job_id)
    print(f"язык    : {result.language or '—'}")
    print(f"сегментов: {result.segment_count}")
    print(f"слов    : {result.word_count}")
    print(f"сцен с текстом: {result.scenes_with_text}")
    print(f"время   : {result.duration_sec:.2f} с")

    if show:
        async with container.uow() as uow:
            transcript = await uow.transcripts.get_for_job(job_id)
        print("\nпервые реплики:")
        for segment in transcript.segments[:show]:
            print(
                f"  {format_timecode(segment.range.start)} → "
                f"{format_timecode(segment.range.end)}  {segment.text}"
            )
    return 0


async def _rank(*, job_id: UUID, show: int) -> int:
    """Оценить сцены языковой моделью."""
    container = get_container()
    settings = container.settings
    print(f"провайдер: {settings.llm_provider}")
    if settings.llm_provider == "openrouter":
        print(f"модель   : {settings.openrouter_model}")

    result = await container.rank_scenes.execute(job_id)

    if result.skipped:
        print("этап пропущен")
        return 0
    if result.reused:
        print("этап уже выполнен ранее")
        return 0

    print(f"сцен     : {result.scenes_total}, отправлено {result.scenes_sent}")
    print(f"оценено  : {result.scenes_scored} за {result.duration_sec:.1f} с")
    print(f"батчей   : {result.batches}")
    print(f"расход   : {dict(result.cost)}")
    if result.degraded:
        print("! модель недоступна, джоб пойдёт по эвристике", file=sys.stderr)
    for error in result.failed:
        print(f"  ! {error}", file=sys.stderr)

    if show:
        async with container.uow() as uow:
            scenes = await uow.scenes.list_for_job(job_id)
        rated = sorted(
            (scene for scene in scenes if scene.llm_score is not None),
            key=lambda scene: -(scene.llm_score or 0),
        )
        print("\nлучшие сцены по мнению модели:")
        for scene in rated[:show]:
            print(
                f"  {scene.llm_score:3.0f} {scene.category or '—':12} "
                f"{format_timecode(scene.range.start)} | {scene.reason or ''}"
            )
    return 0


async def _select(
    *,
    job_id: UUID,
    profile: str | None,
    budget: float | None,
    show: int,
) -> int:
    """Посчитать эвристику и отобрать моменты.

    Оба этапа сразу: по отдельности они бессмысленны — скор без отбора
    никуда не идёт, а отбор без скора не по чему делать.
    """
    container = get_container()
    name = profile or container.settings.default_profile

    if profile:
        # Профиль на один прогон: пересобираем сценарии, а не правим конфиг.
        # Так можно сравнить movie и podcast на одном джобе, ничего не трогая.
        from core.usecases.score import ScoreScenes
        from core.usecases.select import SelectHighlights

        scorer = ScoreScenes(uow_factory=container.uow, scorer=container.detector, profile=name)
        selector = SelectHighlights(
            uow_factory=container.uow, selector=container.detector, profile=name
        )
    else:
        scorer = container.score_scenes
        selector = container.select_highlights

    print(f"профиль : {name}")

    scored = await scorer.execute(job_id)
    if scored.skipped:
        print("скоринг : пропущен (нет сцен)")
        return 1
    if scored.reused:
        print("скоринг : уже выполнен ранее")
    else:
        print(
            f"скоринг : {scored.scenes_scored} сцен, "
            f"средний {scored.mean_score}, максимум {scored.max_score}"
        )

    result = await selector.execute(job_id, total_duration_sec=budget)
    if result.skipped:
        print("отбор   : пропущен")
        return 1
    if result.reused:
        print("отбор   : уже выполнен ранее")
        return 0

    print(f"клипов  : {result.clips} из {result.scenes} сцен")
    print(f"лучший  : {result.top_score}")
    print(f"всего   : {result.total_duration_sec:.1f} с")

    if show:
        async with container.uow() as uow:
            clips = await uow.clips.list_for_job(job_id)
        print("\nотобранные моменты:")
        for clip in clips[:show]:
            print(
                f"  {clip.rank:2d}. {clip.score:5.1f} {clip.category:12} "
                f"{format_timecode(clip.range.start)} → {format_timecode(clip.range.end)} "
                f"({clip.range.duration:5.1f} с)"
            )
            if clip.reason:
                print(f"      {clip.reason}")
    return 0


async def _analyze(*, job_id: UUID, groups: Sequence[str], show: int) -> int:
    """Посчитать признаки по уже разбитому на сцены джобу."""
    container = get_container()
    stages = {
        "audio": container.analyze_audio,
        "speech": container.analyze_speech,
        "vision": container.analyze_vision,
    }

    for group in groups:
        usecase = stages.get(group)
        if usecase is None:
            print(f"Неизвестная группа: {group}", file=sys.stderr)
            return 1

        result = await usecase.execute(job_id)
        status = "пропущен" if result.skipped else ", ".join(result.analyzers) or "—"
        print(f"{group:7}: {status}")
        if result.failed:
            for key, message in sorted(result.failed.items()):
                print(f"         ! {key}: {message}", file=sys.stderr)
        if not result.skipped:
            print(
                f"         признаков {result.feature_count} "
                f"на {result.scenes_with_features} сценах за {result.duration_sec:.2f} с"
            )

    if show:
        async with container.uow() as uow:
            scenes = await uow.scenes.list_for_job(job_id)
        print("\nпризнаки первых сцен:")
        for scene in scenes[:show]:
            values = ", ".join(
                f"{key.split('.', 1)[1]}={value.value:.2f}"
                for key, value in sorted(scene.features.items())
            )
            print(f"  {scene.index:4d}  {format_timecode(scene.range.start)}  {values or '—'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manage", description="Служебные команды проекта")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-user", help="Создать пользователя и выдать API-ключ")
    create.add_argument("--email", required=True)

    sub.add_parser("list-users", help="Показать пользователей")

    scenes = sub.add_parser("detect-scenes", help="Создать джоб и разбить видео на сцены")
    scenes.add_argument("--video-id", required=True, type=UUID)
    scenes.add_argument(
        "--enqueue",
        action="store_true",
        help="Отправить задачу в Celery вместо выполнения на месте",
    )
    scenes.add_argument("--show", type=int, default=10, help="Сколько сцен напечатать")

    speech = sub.add_parser("transcribe", help="Извлечь аудио и распознать речь")
    target = speech.add_mutually_exclusive_group(required=True)
    target.add_argument("--job-id", type=UUID, help="Существующий джоб")
    target.add_argument("--video-id", type=UUID, help="Создать джоб и пройти этапы с нуля")
    speech.add_argument(
        "--enqueue",
        action="store_true",
        help="Отправить цепочку задач в Celery вместо выполнения на месте",
    )
    speech.add_argument("--show", type=int, default=10, help="Сколько реплик напечатать")

    analyze = sub.add_parser("analyze", help="Посчитать признаки по сценам джоба")
    analyze.add_argument("--job-id", required=True, type=UUID)
    analyze.add_argument(
        "--groups",
        default="audio",
        help="Группы через запятую: audio, speech, vision",
    )
    analyze.add_argument("--show", type=int, default=10, help="Сколько сцен напечатать")

    rank = sub.add_parser("rank", help="Оценить сцены языковой моделью")
    rank.add_argument("--job-id", required=True, type=UUID)
    rank.add_argument("--show", type=int, default=10, help="Сколько сцен напечатать")

    select = sub.add_parser("select", help="Посчитать скор и отобрать моменты")
    select.add_argument("--job-id", required=True, type=UUID)
    select.add_argument(
        "--profile", default=None, help="Профиль отбора: default, movie, podcast, lecture, gaming"
    )
    select.add_argument(
        "--budget", type=float, default=None, help="Суммарная длительность клипов, с"
    )
    select.add_argument("--show", type=int, default=10, help="Сколько клипов напечатать")

    args = parser.parse_args(argv)
    configure_logging(level="WARNING", fmt="console")

    if args.command == "create-user":
        return asyncio.run(_create_user(args.email))
    if args.command == "list-users":
        return asyncio.run(_list_users())
    if args.command == "detect-scenes":
        return asyncio.run(_detect_scenes(args.video_id, enqueue=args.enqueue, show=args.show))
    if args.command == "transcribe":
        return asyncio.run(
            _transcribe(
                job_id=args.job_id,
                video_id=args.video_id,
                enqueue=args.enqueue,
                show=args.show,
            )
        )
    if args.command == "analyze":
        groups = [item.strip() for item in args.groups.split(",") if item.strip()]
        return asyncio.run(_analyze(job_id=args.job_id, groups=groups, show=args.show))
    if args.command == "rank":
        return asyncio.run(_rank(job_id=args.job_id, show=args.show))
    if args.command == "select":
        return asyncio.run(
            _select(
                job_id=args.job_id,
                profile=args.profile,
                budget=args.budget,
                show=args.show,
            )
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
