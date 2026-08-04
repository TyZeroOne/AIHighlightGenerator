"""Реализации репозиториев поверх SQLAlchemy.

Репозиторий переводит язык домена («найди видео по контрольной сумме»)
в язык БД («SELECT ... WHERE checksum_sha256 = ...»). Всё, что выше, про SQL
не знает — и это позволяет тестировать сценарии на фейковых репозиториях.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.entities import Asset, Clip, Job, Scene, UploadSession, User, Video
from core.domain.enums import AssetKind, AssetOwnerType, JobStatus, VideoStatus
from core.domain.media import Transcript
from core.infra.db.mappers import (
    asset_to_domain,
    clip_to_domain,
    clip_to_orm_values,
    job_to_domain,
    job_to_orm_values,
    scene_to_domain,
    scene_to_orm_values,
    transcript_segment_to_domain,
    transcript_segment_to_orm_values,
    upload_session_to_domain,
    user_to_domain,
    video_to_domain,
    video_to_orm_values,
)
from core.infra.db.models import (
    AssetORM,
    ClipORM,
    ClipSceneORM,
    JobORM,
    SceneORM,
    TranscriptSegmentORM,
    UploadSessionORM,
    UserORM,
    VideoORM,
)

__all__ = [
    "AssetRepository",
    "SqlAlchemyClipRepository",
    "SqlAlchemyJobRepository",
    "SqlAlchemySceneRepository",
    "SqlAlchemyTranscriptRepository",
    "SqlAlchemyUploadSessionRepository",
    "SqlAlchemyUserRepository",
    "SqlAlchemyVideoRepository",
]


class SqlAlchemyUserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, user_id: UUID) -> User | None:
        row = await self._session.get(UserORM, user_id)
        return user_to_domain(row) if row else None

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(UserORM).where(func.lower(UserORM.email) == email.lower())
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return user_to_domain(row) if row else None

    async def get_by_api_key_hash(self, key_hash: str) -> User | None:
        stmt = select(UserORM).where(
            UserORM.api_key_hash == key_hash,
            UserORM.is_active.is_(True),
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return user_to_domain(row) if row else None

    async def create(
        self,
        *,
        email: str,
        api_key_hash: str,
        password_hash: str | None = None,
    ) -> User:
        row = UserORM(
            id=uuid4(),
            email=email,
            api_key_hash=api_key_hash,
            password_hash=password_hash,
        )
        self._session.add(row)
        await self._session.flush()
        return user_to_domain(row)


class SqlAlchemyVideoRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, video: Video) -> Video:
        row = VideoORM(**video_to_orm_values(video))
        self._session.add(row)
        await self._session.flush()
        return video_to_domain(row)

    async def get(self, video_id: UUID) -> Video | None:
        row = await self._session.get(VideoORM, video_id)
        return video_to_domain(row) if row else None

    async def get_owned(self, video_id: UUID, user_id: UUID) -> Video | None:
        """Видео конкретного пользователя.

        Проверка владения — в запросе, а не после него. Иначе легко забыть
        сравнить ``video.user_id`` в одном из вызовов, и появится доступ
        к чужим данным. Здесь забыть невозможно: другого метода получения нет.
        """
        stmt = select(VideoORM).where(
            VideoORM.id == video_id,
            VideoORM.user_id == user_id,
            VideoORM.status != VideoStatus.DELETED,
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return video_to_domain(row) if row else None

    async def get_by_checksum(self, user_id: UUID, checksum: str) -> Video | None:
        stmt = select(VideoORM).where(
            VideoORM.user_id == user_id,
            VideoORM.checksum_sha256 == checksum,
            VideoORM.status != VideoStatus.DELETED,
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return video_to_domain(row) if row else None

    async def list_for_user(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Video]:
        stmt = (
            select(VideoORM)
            .where(VideoORM.user_id == user_id, VideoORM.status != VideoStatus.DELETED)
            .order_by(VideoORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [video_to_domain(row) for row in rows]

    async def count_for_user(self, user_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(VideoORM)
            .where(
                VideoORM.user_id == user_id,
                VideoORM.status != VideoStatus.DELETED,
            )
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def update(self, video: Video) -> Video:
        values = video_to_orm_values(video)
        values.pop("id", None)
        stmt = update(VideoORM).where(VideoORM.id == video.id).values(**values)
        await self._session.execute(stmt)
        row = await self._session.get(VideoORM, video.id)
        assert row is not None
        await self._session.refresh(row)
        return video_to_domain(row)

    async def mark_deleted(self, video_id: UUID) -> None:
        """Мягкое удаление.

        Строку не удаляем: на неё ссылаются джобы, сцены и клипы, а история
        обработки может понадобиться для разбора инцидента. Файлы при этом
        удаляются по-настоящему — место они занимают реально.
        """
        stmt = update(VideoORM).where(VideoORM.id == video_id).values(status=VideoStatus.DELETED)
        await self._session.execute(stmt)


class SqlAlchemyJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, job: Job) -> Job:
        row = JobORM(**job_to_orm_values(job))
        self._session.add(row)
        await self._session.flush()
        return job_to_domain(row)

    async def get(self, job_id: UUID) -> Job | None:
        row = await self._session.get(JobORM, job_id)
        return job_to_domain(row) if row else None

    async def get_owned(self, job_id: UUID, user_id: UUID) -> Job | None:
        stmt = select(JobORM).where(JobORM.id == job_id, JobORM.user_id == user_id)
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return job_to_domain(row) if row else None

    async def find_succeeded(self, video_id: UUID, params_hash: str) -> Job | None:
        stmt = select(JobORM).where(
            JobORM.video_id == video_id,
            JobORM.params_hash == params_hash,
            JobORM.status == JobStatus.SUCCEEDED,
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return job_to_domain(row) if row else None

    async def update(self, job: Job) -> Job:
        values = job_to_orm_values(job)
        values.pop("id", None)
        await self._session.execute(update(JobORM).where(JobORM.id == job.id).values(**values))
        row = await self._session.get(JobORM, job.id)
        assert row is not None
        await self._session.refresh(row)
        return job_to_domain(row)

    async def list_by_status(self, status: JobStatus, *, limit: int = 50) -> Sequence[Job]:
        stmt = (
            select(JobORM)
            .where(JobORM.status == status)
            .order_by(JobORM.created_at.desc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [job_to_domain(row) for row in rows]


class SqlAlchemySceneRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_add(self, scenes: Sequence[Scene]) -> None:
        """Одна вставка на все сцены.

        ``session.add()`` в цикле на 1200 сценах — это 1200 отдельных INSERT
        и заметные секунды на ровном месте. Массовая форма ``insert().values()``
        отправляет их одним запросом.
        """
        if not scenes:
            return
        await self._session.execute(
            insert(SceneORM).values([scene_to_orm_values(scene) for scene in scenes])
        )

    async def list_for_job(self, job_id: UUID) -> Sequence[Scene]:
        stmt = select(SceneORM).where(SceneORM.job_id == job_id).order_by(SceneORM.index)
        rows = (await self._session.execute(stmt)).scalars().all()
        return [scene_to_domain(row) for row in rows]

    async def count_for_job(self, job_id: UUID) -> int:
        stmt = select(func.count()).select_from(SceneORM).where(SceneORM.job_id == job_id)
        return int((await self._session.execute(stmt)).scalar_one())

    async def delete_for_job(self, job_id: UUID) -> None:
        await self._session.execute(delete(SceneORM).where(SceneORM.job_id == job_id))

    async def bulk_update(self, scenes: Sequence[Scene]) -> None:
        """Обновить изменяемую часть сцен: текст (Этап 4), признаки и скоры (Этап 5+)."""
        for scene in scenes:
            values = scene_to_orm_values(scene)
            values.pop("id", None)
            await self._session.execute(
                update(SceneORM).where(SceneORM.id == scene.id).values(**values)
            )

    async def top_by_score(self, job_id: UUID, *, limit: int) -> Sequence[Scene]:
        stmt = (
            select(SceneORM)
            .where(SceneORM.job_id == job_id)
            .order_by(SceneORM.final_score.desc().nullslast())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [scene_to_domain(row) for row in rows]


class SqlAlchemyTranscriptRepository:
    """Сегменты распознанной речи.

    Транскрипт хранится сегментами, а не одним полем: по ним строятся
    субтитры, к ним привязывается текст сцен, и по ним же идёт полнотекстовый
    поиск (индекс в док 03). Слова с таймингами лежат в JSONB внутри сегмента —
    отдельная таблица на двадцать тысяч слов ради одного видео не окупается.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_for_job(self, job_id: UUID, transcript: Transcript, video_id: UUID) -> int:
        await self._session.execute(
            delete(TranscriptSegmentORM).where(TranscriptSegmentORM.job_id == job_id)
        )
        if not transcript.segments:
            return 0

        await self._session.execute(
            insert(TranscriptSegmentORM).values(
                [
                    transcript_segment_to_orm_values(segment, job_id=job_id, video_id=video_id)
                    for segment in transcript.segments
                ]
            )
        )
        return len(transcript.segments)

    async def get_for_job(self, job_id: UUID) -> Transcript:
        stmt = (
            select(TranscriptSegmentORM)
            .where(TranscriptSegmentORM.job_id == job_id)
            .order_by(TranscriptSegmentORM.start_sec)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        segments = [transcript_segment_to_domain(row) for row in rows]
        # Язык берём у первого сегмента: он один на весь транскрипт, но
        # хранится при каждом сегменте — так проще потом добавить смену языка
        # посреди видео, не меняя схему.
        language = segments[0].language if segments else None
        return Transcript(segments=tuple(segments), language=language)

    async def count_for_job(self, job_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(TranscriptSegmentORM)
            .where(TranscriptSegmentORM.job_id == job_id)
        )
        return int((await self._session.execute(stmt)).scalar_one())


class AssetRepository:
    """Учёт файловых артефактов. Отдельно от видео, потому что артефакты
    принадлежат разным сущностям (видео, джобу, клипу)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        owner_type: AssetOwnerType,
        owner_id: UUID,
        kind: AssetKind,
        storage_key: str,
        mime: str = "application/octet-stream",
        size_bytes: int = 0,
        meta: dict[str, object] | None = None,
        checksum: str | None = None,
        ttl_hours: int | None = None,
    ) -> UUID:
        asset = AssetORM(
            id=uuid4(),
            owner_type=owner_type,
            owner_id=owner_id,
            kind=kind,
            storage_key=storage_key,
            mime=mime,
            size_bytes=size_bytes,
            meta=meta or {},
            checksum_sha256=checksum,
            expires_at=(datetime.now(UTC) + timedelta(hours=ttl_hours) if ttl_hours else None),
        )
        self._session.add(asset)
        await self._session.flush()
        return asset.id

    async def get(self, asset_id: UUID) -> Asset | None:
        row = await self._session.get(AssetORM, asset_id)
        return asset_to_domain(row) if row is not None else None

    async def list_for_owner(
        self,
        owner_type: AssetOwnerType,
        owner_id: UUID,
    ) -> Sequence[Asset]:
        """Артефакты владельца — доменными объектами, а не строками ORM.

        Разница не косметическая. Строка ORM живёт только внутри сессии:
        после выхода из ``async with`` обращение к её полю падает
        ``DetachedInstanceError``, причём выборочно — там, где атрибут
        успел протухнуть после коммита. Ровно так и упал первый прогон
        нарезки, распечатывая готовые клипы.
        """
        stmt = select(AssetORM).where(
            AssetORM.owner_type == owner_type,
            AssetORM.owner_id == owner_id,
            AssetORM.deleted_at.is_(None),
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [asset_to_domain(row) for row in rows]

    async def delete_for_owner(self, owner_type: AssetOwnerType, owner_id: UUID) -> None:
        stmt = delete(AssetORM).where(
            AssetORM.owner_type == owner_type,
            AssetORM.owner_id == owner_id,
        )
        await self._session.execute(stmt)


class SqlAlchemyUploadSessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        user_id: UUID,
        filename: str,
        chunk_size: int,
        total_size: int | None,
        ttl_hours: int = 24,
    ) -> UploadSession:
        row = UploadSessionORM(
            id=uuid4(),
            user_id=user_id,
            filename=filename,
            chunk_size=chunk_size,
            total_size=total_size,
            received_parts=[],
            received_bytes=0,
            expires_at=datetime.now(UTC) + timedelta(hours=ttl_hours),
        )
        self._session.add(row)
        await self._session.flush()
        return upload_session_to_domain(row)

    async def get_owned(self, upload_id: UUID, user_id: UUID) -> UploadSession | None:
        stmt = select(UploadSessionORM).where(
            UploadSessionORM.id == upload_id,
            UploadSessionORM.user_id == user_id,
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return upload_session_to_domain(row) if row else None

    async def register_part(self, upload_id: UUID, part_number: int, size: int) -> UploadSession:
        """Отметить полученную часть.

        Повторная отправка той же части не увеличивает счётчик байтов:
        клиент имеет право переслать часть после обрыва, и это не должно
        ломать учёт. Идемпотентность здесь обязательна.
        """
        row = await self._session.get(UploadSessionORM, upload_id, with_for_update=True)
        if row is None:
            raise LookupError(upload_id)

        parts = set(row.received_parts or [])
        if part_number not in parts:
            parts.add(part_number)
            row.received_bytes += size
        row.received_parts = sorted(parts)
        await self._session.flush()
        return upload_session_to_domain(row)

    async def complete(self, upload_id: UUID, video_id: UUID) -> None:
        stmt = (
            update(UploadSessionORM)
            .where(UploadSessionORM.id == upload_id)
            .values(completed=True, video_id=video_id)
        )
        await self._session.execute(stmt)

    async def delete(self, upload_id: UUID) -> None:
        await self._session.execute(
            delete(UploadSessionORM).where(UploadSessionORM.id == upload_id)
        )


class SqlAlchemyClipRepository:
    """Клипы и их связь со сценами.

    Связка ``clip_scenes`` пишется вместе с клипом, а не отдельным вызовом:
    клип без сцен — потерянная связь, по которой потом не ответить на вопрос
    «из чего собран этот момент».
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_add(self, clips: Sequence[Clip]) -> None:
        if not clips:
            return
        await self._session.execute(
            insert(ClipORM).values([clip_to_orm_values(clip) for clip in clips])
        )
        links = [
            {"clip_id": clip.id, "scene_id": scene_id}
            for clip in clips
            for scene_id in clip.scene_ids
        ]
        if links:
            await self._session.execute(insert(ClipSceneORM).values(links))

    async def get(self, clip_id: UUID) -> Clip | None:
        row = (
            await self._session.execute(select(ClipORM).where(ClipORM.id == clip_id))
        ).scalar_one_or_none()
        if row is None:
            return None
        return clip_to_domain(row, await self._scene_ids(clip_id))

    async def list_for_job(self, job_id: UUID) -> Sequence[Clip]:
        stmt = select(ClipORM).where(ClipORM.job_id == job_id).order_by(ClipORM.rank)
        rows = (await self._session.execute(stmt)).scalars().all()
        return [clip_to_domain(row, await self._scene_ids(row.id)) for row in rows]

    async def update(self, clip: Clip) -> Clip:
        values = clip_to_orm_values(clip)
        values.pop("id", None)
        await self._session.execute(update(ClipORM).where(ClipORM.id == clip.id).values(**values))
        return clip

    async def delete_for_job(self, job_id: UUID) -> None:
        """Убрать клипы джоба перед повторным отбором.

        Нужна ровно та же идемпотентность, что у сцен: повторный запуск
        этапа не должен дописывать второй комплект клипов к первому.
        """
        await self._session.execute(delete(ClipORM).where(ClipORM.job_id == job_id))

    async def _scene_ids(self, clip_id: UUID) -> list[UUID]:
        stmt = select(ClipSceneORM.scene_id).where(ClipSceneORM.clip_id == clip_id)
        return list((await self._session.execute(stmt)).scalars().all())
