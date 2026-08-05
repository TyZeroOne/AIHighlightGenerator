"""Порты репозиториев и Unit of Work.

Repository прячет способ хранения: сценарий говорит «дай мне видео по
контрольной сумме», а не «сделай SELECT». Unit of Work объединяет несколько
операций в одну транзакцию: либо сохранились и видео, и его артефакт,
либо ничего.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from core.domain.entities import (
    Asset,
    Clip,
    Job,
    Scene,
    StageState,
    UploadSession,
    User,
    Video,
)
from core.domain.enums import AssetKind, AssetOwnerType, JobStage, JobStatus
from core.domain.media import Transcript

__all__ = [
    "AssetRecord",
    "AssetRepositoryPort",
    "ClipRepository",
    "JobRepository",
    "SceneRepository",
    "TranscriptRepository",
    "UnitOfWork",
    "UnitOfWorkFactory",
    "UploadSessionRepository",
    "UserRepository",
    "VideoRepository",
]


#: Артефакт наружу отдаётся доменным объектом. Раньше здесь стоял
#: отдельный протокол ``AssetRecord`` — «минимум, который нужно знать
#: об артефакте», — и репозиторий возвращал под ним строку ORM.
#: Это оказалось ошибкой дважды. Во-первых, строка живёт только внутри
#: сессии и за её пределами падает `DetachedInstanceError`. Во-вторых,
#: изменяемые атрибуты в ``Protocol`` инвариантны, и подставить под него
#: неизменяемый объект нельзя — тот же самый урок, что и с репозиториями
#: на Этапе 2. Домен уже описывает артефакт; второе описание не нужно.
type AssetRecord = Asset


class UserRepository(Protocol):
    async def get(self, user_id: UUID) -> User | None: ...
    async def get_by_email(self, email: str) -> User | None: ...
    async def get_by_api_key_hash(self, key_hash: str) -> User | None: ...
    async def create(
        self, *, email: str, api_key_hash: str, password_hash: str | None = None
    ) -> User: ...


class VideoRepository(Protocol):
    async def add(self, video: Video) -> Video: ...
    async def get(self, video_id: UUID) -> Video | None: ...

    async def get_owned(self, video_id: UUID, user_id: UUID) -> Video | None:
        """Видео конкретного пользователя.

        Проверка владения зашита в запрос намеренно: если бы существовал
        только ``get()``, рано или поздно кто-нибудь забыл бы сверить
        ``video.user_id`` — и появился бы доступ к чужим данным.
        """
        ...

    async def get_by_checksum(self, user_id: UUID, checksum: str) -> Video | None: ...
    async def list_for_user(
        self, user_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Video]: ...
    async def count_for_user(self, user_id: UUID) -> int: ...
    async def update(self, video: Video) -> Video: ...
    async def mark_deleted(self, video_id: UUID) -> None: ...


class AssetRepositoryPort(Protocol):
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
    ) -> UUID: ...

    async def get(self, asset_id: UUID) -> AssetRecord | None:
        """Артефакт по идентификатору.

        Нужен скачиванию: клиент приходит с ``asset_id`` из списка клипов,
        а владельца (и, через него, право на скачивание) проверяет уже API.
        """
        ...

    async def list_for_owner(
        self, owner_type: AssetOwnerType, owner_id: UUID
    ) -> Sequence[AssetRecord]: ...
    async def delete_for_owner(self, owner_type: AssetOwnerType, owner_id: UUID) -> None: ...

    async def expired(self, *, limit: int = 500) -> Sequence[AssetRecord]:
        """Артефакты, у которых вышел срок хранения.

        Отдаются пачками: за сутки простоя их накапливаются тысячи,
        а удаление каждого — это поход в хранилище, и делать их все
        в одной транзакции значит держать её открытой минутами.
        """
        ...

    async def mark_deleted(self, asset_ids: Sequence[UUID]) -> None:
        """Пометить удалёнными — после того, как файлы действительно убраны.

        Мягко, а не ``DELETE``: запись о том, что артефакт был и когда
        исчез, стоит одну строку и отвечает на вопрос «куда делся файл»,
        который иначе выясняется просмотром логов за неделю.
        """
        ...


class UploadSessionRepository(Protocol):
    async def create(
        self,
        *,
        user_id: UUID,
        filename: str,
        chunk_size: int,
        total_size: int | None,
        ttl_hours: int = 24,
    ) -> UploadSession: ...

    async def get_owned(self, upload_id: UUID, user_id: UUID) -> UploadSession | None: ...
    async def register_part(
        self, upload_id: UUID, part_number: int, size: int
    ) -> UploadSession: ...
    async def complete(self, upload_id: UUID, video_id: UUID) -> None: ...
    async def delete(self, upload_id: UUID) -> None: ...


class JobRepository(Protocol):
    async def add(self, job: Job) -> Job: ...
    async def get(self, job_id: UUID) -> Job | None: ...

    async def find_succeeded(self, video_id: UUID, params_hash: str) -> Job | None:
        """Основа идемпотентности ``POST /analyze`` (док 04, §4.6)."""
        ...

    async def update(self, job: Job) -> Job: ...

    async def write_stage(
        self,
        job_id: UUID,
        stage: JobStage,
        state: StageState,
        *,
        status: JobStatus,
        error: dict[str, object] | None = None,
    ) -> Job | None:
        """Записать состояние ОДНОГО этапа, не трогая остальные.

        Отдельный метод, а не ``update`` с готовым объектом, потому что
        ветки конвейера идут параллельно. Обычная запись читает джоб
        целиком, меняет своё поле и сохраняет всё обратно — и вторая
        ветка, закончившаяся позже, затирает отметку первой.

        Это не теория: живой прогон показал джоб, застрявший на 52 %,
        у которого пропала запись об успешно отработавшем видеоанализе.
        Классический lost update — тот же, что был с ``jobs.cost``
        на Этапе 6, но здесь он проявился только тогда, когда в графе
        появились настоящие параллельные ветки.
        """
        ...

    async def list_by_status(self, status: JobStatus, *, limit: int) -> Sequence[Job]: ...

    async def list_for_user(
        self,
        user_id: UUID,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Job]:
        """Джобы пользователя, новые сверху.

        Фильтр по владельцу зашит в запрос, а не оставлен вызывающему:
        отдельный ``list_all`` рано или поздно позвали бы без него.
        """
        ...

    async def minutes_used(self, user_id: UUID, *, since: datetime) -> float:
        """Сколько минут видео пользователь отправил в обработку с этого момента.

        Считается по длительности исходников, а не по времени работы
        воркеров: квота выдаётся в минутах материала, и пользователь
        не должен платить за то, что наша модель считает медленно.
        """
        ...

    async def stuck(self, *, older_than_sec: float, limit: int = 100) -> Sequence[Job]:
        """Джобы, застрявшие в работе дольше положенного.

        Нужны периодической уборке: воркер, убитый по OOM, оставляет джоб
        в ``running`` навсегда, и снаружи он неотличим от живого.
        """
        ...


class SceneRepository(Protocol):
    async def bulk_add(self, scenes: Sequence[Scene]) -> None:
        """Массовая вставка: сцен бывают тысячи, по одной — недопустимо медленно."""
        ...

    async def list_for_job(self, job_id: UUID) -> Sequence[Scene]: ...
    async def count_for_job(self, job_id: UUID) -> int: ...

    async def delete_for_job(self, job_id: UUID) -> None:
        """Убрать сцены джоба перед повторным разбиением.

        Нужно ради идемпотентности: у таблицы уникальный индекс
        ``(job_id, index)``, и повторный прогон этапа без очистки упёрся бы
        в конфликт вместо того, чтобы честно переписать результат.
        """
        ...

    async def bulk_update(self, scenes: Sequence[Scene]) -> None:
        """Обновить изменяемую часть сцен: признаки, скоры, текст.

        Один метод на всё перечисленное, а не три: обновления идут пачками
        по тысяче сцен, и каждый отдельный метод означал бы ещё один проход
        по таблице ради одной колонки.
        """
        ...

    async def top_by_score(self, job_id: UUID, *, limit: int) -> Sequence[Scene]: ...


class TranscriptRepository(Protocol):
    async def replace_for_job(self, job_id: UUID, transcript: Transcript, video_id: UUID) -> int:
        """Записать транскрипт джоба целиком, заменив прежний.

        Именно «заменить», а не «добавить»: этап может выполниться повторно
        (``acks_late``, ретрай с этапа), и дописывание дало бы каждому слову
        по второму экземпляру. Возвращает число записанных сегментов.
        """
        ...

    async def get_for_job(self, job_id: UUID) -> Transcript: ...
    async def count_for_job(self, job_id: UUID) -> int: ...


class ClipRepository(Protocol):
    async def bulk_add(self, clips: Sequence[Clip]) -> None: ...
    async def get(self, clip_id: UUID) -> Clip | None: ...
    async def list_for_job(self, job_id: UUID) -> Sequence[Clip]: ...
    async def update(self, clip: Clip) -> Clip: ...

    async def delete_for_job(self, job_id: UUID) -> None:
        """Убрать клипы джоба перед повторным отбором.

        Та же идемпотентность, что у сцен: повторный запуск этапа не должен
        дописывать второй комплект клипов к первому.
        """
        ...


class UnitOfWork(Protocol):
    """Транзакционная граница.

    Использование::

        async with uow_factory() as uow:
            await uow.videos.add(video)
            await uow.assets.add(...)
            await uow.commit()

    Без явного ``commit()`` выход из контекста делает ``rollback()`` —
    так забытый коммит превращается в отсутствие записи, а не в частичную.
    """

    # Репозитории объявлены свойствами, а не атрибутами, и это не стиль.
    # Изменяемый атрибут в Protocol **инвариантен**: реализация обязана иметь
    # ровно объявленный тип, и `assets: AssetRepository` не подошёл бы под
    # `assets: AssetRepositoryPort`, хотя реализует его полностью.
    # Свойство доступно только на чтение и потому ковариантно — подтип годится.
    # Обычный атрибут в реализации свойству протокола соответствует.
    @property
    def users(self) -> UserRepository: ...

    @property
    def videos(self) -> VideoRepository: ...

    @property
    def assets(self) -> AssetRepositoryPort: ...

    @property
    def uploads(self) -> UploadSessionRepository: ...

    @property
    def jobs(self) -> JobRepository: ...

    @property
    def scenes(self) -> SceneRepository: ...

    @property
    def transcripts(self) -> TranscriptRepository: ...

    @property
    def clips(self) -> ClipRepository: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


class UnitOfWorkFactory(Protocol):
    """Фабрика UoW.

    Сценарий получает фабрику, а не готовый UoW: он сам решает, где проходит
    граница транзакции, и может открыть несколько последовательных.
    """

    def __call__(self) -> UnitOfWork: ...
