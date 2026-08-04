"""Unit of Work — транзакционная граница поверх сессии SQLAlchemy.

Смысл: несколько операций либо применяются вместе, либо не применяются вовсе.
Загрузка видео — как раз такой случай: запись в ``videos`` и запись в ``assets``
должны появиться одновременно, иначе в базе окажется видео без файла
или файл без владельца.
"""

from __future__ import annotations

from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.infra.db.repositories import (
    AssetRepository,
    SqlAlchemyClipRepository,
    SqlAlchemyJobRepository,
    SqlAlchemySceneRepository,
    SqlAlchemyTranscriptRepository,
    SqlAlchemyUploadSessionRepository,
    SqlAlchemyUserRepository,
    SqlAlchemyVideoRepository,
)

__all__ = ["SqlAlchemyUnitOfWork"]


class SqlAlchemyUnitOfWork:
    """Реализация порта ``UnitOfWork``.

    Репозитории создаются в ``__aenter__``, а не в конструкторе: все они
    делят одну сессию, а сессия живёт ровно столько, сколько живёт транзакция.
    Создать UoW и не войти в него — значит не получить репозиториев вовсе,
    и это правильно: обращение к БД вне транзакции должно быть невозможно.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> Self:
        self._session = self._session_factory()
        self.users = SqlAlchemyUserRepository(self._session)
        self.videos = SqlAlchemyVideoRepository(self._session)
        self.assets = AssetRepository(self._session)
        self.uploads = SqlAlchemyUploadSessionRepository(self._session)
        self.jobs = SqlAlchemyJobRepository(self._session)
        self.scenes = SqlAlchemySceneRepository(self._session)
        self.transcripts = SqlAlchemyTranscriptRepository(self._session)
        self.clips = SqlAlchemyClipRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._session is not None
        try:
            # Откат по умолчанию. Забытый commit() превращается в отсутствие
            # записи — заметную и легко находимую ошибку. Если бы по умолчанию
            # был commit, забытый rollback давал бы частично применённую
            # транзакцию, которую в проде не отловить.
            await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("UnitOfWork используется вне контекстного менеджера")
        return self._session

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()

    async def flush(self) -> None:
        """Отправить изменения в БД без фиксации.

        Нужно, когда следующий шаг зависит от сгенерированного БД значения
        (например, от проверки уникального индекса), но транзакцию
        завершать рано.
        """
        await self.session.flush()
