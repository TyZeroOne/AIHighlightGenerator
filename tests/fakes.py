"""Фейковые реализации портов для тестов.

Не «моки» в смысле ``unittest.mock``: это настоящие рабочие реализации,
просто в памяти. Разница существенная — мок проверяет, что метод вызвали,
фейк проверяет, что система в целом работает правильно. Первое ломается
при любом рефакторинге, второе — только при поломке поведения.

Ни один из этих классов ничего не наследует из ``core``: соответствие
портам структурное. Ровно так же будет выглядеть сторонняя реализация.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self
from uuid import UUID, uuid4

from core.domain.entities import (
    Clip,
    Job,
    Scene,
    StageState,
    UploadSession,
    User,
    Video,
    compute_progress,
)
from core.domain.enums import (
    AssetKind,
    AssetOwnerType,
    JobStage,
    JobStatus,
    UserRole,
    VideoStatus,
)
from core.domain.errors import ObjectNotFoundError
from core.domain.media import (
    AudioTrack,
    LocalMedia,
    MediaInfo,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)
from core.domain.values import MediaWindow, TimeRange

__all__ = [
    "FakeAssetRepository",
    "FakeAudioExtractor",
    "FakeJobRepository",
    "FakeKeyframeExtractor",
    "FakeProbe",
    "FakeSceneDetector",
    "FakeSceneRepository",
    "FakeSourceProvider",
    "FakeStorage",
    "FakeTranscriber",
    "FakeTranscriptRepository",
    "FakeUnitOfWork",
    "FakeUploadSessionRepository",
    "FakeUserRepository",
    "FakeVideoRepository",
    "make_media_info",
    "make_transcript",
    "uow_factory",
]


def make_media_info(**overrides: object) -> MediaInfo:
    base: dict[str, object] = {
        "duration_sec": 120.0,
        "width": 1920,
        "height": 1080,
        "fps": 25.0,
        "video_codec": "h264",
        "audio_codec": "aac",
        "container": "mp4",
        "size_bytes": 1024,
    }
    base.update(overrides)
    return MediaInfo(**base)  # type: ignore[arg-type]


class FakeProbe:
    """ffprobe, который отвечает заранее заданным результатом."""

    def __init__(self, info: MediaInfo | None = None, error: Exception | None = None) -> None:
        self.info = info or make_media_info()
        self.error = error
        self.calls: list[Path] = []

    async def probe(self, path: Path) -> MediaInfo:
        self.calls.append(path)
        if self.error is not None:
            raise self.error
        # Размер берём реальный, если файл есть: тесты на лимиты должны
        # видеть настоящий размер, а не выдуманный.
        if path.exists():
            return MediaInfo(
                duration_sec=self.info.duration_sec,
                width=self.info.width,
                height=self.info.height,
                fps=self.info.fps,
                video_codec=self.info.video_codec,
                audio_codec=self.info.audio_codec,
                container=self.info.container,
                size_bytes=path.stat().st_size,
                raw=self.info.raw,
            )
        return self.info


class FakeStorage:
    """Хранилище в памяти. Реализует тот же протокол, что Local и S3."""

    def __init__(self, *, supports_signed_urls: bool = False) -> None:
        self.objects: dict[str, bytes] = {}
        self._supports_signed = supports_signed_urls

    async def put(self, key: str, source: Path, *, content_type: str | None = None) -> str:
        self.objects[key] = source.read_bytes()
        return key

    async def get(self, key: str, destination: Path) -> Path:
        if key not in self.objects:
            raise ObjectNotFoundError(f"Объект не найден: {key}", key=key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])
        return destination

    async def _stream(self, key: str, chunk_size: int) -> AsyncIterator[bytes]:
        if key not in self.objects:
            raise ObjectNotFoundError(f"Объект не найден: {key}", key=key)
        data = self.objects[key]
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]

    def open_stream(self, key: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        return self._stream(key, chunk_size)

    async def exists(self, key: str) -> bool:
        return key in self.objects

    async def size(self, key: str) -> int:
        if key not in self.objects:
            raise ObjectNotFoundError(f"Объект не найден: {key}", key=key)
        return len(self.objects[key])

    async def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    async def delete_prefix(self, prefix: str) -> int:
        keys = [key for key in self.objects if key.startswith(prefix)]
        for key in keys:
            del self.objects[key]
        return len(keys)

    async def signed_url(self, key: str, *, ttl_sec: int = 900) -> str | None:
        return f"https://fake/{key}?ttl={ttl_sec}" if self._supports_signed else None

    async def healthcheck(self) -> None:
        return None


class FakeUserRepository:
    def __init__(self, users: dict[UUID, User] | None = None) -> None:
        self.users = users or {}
        self.api_keys: dict[str, UUID] = {}

    async def get(self, user_id: UUID) -> User | None:
        return self.users.get(user_id)

    async def get_by_email(self, email: str) -> User | None:
        return next((u for u in self.users.values() if u.email.lower() == email.lower()), None)

    async def get_by_api_key_hash(self, key_hash: str) -> User | None:
        user_id = self.api_keys.get(key_hash)
        return self.users.get(user_id) if user_id else None

    async def create(
        self, *, email: str, api_key_hash: str, password_hash: str | None = None
    ) -> User:
        user = User(id=uuid4(), email=email, role=UserRole.USER)
        self.users[user.id] = user
        self.api_keys[api_key_hash] = user.id
        return user


class FakeVideoRepository:
    def __init__(self, videos: dict[UUID, Video] | None = None) -> None:
        self.videos = videos if videos is not None else {}

    async def add(self, video: Video) -> Video:
        stored = Video(
            id=video.id,
            user_id=video.user_id,
            source_uri=video.source_uri,
            source_kind=video.source_kind,
            status=video.status,
            storage_key=video.storage_key,
            checksum_sha256=video.checksum_sha256,
            info=video.info,
            original_filename=video.original_filename,
            size_bytes=video.size_bytes,
            created_at=datetime.now(UTC),
        )
        self.videos[video.id] = stored
        return stored

    async def get(self, video_id: UUID) -> Video | None:
        return self.videos.get(video_id)

    async def get_owned(self, video_id: UUID, user_id: UUID) -> Video | None:
        video = self.videos.get(video_id)
        if video is None or video.user_id != user_id or video.status is VideoStatus.DELETED:
            return None
        return video

    async def get_by_checksum(self, user_id: UUID, checksum: str) -> Video | None:
        return next(
            (
                video
                for video in self.videos.values()
                if video.user_id == user_id
                and video.checksum_sha256 == checksum
                and video.status is not VideoStatus.DELETED
            ),
            None,
        )

    async def list_for_user(
        self, user_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Video]:
        items = [
            video
            for video in self.videos.values()
            if video.user_id == user_id and video.status is not VideoStatus.DELETED
        ]
        return items[offset : offset + limit]

    async def count_for_user(self, user_id: UUID) -> int:
        return len(await self.list_for_user(user_id, limit=10**6))

    async def update(self, video: Video) -> Video:
        self.videos[video.id] = video
        return video

    async def mark_deleted(self, video_id: UUID) -> None:
        video = self.videos.get(video_id)
        if video is not None:
            self.videos[video_id] = Video(
                id=video.id,
                user_id=video.user_id,
                source_uri=video.source_uri,
                source_kind=video.source_kind,
                status=VideoStatus.DELETED,
                storage_key=video.storage_key,
                checksum_sha256=video.checksum_sha256,
                info=video.info,
                original_filename=video.original_filename,
                size_bytes=video.size_bytes,
                created_at=video.created_at,
            )


class FakeAssetRepository:
    def __init__(self) -> None:
        self.assets: list[dict[str, object]] = []

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
        asset_id = uuid4()
        self.assets.append(
            {
                "id": asset_id,
                "owner_type": owner_type,
                "owner_id": owner_id,
                "kind": kind,
                "storage_key": storage_key,
                "mime": mime,
                "size_bytes": size_bytes,
            }
        )
        return asset_id

    async def list_for_owner(
        self, owner_type: AssetOwnerType, owner_id: UUID
    ) -> Sequence[dict[str, object]]:
        return [
            asset
            for asset in self.assets
            if asset["owner_type"] == owner_type and asset["owner_id"] == owner_id
        ]

    async def delete_for_owner(self, owner_type: AssetOwnerType, owner_id: UUID) -> None:
        self.assets = [
            asset
            for asset in self.assets
            if not (asset["owner_type"] == owner_type and asset["owner_id"] == owner_id)
        ]


class FakeUploadSessionRepository:
    def __init__(self) -> None:
        self.sessions: dict[UUID, UploadSession] = {}

    async def create(
        self,
        *,
        user_id: UUID,
        filename: str,
        chunk_size: int,
        total_size: int | None,
        ttl_hours: int = 24,
    ) -> UploadSession:
        session = UploadSession(
            id=uuid4(),
            user_id=user_id,
            filename=filename,
            chunk_size=chunk_size,
            total_size=total_size,
            created_at=datetime.now(UTC),
        )
        self.sessions[session.id] = session
        return session

    async def get_owned(self, upload_id: UUID, user_id: UUID) -> UploadSession | None:
        session = self.sessions.get(upload_id)
        return session if session and session.user_id == user_id else None

    async def register_part(self, upload_id: UUID, part_number: int, size: int) -> UploadSession:
        session = self.sessions[upload_id]
        parts = set(session.received_parts)
        received_bytes = session.received_bytes
        if part_number not in parts:
            parts.add(part_number)
            received_bytes += size
        updated = UploadSession(
            id=session.id,
            user_id=session.user_id,
            filename=session.filename,
            chunk_size=session.chunk_size,
            total_size=session.total_size,
            received_parts=tuple(sorted(parts)),
            received_bytes=received_bytes,
            completed=session.completed,
            created_at=session.created_at,
        )
        self.sessions[upload_id] = updated
        return updated

    async def complete(self, upload_id: UUID, video_id: UUID) -> None:
        session = self.sessions[upload_id]
        self.sessions[upload_id] = UploadSession(
            id=session.id,
            user_id=session.user_id,
            filename=session.filename,
            chunk_size=session.chunk_size,
            total_size=session.total_size,
            received_parts=session.received_parts,
            received_bytes=session.received_bytes,
            completed=True,
            created_at=session.created_at,
        )

    async def delete(self, upload_id: UUID) -> None:
        self.sessions.pop(upload_id, None)


class FakeJobRepository:
    def __init__(self, jobs: dict[UUID, Job] | None = None) -> None:
        self.jobs = jobs if jobs is not None else {}

    async def add(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job

    async def get(self, job_id: UUID) -> Job | None:
        return self.jobs.get(job_id)

    async def find_succeeded(self, video_id: UUID, params_hash: str) -> Job | None:
        return next(
            (
                job
                for job in self.jobs.values()
                if job.video_id == video_id
                and job.params_hash == params_hash
                and job.status is JobStatus.SUCCEEDED
            ),
            None,
        )

    async def update(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job

    async def write_stage(
        self,
        job_id: UUID,
        stage: JobStage,
        state: StageState,
        *,
        status: JobStatus,
        error: dict[str, object] | None = None,
    ) -> Job | None:
        """Атомарная запись одного этапа — как в настоящем репозитории.

        Фейк обязан вести себя так же в главном: менять **только** своё
        поле. Если бы он писал джоб целиком, тесты проходили бы, а живой
        конвейер терял отметки параллельных веток — ровно то, что
        и случилось на Этапе 10 до появления этого метода.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return None

        states = dict(job.stages)
        states[stage] = state
        updated = replace(
            job,
            stage=stage,
            stages=states,
            progress=compute_progress(states),
            status=status,
            error=error if error is not None else job.error,
        )
        self.jobs[job_id] = updated
        return updated

    async def list_by_status(self, status: JobStatus, *, limit: int = 50) -> Sequence[Job]:
        return [job for job in self.jobs.values() if job.status is status][:limit]


class FakeSceneRepository:
    def __init__(self) -> None:
        self.scenes: list[Scene] = []

    async def bulk_add(self, scenes: Sequence[Scene]) -> None:
        self.scenes.extend(scenes)

    async def list_for_job(self, job_id: UUID) -> Sequence[Scene]:
        return sorted(
            (scene for scene in self.scenes if scene.job_id == job_id),
            key=lambda scene: scene.index,
        )

    async def count_for_job(self, job_id: UUID) -> int:
        return sum(1 for scene in self.scenes if scene.job_id == job_id)

    async def delete_for_job(self, job_id: UUID) -> None:
        self.scenes = [scene for scene in self.scenes if scene.job_id != job_id]

    async def bulk_update(self, scenes: Sequence[Scene]) -> None:
        by_id = {scene.id: scene for scene in scenes}
        self.scenes = [by_id.get(scene.id, scene) for scene in self.scenes]

    async def top_by_score(self, job_id: UUID, *, limit: int) -> Sequence[Scene]:
        ranked = sorted(
            (scene for scene in self.scenes if scene.job_id == job_id),
            key=lambda scene: scene.final_score or -1.0,
            reverse=True,
        )
        return ranked[:limit]


class FakeTranscriptRepository:
    def __init__(self) -> None:
        self.by_job: dict[UUID, Transcript] = {}
        self.writes = 0

    async def replace_for_job(self, job_id: UUID, transcript: Transcript, video_id: UUID) -> int:
        self.by_job[job_id] = transcript
        self.writes += 1
        return len(transcript.segments)

    async def get_for_job(self, job_id: UUID) -> Transcript:
        return self.by_job.get(job_id, Transcript())

    async def count_for_job(self, job_id: UUID) -> int:
        return len(self.by_job.get(job_id, Transcript()).segments)


class FakeAudioExtractor:
    """Извлечение аудио, которое пишет на диск заранее заданные байты."""

    def __init__(
        self,
        *,
        duration_sec: float = 120.0,
        error: Exception | None = None,
        payload: bytes = b"RIFF" + b"\x00" * 60,
    ) -> None:
        self.duration_sec = duration_sec
        self.error = error
        self.payload = payload
        self.calls: list[MediaWindow] = []

    async def extract(self, media: LocalMedia, window: MediaWindow) -> AudioTrack:
        self.calls.append(window)
        if self.error is not None:
            raise self.error
        path = media.path.with_suffix(".wav")
        path.write_bytes(self.payload)
        return AudioTrack(
            path=path,
            sample_rate=16000,
            channels=1,
            duration_sec=self.duration_sec,
        )


class FakeTranscriber:
    """Транскрайбер, отдающий заранее заданный результат."""

    def __init__(
        self,
        transcript: Transcript | None = None,
        error: Exception | None = None,
    ) -> None:
        self.transcript = transcript if transcript is not None else Transcript()
        self.error = error
        self.calls: list[AudioTrack] = []
        self.progress_reports: list[float] = []

    async def transcribe(
        self,
        audio: AudioTrack,
        *,
        language: str | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> Transcript:
        self.calls.append(audio)
        if self.error is not None:
            raise self.error
        if progress is not None:
            for fraction in (0.5, 1.0):
                progress(fraction)
                self.progress_reports.append(fraction)
        return self.transcript


def make_transcript(*spans: tuple[float, float, str], language: str = "ru") -> Transcript:
    """Транскрипт из троек «начало, конец, текст» — с пословными таймингами.

    Слова раскладываются равномерно по сегменту: тестам важно, что они
    есть и монотонны, а не то, где именно проходит граница между ними.
    """
    segments: list[TranscriptSegment] = []
    for start, end, text in spans:
        tokens = text.split()
        step = (end - start) / max(len(tokens), 1)
        words = tuple(
            TranscriptWord(
                text=token,
                start=start + step * index,
                end=start + step * (index + 1),
                probability=0.9,
            )
            for index, token in enumerate(tokens)
        )
        segments.append(
            TranscriptSegment(
                text=text,
                range=TimeRange(start, end),
                words=words,
                language=language,
                avg_logprob=-0.3,
                no_speech_prob=0.05,
            )
        )
    return Transcript(segments=tuple(segments), language=language)


class FakeSceneDetector:
    """Детектор, отдающий заранее заданные сцены.

    Умеет и «сломаться» — это нужно проверке того, что провалившийся этап
    помечает джоб, а не оставляет его вечно в ``running``.
    """

    def __init__(
        self,
        scenes: Sequence[TimeRange] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.scenes = list(scenes) if scenes is not None else [TimeRange(0.0, 120.0)]
        self.error = error
        self.calls: list[MediaWindow] = []

    async def detect(self, media: LocalMedia, window: MediaWindow) -> Sequence[TimeRange]:
        self.calls.append(window)
        if self.error is not None:
            raise self.error
        return self.scenes


class FakeKeyframeExtractor:
    def __init__(self, *, fail_indexes: frozenset[int] = frozenset()) -> None:
        self.fail_indexes = fail_indexes
        self.moments: list[float] = []

    async def extract(
        self,
        media: LocalMedia,
        moments: Sequence[float],
        destination: Path,
    ) -> Sequence[Path | None]:
        self.moments = list(moments)
        destination.mkdir(parents=True, exist_ok=True)
        frames: list[Path | None] = []
        for index, _ in enumerate(moments):
            if index in self.fail_indexes:
                frames.append(None)
                continue
            path = destination / f"{index:06d}.jpg"
            path.write_bytes(b"\xff\xd8\xff\xdb-fake-jpeg")
            frames.append(path)
        return frames


class FakeSourceProvider:
    """Провайдер, отдающий заранее известный файл. Считает release()."""

    scheme = "fake"

    def __init__(self, media: LocalMedia) -> None:
        self.media = media
        self.acquired = 0
        self.released = 0

    @classmethod
    def supports(cls, uri: str) -> bool:
        return True

    async def describe(self, uri: str) -> object:
        return self.media

    async def acquire(self, uri: str, window: TimeRange | None = None) -> LocalMedia:
        self.acquired += 1
        return self.media

    async def release(self, media: LocalMedia) -> None:
        self.released += 1


class FakeClipRepository:
    def __init__(self) -> None:
        self.clips: list[Clip] = []

    async def bulk_add(self, clips: Sequence[Clip]) -> None:
        self.clips.extend(clips)

    async def get(self, clip_id: UUID) -> Clip | None:
        return next((clip for clip in self.clips if clip.id == clip_id), None)

    async def list_for_job(self, job_id: UUID) -> Sequence[Clip]:
        return sorted((clip for clip in self.clips if clip.job_id == job_id), key=lambda c: c.rank)

    async def update(self, clip: Clip) -> Clip:
        self.clips = [clip if item.id == clip.id else item for item in self.clips]
        return clip

    async def delete_for_job(self, job_id: UUID) -> None:
        self.clips = [clip for clip in self.clips if clip.job_id != job_id]


class FakeUnitOfWork:
    """UoW без транзакций: репозитории общие, коммит только считается.

    Отсутствие настоящего отката — сознательное упрощение. Транзакционность
    проверяется интеграционными тестами на живой БД; здесь важно другое —
    что сценарий вызывает ``commit()`` там, где должен.
    """

    def __init__(
        self,
        *,
        users: FakeUserRepository | None = None,
        videos: FakeVideoRepository | None = None,
        assets: FakeAssetRepository | None = None,
        uploads: FakeUploadSessionRepository | None = None,
        jobs: FakeJobRepository | None = None,
        scenes: FakeSceneRepository | None = None,
        transcripts: FakeTranscriptRepository | None = None,
        clips: FakeClipRepository | None = None,
    ) -> None:
        self.users = users or FakeUserRepository()
        self.videos = videos or FakeVideoRepository()
        self.assets = assets or FakeAssetRepository()
        self.uploads = uploads or FakeUploadSessionRepository()
        self.jobs = jobs or FakeJobRepository()
        self.scenes = scenes or FakeSceneRepository()
        self.transcripts = transcripts or FakeTranscriptRepository()
        self.clips = clips or FakeClipRepository()
        self.commits = 0
        self.rollbacks = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


def uow_factory(uow: FakeUnitOfWork):  # type: ignore[no-untyped-def]
    """Фабрика, всегда отдающая один и тот же UoW.

    В проде каждый вызов создаёт новый UoW со своей транзакцией. В тесте
    нужен общий, чтобы после выполнения сценария заглянуть в репозитории
    и проверить, что там оказалось.
    """

    def _factory() -> FakeUnitOfWork:
        return uow

    return _factory


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
