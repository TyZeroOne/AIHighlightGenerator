# 02 — Архитектура

## 2.1 Правило зависимостей

```
        api/            workers/          (адаптеры доставки)
          \                /
           \              /
            v            v
         core/usecases/                   (сценарии приложения)
                |
                v
    core/domain + core/ports              (сущности + интерфейсы)
                ^
                |
   ┌────────────┴─────────────────────────────────────────┐
   │ video_loader  audio  speech  vision  highlight_detector│  (адаптеры-реализации)
   │ llm  video_editor  subtitles  export  core/infra       │
   └────────────────────────────────────────────────────────┘
```

**Инвариант, который проверяется автотестом (`tests/test_architecture.py`):**

- `core/domain` и `core/ports` не импортируют **ничего** из проекта и ничего тяжёлого из внешнего мира
  (нет `torch`, `cv2`, `sqlalchemy`, `fastapi`, `celery`).
- Модули-адаптеры (`audio`, `vision`, `speech`, …) импортируют **только** `core`, никогда — друг друга.
- `api/` и `workers/` импортируют `core/usecases` и композиционный корень, но не лезут в адаптеры напрямую.

Проверка реализуется обходом AST всех файлов пакета — это дешевле, чем полагаться на дисциплину.

## 2.2 Структура репозитория

```
D:\Slayer\
├─ docs/                      # эта документация
├─ pyproject.toml             # зависимости, extras по модулям
├─ docker-compose.yml
├─ docker/
│  ├─ api.Dockerfile
│  ├─ worker.Dockerfile       # + ffmpeg, + опционально CUDA
│  └─ entrypoint.sh
├─ alembic/                   # миграции
├─ src/
│  ├─ core/
│  │  ├─ domain/              # сущности, VO, доменные ошибки  (чистый Python)
│  │  ├─ ports/               # Protocol-интерфейсы            (чистый Python)
│  │  ├─ usecases/            # сценарии, оркестрация          (зависит от ports)
│  │  ├─ registry.py          # реестр плагинов-анализаторов
│  │  ├─ container.py         # композиционный корень (DI)
│  │  ├─ config.py            # Pydantic Settings
│  │  ├─ logging.py           # structlog, trace_id
│  │  └─ infra/               # SQLAlchemy-репозитории, storage, кеш
│  ├─ video_loader/           # SourceProvider'ы, probe, валидация, разбиение на сцены
│  ├─ audio/                  # извлечение аудио, громкость, паузы, аудиособытия
│  ├─ speech/                 # faster-whisper, темп речи, эмоции речи
│  ├─ vision/                 # движение, лица, люди, динамика кадра
│  ├─ highlight_detector/     # нормализация признаков, скоринг, отбор моментов
│  ├─ llm/                    # провайдер Claude, промпты, схемы, батчи
│  ├─ video_editor/           # нарезка, вертикаль, перекодирование
│  ├─ subtitles/              # SRT/ASS, burn-in
│  ├─ export/                 # упаковка результата, манифест, публикация
│  ├─ api/                    # FastAPI: роутеры, схемы, зависимости
│  └─ workers/                # Celery app, задачи-обёртки над usecases
└─ tests/
   ├─ unit/  integration/  e2e/  fixtures/
```

## 2.3 Доменные сущности (`core/domain`)

Неизменяемые dataclass'ы, без ORM и без I/O.

```python
# core/domain/values.py
@dataclass(frozen=True, slots=True)
class TimeRange:
    start: float  # секунды от начала медиа
    end: float

    def __post_init__(self) -> None: ...  # end > start, оба >= 0
    @property
    def duration(self) -> float: ...
    def overlaps(self, other: "TimeRange") -> bool: ...
    def intersection(self, other: "TimeRange") -> "TimeRange | None": ...


@dataclass(frozen=True, slots=True)
class MediaWindow:
    """Окно медиа, над которым работает конвейер.
    Для файла — одно окно [0, duration]. Для стрима — бесконечная последовательность окон.
    Именно эта абстракция делает поддержку live бесплатной."""

    source_id: str
    range: TimeRange
    is_final: bool = True  # False → за окном придут ещё данные


@dataclass(frozen=True, slots=True)
class FeatureValue:
    key: str  # "audio.loudness.peak"
    value: float
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class Score:
    value: float  # 0..100
    breakdown: Mapping[str, float]  # вклад каждой группы признаков
```

```python
# core/domain/entities.py
@dataclass(frozen=True, slots=True)
class Scene:
    id: UUID
    video_id: UUID
    index: int
    range: TimeRange
    features: Mapping[str, FeatureValue]
    heuristic_score: Score | None
    llm_score: float | None
    category: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class Highlight:
    range: TimeRange
    final_score: float
    category: str
    reason: str
    scene_ids: tuple[UUID, ...]  # highlight может покрывать несколько сцен
```

## 2.4 Порты (`core/ports`)

Всё — `typing.Protocol`, чтобы адаптеры не наследовались и не тянули зависимость обратно.

```python
class SourceProvider(Protocol):
    scheme: ClassVar[str]  # "file", "upload", "youtube", "twitch", "kick"

    def supports(self, uri: str) -> bool: ...
    async def describe(self, uri: str) -> SourceDescriptor: ...
    async def acquire(self, uri: str, window: TimeRange | None) -> LocalMedia: ...
    async def windows(self, uri: str) -> AsyncIterator[MediaWindow]: ...
    async def release(self, media: LocalMedia) -> None: ...


class SceneDetector(Protocol):
    async def detect(self, media: LocalMedia, window: MediaWindow) -> Sequence[TimeRange]: ...


class AudioExtractor(Protocol):
    async def extract(self, media: LocalMedia, window: MediaWindow) -> AudioTrack: ...


class Transcriber(Protocol):
    async def transcribe(self, audio: AudioTrack) -> Transcript: ...  # word-level тайминги


class FeatureAnalyzer(Protocol):
    key: ClassVar[str]  # "vision.motion"
    produces: ClassVar[frozenset[str]]  # {"vision.motion.mean", "vision.motion.peak"}
    requires: ClassVar[frozenset[str]]  # {"artifact.video"} | {"artifact.transcript"} ...
    cost: ClassVar[int]  # относительная стоимость, для планировщика

    async def analyze(
        self, ctx: AnalysisContext, scenes: Sequence[TimeRange]
    ) -> Mapping[int, list[FeatureValue]]: ...


class ScoringStrategy(Protocol):
    async def score(self, scenes: Sequence[Scene], profile: ScoringProfile) -> Sequence[Score]: ...


class LLMProvider(Protocol):
    async def rank_scenes(
        self, batch: Sequence[SceneSummary], context: VideoContext
    ) -> Sequence[LLMVerdict]: ...
    async def generate_metadata(self, highlight: HighlightSummary) -> ClipMetadata: ...


class ClipRenderer(Protocol):
    async def cut(self, media: LocalMedia, rng: TimeRange, spec: RenderSpec) -> Path: ...
    async def reframe_vertical(self, src: Path, track: Sequence[CropBox]) -> Path: ...


class SubtitleWriter(Protocol):
    def write(self, transcript: Transcript, rng: TimeRange, fmt: SubtitleFormat) -> Path: ...


class ObjectStorage(Protocol):
    async def put(self, key: str, path: Path) -> str: ...
    async def get(self, key: str, dest: Path) -> Path: ...
    async def signed_url(self, key: str, ttl: int) -> str: ...


class UnitOfWork(Protocol):
    videos: VideoRepository
    scenes: SceneRepository
    clips: ClipRepository
    jobs: JobRepository

    async def __aenter__(self) -> "UnitOfWork": ...
    async def commit(self) -> None: ...
```

## 2.5 Механизм плагинов для анализаторов

Три уровня расширения — от «внутри репозитория» до «сторонний пакет».

```python
# core/registry.py
_ANALYZERS: dict[str, type[FeatureAnalyzer]] = {}


def register_analyzer(cls: type[FeatureAnalyzer]) -> type[FeatureAnalyzer]:
    if cls.key in _ANALYZERS:
        raise DuplicateAnalyzerError(cls.key)
    _ANALYZERS[cls.key] = cls
    return cls


def discover() -> None:
    """1) импортирует встроенные модули-анализаторы;
    2) подхватывает внешние через entry points группы 'highlight.analyzers'."""
    for ep in entry_points(group="highlight.analyzers"):
        register_analyzer(ep.load())


def build_plan(enabled: Collection[str], available: Collection[str]) -> list[list[FeatureAnalyzer]]:
    """Топологическая сортировка по requires/produces.
    Возвращает «слои»: анализаторы внутри слоя независимы → идут в Celery group()."""
```

Использование:

```python
# vision/motion.py
@register_analyzer
class MotionAnalyzer:
    key = "vision.motion"
    produces = frozenset({"vision.motion.mean", "vision.motion.peak", "vision.camera.pan"})
    requires = frozenset({"artifact.video"})
    cost = 3

    async def analyze(self, ctx, scenes): ...
```

Добавление нового признака = один новый файл + декоратор. Ни `core`, ни конвейер,
ни API не меняются. Отключение — через конфиг (`ANALYZERS_DISABLED=vision.faces`).

## 2.6 Как добавится YouTube / Twitch / Kick

Ключ в том, что конвейер уже сейчас работает не с «файлом», а с парой
`(LocalMedia, MediaWindow)`.

| | Файл (Этап 2) | Стрим (будущее) |
|---|---|---|
| `describe()` | `ffprobe` | API платформы / `yt-dlp -J` |
| `acquire()` | путь на диске | скачивание сегмента HLS/DASH во временный файл |
| `windows()` | одно окно `[0, duration]`, `is_final=True` | бесконечный генератор окон по 60–120 с, `is_final=False` |
| `release()` | no-op | удаление временного сегмента |

Что **не** меняется: детектор сцен, анализаторы, скоринг, LLM, нарезка, API.
Для live добавится только режим `JobMode.STREAMING`, при котором `workers` подписывается
на `windows()` и запускает конвейер по каждому окну, а не один раз.

Регистрация провайдера — тот же реестр:

```python
@register_source("youtube")
class YouTubeProvider: ...
```

Резолвер выбирает провайдер по схеме URI: `file://…`, `upload://<video_id>`,
`https://youtube.com/…`, `https://twitch.tv/…`, `https://kick.com/…`.

> Юридическая оговорка: скачивание с платформ должно соответствовать их ToS.
> Провайдеры-заглушки в архитектуре есть, реализация — по факту наличия прав/API-ключей.

## 2.7 Композиционный корень

Единственное место, где конкретные классы связываются с портами.

```python
# core/container.py
class Container:
    def __init__(self, settings: Settings) -> None: ...

    @cached_property
    def storage(self) -> ObjectStorage:
        return LocalStorage(...) if self.s.storage_backend == "local" else S3Storage(...)

    @cached_property
    def transcriber(self) -> Transcriber:
        return FasterWhisperTranscriber(model=self.s.whisper_model, device=self.s.device)

    @cached_property
    def llm(self) -> LLMProvider:
        return ClaudeProvider(model=self.s.llm_model, api_key=self.s.anthropic_api_key)

    ...
```

`api/` получает контейнер через FastAPI `Depends`, `workers/` — через модульный синглтон,
инициализируемый в сигнале `worker_process_init`. Тесты подменяют любой порт фейком.

## 2.8 Обработка ошибок

Иерархия в `core/domain/errors.py`:

```
HighlightError
├─ ValidationError        → HTTP 422, Celery: не ретраить
├─ SourceError            → HTTP 400/404
├─ UnsupportedMediaError  → HTTP 415
├─ ProcessingError        → HTTP 500, Celery: ретрай с backoff
│  ├─ FFmpegError
│  ├─ ModelLoadError
│  └─ TransientError      → ретрай агрессивнее
└─ ExternalServiceError   → LLM/S3 недоступны; ретрай + circuit breaker
```

Единый exception handler в `api/`, единый `autoretry_for` в базовом классе задачи `workers/`.
