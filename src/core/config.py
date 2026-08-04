"""Конфигурация приложения.

Один источник правды на все процессы. Читается из переменных окружения
и файла ``.env``, валидируется Pydantic при старте: процесс с некорректным
конфигом не поднимется вообще, вместо того чтобы упасть через час работы
на строке ``int(os.environ["FOO"])``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["AppEnv", "Settings", "get_settings"]


class AppEnv(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # лишние переменные в окружении — не повод падать
        case_sensitive=False,
    )

    # --- приложение ---
    app_env: AppEnv = AppEnv.DEV
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"
    secret_key: str = "dev-insecure-change-me"

    # --- база данных ---
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "highlights"
    postgres_user: str = "hg"
    postgres_password: str = "hg-dev-password"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # --- redis / celery ---
    redis_url: RedisDsn = Field(default="redis://redis:6379/0")  # type: ignore[assignment]
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"
    celery_task_time_limit: int = 3600

    # --- хранилище ---
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_root: Path = Path("/media")
    s3_endpoint_url: str | None = None
    #: Адрес хранилища, видимый КЛИЕНТУ. Отличается от внутреннего, когда
    #: сервер ходит в S3 по адресу docker-сети, а браузер — по публичному
    #: домену. Хост входит в подпись presigned-ссылки, поэтому подписывать
    #: её нужно для того адреса, по которому пойдёт клиент.
    s3_public_endpoint_url: str | None = None
    s3_bucket: str = "highlights"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_signed_url_ttl: int = 900

    # --- медиа ---
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    max_upload_bytes: int = 20 * 1024**3
    max_duration_sec: int = 6 * 3600
    allowed_containers: str = "mp4,mkv,mov"
    upload_chunk_size: int = 32 * 1024 * 1024
    workdir: Path = Path("/tmp/highlight-generator")
    #: Каталоги, из которых разрешено регистрировать локальные файлы.
    #: Пустой список означает «нельзя ниоткуда»: включать эту возможность
    #: нужно осознанно, иначе `file:///etc/passwd` станет валидным запросом.
    local_source_roots_raw: str = ""

    # --- детекция сцен ---
    scene_detector: Literal["pyscenedetect", "ffmpeg"] = "pyscenedetect"
    #: Порог различия соседних кадров в шкале PySceneDetect (0..255).
    #: Для ffmpeg-детектора пересчитывается в проценты внутри адаптера:
    #: у фильтра scdet своя шкала 0..100, и общий конфиг иначе означал бы
    #: разную чувствительность у двух «взаимозаменяемых» реализаций.
    scene_threshold: float = 27.0
    scene_min_len_sec: float = 1.5
    #: Верхняя граница длительности сцены. Статичный план на 12 минут —
    #: это не «одна сцена», а кусок, внутри которого искать нечего:
    #: скоринг усреднит признаки по всему отрезку и утопит любой всплеск.
    scene_max_len_sec: float = 90.0
    #: Кадры, с которых считается разница. 0 = отдать решение детектору
    #: (PySceneDetect сам уменьшает большие кадры). Значение > 1 ускоряет
    #: обработку кратно, ценой пропуска мелких деталей.
    scene_downscale: int = 0
    scene_keyframes_enabled: bool = True
    scene_keyframe_width: int = 640
    #: Сколько кадров тянуть из файла параллельно. Каждый кадр — отдельный
    #: запуск ffmpeg с быстрой перемоткой; на тысяче сцен последовательный
    #: цикл занимает минуты, а параллельный укладывается в десятки секунд.
    scene_keyframe_workers: int = 8

    # --- аудио ---
    #: 16 кГц — не «качество похуже», а формат, на котором обучался Whisper.
    #: Подавать 48 кГц бессмысленно: модель всё равно ресемплит вниз.
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    # --- модели и распознавание речи ---
    device: Literal["auto", "cpu", "cuda"] = "auto"
    #: ``medium``, а не ``large-v3``, и это результат замера, а не осторожность.
    #: На разговорной русской речи обе дали одинаковый WER 0.096, но large-v3
    #: на процессоре обрабатывает час речи 30 минут против 17 у medium
    #: и тянет 3 ГБ весов при первой же задаче. Развёртывание по умолчанию —
    #: CPU-only, и дефолт обязан быть пригодным именно там. Есть видеокарта —
    #: ставьте large-v3 явно: на ней разница в скорости несущественна.
    whisper_model: str = "medium"
    #: ``auto`` — float16 на GPU, int8 на CPU. Явное ``float16`` на процессоре
    #: не ошибка конфигурации, но плохая идея: CTranslate2 молча откатится
    #: на float32 и будет считать вчетверо дольше, чем в int8.
    whisper_compute_type: str = "auto"
    whisper_beam_size: int = 5
    whisper_vad: bool = True
    #: Порог вероятности «здесь нет речи», выше которого сегмент выбрасывается.
    #: Музыка и шум дают у Whisper уверенные галлюцинации — «Продолжение следует...»,
    #: «Субтитры сделал DimaTorzok». Без фильтра они попадут в скоринг как речь.
    whisper_no_speech_threshold: float = 0.6
    #: Язык распознавания; пусто — автоопределение по первым секундам.
    whisper_language: str = ""
    #: Длина куска, которым файл скармливается модели, и перекрытие кусков.
    #: 0 — не резать вовсе (модель сама пройдёт файл целиком).
    whisper_chunk_sec: float = 600.0
    whisper_chunk_overlap_sec: float = 2.0
    #: Сколько окон модель обрабатывает за один проход. 0 — решает устройство:
    #: на GPU 16 (замер: час аудио за 0.9 мин против 6.3 мин без батчей),
    #: на CPU 1 — там batching почти не даёт выигрыша, потому что CTranslate2
    #: и так занимает все ядра одним окном, зато память растёт кратно.
    whisper_batch_size: int = 0

    # --- анализаторы (Этап 5+) ---
    analyzers_disabled: str = ""

    #: Классификатор звуковых событий. AudioSet-модель, а не специализированный
    #: детектор смеха: одна модель закрывает все пять нужных событий сразу,
    #: и добавить шестое — это строчка в таблице соответствий, а не новые веса.
    audio_event_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    #: Длина окна классификации. 10.24 с — размер входа, на котором обучалась
    #: модель; отклоняться от него можно, но качество падает.
    audio_event_window_sec: float = 10.0
    #: Сколько окон разбирать на сцену. Событие ищется по всей сцене, но
    #: платить за длинную сцену линейно нельзя: три окна на двухчасовом
    #: фильме — это уже тысячи прогонов модели.
    audio_event_max_windows: int = 3
    #: Порог, ниже которого событие не считается найденным и не пишется
    #: в признаки вовсе. Разреженность здесь полезна: скорингу нужны
    #: события, а не 527 вероятностей около нуля.
    audio_event_threshold: float = 0.1

    #: Распознавание эмоции в голосе. Модель даёт четыре класса
    #: (neutral/happy/angry/sad); arousal и valence выводятся из них
    #: проекцией — см. разбор Этапа 5.
    ser_model: str = "superb/wav2vec2-base-superb-er"
    #: Сколько секунд речи на сцену подавать модели. Вся речь двухчасового
    #: фильма — это часы работы wav2vec2; пяти секунд хватает, чтобы
    #: определить эмоцию, и бюджет остаётся конечным.
    ser_max_speech_sec: float = 5.0

    #: Детектор лиц: ``yunet`` (веса 232 КБ, качаются в HF_HOME) или
    #: ``haar`` (каскад из комплекта OpenCV, ничего не качает, но заметно
    #: хуже на неанфасных лицах).
    face_detector: Literal["yunet", "haar"] = "yunet"
    face_model_repo: str = "opencv/face_detection_yunet"
    face_model_file: str = "face_detection_yunet_2023mar.onnx"
    #: Порог уверенности детектора лиц.
    face_score_threshold: float = 0.7
    #: Сколько кадров сцены разбирать детекторами. Лица и люди меняются
    #: медленнее, чем движение: пяти кадров на сцену достаточно, а полный
    #: проход по всем сэмплам стоил бы вчетверо дороже всей группы.
    vision_detect_frames: int = 5
    #: Детектор людей (шире, чем лица: человек со спины лицом не считается).
    #: Пусто — не считать вовсе.
    person_detector: str = "ssdlite320_mobilenet_v3_large"
    person_score_threshold: float = 0.5
    #: Сколько кадров в секунду разбирают видеоанализаторы. Больше почти
    #: не добавляет информации о сцене, а время растёт линейно: на двухчасовом
    #: фильме разница между 3 и 25 кадрами — это часы.
    vision_sample_fps: float = 3.0
    #: Длинная сторона кадра при анализе. Оптическому потоку и гистограммам
    #: полного разрешения не нужно, а памяти и времени оно стоит вчетверо.
    vision_max_side: int = 320

    # --- скоринг (Этап 7+) ---
    default_profile: str = "default"
    score_alpha: float = 0.4
    score_beta: float = 0.5
    score_gamma: float = 0.1

    # --- нарезка клипов (Этап 8) ---
    #: x264, а не x265/AV1: клип уходит на площадку, которая всё равно
    #: перекодирует его под себя, и единственное, что нам нужно от кодека, —
    #: чтобы файл открылся везде без разговоров.
    render_video_codec: str = "libx264"
    render_audio_codec: str = "aac"
    #: 20 — визуально без потерь на материале, который уже пережил одно
    #: сжатие. Меньше смысла нет: улучшать нечего, а вес растёт.
    render_crf: int = 20
    render_preset: str = "medium"
    render_audio_bitrate: str = "192k"
    #: Нормализовать громкость к −14 LUFS (уровень YouTube и Spotify).
    #: Выключать осмысленно, если исходник уже сведён.
    render_normalize_audio: bool = True
    render_loudness_lufs: float = -14.0
    #: Собирать ли вертикальную версию 1080×1920.
    render_vertical: bool = True
    render_thumbnails: bool = True
    render_thumbnail_width: int = 640
    #: Сколько клипов рендерится одновременно. Не «по числу ядер»: ffmpeg
    #: сам занимает все ядра одним файлом, и десять параллельных процессов
    #: не ускорят работу, а поделят те же ядра и умножат память.
    render_concurrency: int = 2
    #: Потоки одного ffmpeg. 0 — пусть решает сам.
    render_threads: int = 0
    #: Резать копированием потока, когда это возможно. Выключено: замер
    #: показал отклонение длительности +0.123 с против 0.000 с при
    #: перекодировании — копия режет по границам пакетов, а не кадров,
    #: и в требуемые ±0.05 с не укладывается (см. разбор Этапа 8).
    render_allow_stream_copy: bool = False

    #: Как часто пересчитывается положение вертикального кропа, кадров/с.
    #: Лица не телепортируются: четырёх измерений в секунду хватает,
    #: а разбор каждого кадра стоил бы дороже самого кодирования.
    reframe_sample_fps: float = 4.0
    #: Окно сглаживания траектории, секунды.
    reframe_smooth_sec: float = 1.2
    #: Предел скорости кропа в долях ширины исходного кадра за секунду.
    #: Без него камера дёргается за каждым ложным срабатыванием детектора.
    reframe_max_speed: float = 0.25

    # --- LLM (Этап 6) ---
    #: Какой провайдер отвечает за ранжирование. Порт ``LLMProvider`` один,
    #: реализаций может быть несколько: ``openrouter`` даёт бесплатные модели
    #: и годится для разработки, ``claude`` — качество и гарантию схемы.
    llm_provider: Literal["openrouter", "claude"] = "openrouter"
    llm_enabled: bool = True

    #: Общие для всех провайдеров параметры запроса.
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    #: Потолок на ответ. У моделей с «думанием» он ограничивает сумму
    #: размышления и ответа, поэтому запас нужен заметный: батч из 40 сцен
    #: это до 2400 токенов одних вердиктов.
    llm_max_tokens: int = 16000
    #: Сколько сцен уходит в один запрос. Больше — дешевле по системному
    #: промпту, но выше риск, что модель собьётся с нумерации.
    llm_batch_size: int = 40
    llm_timeout_sec: float = 180.0
    llm_max_retries: int = 3

    # --- OpenRouter ---
    openrouter_api_key: str | None = None
    #: Замер на задаче ранжирования (Этап 6): точные категории, лаконичные
    #: обоснования, 16.6 с на батч. Строгие схемы поддерживает — но, в отличие
    #: от Claude, соблюдает их не API, а сама модель.
    openrouter_model: str = "google/gemma-4-26b-a4b-it:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Claude ---
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5"
    llm_use_batch_api: bool = True

    # --- лимиты ---
    max_concurrent_jobs: int = 2
    rate_limit_rpm: int = 60

    # ------------------------------------------------------------ валидация
    @field_validator("secret_key")
    @classmethod
    def _reject_default_secret_in_prod(cls, value: str, info: object) -> str:
        # Проверку окружения делаем в model_post_init: здесь ещё нет app_env.
        return value

    def model_post_init(self, __context: object) -> None:
        if self.app_env is AppEnv.PROD:
            if self.secret_key == "dev-insecure-change-me":
                raise ValueError("SECRET_KEY обязателен в проде")
            if self.llm_enabled and not self.llm_api_key:
                raise ValueError(
                    f"Ключ провайдера {self.llm_provider!r} обязателен при LLM_ENABLED=true в проде"
                )

        if self.scene_min_len_sec <= 0:
            raise ValueError(
                f"SCENE_MIN_LEN_SEC должен быть положительным: {self.scene_min_len_sec}"
            )
        if self.scene_max_len_sec <= self.scene_min_len_sec:
            raise ValueError(
                "SCENE_MAX_LEN_SEC должен быть больше SCENE_MIN_LEN_SEC, получено "
                f"{self.scene_max_len_sec} <= {self.scene_min_len_sec}"
            )

        if self.whisper_chunk_sec and self.whisper_chunk_overlap_sec >= self.whisper_chunk_sec:
            raise ValueError(
                "WHISPER_CHUNK_OVERLAP_SEC должен быть меньше WHISPER_CHUNK_SEC, получено "
                f"{self.whisper_chunk_overlap_sec} >= {self.whisper_chunk_sec}"
            )

        if not 0 <= self.render_crf <= 51:
            raise ValueError(f"RENDER_CRF вне диапазона x264 (0..51): {self.render_crf}")
        if self.render_concurrency < 1:
            raise ValueError(
                f"RENDER_CONCURRENCY должен быть не меньше 1: {self.render_concurrency}"
            )
        if self.reframe_sample_fps <= 0:
            raise ValueError(
                f"REFRAME_SAMPLE_FPS должен быть положительным: {self.reframe_sample_fps}"
            )

        total = self.score_alpha + self.score_beta + self.score_gamma
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"SCORE_ALPHA + SCORE_BETA + SCORE_GAMMA должны давать 1.0, получено {total}"
            )

    # ---------------------------------------------------------- производные
    @property
    def database_url(self) -> str:
        """DSN для asyncpg-драйвера SQLAlchemy."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_db,
            )
        )

    @property
    def sync_database_url(self) -> str:
        """DSN для psycopg — нужен Alembic, который работает синхронно."""
        return self.database_url.replace("postgresql+asyncpg", "postgresql+psycopg")

    @property
    def allowed_containers_set(self) -> frozenset[str]:
        return frozenset(x.strip().lower() for x in self.allowed_containers.split(",") if x.strip())

    @property
    def disabled_analyzers_set(self) -> frozenset[str]:
        return frozenset(x.strip() for x in self.analyzers_disabled.split(",") if x.strip())

    @property
    def llm_api_key(self) -> str | None:
        """Ключ выбранного провайдера.

        Отдельное свойство, чтобы вызывающему коду не приходилось помнить,
        какая переменная окружения относится к какому провайдеру.
        """
        return (
            self.openrouter_api_key if self.llm_provider == "openrouter" else self.anthropic_api_key
        )

    @property
    def local_source_roots(self) -> tuple[Path, ...]:
        return tuple(
            Path(item.strip()) for item in self.local_source_roots_raw.split(",") if item.strip()
        )

    @property
    def is_production(self) -> bool:
        return self.app_env is AppEnv.PROD


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Синглтон настроек.

    ``lru_cache`` вместо глобальной переменной: тест может вызвать
    ``get_settings.cache_clear()`` и получить свежий объект с подменённым
    окружением, а в проде разбор ``.env`` происходит ровно один раз.
    """
    return Settings()
