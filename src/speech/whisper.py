"""Распознавание речи через faster-whisper.

Почему faster-whisper, а не оригинальный `openai-whisper`: под капотом
CTranslate2 вместо PyTorch. То же качество (веса те же), вчетверо меньше
памяти и вдвое быстрее — а главное, на CPU он работает в int8 и остаётся
пригодным к использованию, тогда как оригинал на процессоре считает часами.

Модель грузится **лениво и один раз на процесс**. Large-v3 — это три гигабайта
на диске и полторы минуты на инициализацию; загружать её на каждый вызов
означало бы тратить на инициализацию больше, чем на работу.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from core.domain.errors import ModelLoadError, ProcessingError
from core.domain.media import AudioTrack, Transcript, TranscriptSegment, TranscriptWord
from core.domain.values import TimeRange
from core.logging import get_logger
from speech.postprocess import (
    Chunk,
    chunk_windows,
    drop_non_speech,
    merge_chunks,
    normalize_words,
    shift,
)

__all__ = [
    "FasterWhisperTranscriber",
    "resolve_batch_size",
    "resolve_compute_type",
    "resolve_device",
]

log = get_logger(__name__)

#: Обратный вызов прогресса: доля выполненного, 0..1.
type ProgressCallback = Callable[[float], None]


def resolve_device(requested: str) -> str:
    """``auto`` → ``cuda``, если она есть, иначе ``cpu``.

    Проверяем через CTranslate2, а не через torch: torch в образе воркера нет
    и до Этапа 5 не будет, а тащить 2 ГБ зависимостей ради одной проверки —
    плохая сделка.
    """
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception as error:  # pragma: no cover — зависит от сборки CTranslate2
        log.warning("whisper.cuda_probe_failed", error=str(error))
        return "cpu"


def resolve_batch_size(requested: int, device: str) -> int:
    """Размер батча под устройство.

    На GPU батчи — главный рычаг скорости: видеокарта простаивает, пока одно
    тридцатисекундное окно идёт через модель, и шестнадцать окон разом
    обрабатываются почти за то же время. Замер на RTX 5070 и large-v3:
    час аудио за 0.9 минуты против 6.3 без батчей.

    На CPU выигрыша почти нет — CTranslate2 и так распараллеливает одно окно
    по всем ядрам, — зато память растёт кратно размеру батча. Поэтому там 1.
    """
    if requested > 0:
        return requested
    return 16 if device == "cuda" else 1


def resolve_compute_type(requested: str, device: str) -> str:
    """Тип вычислений под устройство.

    ``float16`` на процессоре либо не поддерживается вовсе, либо эмулируется
    и работает вчетверо медленнее ``int8``. Обратное тоже верно: int8 на GPU
    отдаёт часть качества без выигрыша в скорости. Поэтому ``auto`` — разумный
    дефолт, а явное значение остаётся возможным.
    """
    if requested and requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


class FasterWhisperTranscriber:
    """Реализация порта ``Transcriber``."""

    def __init__(
        self,
        *,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        use_vad: bool = True,
        language: str | None = None,
        no_speech_threshold: float = 0.6,
        chunk_sec: float = 600.0,
        chunk_overlap_sec: float = 2.0,
        batch_size: int = 0,
        download_root: Path | None = None,
        cpu_threads: int = 0,
    ) -> None:
        self._model_size = model_size
        self._device = resolve_device(device)
        self._compute_type = resolve_compute_type(compute_type, self._device)
        self._beam_size = beam_size
        self._use_vad = use_vad
        self._language = language or None
        self._no_speech_threshold = no_speech_threshold
        self._chunk_sec = chunk_sec
        self._chunk_overlap_sec = chunk_overlap_sec
        self._batch_size = resolve_batch_size(batch_size, self._device)
        self._download_root = download_root
        self._cpu_threads = cpu_threads

        self._model: Any | None = None
        self._pipeline: Any | None = None
        # Загрузку модели защищаем замком: воркер может выполнять задачу
        # в нескольких потоках, и две одновременные загрузки large-v3 —
        # это шесть гигабайт памяти вместо трёх.
        self._lock = threading.Lock()

    # ------------------------------------------------------------- модель
    def _load(self) -> Any:
        # Двойная проверка: снаружи замка — чтобы не платить за него на каждом
        # вызове, внутри — чтобы две задачи не начали грузить модель разом.
        # Читаем в локальную переменную, а не через self: атрибут может
        # измениться между проверкой и возвратом.
        model = self._model
        if model is not None:
            return model

        with self._lock:
            model = self._model
            if model is not None:
                return model
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:  # pragma: no cover — зависит от окружения
                raise ModelLoadError(
                    "faster-whisper не установлен. Поставьте extra 'worker'.",
                ) from error

            log.info(
                "whisper.loading",
                model=self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                batch_size=self._batch_size,
            )
            try:
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                    download_root=str(self._download_root) if self._download_root else None,
                    cpu_threads=self._cpu_threads,
                )
                if self._batch_size > 1:
                    from faster_whisper import BatchedInferencePipeline

                    self._pipeline = BatchedInferencePipeline(model=self._model)
            except Exception as error:
                raise ModelLoadError(
                    f"Не удалось загрузить модель {self._model_size} "
                    f"({self._device}/{self._compute_type}): {type(error).__name__}: {error}",
                    model=self._model_size,
                    device=self._device,
                ) from error
            return self._model

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # ---------------------------------------------------------- транскрипция
    async def transcribe(
        self,
        audio: AudioTrack,
        *,
        language: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> Transcript:
        """Распознать дорожку целиком.

        Работа синхронная и долгая, поэтому уходит в отдельный поток: держать
        event loop заблокированным на минуты нельзя — в нём же живут отмена
        задачи и heartbeat воркера.
        """
        windows = chunk_windows(
            audio.duration_sec,
            chunk_sec=self._chunk_sec,
            overlap_sec=self._chunk_overlap_sec,
        )
        if not windows:
            return Transcript()

        return await asyncio.to_thread(
            self._transcribe_sync, audio, windows, language or self._language, progress
        )

    def _transcribe_sync(
        self,
        audio: AudioTrack,
        windows: Sequence[TimeRange],
        language: str | None,
        progress: ProgressCallback | None,
    ) -> Transcript:
        model = self._load()
        samples = self._decode(audio)

        chunks: list[Chunk] = []
        detected = language
        for index, window in enumerate(windows):
            piece = self._slice(samples, audio.sample_rate, window)
            transcript, detected = self._run_model(model, piece, detected, window)
            chunks.append(Chunk(range=window, transcript=transcript))
            if progress is not None:
                progress((index + 1) / len(windows))

        merged = merge_chunks(chunks)
        cleaned = drop_non_speech(merged, no_speech_threshold=self._no_speech_threshold)
        log.info(
            "whisper.transcribed",
            language=cleaned.language,
            chunks=len(chunks),
            segments_raw=len(merged.segments),
            segments=len(cleaned.segments),
            audio_sec=round(audio.duration_sec, 1),
        )
        return cleaned

    def _decode(self, audio: AudioTrack) -> Any:
        """Прочитать дорожку в массив отсчётов.

        Читаем один раз и режем в памяти, а не запускаем декодирование на
        каждый кусок: час аудио — это 230 МБ float32, зато никакого
        повторного разбора файла двенадцать раз подряд.
        """
        try:
            from faster_whisper.audio import decode_audio
        except ImportError as error:  # pragma: no cover
            raise ModelLoadError("faster-whisper не установлен") from error

        try:
            return decode_audio(str(audio.path), sampling_rate=audio.sample_rate)
        except Exception as error:
            raise ProcessingError(
                f"Не удалось прочитать аудио для распознавания: {type(error).__name__}",
                path=audio.path.name,
            ) from error

    @staticmethod
    def _slice(samples: Any, sample_rate: int, window: TimeRange) -> Any:
        start = int(window.start * sample_rate)
        end = int(window.end * sample_rate)
        return samples[start:end]

    def _run_model(
        self,
        model: Any,
        samples: Any,
        language: str | None,
        window: TimeRange,
    ) -> tuple[Transcript, str | None]:
        options: dict[str, Any] = {
            "language": language,
            "beam_size": self._beam_size,
            # Без пословных таймингов не будет ни караоке, ни снапа границ.
            "word_timestamps": True,
            "vad_filter": self._use_vad,
            "no_speech_threshold": self._no_speech_threshold,
        }
        if self._pipeline is None:
            # Ключевая настройка против галлюцинаций: с ней модель тянет
            # контекст предыдущего сегмента в следующий, и одна ошибка
            # на музыкальной паузе размножается до конца файла. У батчевого
            # конвейера такого параметра нет — он и так не переносит контекст
            # между окнами, потому что считает их одновременно.
            options["condition_on_previous_text"] = False

        try:
            if self._pipeline is not None:
                segments, info = self._pipeline.transcribe(
                    samples, batch_size=self._batch_size, **options
                )
            else:
                segments, info = model.transcribe(samples, **options)
            collected = [self._to_segment(segment, info) for segment in segments]
        except Exception as error:
            raise ProcessingError(
                f"Распознавание не удалось на окне {window}: {type(error).__name__}: {error}",
                window=str(window),
            ) from error

        detected = getattr(info, "language", None) or language
        transcript = Transcript(segments=tuple(collected), language=detected)
        # Сдвигаем в абсолютное время видео: модель отсчитывает от начала куска.
        return shift(transcript, window.start), detected

    def _to_segment(self, segment: Any, info: Any) -> TranscriptSegment:
        span = TimeRange(float(segment.start), max(float(segment.end), float(segment.start) + 1e-3))
        words = tuple(
            TranscriptWord(
                text=word.word,
                start=float(word.start),
                end=float(word.end),
                probability=float(getattr(word, "probability", 1.0)),
            )
            for word in (segment.words or [])
        )
        return TranscriptSegment(
            text=segment.text.strip(),
            range=span,
            words=normalize_words(words, bounds=span),
            language=getattr(info, "language", None),
            avg_logprob=_float_or_none(getattr(segment, "avg_logprob", None)),
            no_speech_prob=_float_or_none(getattr(segment, "no_speech_prob", None)),
        )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
