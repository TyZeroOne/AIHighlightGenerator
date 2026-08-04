"""Громкость по EBU R128.

Почему именно этот стандарт, а не «средняя амплитуда»: R128 меряет
**воспринимаемую** громкость — с учётом того, что ухо слышит середину
спектра лучше краёв, и с интегрированием по времени. Два фрагмента
с одинаковым пиком амплитуды могут звучать совершенно по-разному, и для
поиска «эмоциональных всплесков» важно именно ощущение, а не физика.

Один проход ffmpeg на весь файл, а не по вызову на сцену. На тысяче сцен
второй вариант означал бы тысячу запусков процесса — минуты накладных
расходов ради арифметики, которая делается за миллисекунды.

Разбор и агрегация — чистые функции. Проверять их на записанном выводе
ffmpeg дешевле и надёжнее, чем гонять процесс в каждом тесте.
"""

from __future__ import annotations

import asyncio
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Final

from core.domain.analysis import ARTIFACT_AUDIO, AnalysisContext
from core.domain.errors import FFmpegError, TransientError
from core.domain.values import FeatureValue, TimeRange
from core.infra.proc import death_signal
from core.logging import get_logger
from core.registry import register_analyzer

__all__ = ["LoudnessAnalyzer", "LoudnessSample", "aggregate", "parse_ebur128"]

log = get_logger(__name__)

#: Вывод `ametadata=print`: заголовок кадра, затем метрики этого кадра.
#:
#:     frame:12   pts:19200   pts_time:1.2
#:     lavfi.r128.M=-22.500
#:     lavfi.r128.S=-23.100
#:     lavfi.r128.true_peak=0.125
#:
#: Разбор идёт по метаданным, а не по журналу фильтра: журнальный формат
#: (`framelog=verbose`) в ffmpeg 7.1 на stderr не печатается вовсе — тот же
#: приём с метаданными уже пригодился на Этапе 3 для `scdet`.
_FRAME: Final = re.compile(r"^frame:\s*\d+\s+pts:\s*\S+\s+pts_time:\s*(?P<t>[-\d.]+)")
_METRIC: Final = re.compile(r"^lavfi\.r128\.(?P<key>[\w.]+)=(?P<value>\S+)")

#: Порог тишины у ffmpeg: momentary loudness ниже этого значения означает,
#: что мерить нечего. Считать такие отсчёты наравне с остальными нельзя —
#: одна секунда полной тишины утянет среднее по сцене на десятки LUFS.
_SILENCE_FLOOR: Final = -70.0

#: Momentary loudness — среднее по 400 мс, и метка времени указывает
#: не на границу окна, а куда-то внутрь него. Замер на ffmpeg 7.1: файл,
#: где тон возвращается ровно на 10.000 с, даёт на отсчёте с меткой 10.000
#: значение на 6 дБ ниже полного тона — то есть окно уже захватило 100 мс
#: нового звука. Симметрично на входе в тишину: метка 8.100 показывает
#: половину энергии тона, 8.200 — четверть. Отсюда фактическое окно
#: примерно ``[t − 0.3, t + 0.1]``.
#:
#: Точную формулу закладывать не будем: она деталь реализации фильтра
#: и может измениться с версией. Берём заведомо покрывающие границы
#: и учитываем отсчёт в сцене, только если **всё его окно** лежит внутри.
#:
#: Зачем так строго: иначе громкая сцена протекает в соседнюю. Тишина сразу
#: после взрыва измеряется как взрыв, а сцена целиком в тишине получает
#: осмысленную громкость из хвоста соседа — ровно это и показал тест
#: на синтетике с заранее известной раскладкой.
#:
#: Плата — полсекунды по краям каждой сцены не участвуют в расчёте.
#: При минимальной длине сцены 1.5 с (``SCENE_MIN_LEN_SEC``) остаётся
#: достаточно отсчётов, а на совсем коротких сценах признака не будет —
#: и это честнее числа, посчитанного по соседям.
_WINDOW_BEFORE: Final = 0.4
_WINDOW_AFTER: Final = 0.1


@dataclass(frozen=True, slots=True)
class LoudnessSample:
    """Один отсчёт временного ряда громкости (ffmpeg выдаёт их каждые 100 мс)."""

    at_sec: float
    momentary: float
    short_term: float
    true_peak: float | None = None


def parse_ebur128(text: str) -> list[LoudnessSample]:
    """Вытащить временной ряд из метаданных фильтра ``ebur128``.

    Разбор терпимый: строки, которые не разобрались, пропускаются молча.
    Формат вывода ffmpeg менялся между версиями, и падать из-за лишнего
    пробела в диагностике — не то поведение, которого хочется от этапа,
    считающего вспомогательный признак.
    """
    samples: list[LoudnessSample] = []
    at: float | None = None
    metrics: dict[str, float] = {}

    def flush() -> None:
        if at is None or "M" not in metrics:
            return
        samples.append(
            LoudnessSample(
                at_sec=at,
                momentary=metrics["M"],
                short_term=metrics.get("S", metrics["M"]),
                true_peak=_peak_to_db(metrics.get("true_peak")),
            )
        )

    for line in text.splitlines():
        frame = _FRAME.match(line.strip())
        if frame is not None:
            flush()
            try:
                at = float(frame["t"])
            except ValueError:
                at = None
            metrics = {}
            continue

        metric = _METRIC.match(line.strip())
        if metric is not None and at is not None:
            try:
                metrics[metric["key"]] = _to_float(metric["value"])
            except ValueError:
                continue

    flush()
    return samples


def aggregate(
    samples: Sequence[LoudnessSample],
    scenes: Sequence[TimeRange],
) -> dict[int, list[FeatureValue]]:
    """Временной ряд → признаки по сценам.

    ``delta`` считается относительно медианы по всему видео: «громко» —
    понятие относительное, и в фильме ужасов оно другое, чем в лекции
    (док 05, §5.3). Абсолютные LUFS остаются в ``mean`` для тех, кому
    нужна физика.
    """
    if not samples or not scenes:
        return {}

    by_scene = _split(samples, scenes)
    means: dict[int, float] = {}
    for index, group in by_scene.items():
        mean = _energy_mean([sample.momentary for sample in group]) if group else None
        if mean is not None:
            means[index] = mean
    # Медиана по сценам, а не по отсчётам: иначе одна длинная громкая сцена
    # перетянет базовый уровень на себя, и «тише обычного» станет неверным
    # для всего остального видео.
    baseline = statistics.median(means.values()) if means else None

    features: dict[int, list[FeatureValue]] = {}
    for index, group in by_scene.items():
        if not group:
            continue
        values: list[FeatureValue] = []
        mean = means.get(index)
        if mean is not None:
            values.append(FeatureValue("audio.loudness.mean", mean))
            if baseline is not None:
                values.append(FeatureValue("audio.loudness.delta", mean - baseline))

        peaks = [sample.true_peak for sample in group if sample.true_peak is not None]
        if peaks:
            values.append(FeatureValue("audio.loudness.peak", max(peaks)))

        spread = _loudness_range([sample.short_term for sample in group])
        if spread is not None:
            values.append(FeatureValue("audio.loudness.dynamic_range", spread))

        if values:
            features[index] = values
    return features


@register_analyzer
class LoudnessAnalyzer:
    """Реализация порта ``FeatureAnalyzer`` для громкости."""

    key: ClassVar[str] = "audio.loudness"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_AUDIO})
    produces: ClassVar[frozenset[str]] = frozenset(
        {
            "audio.loudness.mean",
            "audio.loudness.peak",
            "audio.loudness.delta",
            "audio.loudness.dynamic_range",
        }
    )
    #: Дешевле распознавания на порядки: один проход ffmpeg без декодирования видео.
    cost: ClassVar[int] = 1

    def __init__(self, *, binary: str = "ffmpeg", timeout_sec: float = 1800.0) -> None:
        self._binary = binary
        self._timeout = timeout_sec

    async def analyze(
        self,
        context: AnalysisContext,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        if context.audio is None:
            return {}

        raw = await self._measure(str(context.audio.path))
        samples = parse_ebur128(raw)
        log.debug("loudness.measured", samples=len(samples), scenes=len(scenes))
        return aggregate(samples, scenes)

    async def _measure(self, path: str) -> str:
        args = [
            self._binary,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-i",
            path,
            # Метрики забираем метаданными, а не из журнала фильтра:
            # журнальный формат менялся между версиями ffmpeg и в 7.1
            # не печатается вовсе, а метаданные стабильны и разбираются
            # однозначно.
            "-af",
            "ebur128=peak=true:metadata=1,ametadata=print:file=-",
            "-f",
            "null",
            "-",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Дочерний ffmpeg не должен пережить воркера: при отмене джоба
                # осиротевший процесс доедает ядра, и никто о нём уже не знает.
                preexec_fn=death_signal(),
            )
        except FileNotFoundError as error:
            raise TransientError(
                f"ffmpeg не найден: {self._binary}. Проверьте FFMPEG_BIN и образ.",
                binary=self._binary,
            ) from error

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self._timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise FFmpegError(f"ebur128 не уложился в {self._timeout} с") from None

        if process.returncode != 0:
            raise FFmpegError(f"ebur128 завершился с кодом {process.returncode}")

        return stdout.decode("utf-8", errors="replace")


def _split(
    samples: Sequence[LoudnessSample],
    scenes: Sequence[TimeRange],
) -> dict[int, list[LoudnessSample]]:
    """Разложить отсчёты по сценам одним проходом.

    Сцены упорядочены и не пересекаются (контракт ``SceneDetector``),
    поэтому достаточно двигать указатель, а не искать сцену для каждого
    отсчёта бинарным поиском: отсчётов десятки тысяч, сцен — тысячи.
    """
    buckets: dict[int, list[LoudnessSample]] = {index: [] for index in range(len(scenes))}
    index = 0
    for sample in samples:
        if sample.momentary <= _SILENCE_FLOOR:
            continue
        # У самого начала файла окно обрезаем нулём: там оно короче.
        window_start = max(sample.at_sec - _WINDOW_BEFORE, 0.0)
        window_end = sample.at_sec + _WINDOW_AFTER

        while index < len(scenes) and window_end > scenes[index].end:
            index += 1
        if index >= len(scenes):
            break
        if window_start >= scenes[index].start:
            buckets[index].append(sample)
    return buckets


def _energy_mean(values: Sequence[float]) -> float | None:
    """Среднее в энергетической области, а не по децибелам.

    Громкость логарифмическая: арифметическое среднее −40 и −10 LUFS даёт
    −25, хотя на слух отрезок почти целиком определяется громким куском.
    Правильно усреднять энергию и логарифмировать обратно.
    """
    usable = [value for value in values if value > _SILENCE_FLOOR and math.isfinite(value)]
    if not usable:
        return None
    energy = sum(10 ** (value / 10) for value in usable) / len(usable)
    return round(10 * math.log10(energy), 3) if energy > 0 else None


def _loudness_range(values: Sequence[float]) -> float | None:
    """LRA — разброс громкости внутри сцены.

    Определение EBU: разница между 95-м и 10-м процентилями кратковременной
    (short-term) громкости после гейтинга. Считаем по той же схеме;
    на коротких сценах (меньше десяти отсчётов) процентили бессмысленны,
    и признак не выдаётся.
    """
    usable = sorted(value for value in values if value > _SILENCE_FLOOR and math.isfinite(value))
    if len(usable) < 10:
        return None
    low = usable[int(len(usable) * 0.10)]
    high = usable[min(int(len(usable) * 0.95), len(usable) - 1)]
    return round(high - low, 3)


def _peak_to_db(linear: float | None) -> float | None:
    """True peak из линейной амплитуды в dBTP.

    ffmpeg отдаёт пик в метаданных как отсчёт (0.125), а не в децибелах,
    и класть это число рядом с LUFS нельзя: скоринг сравнивал бы величины
    в разных шкалах и «тихая» сцена оказалась бы громче взрыва.
    """
    if linear is None or not math.isfinite(linear) or linear <= 0:
        return None
    return round(20 * math.log10(linear), 3)


def _to_float(raw: str) -> float:
    """``-inf`` встречается в выводе штатно — на тишине."""
    text = raw.strip().lower()
    if text in {"-inf", "inf", "nan"}:
        return -math.inf if text == "-inf" else math.inf
    return float(text)
