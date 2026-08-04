"""Сцена → компактный текстовый блок для модели.

Это самый недооценённый файл этапа. От него зависят и качество ответа,
и его цена — линейно: каждый лишний десяток токенов на сцену превращается
в лишние доллары на фильме из полутора тысяч сцен.

## Что уходит в модель и почему именно это

Не сырые признаки (их два десятка на сцену, и модель не знает, что такое
`vision.camera.shake = 0.15`) и не транскрипт целиком. Уходит короткая
человекочитаемая сводка:

```
### Сцена 47 | 00:15:21 – 00:16:05 (44.0 с)
Аудио: громкость +8.2 дБ к медиане, пауза до 1.4 с, музыка 0.62
Речь: 178 слов/мин (+34% к обычному), радость 0.81
Видео: склеек 12 (0.27/с), движение 0.84, лиц до 3, крупный план
Реплика: «Держись! Они прямо за нами — сворачивай направо, быстро!»
```

Три правила, которые за этим стоят.

**Числа переводятся в слова там, где слово короче.** «крупный план»
вместо `vision.faces.closeup = 1.0`; «пауза до 1.4 с» вместо трёх ключей
про паузы. Модель читает описание, а не таблицу.

**Признак, которого нет, не упоминается вовсе.** Строчка «лиц: нет данных»
стоит токенов и не сообщает ничего: анализатор мог быть отключён,
упасть или просто не найти лиц — модели эта разница не нужна.

**Речь важнее всех признаков вместе взятых.** Она идёт последней (ближе
к концу контекста — там внимание модели выше) и получает больше всего
места. Признаки описывают форму, реплика — содержание, а решение
про «интересно ли» принимается по содержанию.

## Фильтр перед отправкой

Сцены без речи и без единого заметного сигнала в модель не отправляются:
они заведомо не хайлайты, а платим мы за каждую. На типичном материале
это снимает от трети до половины батча (док 06, §6.4).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from core.domain.analysis import SceneSummary, VideoContext
from core.domain.entities import Scene
from core.domain.timecode import format_timecode

__all__ = [
    "MAX_SPEECH_CHARS",
    "format_batch",
    "format_video_context",
    "is_worth_ranking",
    "summarize",
]

#: Сколько символов реплики отдавать модели. 600 — это примерно 150 токенов
#: и полминуты живой речи: достаточно, чтобы понять, о чём сцена, и вдвое
#: меньше, чем занимал бы полный текст длинной сцены.
MAX_SPEECH_CHARS: Final = 600

#: Порог, ниже которого сцена не отправляется в модель. Сравнивается
#: с максимумом по «интересным» признакам, приведённым к 0..1.
DEFAULT_MIN_SIGNAL: Final = 0.15

#: Признаки, по которым решается «есть ли тут вообще что-то». Каждый уже
#: лежит в 0..1 или приводится к нему функцией ниже.
_SIGNAL_KEYS: Final = (
    "audio.event.laughter",
    "audio.event.scream",
    "audio.event.applause",
    "audio.event.impact",
    "vision.motion.mean",
    "vision.cuts.rate",
    "vision.faces.closeup",
    "speech.emotion.arousal",
)


def _clip_speech(text: str, limit: int = MAX_SPEECH_CHARS) -> str:
    """Длинная реплика обрезается серединой, а не концом.

    Конец фразы часто и есть соль момента — панчлайн, признание, приказ.
    Отрезав хвост, мы выкинули бы ровно то, ради чего сцена может стать
    клипом. Поэтому берётся начало и конец, а вырезается середина.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text

    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head].rstrip()} … {text[-tail:].lstrip()}"


def _describe_audio(scene: Scene) -> str | None:
    parts: list[str] = []

    delta = scene.features.get("audio.loudness.delta")
    if delta is not None and abs(delta.value) >= 1.0:
        parts.append(f"громкость {delta.value:+.1f} дБ к медиане")

    longest_pause = scene.features.get("audio.pause.max")
    if longest_pause is not None and longest_pause.value >= 1.0:
        parts.append(f"пауза до {longest_pause.value:.1f} с")

    for event in ("laughter", "applause", "scream", "impact", "music"):
        value = scene.features.get(f"audio.event.{event}")
        if value is not None and value.value >= 0.3:
            parts.append(f"{_EVENT_NAMES[event]} {value.value:.2f}")

    return ", ".join(parts) or None


_EVENT_NAMES: Final[Mapping[str, str]] = {
    "laughter": "смех",
    "applause": "аплодисменты",
    "scream": "крик",
    "impact": "удар",
    "music": "музыка",
}

_EMOTION_NAMES: Final[Mapping[str, str]] = {
    "hap": "радость",
    "ang": "злость",
    "sad": "грусть",
}


def _describe_speech(scene: Scene) -> str | None:
    parts: list[str] = []

    wpm = scene.features.get("speech.rate.wpm")
    delta = scene.features.get("speech.rate.delta")
    if wpm is not None:
        pace = f"{wpm.value:.0f} слов/мин"
        # Отклонение от медианы этого же видео говорит больше, чем абсолютный
        # темп: 180 wpm — это стремительно для лекции и норма для стендапа.
        if delta is not None and abs(delta.value) >= 15.0:
            share = delta.value / max(wpm.value - delta.value, 1.0) * 100
            pace += f" ({share:+.0f}% к обычному)"
        parts.append(pace)

    turns = scene.features.get("speech.dialogue_turns")
    if turns is not None and turns.value >= 1:
        parts.append(f"смен реплик {turns.value:.0f}")

    strongest = max(
        (
            (label, scene.features[f"speech.emotion.{label}"].value)
            for label in _EMOTION_NAMES
            if f"speech.emotion.{label}" in scene.features
        ),
        key=lambda item: item[1],
        default=None,
    )
    if strongest is not None and strongest[1] >= 0.5:
        parts.append(f"{_EMOTION_NAMES[strongest[0]]} {strongest[1]:.2f}")

    sentiment = scene.features.get("speech.text.sentiment")
    if sentiment is not None and abs(sentiment.value) >= 0.5:
        parts.append("тон текста " + ("позитивный" if sentiment.value > 0 else "негативный"))

    return ", ".join(parts) or None


def _describe_vision(scene: Scene) -> str | None:
    parts: list[str] = []

    cuts = scene.features.get("vision.cuts.count")
    rate = scene.features.get("vision.cuts.rate")
    if cuts is not None and cuts.value >= 1:
        piece = f"склеек {cuts.value:.0f}"
        if rate is not None:
            piece += f" ({rate.value:.2f}/с)"
        parts.append(piece)

    motion = scene.features.get("vision.motion.mean")
    if motion is not None and motion.value >= 0.01:
        parts.append(f"движение {motion.value:.2f}")

    shake = scene.features.get("vision.camera.shake")
    if shake is not None and shake.value >= 0.15:
        parts.append("тряска камеры")

    faces = scene.features.get("vision.faces.count_max")
    if faces is not None and faces.value >= 1:
        parts.append(f"лиц до {faces.value:.0f}")

    closeup = scene.features.get("vision.faces.closeup")
    if closeup is not None and closeup.value >= 1.0:
        parts.append("крупный план")

    people = scene.features.get("vision.people.count")
    if people is not None and people.value >= 2:
        parts.append(f"людей {people.value:.0f}")

    return ", ".join(parts) or None


def signal_strength(scene: Scene) -> float:
    """Насколько сцена вообще выделяется — 0..1.

    Максимум, а не сумма: одного яркого признака достаточно, чтобы сцену
    стоило показать модели, а суммирование размывало бы редкое событие
    среди десятка нулей.
    """
    values: list[float] = []
    for key in _SIGNAL_KEYS:
        value = scene.features.get(key)
        if value is None:
            continue
        # cuts.rate — единственный признак не в 0..1: две склейки в секунду
        # это уже очень быстрый монтаж, по нему и нормируем.
        values.append(min(value.value / 2.0, 1.0) if key == "vision.cuts.rate" else value.value)
    return max(values, default=0.0)


def is_worth_ranking(scene: Scene, *, min_signal: float = DEFAULT_MIN_SIGNAL) -> bool:
    """Стоит ли платить за оценку этой сцены.

    Речь перевешивает всё: сцена с репликой отправляется даже при нулевых
    признаках — именно в словах чаще всего и находится момент. Молчаливая
    сцена должна хотя бы чем-то выделяться.
    """
    if scene.transcript_text and scene.transcript_text.strip():
        return True
    return signal_strength(scene) >= min_signal


def summarize(scene: Scene) -> SceneSummary:
    """Сцена → сводка для модели."""
    notes = [
        f"{title}: {body}"
        for title, body in (
            ("Аудио", _describe_audio(scene)),
            ("Речь", _describe_speech(scene)),
            ("Видео", _describe_vision(scene)),
        )
        if body
    ]

    return SceneSummary(
        index=scene.index,
        range=scene.range,
        signals={"signal": round(signal_strength(scene), 3)},
        transcript_excerpt=_clip_speech(scene.transcript_text or ""),
        notes=tuple(notes),
    )


def render(summary: SceneSummary) -> str:
    """Сводка → текстовый блок.

    Формат заголовка неслучаен: `scene_index`, а следом тайм-коды ровно
    в том виде, в каком модель обязана вернуть их в ответе. Чем меньше
    ей нужно преобразовывать, тем меньше поводов ошибиться.
    """
    header = (
        f"### Сцена {summary.index} | "
        f"{format_timecode(summary.range.start, with_millis=False)} – "
        f"{format_timecode(summary.range.end, with_millis=False)} "
        f"({summary.range.duration:.1f} с)"
    )
    lines = [header, *summary.notes]
    if summary.transcript_excerpt:
        lines.append(f"Реплика: «{summary.transcript_excerpt}»")
    else:
        lines.append("Реплика: нет речи")
    return "\n".join(lines)


def format_batch(summaries: Iterable[SceneSummary]) -> str:
    """Батч сцен одним сообщением пользователя."""
    return "\n\n".join(render(summary) for summary in summaries)


def format_video_context(context: VideoContext) -> str:
    """Контекст видео — вторая точка кеширования.

    Одинаков для всех батчей одного видео, поэтому кешируется целиком.
    Ни `job_id`, ни даты здесь нет и быть не может: они меняются между
    джобами и обнулили бы кеш при повторном запуске того же видео.
    """
    lines = [
        "Контекст видео:",
        f"- длительность: {format_timecode(context.duration_sec, with_millis=False)}",
        f"- сцен всего: {context.scene_count}",
    ]
    if context.title:
        lines.append(f"- название: {context.title}")
    if context.language:
        lines.append(f"- язык речи: {context.language}")
    lines.append(f"- профиль отбора: {context.profile}")
    return "\n".join(lines)


def prepare(
    scenes: Sequence[Scene],
    *,
    min_signal: float = DEFAULT_MIN_SIGNAL,
) -> list[SceneSummary]:
    """Сцены → сводки тех, за которые стоит платить."""
    return [summarize(scene) for scene in scenes if is_worth_ranking(scene, min_signal=min_signal)]
