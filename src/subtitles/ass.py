"""Advanced SubStation Alpha: стиль, положение и караоке-подсветка.

SRT умеет только текст и тайминги. Всё остальное — шрифт, размер,
обводку, положение в кадре — решает плеер, и решает по-разному. Для
файла, который уйдёт на площадку или будет вжжён в картинку, этого мало:
нужен предсказуемый вид.

## Почему ASS, а не SSA/WebVTT-стили

WebVTT со своим CSS понимает браузер и почти никто больше; вжечь его
в видео нельзя. ASS понимает libass, встроенный в ffmpeg, VLC и mpv —
то есть один файл годится и для скачивания, и для вжигания, и выглядит
он везде одинаково.

## Караоке

Тег ``\\k<сантисекунды>`` перед словом означает «через столько это слово
перекрасится из вторичного цвета в основной». Отсюда договорённость
о цветах, обратная интуиции: **основной** цвет — у уже прозвучавших слов
(белый, яркий), **вторичный** — у ещё не прозвучавших (приглушённый).

Пауза между словами включается в ``\\k`` следующего слова, а не теряется:
иначе подсветка убегает вперёд и к концу реплики опережает речь
на несколько десятых секунды.

## PlayRes и почему их два

libass рисует в системе координат ``PlayResX × PlayResY`` и масштабирует
результат к реальному кадру. Если объявить 1920×1080, а вжечь в вертикаль
1080×1920, текст растянется по горизонтали. Поэтому файл собирается
под конкретный размер кадра, а кегль пересчитывается от высоты.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from subtitles.cues import Cue

__all__ = [
    "REFERENCE_HEIGHT",
    "AssStyle",
    "ass_timestamp",
    "chars_that_fit",
    "font_size_for",
    "render_ass",
]

#: Высота кадра, для которой задан кегль по умолчанию. Для других высот
#: размер шрифта пересчитывается пропорционально — иначе текст, читаемый
#: на горизонтали, в вертикали 1080×1920 окажется вдвое мельче.
REFERENCE_HEIGHT: Final = 1080

#: Средняя ширина символа в долях кегля. Для пропорциональных шрифтов
#: вроде DejaVu Sans около половины: узкие «i» и «л» компенсируют широкие
#: «ш» и «m». Точность здесь не нужна — число служит оценкой сверху для
#: того, сколько символов влезет в строку.
CHAR_WIDTH_RATIO: Final = 0.52

#: Какую долю ширины кадра разрешено занимать строке. Не всю: по краям
#: остаются поля, а у вертикальных площадок там ещё и элементы интерфейса.
TEXT_WIDTH_RATIO: Final = 0.88

#: Цвета в ASS записываются как ``&HAABBGGRR``: альфа, синий, зелёный,
#: красный — задом наперёд относительно привычного RGB. ``00`` в альфе
#: означает непрозрачный.
WHITE: Final = "&H00FFFFFF"
DIMMED: Final = "&H00B0B0B0"
BLACK: Final = "&H00000000"
SHADOW: Final = "&H80000000"


@dataclass(frozen=True, slots=True)
class AssStyle:
    """Оформление субтитров.

    Обводка, а не подложка: плашка под текстом надёжнее читается,
    но закрывает картинку, а у нас в кадре и так осталась треть
    исходной ширины.
    """

    font: str = "DejaVu Sans"
    #: Кегль при высоте кадра 1080.
    size: int = 42
    primary: str = WHITE
    secondary: str = DIMMED
    outline_colour: str = BLACK
    back_colour: str = SHADOW
    outline: float = 2.5
    shadow: float = 1.0
    #: 2 — по центру внизу (нумерация как на цифровой клавиатуре).
    alignment: int = 2
    #: Поля в долях соответствующей стороны, а не в пикселях. В пикселях
    #: их пришлось бы пересчитывать при каждом изменении кадра, причём
    #: горизонтальное — от ширины, вертикальное — от высоты; доля
    #: не требует помнить, от чего именно.
    margin_h_ratio: float = 0.06
    #: Отступ снизу больше обычного: в вертикальном формате нижнюю полосу
    #: кадра занимает интерфейс площадки — подписи, кнопки, ник.
    margin_v_ratio: float = 0.13


def render_ass(
    cues: Sequence[Cue],
    *,
    width: int,
    height: int,
    style: AssStyle | None = None,
    karaoke: bool = True,
) -> str:
    """Реплики → текст файла ``.ass`` под кадр ``width × height``."""
    style = style or AssStyle()
    size = font_size_for(height, style)
    margin_v = max(round(style.margin_v_ratio * height), 10)
    margin_h = max(round(style.margin_h_ratio * width), 10)

    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            # Переносы расставлены нами, и libass не должен их трогать:
            # иначе к нашим двум строкам он добавит свои.
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: None",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{style.font},{size},{style.primary},{style.secondary},"
            f"{style.outline_colour},{style.back_colour},0,0,0,0,100,100,0,0,1,"
            f"{style.outline:g},{style.shadow:g},{style.alignment},"
            f"{margin_h},{margin_h},{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    events = [
        f"Dialogue: 0,{ass_timestamp(cue.range.start)},{ass_timestamp(cue.range.end)},"
        f"Default,,0,0,0,,{_body(cue, karaoke=karaoke)}"
        for cue in cues
    ]
    return header + "\n" + "\n".join(events) + "\n"


def font_size_for(height: int, style: AssStyle | None = None) -> int:
    """Кегль под высоту кадра.

    От высоты, а не от ширины: читаемость текста определяется тем, какую
    долю экрана он занимает по вертикали, и в вертикальном формате шрифт
    обязан вырасти вместе с кадром, иначе субтитры превращаются в сноску.
    """
    style = style or AssStyle()
    return max(round(style.size * height / REFERENCE_HEIGHT), 8)


def chars_that_fit(width: int, height: int, style: AssStyle | None = None) -> int:
    """Сколько символов помещается в строку кадра ``width × height``.

    Это не второй способ задать те же 42 символа из стандарта вещания,
    а ограничение сверху, о котором стандарт молчит. 42 символа
    рассчитаны на широкий кадр: в 1920×1080 при кегле 42 строка занимает
    чуть больше половины ширины. В вертикали 1080×1920 кегль вырастает
    до 75 (кадр вдвое выше), и те же 42 символа требуют вдвое больше
    места, чем есть, — libass с ``WrapStyle: 2`` не переносит их сам,
    а честно выводит за край кадра.

    Поэтому для вертикали реплики пересобираются с меньшей длиной
    строки. Тексту от этого хуже не становится: строк по-прежнему две,
    просто короче.
    """
    size = font_size_for(height, style)
    usable = width * TEXT_WIDTH_RATIO
    return max(int(usable / (size * CHAR_WIDTH_RATIO)), 8)


def ass_timestamp(seconds: float) -> str:
    """``4.29`` → ``0:00:04.29``.

    У ASS своя форма: часы без ведущего нуля и **сантисекунды**, а не
    миллисекунды. Лишний разряд формат молча проглотит и покажет
    не то время.
    """
    total = max(round(seconds * 100), 0)
    centis = total % 100
    total_seconds = total // 100
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _body(cue: Cue, *, karaoke: bool) -> str:
    """Текст события: с караоке-разметкой или без неё."""
    if not karaoke or not cue.words:
        return _escape(cue.text).replace("\n", "\\N")

    # Караоке размечает поток слов, а не строки, поэтому переносы
    # расставляются заново — по тем же границам, что и в ``lines``.
    breaks = _line_breaks(cue)
    parts: list[str] = []
    cursor = cue.range.start
    for index, word in enumerate(cue.words):
        # Пауза перед словом отходит ему же: иначе подсветка убегает
        # вперёд и к концу реплики опережает речь.
        centis = max(round((word.end - cursor) * 100), 1)
        cursor = word.end
        prefix = "\\N" if index in breaks else ("" if index == 0 else " ")
        parts.append(f"{prefix}{{\\k{centis}}}{_escape(word.text.strip())}")
    return "".join(parts)


def _line_breaks(cue: Cue) -> set[int]:
    """Индексы слов, с которых начинается новая строка.

    Считаются по уже готовому разбиению ``lines``: два разных алгоритма
    переноса для SRT и ASS означали бы, что один и тот же клип выглядит
    по-разному в зависимости от формата.
    """
    breaks: set[int] = set()
    index = 0
    for line in cue.lines[:-1]:
        index += len(line.split())
        breaks.add(index)
    return breaks


def _escape(text: str) -> str:
    """Обезвредить символы разметки ASS.

    Фигурные скобки открывают блок тегов, обратный слеш начинает тег.
    В речи они не встречаются почти никогда — но «почти» здесь означает
    ровно один клип, в котором субтитры молча исчезнут.
    """
    return text.replace("\\", "／").replace("{", "(").replace("}", ")")
