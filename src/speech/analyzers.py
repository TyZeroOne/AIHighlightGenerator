"""Манифест речевых анализаторов.

Реестр импортирует этот модуль при ``discover()``; импорт исполняет
декораторы ``@register_analyzer``. Список явный по той же причине, что
и у ``audio/analyzers.py``: промах в имени файла виден глазами, а не
на живом воркере.
"""

from __future__ import annotations

from speech.emotion import EmotionAnalyzer
from speech.rate import SpeechRateAnalyzer

__all__ = ["EmotionAnalyzer", "SpeechRateAnalyzer"]
