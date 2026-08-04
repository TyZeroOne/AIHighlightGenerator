"""Манифест аудиоанализаторов.

Реестр импортирует этот модуль при ``discover()``, а импорт исполняет
декораторы ``@register_analyzer`` — так классы попадают в реестр.

Список явный, а не обход каталога: промах в имени файла виден глазами
и находится грепом, тогда как «почему-то не зарегистрировался» при
автообходе выясняется на живом воркере. Ровно то же соображение,
что и у ``TASK_MODULES`` в ``workers/app.py``.

Каждый новый анализатор добавляется сюда одной строкой — и это
единственное место в проекте, которое при этом меняется.
"""

from __future__ import annotations

from audio.events import EventAnalyzer
from audio.loudness import LoudnessAnalyzer
from audio.pauses import PauseAnalyzer

__all__ = ["EventAnalyzer", "LoudnessAnalyzer", "PauseAnalyzer"]
