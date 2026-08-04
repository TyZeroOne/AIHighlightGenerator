"""Манифест видеоанализаторов.

Реестр импортирует этот модуль при ``discover()``; импорт исполняет
декораторы ``@register_analyzer``.

Импорт лёгкий: OpenCV и NumPy подтягиваются внутри функций, а не на уровне
модуля. Это важно — ``discover()`` вызывается и в API-процессе, которому
машинное зрение не нужно вовсе, а импорт cv2 стоит секунды и десятки
мегабайт.
"""

from __future__ import annotations

from vision.cuts import CutsAnalyzer
from vision.dynamics import DynamicsAnalyzer
from vision.faces import FaceAnalyzer
from vision.motion import MotionAnalyzer

__all__ = ["CutsAnalyzer", "DynamicsAnalyzer", "FaceAnalyzer", "MotionAnalyzer"]
