"""Выбор устройства и типа вычислений для Whisper.

Чистые функции, но цена ошибки в них высокая: неверная пара «устройство +
compute_type» либо не даёт модели загрузиться вовсе, либо роняет скорость
в разы — и то и другое обнаруживается уже на живом файле.
"""

from __future__ import annotations

import pytest

from speech.whisper import resolve_batch_size, resolve_compute_type, resolve_device

pytestmark = pytest.mark.unit


class TestDevice:
    def test_explicit_value_wins(self) -> None:
        """Явный выбор не должен молча подменяться: если в конфиге стоит cpu,
        значит, у GPU есть занятие поважнее."""
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"

    def test_auto_always_gives_a_working_device(self) -> None:
        """``auto`` обязан вернуть рабочий режим при любом окружении:
        отсутствие CUDA — это не ошибка конфигурации."""
        assert resolve_device("auto") in {"cpu", "cuda"}


class TestComputeType:
    def test_auto_matches_the_device(self) -> None:
        """float16 на процессоре либо не поддерживается, либо эмулируется
        и работает вчетверо медленнее int8; int8 на GPU отдаёт качество
        без выигрыша в скорости."""
        assert resolve_compute_type("auto", "cpu") == "int8"
        assert resolve_compute_type("auto", "cuda") == "float16"

    def test_explicit_value_is_kept(self) -> None:
        assert resolve_compute_type("int8_float16", "cuda") == "int8_float16"
        assert resolve_compute_type("float32", "cpu") == "float32"

    def test_empty_value_behaves_as_auto(self) -> None:
        """Пустая переменная окружения — это «не указано», а не «пустой тип»."""
        assert resolve_compute_type("", "cpu") == "int8"


class TestBatchSize:
    def test_auto_batches_on_gpu_only(self) -> None:
        """Замер на RTX 5070 и large-v3: час аудио за 0.9 мин с батчами
        против 6.3 мин без них. На CPU выигрыша нет — CTranslate2 и так
        занимает все ядра одним окном, зато память растёт кратно."""
        assert resolve_batch_size(0, "cuda") == 16
        assert resolve_batch_size(0, "cpu") == 1

    def test_explicit_value_wins_on_any_device(self) -> None:
        assert resolve_batch_size(4, "cpu") == 4
        assert resolve_batch_size(32, "cuda") == 32
