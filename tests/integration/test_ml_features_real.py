"""Анализаторы Этапа 5 на настоящих весах.

Здесь проверяется не качество моделей — их обучали не мы, — а **стыки
с ними**, то есть ровно то, что ломается молча:

* имена меток. Опечатка в таблице соответствий не вызовет ошибки: метка
  просто никогда не найдётся, и признак навсегда останется нулём;
* набор классов SER. Если модель отвечает другими именами, проекция
  в arousal/valence подставит значения по умолчанию — и снова тихо;
* отсутствие ложных срабатываний на пустоте: шум и тишина не должны
  порождать смех, крик и аплодисменты.

Тесты пропускаются, если torch/transformers не установлены (окружение
разработчика их не содержит намеренно) или если веса недоступны.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from audio.events import EVENT_LABELS, AudioSetClassifier, events_from_scores
from core.domain.values import TimeRange
from core.infra.wav import read_window
from speech.emotion import AROUSAL, VALENCE, SpeechEmotionModel, dimensions
from vision.faces import PersonDetector, YuNetFaceDetector

pytestmark = [pytest.mark.integration, pytest.mark.slow]

torch = pytest.importorskip("torch", reason="torch не установлен: ML-анализаторы не проверить")
pytest.importorskip("transformers", reason="transformers не установлен")

np = pytest.importorskip("numpy")

RATE = 16000

#: Видео с человеком в кадре. Задаётся окружением, как и эталон для ASR:
#: класть в репозиторий десятки мегабайт ради одного теста незачем.
REFERENCE_MEDIA = "VISION_REFERENCE_MEDIA"


def _write(path: Path, samples: object) -> Path:
    payload = np.clip(np.asarray(samples), -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((payload * 32767).astype(np.int16).tobytes())
    return path


@pytest.fixture(scope="module")
def classifier() -> AudioSetClassifier:
    return AudioSetClassifier(model_name="MIT/ast-finetuned-audioset-10-10-0.4593")


@pytest.fixture(scope="module")
def ser() -> SpeechEmotionModel:
    return SpeechEmotionModel(model_name="superb/wav2vec2-base-superb-er")


@pytest.fixture(scope="module")
def noise(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(0)
    return _write(tmp_path_factory.mktemp("ml") / "noise.wav", rng.normal(0, 0.15, RATE * 11))


@pytest.fixture(scope="module")
def silence(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write(tmp_path_factory.mktemp("ml") / "silence.wav", np.zeros(RATE * 11))


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Дорожка эталонного ролика — настоящая речь с фоновой музыкой."""
    import os
    import shutil
    import subprocess

    source = os.environ.get(REFERENCE_MEDIA)
    if not source or not Path(source).exists():
        pytest.skip(f"{REFERENCE_MEDIA} не задан: нет настоящего звука")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg не установлен")

    path = tmp_path_factory.mktemp("ml") / "speech.wav"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            source,
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        timeout=180,
    )
    return path


class TestAudioSetLabels:
    def test_every_label_in_the_table_exists_in_the_model(
        self, classifier: AudioSetClassifier
    ) -> None:
        """Главный инвариант модуля.

        Метка, которой нет в модели, не вызывает ошибки — она просто
        не находится, и признак молча остаётся нулевым навсегда.
        Именно так `Crash cymbal` из первой версии таблицы и не работал бы.
        """
        scores = classifier.probabilities(np.zeros(RATE * 2, dtype=np.float32), RATE)

        missing = {
            label for labels in EVENT_LABELS.values() for label in labels if label not in scores
        }
        assert not missing, f"этих меток нет в модели: {sorted(missing)}"

    def test_probabilities_are_independent_not_a_distribution(
        self, classifier: AudioSetClassifier, speech_wav: Path
    ) -> None:
        """Разметка мультилейбловая, и проверяется это на реальном звуке.

        В эталонном ролике одновременно звучат голос и фоновая музыка,
        и модель это отражает: ``Speech`` 0.82 и ``Music`` 0.78 — вместе
        больше единицы. Через softmax такое невозможно в принципе, там
        сумма по всем 527 классам равна единице, и каждое событие получило
        бы сотые доли.

        Проверять этот инвариант на тишине бесполезно: там уверенных меток
        мало и сумма меньше единицы сама по себе — первая версия теста
        именно на этом и провалилась.
        """
        scores = classifier.probabilities(read_window(speech_wav, TimeRange(20.0, 30.0)), RATE)

        assert all(0.0 <= value <= 1.0 for value in scores.values())
        assert scores["Speech"] + scores["Music"] > 1.0


class TestNoFalseEvents:
    def test_noise_produces_no_human_events(
        self, classifier: AudioSetClassifier, noise: Path
    ) -> None:
        """Белый шум — это не смех, не крик и не овации."""
        scores = classifier.probabilities(read_window(noise, TimeRange(0.0, 10.0)), RATE)
        found = {value.key for value in events_from_scores(scores, threshold=0.1)}

        assert (
            found
            & {
                "audio.event.laughter",
                "audio.event.scream",
                "audio.event.applause",
            }
            == set()
        )

    def test_silence_is_vetoed_rather_than_called_music(
        self, classifier: AudioSetClassifier, silence: Path
    ) -> None:
        """Замер, из-за которого появилось вето: на ровной тишине модель
        даёт ``Silence`` 0.61 и одновременно ``Music`` 0.27 — то есть выше
        порога. Без вето каждая тихая сцена получала бы «музыку»."""
        scores = classifier.probabilities(read_window(silence, TimeRange(0.0, 10.0)), RATE)

        assert scores["Silence"] >= 0.5, "модель перестала уверенно узнавать тишину"
        assert scores["Music"] >= 0.1, "ложное срабатывание пропало — вето можно упростить"


class TestSpeechEmotionModel:
    def test_classes_match_the_projection_table(self, ser: SpeechEmotionModel, noise: Path) -> None:
        """Второй тихий инвариант: имена классов.

        Ответь модель `neutral` вместо `neu`, и проекция подставила бы
        значения по умолчанию — arousal 0.5 на всех сценах подряд,
        без единой ошибки в логах.
        """
        probabilities = ser.probabilities(read_window(noise, TimeRange(0.0, 3.0)), RATE)

        assert set(probabilities) == set(AROUSAL) == set(VALENCE)

    def test_probabilities_form_a_distribution(self, ser: SpeechEmotionModel, noise: Path) -> None:
        """Классы взаимоисключающие — здесь softmax, а не сигмоида."""
        probabilities = ser.probabilities(read_window(noise, TimeRange(0.0, 3.0)), RATE)

        assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_dimensions_stay_in_their_ranges(self, ser: SpeechEmotionModel, noise: Path) -> None:
        arousal, valence = dimensions(
            ser.probabilities(read_window(noise, TimeRange(0.0, 3.0)), RATE)
        )

        assert 0.0 <= arousal <= 1.0
        assert -1.0 <= valence <= 1.0


class TestDetectors:
    def test_yunet_weights_load_and_empty_frame_has_no_faces(self) -> None:
        detector = YuNetFaceDetector(
            repo_id="opencv/face_detection_yunet",
            filename="face_detection_yunet_2023mar.onnx",
        )

        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        assert list(detector.detect(frame)) == []

    def test_person_detector_finds_nobody_on_an_empty_frame(self) -> None:
        detector = PersonDetector()

        assert detector.count(np.zeros((240, 320, 3), dtype=np.uint8)) == 0

    def test_face_is_found_on_real_footage(self, request: pytest.FixtureRequest) -> None:
        """Единственная проверка, где важно качество детектора: на кадре
        с человеком лицо обязано найтись, иначе признак бесполезен."""
        import os

        source = os.environ.get(REFERENCE_MEDIA)
        if not source or not Path(source).exists():
            pytest.skip(f"{REFERENCE_MEDIA} не задан: не на чем искать лица")

        import cv2

        capture = cv2.VideoCapture(source)
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 25)
            ok, frame = capture.read()
        finally:
            capture.release()
        assert ok, "не удалось прочитать кадр из эталонного видео"

        height, width = frame.shape[:2]
        scale = 320 / max(height, width)
        small = cv2.resize(frame, (int(width * scale), int(height * scale)))

        detector = YuNetFaceDetector(
            repo_id="opencv/face_detection_yunet",
            filename="face_detection_yunet_2023mar.onnx",
        )

        assert len(detector.detect(small)) >= 1
