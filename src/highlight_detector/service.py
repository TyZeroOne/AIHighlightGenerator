"""Реализация портов ``SceneScorer`` и ``HighlightSelector``.

Здесь сходятся четыре чистых модуля пакета: нормализация, структура,
сигналы, скоринг — и отбор. Сценарии этапов видят только этот файл через
порты и не знают ни про IQR, ни про MMR.

Разделение по границе изменчивости: формулы сигналов и пороги отбора будут
правиться по мере накопления материала, место этапов в конвейере — нет.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID, uuid4

from core.domain.entities import Clip, Highlight, Scene
from core.domain.values import FeatureValue, TimeRange
from core.logging import get_logger
from highlight_detector.normalize import normalize_scenes
from highlight_detector.profiles import get_profile
from highlight_detector.scoring import fuse, heuristic_score
from highlight_detector.selector import select_highlights
from highlight_detector.signals import compute_signals
from highlight_detector.structure import compute_structure

__all__ = ["DetectorService"]

log = get_logger(__name__)


class DetectorService:
    """Скоринг сцен и отбор моментов."""

    def score(self, scenes: Sequence[Scene], *, profile: str) -> Sequence[Scene]:
        if not scenes:
            return []

        settings = get_profile(profile)
        ordered = sorted(scenes, key=lambda scene: scene.index)

        raw = [{key: value.value for key, value in scene.features.items()} for scene in ordered]
        normalized = normalize_scenes(raw)
        structure = compute_structure(normalized, [scene.range.duration for scene in ordered])

        updated: list[Scene] = []
        for scene, norm, struct in zip(ordered, normalized, structure, strict=True):
            signals = compute_signals({**norm, **struct}, settings.signal_weights)
            score = heuristic_score(signals, settings)

            # Признаки дописываются, а не заменяются: в той же колонке лежат
            # результаты трёх групп анализаторов Этапа 5.
            features = dict(scene.features)
            features.update({key: FeatureValue(key, value) for key, value in struct.items()})
            features.update(
                {
                    f"signal.{name}": FeatureValue(f"signal.{name}", value)
                    for name, value in signals.items()
                }
            )

            updated.append(
                replace(
                    scene,
                    features=features,
                    heuristic_score=score,
                    # Итоговый скор пока равен эвристике: вердиктов модели
                    # на этом этапе ещё нет, они появятся следующим этапом.
                    final_score=score.value,
                )
            )

        log.debug("scoring.done", scenes=len(updated), profile=profile)
        return updated

    def select(
        self,
        scenes: Sequence[Scene],
        *,
        profile: str,
        pauses: Sequence[TimeRange] = (),
        total_duration_sec: float | None = None,
    ) -> Sequence[Highlight]:
        settings = get_profile(profile)
        fused = self._fuse_scores(scenes)
        return select_highlights(
            fused,
            settings,
            pauses=pauses,
            total_duration_sec=total_duration_sec,
        )

    def fuse_scores(self, scenes: Sequence[Scene]) -> Sequence[Scene]:
        """Публичная обёртка: слияние эвристики с оценкой модели.

        Отдельным методом, потому что результат надо сохранить в базу —
        `final_score` показывается в API и объясняет порядок клипов.
        """
        return self._fuse_scores(scenes)

    def to_clips(
        self,
        highlights: Sequence[Highlight],
        *,
        job_id: object,
        video_id: object,
    ) -> Sequence[Clip]:
        """Моменты → записи клипов.

        Файлов ещё нет: на этом этапе клип — это решение «вот этот отрезок
        стоит вырезать», а резать будет Этап 8. Ранг проставляется здесь,
        потому что он часть решения, а не рендера.
        """
        return [
            Clip(
                id=uuid4(),
                job_id=_as_uuid(job_id),
                video_id=_as_uuid(video_id),
                rank=rank,
                range=highlight.range,
                score=highlight.final_score,
                category=highlight.category,
                reason=highlight.reason,
                scene_ids=highlight.scene_ids,
            )
            for rank, highlight in enumerate(highlights, start=1)
        ]

    # ------------------------------------------------------------ внутреннее
    def _fuse_scores(self, scenes: Sequence[Scene]) -> list[Scene]:
        result: list[Scene] = []
        for scene in scenes:
            heuristic = scene.heuristic_score.value if scene.heuristic_score else 0.0
            result.append(replace(scene, final_score=fuse(heuristic, scene.llm_score)))
        return result


def _as_uuid(value: object) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))
