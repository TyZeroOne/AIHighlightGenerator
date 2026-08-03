"""Тесты конфигурации, ошибок и перечислений."""

from __future__ import annotations

import re

import pytest

from core.config import AppEnv, Settings
from core.domain.enums import STAGE_WEIGHTS, JobStage, JobStatus
from core.domain.errors import (
    ExternalServiceError,
    HighlightError,
    TransientError,
    UnsupportedMediaError,
    ValidationError,
)

pytestmark = pytest.mark.unit


class TestSettings:
    def test_defaults_are_usable_in_dev(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.app_env is AppEnv.DEV
        assert settings.llm_model == "claude-opus-5"

    def test_database_url_is_async_dsn(self) -> None:
        settings = Settings(_env_file=None, postgres_host="db", postgres_db="x")  # type: ignore[call-arg]
        assert settings.database_url.startswith("postgresql+asyncpg://")
        assert settings.sync_database_url.startswith("postgresql+psycopg://")

    def test_score_weights_must_sum_to_one(self) -> None:
        """Иначе итоговый скор молча вылезет за 100 или не дотянет до него."""
        with pytest.raises(ValueError, match=re.escape("1.0")):
            Settings(_env_file=None, score_alpha=0.5, score_beta=0.5, score_gamma=0.5)  # type: ignore[call-arg]

    def test_prod_requires_real_secret(self) -> None:
        with pytest.raises(ValueError, match="SECRET_KEY"):
            Settings(_env_file=None, app_env="prod")  # type: ignore[call-arg]

    def test_prod_requires_llm_key_when_llm_enabled(self) -> None:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            Settings(_env_file=None, app_env="prod", secret_key="s3cret", llm_enabled=True)  # type: ignore[call-arg]

    def test_prod_without_llm_does_not_require_key(self) -> None:
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None, app_env="prod", secret_key="s3cret", llm_enabled=False
        )
        assert settings.is_production

    def test_list_like_settings_are_parsed(self) -> None:
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            allowed_containers="mp4, mkv , mov",
            analyzers_disabled="vision.faces,audio.events",
        )
        assert settings.allowed_containers_set == {"mp4", "mkv", "mov"}
        assert settings.disabled_analyzers_set == {"vision.faces", "audio.events"}

    def test_unknown_env_vars_are_ignored(self) -> None:
        """Лишняя переменная в окружении не должна ронять процесс."""
        Settings(_env_file=None, SOME_UNRELATED_VAR="x")  # type: ignore[call-arg]


class TestErrors:
    def test_hierarchy_maps_to_http(self) -> None:
        assert ValidationError("x").http_status == 422
        assert UnsupportedMediaError("x").http_status == 415
        assert ExternalServiceError("x").http_status == 503

    def test_only_transient_errors_are_retriable(self) -> None:
        """Ретрай битого файла — пустая трата воркера (док 04, §4.4)."""
        assert TransientError("disk busy").retriable
        assert ExternalServiceError("llm down").retriable
        assert not ValidationError("bad input").retriable
        assert not UnsupportedMediaError("avi").retriable

    def test_context_is_preserved(self) -> None:
        error = UnsupportedMediaError("Формат не поддерживается", container="avi", allowed=["mp4"])
        problem = error.to_problem()
        assert problem["code"] == "UNSUPPORTED_MEDIA"
        assert problem["context"]["container"] == "avi"

    def test_all_inherit_base(self) -> None:
        for cls in (ValidationError, UnsupportedMediaError, TransientError, ExternalServiceError):
            assert issubclass(cls, HighlightError)


class TestEnums:
    def test_stage_weights_cover_every_stage(self) -> None:
        """Новый этап нельзя добавить, забыв про его вклад в прогресс."""
        assert set(STAGE_WEIGHTS) == set(JobStage)

    def test_stage_weights_sum_to_100(self) -> None:
        assert sum(STAGE_WEIGHTS.values()) == 100

    def test_terminal_statuses(self) -> None:
        assert JobStatus.SUCCEEDED.is_terminal
        assert JobStatus.PARTIAL.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert not JobStatus.RUNNING.is_terminal
        assert not JobStatus.QUEUED.is_terminal

    def test_enum_values_are_lowercase_strings(self) -> None:
        """Значения уходят в PostgreSQL как есть — регистр важен."""
        for status in JobStatus:
            assert status.value == status.value.lower()
            assert isinstance(status.value, str)
