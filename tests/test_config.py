"""Тесты конфигурации приложения."""

import pytest
from pydantic import ValidationError

from chgk_agent.config import Settings


def test_external_source_enabled_by_default() -> None:
    settings = Settings(_env_file=None)

    assert settings.external.enabled is True
    assert settings.external.base_url == "https://gotquestions.online"


def test_user_agent_is_required_and_non_empty() -> None:
    settings = Settings(_env_file=None)

    assert settings.external.user_agent.strip()


def test_query_limit_matches_external_site_limit() -> None:
    settings = Settings(_env_file=None)

    assert settings.external.max_query_chars == 50


def test_query_limit_above_site_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, external={"max_query_chars": 51})


def test_settings_read_nested_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHGK_EXTERNAL__ENABLED", "false")
    monkeypatch.setenv("CHGK_EXTERNAL__BASE_URL", "https://example.test")
    monkeypatch.setenv("CHGK_DATABASE__DSN", "postgresql+asyncpg://u:p@h:5432/db")
    monkeypatch.setenv("CHGK_SEARCH__TOP_K", "7")

    settings = Settings(_env_file=None)

    assert settings.external.enabled is False
    assert settings.external.base_url == "https://example.test"
    assert settings.database.dsn == "postgresql+asyncpg://u:p@h:5432/db"
    assert settings.search.top_k == 7


def test_empty_user_agent_from_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHGK_EXTERNAL__USER_AGENT", "")

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    locations = {error["loc"] for error in excinfo.value.errors()}
    assert ("external", "user_agent") in locations
