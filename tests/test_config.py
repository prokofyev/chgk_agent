"""Тесты конфигурации приложения."""

import pytest
from pydantic import ValidationError

from chgk_agent.config import Settings


@pytest.fixture(autouse=True)
def _clear_test_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Убрать влияние окружения на проверки вывода тестового DSN."""

    monkeypatch.delenv("CHGK_DATABASE__TEST_DSN", raising=False)


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


def test_external_query_budget_is_setting() -> None:
    """Бюджет коротких запросов к внешнему источнику задаётся настройкой."""

    settings = Settings(_env_file=None)

    assert settings.external.max_query_terms == 3


def test_external_query_term_limit_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, external={"max_query_terms": 0})


def test_corpus_index_timeout_is_setting() -> None:
    settings = Settings(_env_file=None)

    assert settings.search.corpus_index_timeout_seconds == pytest.approx(2.0)


def test_corpus_index_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, search={"corpus_index_timeout_seconds": 0})


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


def test_lexical_defaults_keep_score_in_unit_range() -> None:
    """Вес лексики подобран так, чтобы сумма не вырождалась в потолок."""

    settings = Settings(_env_file=None)

    assert settings.search.lexical_weight == pytest.approx(0.15)
    assert settings.search.lexical_saturation > 0


def test_empty_user_agent_from_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHGK_EXTERNAL__USER_AGENT", "")

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    locations = {error["loc"] for error in excinfo.value.errors()}
    assert ("external", "user_agent") in locations


def test_test_dsn_is_derived_from_working_dsn() -> None:
    settings = Settings(_env_file=None)

    assert settings.database.test_dsn == (
        "postgresql+asyncpg://prokofyev@localhost:5432/chgk_agent_test"
    )


def test_test_dsn_preserves_server_port_and_credentials() -> None:
    settings = Settings(
        _env_file=None,
        database={"dsn": "postgresql+asyncpg://user:secret@db.example:6000/questions"},
    )

    assert settings.database.test_dsn == (
        "postgresql+asyncpg://user:secret@db.example:6000/questions_test"
    )


def test_test_dsn_requires_database_name() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database={"dsn": "postgresql+asyncpg://user@host:5432"})


def test_test_dsn_can_be_set_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CHGK_DATABASE__TEST_DSN", "postgresql+asyncpg://user@other-host:5432/isolated"
    )

    settings = Settings(_env_file=None)

    assert settings.database.test_dsn == (
        "postgresql+asyncpg://user@other-host:5432/isolated"
    )


def test_generation_switch_is_not_a_setting() -> None:
    """Признака отключения генерации в настройках больше нет."""

    settings = Settings(_env_file=None)

    assert not hasattr(settings.search, "generate_answer")
