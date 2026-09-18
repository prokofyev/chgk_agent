"""Тесты подготовки отдельной базы для интеграционных тестов."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from chgk_agent.config import Settings
from chgk_agent.db.base import Source
from db_setup import (
    DatabaseSetupConfigurationError,
    DatabaseSetupUnavailable,
    admin_dsn_for,
    disposable_database,
    ensure_test_database,
    plan_test_database,
    prepare_test_database,
)

WORKING = "postgresql+asyncpg://user:secret@db.example:5432/chgk_agent"
TEST = "postgresql+asyncpg://user:secret@db.example:5432/chgk_agent_test"


def test_admin_dsn_points_to_admin_database() -> None:
    assert admin_dsn_for(TEST) == "postgresql+asyncpg://user:secret@db.example:5432/postgres"


def test_plan_keeps_test_address_and_database_name() -> None:
    plan = plan_test_database(TEST, WORKING)

    assert plan.dsn == TEST
    assert plan.database == "chgk_agent_test"
    assert plan.admin_dsn.endswith("/postgres")


def test_plan_rejects_working_database() -> None:
    with pytest.raises(DatabaseSetupConfigurationError) as excinfo:
        plan_test_database(WORKING, WORKING)

    message = str(excinfo.value)
    assert "совпадает с рабочей" in message
    assert "CHGK_DATABASE__TEST_DSN" in message


def test_plan_rejects_same_database_name_on_same_host_with_different_user() -> None:
    """Смена пользователя не делает базу другой: имя и сервер те же."""

    other_user = "postgresql+asyncpg://other:secret@db.example:5432/chgk_agent"

    with pytest.raises(DatabaseSetupConfigurationError):
        plan_test_database(other_user, WORKING)


def test_plan_allows_same_name_on_different_server() -> None:
    plan = plan_test_database(
        "postgresql+asyncpg://user:secret@other-host:5432/chgk_agent", WORKING
    )

    assert plan.database == "chgk_agent"


def test_plan_rejects_dsn_without_database_name() -> None:
    with pytest.raises(DatabaseSetupConfigurationError) as excinfo:
        plan_test_database("postgresql+asyncpg://user:secret@db.example:5432", WORKING)

    assert "не указано имя базы" in str(excinfo.value)


def test_prepare_reports_unavailable_database() -> None:
    """Недоступный сервер приводит к пропуску, а не к ошибке теста."""

    unreachable = "postgresql+asyncpg://user@localhost:1/chgk_agent_test"

    with pytest.raises(DatabaseSetupUnavailable) as excinfo:
        prepare_test_database(unreachable, WORKING)

    message = str(excinfo.value)
    assert "недоступна" in message
    assert "CHGK_DATABASE__TEST_DSN" in message
    assert "Рабочая база не используется" in message


@pytest.mark.integration
def test_prepare_creates_database_with_current_schema() -> None:
    """После подготовки в тестовой базе есть схема с нужной размерностью."""

    settings = Settings(_env_file=None)
    plan = prepare_test_database(
        settings.database.test_dsn, settings.database.dsn
    )

    import asyncio

    async def schema() -> tuple[str, str]:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(plan.dsn)
        try:
            async with engine.connect() as connection:
                column = await connection.scalar(
                    text(
                        "SELECT format_type(a.atttypid, a.atttypmod) "
                        "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
                        "WHERE c.relname = 'questions' AND a.attname = 'embedding'"
                    )
                )
                version = await connection.scalar(
                    text("SELECT version_num FROM alembic_version")
                )
        finally:
            await engine.dispose()
        return str(column), str(version)

    column_type, version = asyncio.run(schema())

    assert column_type == f"vector({settings.gigachat.embedding_dim})"
    assert version


def test_ensure_reports_unavailable_as_skip() -> None:
    """Недоступная база превращается в пропуск, а не в падение."""

    unreachable = "postgresql+asyncpg://user@localhost:1/chgk_agent_test"

    with pytest.raises(pytest.skip.Exception) as excinfo:
        ensure_test_database(unreachable, WORKING)

    assert "недоступна" in str(excinfo.value)


def test_ensure_does_not_skip_on_working_database() -> None:
    """Совпадение с рабочей базой остаётся ошибкой, а не пропуском."""

    with pytest.raises(DatabaseSetupConfigurationError):
        ensure_test_database(WORKING, WORKING)


@pytest.mark.integration
async def test_integration_tests_run_on_test_database(session) -> None:
    """Фикстуры подключают тесты к тестовой базе, а не к рабочей."""

    settings = Settings(_env_file=None)

    current = await session.scalar(text("SELECT current_database()"))

    assert current == settings.database.test_dsn.rsplit("/", 1)[-1]
    assert current != settings.database.dsn.rsplit("/", 1)[-1]


async def test_working_database_address_is_rejected_before_any_query() -> None:
    """Проверка совпадения срабатывает до подключения к базе."""

    settings = Settings(_env_file=None)

    with pytest.raises(DatabaseSetupConfigurationError):
        plan_test_database(settings.database.dsn, settings.database.dsn)


ROLLBACK_LOCATION = "session-fixture-rollback.html"


@pytest.mark.integration
async def test_session_fixture_does_not_commit_writes(session, engine) -> None:
    """Фикстура `session` не коммитит записи теста.

    Изменения видны внутри транзакции теста, но не видны отдельному
    соединению — значит, коммита не было и тестовая база не накапливает
    данные между тестами.
    """

    session.add(Source(kind="html", location=ROLLBACK_LOCATION))
    await session.flush()

    inside = await session.scalar(
        text("SELECT count(*) FROM sources WHERE location = :location"),
        {"location": ROLLBACK_LOCATION},
    )
    assert inside == 1

    async with engine.connect() as other:
        outside = await other.scalar(
            text("SELECT count(*) FROM sources WHERE location = :location"),
            {"location": ROLLBACK_LOCATION},
        )

    assert outside == 0

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_pytest(test_dsn: str, *targets: str) -> subprocess.CompletedProcess[str]:
    """Запустить pytest в подпроцессе с заданным адресом тестовой базы."""

    environment = {**os.environ, "CHGK_DATABASE__TEST_DSN": test_dsn}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_unreachable_test_database_skips_integration_and_keeps_unit_tests() -> None:
    """Недоступная тестовая база пропускает интеграционные тесты, не ломая модульные."""

    unreachable = "postgresql+asyncpg://user@localhost:1/chgk_agent_test"

    result = _run_pytest(unreachable, "tests/test_db_session.py", "tests/test_config.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "failed" not in result.stdout
    assert "skipped" in result.stdout


def test_working_database_as_test_database_stops_the_run() -> None:
    """Совпадение с рабочей базой останавливает прогон до первого запроса."""

    settings = Settings(_env_file=None)

    result = _run_pytest(settings.database.dsn, "tests/test_db_session.py")

    assert result.returncode != 0
    assert "совпадает с рабочей" in result.stdout

ALTERNATIVE_DIM = 384


@pytest.mark.integration
def test_disposable_database_is_removed_after_use() -> None:
    """Одноразовая база удаляется после проверки миграции."""

    settings = Settings(_env_file=None)
    created: list[str] = []

    with disposable_database(
        settings.database.test_dsn, embedding_dim=ALTERNATIVE_DIM
    ) as plan:
        created.append(plan.database)
        assert plan.database != settings.database.test_dsn.rsplit("/", 1)[-1]

        column = asyncio.run(_embedding_column_type(plan.dsn))
        assert column == f"vector({ALTERNATIVE_DIM})"

    assert asyncio.run(_database_exists(settings, created[0])) is False


@pytest.mark.integration
def test_disposable_database_does_not_touch_other_databases() -> None:
    """Проверка на одноразовой базе не меняет рабочую и основную тестовую."""

    settings = Settings(_env_file=None)
    before = asyncio.run(_database_fingerprints(settings))

    with disposable_database(
        settings.database.test_dsn, embedding_dim=ALTERNATIVE_DIM
    ) as plan:
        assert plan.database not in before

    after = asyncio.run(_database_fingerprints(settings))

    assert after == before


async def _embedding_column_type(dsn: str) -> str:
    """Тип колонки `questions.embedding` в базе по адресу."""

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(dsn)
    try:
        async with engine.connect() as connection:
            value = await connection.scalar(
                text(
                    "SELECT format_type(a.atttypid, a.atttypmod) "
                    "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
                    "WHERE c.relname = 'questions' AND a.attname = 'embedding'"
                )
            )
    finally:
        await engine.dispose()
    return str(value)


async def _database_exists(settings: Settings, name: str) -> bool:
    """Существует ли база с таким именем."""

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(admin_dsn_for(settings.database.test_dsn))
    try:
        async with engine.connect() as connection:
            found = await connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
            )
    finally:
        await engine.dispose()
    return found is not None


async def _database_fingerprints(settings: Settings) -> dict[str, str]:
    """Отпечатки схемы рабочей и основной тестовой базы."""

    fingerprints: dict[str, str] = {}
    for dsn in (settings.database.dsn, settings.database.test_dsn):
        name = dsn.rsplit("/", 1)[-1]
        column = await _embedding_column_type(dsn)
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(dsn)
        try:
            async with engine.connect() as connection:
                tables = await connection.scalar(
                    text(
                        "SELECT string_agg(tablename, ',' ORDER BY tablename) "
                        "FROM pg_tables WHERE schemaname = 'public'"
                    )
                )
        finally:
            await engine.dispose()
        fingerprints[name] = f"{column}|{tables}"
    return fingerprints
