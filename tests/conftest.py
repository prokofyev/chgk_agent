"""Общие фикстуры тестов.

Интеграционные тесты работают на отдельной базе: её адрес берётся из
`CHGK_DATABASE__TEST_DSN`, база создаётся при необходимости и доводится до
актуальной схемы миграциями. Рабочая база в тестах не используется —
совпадение адресов останавливает прогон до первого запроса.
"""

from collections.abc import AsyncIterator, Iterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from chgk_agent.config import Settings
from chgk_agent.db.session import create_engine, create_session_factory
from chgk_agent.logging_setup import configure_logging
from db_setup import DatabasePlan, ensure_test_database


@pytest.fixture(autouse=True)
def _quiet_logging() -> None:
    """Включить предсказуемое логирование в тестах."""

    configure_logging(json_logs=True, level="WARNING")


@pytest.fixture(scope="session")
def test_database() -> AsyncIterator[DatabasePlan]:
    """Подготовить тестовую базу один раз за прогон.

    Если базу нельзя создать или к ней нет доступа, интеграционные тесты
    пропускаются: без локального PostgreSQL прогон остаётся зелёным.
    """

    settings = Settings(_env_file=None)
    plan = ensure_test_database(
        settings.database.test_dsn, settings.database.dsn
    )
    yield plan


@pytest.fixture(autouse=True)
def _require_database_for_integration(request: pytest.FixtureRequest) -> Iterator[None]:
    """Подготовить тестовую базу только для интеграционных тестов.

    Модульные тесты не должны зависеть от локального PostgreSQL, поэтому
    фикстура запрашивает подготовку базы лениво — лишь когда тест помечен
    маркером `integration`.
    """

    if request.node.get_closest_marker("integration") is None:
        yield
        return

    request.getfixturevalue("test_database")
    yield


@pytest.fixture
def settings() -> Settings:
    """Настройки с изолированными значениями по умолчанию."""

    return Settings(_env_file=None)


@pytest.fixture
async def engine(test_database: DatabasePlan) -> AsyncIterator[AsyncEngine]:
    """Асинхронный движок на тестовой базе, изолированный на тест."""

    engine = create_engine(Settings(_env_file=None, database={"dsn": test_database.dsn}))
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def session_factory(
    engine: AsyncEngine,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Фабрика сессий, изолированная на тест."""

    yield create_session_factory(engine)


@pytest.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Сессия с откатом изменений после теста."""

    async with session_factory() as session:
        transaction = await session.begin()
        try:
            yield session
        finally:
            await transaction.rollback()
