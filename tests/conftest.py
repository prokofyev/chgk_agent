"""Общие фикстуры тестов."""

from collections.abc import AsyncIterator, Iterator
from functools import lru_cache

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from chgk_agent.config import Settings
from chgk_agent.db.session import create_engine, create_session_factory
from chgk_agent.logging_setup import configure_logging


@lru_cache(maxsize=1)
def _database_available() -> bool:
    """Проверить доступность PostgreSQL из настроек."""

    import asyncio

    async def probe() -> bool:
        engine = create_engine(Settings(_env_file=None))
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception:
            return False
        finally:
            await engine.dispose()
        return True

    try:
        return asyncio.run(probe())
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _quiet_logging() -> None:
    """Включить предсказуемое логирование в тестах."""

    configure_logging(json_logs=True, level="WARNING")


@pytest.fixture(autouse=True)
def _require_database_for_integration(request: pytest.FixtureRequest) -> Iterator[None]:
    """Пропустить интеграционные тесты, если локальная база недоступна."""

    if request.node.get_closest_marker("integration") and not _database_available():
        pytest.skip("PostgreSQL с pgvector недоступен")
    yield


@pytest.fixture
def settings() -> Settings:
    """Настройки с изолированными значениями по умолчанию."""

    return Settings(_env_file=None)


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """Асинхронный движок, изолированный на тест."""

    engine = create_engine(Settings(_env_file=None))
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
