"""Асинхронная сессия SQLAlchemy и фабрика движка."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from chgk_agent.config import Settings, get_settings


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    """Создать асинхронный движок по настройкам приложения."""

    current = settings or get_settings()
    return create_async_engine(
        current.database.dsn,
        echo=current.database.echo,
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Создать фабрику асинхронных сессий."""

    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Открыть сессию с автоматическим откатом при ошибке."""

    session = factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
