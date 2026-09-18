"""Подготовка отдельной базы для интеграционных тестов.

Модуль отвечает за три вещи: вывести адрес тестовой базы и убедиться, что
он не указывает на рабочую базу; создать базу при отсутствии; довести её
схему до актуальной миграциями Alembic. Ошибки разделены на два вида:
неверная конфигурация (тестовая база совпала с рабочей) и недоступность
базы, из-за которой интеграционные тесты пропускаются.
"""

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from chgk_agent.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADMIN_DATABASE = "postgres"


class DatabaseSetupError(RuntimeError):
    """Базовая ошибка подготовки тестовой базы."""


class DatabaseSetupConfigurationError(DatabaseSetupError):
    """Тестовая база не отличается от рабочей."""


class DatabaseSetupUnavailable(DatabaseSetupError):
    """Тестовую базу нельзя создать или к ней нет доступа."""


@dataclass(frozen=True, slots=True)
class DatabasePlan:
    """Разобранный адрес тестовой базы."""

    dsn: str
    admin_dsn: str
    database: str


def admin_dsn_for(dsn: str) -> str:
    """Вернуть адрес административной базы на том же сервере."""

    url = make_url(dsn)
    return url.set(database=ADMIN_DATABASE).render_as_string(hide_password=False)


def plan_test_database(test_dsn: str, working_dsn: str) -> DatabasePlan:
    """Разобрать адреса и убедиться, что тестовая база не рабочая.

    Совпадение определяется тройкой «сервер, порт, имя базы»: одинаковая
    база на разных серверах безопасна, а та же база под тем же адресом —
    это ровно тот случай, из-за которого портились данные.
    """

    test_url = make_url(test_dsn)
    working_url = make_url(working_dsn)

    if not test_url.database:
        raise DatabaseSetupConfigurationError(
            "В адресе тестовой базы не указано имя базы: "
            f"{test_dsn!r}. Задайте CHGK_DATABASE__TEST_DSN с именем базы."
        )

    if (
        test_url.database == working_url.database
        and test_url.host == working_url.host
        and test_url.port == working_url.port
    ):
        raise DatabaseSetupConfigurationError(
            "Тестовая база совпадает с рабочей: "
            f"{test_url.host}:{test_url.port}/{test_url.database}. "
            "Интеграционные тесты не выполняются, чтобы не испортить данные. "
            "Укажите другой адрес в CHGK_DATABASE__TEST_DSN."
        )

    return DatabasePlan(
        dsn=test_dsn,
        admin_dsn=admin_dsn_for(test_dsn),
        database=test_url.database,
    )


def prepare_test_database(test_dsn: str, working_dsn: str) -> DatabasePlan:
    """Подготовить тестовую базу: проверить адрес, создать базу, применить миграции."""

    plan = plan_test_database(test_dsn, working_dsn)
    asyncio.run(_ensure_database(plan))
    apply_migrations(plan)
    return plan


def apply_migrations(
    plan: DatabasePlan,
    *,
    embedding_dim: int | None = None,
    create: bool = False,
) -> None:
    """Довести схему тестовой базы до актуальной через Alembic.

    `embedding_dim` переопределяет настройку размерности на время миграций,
    `create` создаёт базу, если её ещё нет.
    """

    if create:
        asyncio.run(_ensure_database(plan))

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", plan.dsn)
    config.set_main_option("path_separator", "os")

    previous = os.environ.get("CHGK_GIGACHAT__EMBEDDING_DIM")
    if embedding_dim is not None:
        os.environ["CHGK_GIGACHAT__EMBEDDING_DIM"] = str(embedding_dim)
    # Миграция читает настройки через кэширующий `get_settings`, поэтому
    # без сброса кэша изменение окружения до неё не дойдёт.
    get_settings.cache_clear()
    try:
        command.upgrade(config, "head")
    finally:
        if embedding_dim is not None:
            if previous is None:
                os.environ.pop("CHGK_GIGACHAT__EMBEDDING_DIM", None)
            else:
                os.environ["CHGK_GIGACHAT__EMBEDDING_DIM"] = previous
            get_settings.cache_clear()


async def _ensure_database(plan: DatabasePlan) -> None:
    """Создать базу, если её нет, и убедиться, что она доступна."""

    if await _is_reachable(plan.dsn):
        return

    engine = create_async_engine(
        plan.admin_dsn, isolation_level="AUTOCOMMIT", poolclass=NullPool
    )
    try:
        async with engine.connect() as connection:
            exists = await connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": plan.database},
            )
            if exists is None:
                await connection.execute(text(f"CREATE DATABASE {_quote(plan.database)}"))
    except Exception as error:
        raise DatabaseSetupUnavailable(_unavailable_message(plan, error)) from error
    finally:
        await engine.dispose()

    if not await _is_reachable(plan.dsn):
        raise DatabaseSetupUnavailable(
            _unavailable_message(plan, "база создана, но подключиться к ней не удалось")
        )


async def _is_reachable(dsn: str) -> bool:
    """Проверить, можно ли подключиться к базе по этому адресу."""

    engine = create_async_engine(dsn, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception:
        return False
    finally:
        await engine.dispose()
    return True


@contextmanager
def disposable_database(
    test_dsn: str, *, embedding_dim: int | None = None
) -> Iterator[DatabasePlan]:
    """Создать одноразовую базу, применить миграции и удалить её после.

    Нужна для проверок, которые требуют другой размерности вектора или
    иной конфигурации схемы: такие проверки не должны трогать ни рабочую,
    ни основную тестовую базу.
    """

    base = make_url(test_dsn)
    name = f"{base.database}_probe_{uuid4().hex[:8]}"
    dsn = base.set(database=name).render_as_string(hide_password=False)
    plan = DatabasePlan(dsn=dsn, admin_dsn=admin_dsn_for(dsn), database=name)

    apply_migrations(plan, embedding_dim=embedding_dim, create=True)
    try:
        yield plan
    finally:
        asyncio.run(_drop_database(plan))


def drop_database(plan: DatabasePlan) -> None:
    """Удалить базу, если она существует."""

    asyncio.run(_drop_database(plan))


async def _drop_database(plan: DatabasePlan) -> None:
    """Удалить базу через административное подключение."""

    engine = create_async_engine(
        plan.admin_dsn, isolation_level="AUTOCOMMIT", poolclass=NullPool
    )
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text(f"DROP DATABASE IF EXISTS {_quote(plan.database)} WITH (FORCE)")
            )
    except Exception as error:  # база могла не создаться
        raise DatabaseSetupUnavailable(
            f"Не удалось удалить одноразовую базу {plan.database!r}: {error}"
        ) from error
    finally:
        await engine.dispose()


def ensure_test_database(test_dsn: str, working_dsn: str) -> DatabasePlan:
    """Подготовить тестовую базу или пропустить интеграционные тесты.

    Недоступность базы не является ошибкой теста: без локального
    PostgreSQL прогон должен оставаться зелёным. А вот совпадение с
    рабочей базой — ошибка конфигурации, и её пропускать нельзя.
    """

    try:
        return prepare_test_database(test_dsn, working_dsn)
    except DatabaseSetupUnavailable as error:
        pytest.skip(str(error))


def _quote(name: str) -> str:
    """Заключить имя базы в кавычки, экранировав внутренние."""

    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _unavailable_message(plan: DatabasePlan, error: object) -> str:
    """Собрать сообщение о недоступности с подсказкой."""

    return (
        f"Тестовая база {plan.database!r} недоступна ({error}), поэтому "
        "интеграционные тесты пропущены. Создайте базу вручную "
        f"(CREATE DATABASE {_quote(plan.database)}) или задайте готовую "
        "через CHGK_DATABASE__TEST_DSN. Рабочая база не используется."
    )


__all__ = [
    "DatabaseSetupConfigurationError",
    "DatabaseSetupError",
    "DatabasePlan",
    "DatabaseSetupUnavailable",
    "admin_dsn_for",
    "apply_migrations",
    "disposable_database",
    "drop_database",
    "ensure_test_database",
    "plan_test_database",
    "prepare_test_database",
]
