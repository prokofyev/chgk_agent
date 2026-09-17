"""Проверка согласованности размерности эмбеддингов со схемой БД."""

import re
from typing import TYPE_CHECKING

from sqlalchemy import text

from chgk_agent.logging_setup import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.config import Settings, get_settings
from chgk_agent.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

_EMBEDDING_COLUMN_TYPE = text(
    "SELECT format_type(a.atttypid, a.atttypmod) "
    "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
    "WHERE c.relname = 'questions' AND a.attname = 'embedding'"
)


class EmbeddingDimensionMismatchError(RuntimeError):
    """Размерность вектора провайдера не совпадает со схемой БД."""


def verify_embedding_dimension(
    provider: EmbeddingProvider,
    settings: Settings | None = None,
) -> None:
    """Убедиться, что размерность провайдера совпадает с настройкой схемы.

    Raises:
        EmbeddingDimensionMismatchError: при несовпадении размерностей.
    """

    current = settings or get_settings()
    expected = current.gigachat.embedding_dim
    actual = provider.dimension

    if actual != expected:
        raise EmbeddingDimensionMismatchError(
            "Размерность эмбеддингов не совпадает со схемой базы: "
            f"провайдер {provider.model!r} возвращает {actual}, "
            f"а колонка questions.embedding ожидает vector({expected}). "
            "Измените CHGK_GIGACHAT__EMBEDDING_DIM и примените миграцию "
            "либо укажите другую модель эмбеддингов."
        )


async def verify_schema_embedding_dimension(
    session_factory: "async_sessionmaker[AsyncSession]",
    settings: Settings | None = None,
) -> int | None:
    """Сверить тип колонки `questions.embedding` с настроенной размерностью.

    Размерность вектора берётся из конфигурации, поэтому смена
    `CHGK_GIGACHAT__EMBEDDING_DIM` без миграции оставляет схему с колонкой
    старой размерности. Проверка ловит это при старте сервиса, до того как
    поиск начнёт возвращать ошибки.

    Returns:
        Фактическая размерность колонки или `None`, если колонки нет
        (например, миграции ещё не применены).

    Raises:
        EmbeddingDimensionMismatchError: при несовпадении размерностей.
    """

    current = settings or get_settings()
    expected = current.gigachat.embedding_dim

    try:
        async with session_factory() as session:
            result = await session.execute(_EMBEDDING_COLUMN_TYPE)
            column_type = result.scalar_one_or_none()
    except Exception as error:  # недоступная база не должна блокировать старт
        logger.warning(
            "не удалось сверить размерность эмбеддингов со схемой базы",
            error=str(error),
        )
        return None

    if column_type is None:
        logger.warning("колонка questions.embedding не найдена")
        return None

    actual = _parse_vector_dimension(str(column_type))
    if actual is not None and actual != expected:
        raise EmbeddingDimensionMismatchError(
            "Размерность эмбеддингов не совпадает со схемой базы: "
            f"колонка questions.embedding объявлена как vector({actual}), "
            f"а настройка CHGK_GIGACHAT__EMBEDDING_DIM равна {expected}. "
            "Примените миграцию под новую размерность или верните прежнее "
            "значение настройки."
        )
    return actual


def _parse_vector_dimension(column_type: str) -> int | None:
    """Извлечь размерность из строки вида `vector(1024)`."""

    match = re.fullmatch(r"vector\s*\(\s*(\d+)\s*\)", column_type.strip())
    if match is None:
        return None
    return int(match.group(1))


__all__ = [
    "EmbeddingDimensionMismatchError",
    "verify_embedding_dimension",
    "verify_schema_embedding_dimension",
]
