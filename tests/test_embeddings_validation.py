"""Тесты проверки размерности эмбеддингов и её согласованности со схемой."""

from typing import Any

import pytest

from chgk_agent.config import Settings
from chgk_agent.embeddings.base import ChatProvider, EmbeddingProvider
from chgk_agent.embeddings.validation import (
    EmbeddingDimensionMismatchError,
    verify_embedding_dimension,
    verify_schema_embedding_dimension,
)


class _FakeEmbeddingProvider:
    def __init__(self, dimension: int, model: str = "FakeEmbeddings") -> None:
        self._dimension = dimension
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimension for _ in texts]


def test_provider_satisfies_protocol() -> None:
    provider = _FakeEmbeddingProvider(dimension=1024)

    assert isinstance(provider, EmbeddingProvider)


def test_matching_dimension_passes() -> None:
    settings = Settings(_env_file=None)
    provider = _FakeEmbeddingProvider(dimension=settings.gigachat.embedding_dim)

    verify_embedding_dimension(provider, settings)


def test_mismatched_dimension_reports_actionable_error() -> None:
    settings = Settings(_env_file=None)
    provider = _FakeEmbeddingProvider(dimension=settings.gigachat.embedding_dim + 1)

    with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
        verify_embedding_dimension(provider, settings)

    message = str(excinfo.value)
    assert str(provider.dimension) in message
    assert f"vector({settings.gigachat.embedding_dim})" in message
    assert "CHGK_GIGACHAT__EMBEDDING_DIM" in message


def test_chat_provider_protocol_is_usable() -> None:
    class _FakeChat:
        async def complete(self, prompt: str, *, system: str | None = None) -> str:
            return f"{system}:{prompt}"

    assert isinstance(_FakeChat(), ChatProvider)

class _SchemaSession:
    """Сессия-заглушка, отдающая заданный тип колонки или ошибку."""

    def __init__(self, column_type: str | None = None, error: Exception | None = None) -> None:
        self._column_type = column_type
        self._error = error

    async def __aenter__(self) -> "_SchemaSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, statement: Any) -> Any:
        if self._error is not None:
            raise self._error

        column_type = self._column_type

        class _Result:
            def scalar_one_or_none(self) -> str | None:
                return column_type

        return _Result()


def _factory(session: _SchemaSession):
    return lambda: session


async def test_schema_dimension_matches_settings() -> None:
    settings = Settings(_env_file=None)
    session = _SchemaSession(f"vector({settings.gigachat.embedding_dim})")

    actual = await verify_schema_embedding_dimension(_factory(session), settings)

    assert actual == settings.gigachat.embedding_dim


async def test_schema_dimension_mismatch_is_reported() -> None:
    """Проверка ловит расхождение настройки и типа колонки в схеме."""

    settings = Settings(_env_file=None)
    session = _SchemaSession("vector(384)")

    with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
        await verify_schema_embedding_dimension(_factory(session), settings)

    message = str(excinfo.value)
    assert "vector(384)" in message
    assert str(settings.gigachat.embedding_dim) in message


async def test_missing_embedding_column_is_not_an_error() -> None:
    """Неприменённые миграции не должны блокировать старт сервиса."""

    settings = Settings(_env_file=None)

    actual = await verify_schema_embedding_dimension(_factory(_SchemaSession(None)), settings)

    assert actual is None


async def test_unavailable_database_is_not_an_error() -> None:
    """Недоступная база не должна валить старт до проверки готовности."""

    settings = Settings(_env_file=None)
    session = _SchemaSession(error=RuntimeError("connection refused"))

    actual = await verify_schema_embedding_dimension(_factory(session), settings)

    assert actual is None


def test_vector_dimension_parser() -> None:
    from chgk_agent.embeddings.validation import _parse_vector_dimension

    assert _parse_vector_dimension("vector(1024)") == 1024
    assert _parse_vector_dimension("halfvec(1024)") is None
    assert _parse_vector_dimension("vector") is None
