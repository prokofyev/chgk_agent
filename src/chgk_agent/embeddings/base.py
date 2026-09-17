"""Интерфейсы провайдеров эмбеддингов и генерации ответа."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Провайдер векторных представлений текста."""

    @property
    def model(self) -> str:
        """Имя модели эмбеддингов."""

    @property
    def dimension(self) -> int:
        """Размерность вектора."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Построить эмбеддинги для списка текстов."""


@runtime_checkable
class ChatProvider(Protocol):
    """Провайдер генерации ответа."""

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Сгенерировать ответ по промпту."""
