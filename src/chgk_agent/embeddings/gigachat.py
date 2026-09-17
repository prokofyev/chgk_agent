"""Адаптеры GigaChat поверх langchain_gigachat."""

import asyncio

from langchain_gigachat import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

from chgk_agent.config import GigaChatSettings
from chgk_agent.embeddings.usage import (
    EMBEDDING_OPERATION,
    GENERATION_OPERATION,
    record_call_error,
    record_call_success,
)
from chgk_agent.logging_setup import get_logger

logger = get_logger(__name__)


def _usage_metadata(response: object) -> dict[str, object] | None:
    """Извлечь сведения о токенах из ответа chat-модели."""

    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return dict(usage)
    return None


def _client_kwargs(settings: GigaChatSettings) -> dict[str, object]:
    """Собрать общие параметры клиентов GigaChat."""

    kwargs: dict[str, object] = {
        "scope": settings.scope,
        "timeout": settings.timeout_seconds,
        "verify_ssl_certs": settings.verify_ssl,
        "max_retries": settings.max_retries,
    }
    credentials = settings.credentials.get_secret_value()
    if credentials:
        kwargs["credentials"] = credentials
    return kwargs


class GigaChatEmbeddingProvider:
    """Провайдер эмбеддингов на базе GigaChatEmbeddings."""

    def __init__(
        self,
        settings: GigaChatSettings,
        *,
        client: GigaChatEmbeddings | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or GigaChatEmbeddings(
            model=settings.embedding_model,
            **_client_kwargs(settings),
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    @property
    def model(self) -> str:
        """Имя модели эмбеддингов."""

        return self._settings.embedding_model

    @property
    def dimension(self) -> int:
        """Ожидаемая размерность вектора."""

        return self._settings.embedding_dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Построить эмбеддинги для списка текстов.

        Пустой ввод не приводит к обращению к GigaChat.
        """

        if not texts:
            return []

        try:
            async with self._semaphore:
                vectors = await self._client.aembed_documents(texts)
        except Exception as error:
            record_call_error(EMBEDDING_OPERATION, error)
            logger.warning(
                "ошибка эмбеддингов GigaChat",
                operation=EMBEDDING_OPERATION,
                items=len(texts),
                error=str(error),
            )
            raise

        record_call_success(EMBEDDING_OPERATION)
        return [list(vector) for vector in vectors]


class GigaChatChatProvider:
    """Провайдер генерации ответа на базе chat-модели GigaChat."""

    def __init__(
        self,
        settings: GigaChatSettings,
        *,
        client: GigaChat | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or GigaChat(
            model=settings.model,
            **_client_kwargs(settings),
        )

    @property
    def model(self) -> str:
        """Имя chat-модели."""

        return self._settings.model

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Сгенерировать ответ по промпту."""

        messages: list[tuple[str, str]] = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))

        try:
            response = await self._client.ainvoke(messages)
        except Exception as error:
            record_call_error(GENERATION_OPERATION, error)
            logger.warning(
                "ошибка генерации GigaChat",
                operation=GENERATION_OPERATION,
                error=str(error),
            )
            raise

        record_call_success(GENERATION_OPERATION, usage=_usage_metadata(response))

        content = response.content
        if isinstance(content, list):
            return "".join(str(part) for part in content)
        return str(content)
