"""Тесты адаптеров GigaChat поверх langchain_gigachat."""

import asyncio

import pytest

from chgk_agent.config import GigaChatSettings
from chgk_agent.embeddings.base import ChatProvider, EmbeddingProvider
from chgk_agent.embeddings.gigachat import (
    GigaChatChatProvider,
    GigaChatEmbeddingProvider,
)


class _FakeEmbeddings:
    """Подмена GigaChatEmbeddings."""

    def __init__(self, *, dimension: int = 4, fail_times: int = 0) -> None:
        self.dimension = dimension
        self.fail_times = fail_times
        self.calls: list[list[str]] = []
        self.active = 0
        self.max_active = 0

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            if self.fail_times > 0:
                self.fail_times -= 1
                raise RuntimeError("GigaChat недоступен")
            self.calls.append(list(texts))
            return [[float(index)] * self.dimension for index, _ in enumerate(texts)]
        finally:
            self.active -= 1


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChat:
    """Подмена chat-модели GigaChat."""

    def __init__(self, content: str = "ответ", *, fail: bool = False) -> None:
        self.content = content
        self.fail = fail
        self.messages: list[list[tuple[str, str]]] = []

    async def ainvoke(self, messages: list[tuple[str, str]]) -> _FakeMessage:
        if self.fail:
            raise RuntimeError("GigaChat недоступен")
        self.messages.append(list(messages))
        return _FakeMessage(self.content)


def _settings(**overrides: object) -> GigaChatSettings:
    return GigaChatSettings(_env_file=None, **overrides)


async def test_embedding_provider_satisfies_protocol() -> None:
    provider = GigaChatEmbeddingProvider(_settings(), client=_FakeEmbeddings())

    assert isinstance(provider, EmbeddingProvider)
    assert provider.model == "Embeddings"
    assert provider.dimension == 1024


async def test_embedding_provider_returns_vectors() -> None:
    client = _FakeEmbeddings(dimension=3)
    provider = GigaChatEmbeddingProvider(_settings(), client=client)

    vectors = await provider.embed(["первый", "второй"])

    assert len(vectors) == 2
    assert all(len(vector) == 3 for vector in vectors)
    assert client.calls == [["первый", "второй"]]


async def test_embedding_provider_skips_empty_input() -> None:
    client = _FakeEmbeddings()
    provider = GigaChatEmbeddingProvider(_settings(), client=client)

    assert await provider.embed([]) == []
    assert client.calls == []


async def test_embedding_provider_batches_input() -> None:
    client = _FakeEmbeddings()
    provider = GigaChatEmbeddingProvider(_settings(), client=client)

    await provider.embed([f"текст {index}" for index in range(10)])

    assert client.calls == [[f"текст {index}" for index in range(10)]]


async def test_embedding_provider_limits_concurrency() -> None:
    client = _FakeEmbeddings()
    provider = GigaChatEmbeddingProvider(
        _settings(max_concurrency=1), client=client
    )

    await asyncio.gather(
        provider.embed(["a"]),
        provider.embed(["b"]),
        provider.embed(["c"]),
    )

    assert client.max_active == 1


async def test_embedding_provider_propagates_error() -> None:
    client = _FakeEmbeddings(fail_times=1)
    provider = GigaChatEmbeddingProvider(_settings(), client=client)

    with pytest.raises(RuntimeError, match="GigaChat недоступен"):
        await provider.embed(["текст"])

    vectors = await provider.embed(["текст"])
    assert len(vectors) == 1


async def test_chat_provider_satisfies_protocol() -> None:
    provider = GigaChatChatProvider(_settings(), client=_FakeChat())

    assert isinstance(provider, ChatProvider)
    assert provider.model == "GigaChat"


async def test_chat_provider_sends_system_and_human_messages() -> None:
    client = _FakeChat(content="ответ модели")
    provider = GigaChatChatProvider(_settings(), client=client)

    answer = await provider.complete("вопрос пользователя", system="системная роль")

    assert answer == "ответ модели"
    assert client.messages == [
        [("system", "системная роль"), ("human", "вопрос пользователя")]
    ]


async def test_chat_provider_without_system_message() -> None:
    client = _FakeChat()
    provider = GigaChatChatProvider(_settings(), client=client)

    await provider.complete("только вопрос")

    assert client.messages == [[("human", "только вопрос")]]


async def test_chat_provider_propagates_error() -> None:
    provider = GigaChatChatProvider(_settings(), client=_FakeChat(fail=True))

    with pytest.raises(RuntimeError, match="GigaChat недоступен"):
        await provider.complete("вопрос")


async def test_embedding_provider_records_usage_on_success(monkeypatch) -> None:
    import chgk_agent.embeddings.gigachat as module

    calls: list[str] = []
    monkeypatch.setattr(
        module, "record_call_success", lambda operation, **_: calls.append(operation)
    )
    provider = GigaChatEmbeddingProvider(_settings(), client=_FakeEmbeddings())

    await provider.embed(["текст"])

    assert calls == ["embedding"]


async def test_embedding_provider_records_error(monkeypatch) -> None:
    import chgk_agent.embeddings.gigachat as module

    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        module,
        "record_call_error",
        lambda operation, error=None, **_: recorded.append((operation, str(error))),
    )
    provider = GigaChatEmbeddingProvider(
        _settings(), client=_FakeEmbeddings(fail_times=1)
    )

    with pytest.raises(RuntimeError):
        await provider.embed(["текст"])

    assert recorded == [("embedding", "GigaChat недоступен")]


async def test_chat_provider_records_tokens(monkeypatch) -> None:
    import chgk_agent.embeddings.gigachat as module

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "record_call_success",
        lambda operation, **kwargs: captured.append(
            {"operation": operation, **kwargs}
        ),
    )

    class _FakeChatWithUsage(_FakeChat):
        async def ainvoke(self, messages):
            message = await super().ainvoke(messages)
            message.usage_metadata = {"total_tokens": 42}
            return message

    provider = GigaChatChatProvider(_settings(), client=_FakeChatWithUsage())

    await provider.complete("вопрос")

    assert captured == [
        {"operation": "generation", "usage": {"total_tokens": 42}}
    ]


async def test_chat_provider_records_error(monkeypatch) -> None:
    import chgk_agent.embeddings.gigachat as module

    recorded: list[str] = []
    monkeypatch.setattr(
        module,
        "record_call_error",
        lambda operation, error=None, **_: recorded.append(operation),
    )
    provider = GigaChatChatProvider(_settings(), client=_FakeChat(fail=True))

    with pytest.raises(RuntimeError):
        await provider.complete("вопрос")

    assert recorded == ["generation"]
