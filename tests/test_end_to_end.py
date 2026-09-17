"""Сквозные интеграционные тесты: импорт, поиск, генерация, внешний источник."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from prometheus_client import CollectorRegistry
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.api.app import create_app
from chgk_agent.api.deps import AppDeps
from chgk_agent.config import ExternalSourceSettings, Settings
from chgk_agent.db.base import Question, Source, get_embedding_dim
from chgk_agent.external.base import ExternalSearchResult, ExternalStatus
from chgk_agent.external.gotquestions import GotQuestionsSource
from chgk_agent.ingestion.service import IngestionService
from chgk_agent.observability.metrics import Metrics
from chgk_agent.ui.client import SearchApiClient
from chgk_agent.ui.view import SearchState

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "telegram_export.html"
EXTERNAL_FIXTURES = Path(__file__).parent / "fixtures" / "external"
LOCATION = str(FIXTURE)
BASE_URL = "https://gotquestions.online"
USER_AGENT = "chgk-agent/0.1 (+https://localhost/chgk-agent)"
DIMENSION = get_embedding_dim()


class _KeywordEmbedder:
    """Детерминированные эмбеддинги по словарю терминов."""

    VOCABULARY = (
        "тьюринг",
        "машина",
        "порошок",
        "пороховница",
        "гоголь",
        "жираф",
        "камелопард",
        "многострочный",
    )

    @property
    def model(self) -> str:
        return "KeywordEmbeddings"

    @property
    def dimension(self) -> int:
        return DIMENSION

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        lowered = text.casefold()
        vector = [0.01] * DIMENSION
        for index, term in enumerate(self.VOCABULARY):
            if term in lowered:
                vector[index] = 1.0
        return vector


class FakeChatProvider:
    """Фейковый GigaChat: считает вызовы и возвращает фиксированный ответ."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._fail = fail

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(prompt)
        if self._fail:
            raise RuntimeError("GigaChat недоступен")
        return "Похоже, ответ — Тарас Бульба."


class OfflineExternalSource:
    """Внешний источник, полностью отключённый от сети."""

    name = "gotquestions"
    enabled = True

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, description: str, *, limit: int = 20) -> ExternalSearchResult:
        self.queries.append(description)
        return ExternalSearchResult(
            status=ExternalStatus.UNAVAILABLE,
            query=description,
            error="внешний источник отключён в тесте",
        )


class UnavailableTransport(httpx.AsyncBaseTransport):
    """Транспорт, эмулирующий недоступность внешнего сайта."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise self._error


class FixtureTransport(httpx.AsyncBaseTransport):
    """Транспорт, отдающий сохранённый HTML выдачи и запоминающий запросы."""

    def __init__(self, html: str, *, status_code: int = 200) -> None:
        self._html = html
        self._status_code = status_code
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self._status_code, text=self._html)


def _external_source(
    transport: httpx.AsyncBaseTransport,
    metrics: Metrics,
) -> GotQuestionsSource:
    """Собрать настоящий адаптер поверх подменённого транспорта.

    Rate limiting и ретраи выключены: в тестах они дают только паузы.
    """

    settings = ExternalSourceSettings(
        _env_file=None,
        min_interval_seconds=0.0,
        max_attempts=1,
    )
    client = httpx.AsyncClient(
        transport=transport,
        base_url=BASE_URL,
        headers={"User-Agent": USER_AGENT},
    )
    return GotQuestionsSource(settings, client=client, metrics=metrics)


def _fixture(name: str) -> str:
    """Прочитать сохранённую страницу выдачи внешнего источника."""

    return (EXTERNAL_FIXTURES / name).read_text(encoding="utf-8")


async def _purge(session_factory: async_sessionmaker[AsyncSession]) -> None:
    async with session_factory() as session:
        await session.execute(delete(Source).where(Source.location == LOCATION))
        await session.execute(delete(Question).where(~Question.occurrences.any()))
        await session.commit()


@pytest.fixture
async def clean_data(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    """Чистое состояние до и после сквозного теста."""

    await _purge(session_factory)
    try:
        yield
    finally:
        await _purge(session_factory)


@pytest.fixture
async def seeded(
    session_factory: async_sessionmaker[AsyncSession],
    clean_data: None,
) -> Metrics:
    """Импортировать HTML-фикстуру с векторизацией и вернуть метрики."""

    metrics = Metrics(CollectorRegistry())
    service = IngestionService(
        session_factory,
        embedding_provider=_KeywordEmbedder(),
        metrics=metrics,
    )
    report = await service.run_import([FIXTURE])
    assert report.processed > 0
    assert report.unembedded == 0
    return metrics


def _build_client(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    external: object | None = None,
    chat: FakeChatProvider | None = None,
    metrics: Metrics | None = None,
) -> httpx.AsyncClient:
    """Собрать тестовый клиент на реальной базе и фейковых провайдерах."""

    deps = AppDeps(
        session_factory=session_factory,
        embedding_provider=_KeywordEmbedder(),
        settings=Settings(_env_file=None),
        external_source=external if external is not None else OfflineExternalSource(),
        chat_provider=chat,
        metrics=metrics,
    )
    app = create_app(deps, with_lifespan=False)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


async def test_end_to_end_import_search_and_generate(
    session_factory: async_sessionmaker[AsyncSession],
    seeded: Metrics,
) -> None:
    chat = FakeChatProvider()
    async with _build_client(session_factory, chat=chat) as client:
        response = await client.post(
            "/search",
            json={
                "query": "В них хранилась доза порошка, но остался ли он ещё?",
                "limit": 5,
                "generate_answer": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    local = [report for report in payload["sources"] if report["source"] == "local"]
    assert local and local[0]["status"] == "ok"
    assert payload["answer"]["available"] is True
    assert payload["answer"]["text"]
    assert payload["answer"]["used_matches"]
    assert chat.calls


async def test_end_to_end_generation_failure_keeps_matches(
    session_factory: async_sessionmaker[AsyncSession],
    seeded: Metrics,
) -> None:
    chat = FakeChatProvider(fail=True)
    async with _build_client(session_factory, chat=chat) as client:
        response = await client.post(
            "/search",
            json={"query": "доза порошка в пороховнице", "limit": 5},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert payload["answer"]["available"] is False
    assert payload["answer"]["text"] is None


async def test_end_to_end_external_unavailable_returns_local_matches(
    session_factory: async_sessionmaker[AsyncSession],
    seeded: Metrics,
) -> None:
    metrics = Metrics(CollectorRegistry())
    external = _external_source(
        UnavailableTransport(httpx.ConnectTimeout("таймаут")), metrics
    )
    async with _build_client(
        session_factory, external=external, metrics=metrics
    ) as client:
        response = await client.post(
            "/search", json={"query": "доза порошка в пороховнице", "limit": 5}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert payload["partial"] is True
    statuses = {item["source"]: item["status"] for item in payload["sources"]}
    assert statuses["gotquestions"] == "unavailable"
    assert (
        metrics.search.requests.labels(source="gotquestions", status="unavailable")._value.get()
        == 1
    )
    assert metrics.search.partial.labels(source="gotquestions")._value.get() == 1
    assert (
        metrics.search.errors.labels(
            source="gotquestions", error_type="timeout"
        )._value.get()
        == 1
    )


async def test_end_to_end_empty_and_rejected_are_distinct(
    session_factory: async_sessionmaker[AsyncSession],
    seeded: Metrics,
) -> None:
    metrics = Metrics(CollectorRegistry())
    for fixture, expected in (
        ("search_empty.html", "empty"),
        ("search_rejected.html", "rejected"),
    ):
        transport = FixtureTransport(_fixture(fixture))
        external = _external_source(transport, metrics)
        async with _build_client(
            session_factory, external=external, metrics=metrics
        ) as client:
            response = await client.post(
                "/search", json={"query": "порошок", "limit": 5}
            )

        assert response.status_code == 200
        payload = response.json()
        statuses = {item["source"]: item["status"] for item in payload["sources"]}
        assert statuses["gotquestions"] == expected

    assert metrics.search.external_empty._value.get() == 1
    assert metrics.search.external_rejected._value.get() == 1


async def test_end_to_end_long_description_uses_short_external_query(
    session_factory: async_sessionmaker[AsyncSession],
    seeded: Metrics,
) -> None:
    long_description = (
        "В докладе 1947 года этот человек предложил использовать две комнаты, "
        "двух не очень сильных игроков в шахматы и оператора. Назовите этого человека."
    )
    transport = FixtureTransport(_fixture("search_ok.html"))
    external = _external_source(transport, Metrics(CollectorRegistry()))

    async with _build_client(session_factory, external=external) as client:
        response = await client.post(
            "/search", json={"query": long_description, "limit": 5}
        )

    assert response.status_code == 200
    payload = response.json()
    # Локальный поиск получил полное описание.
    assert payload["query"] == long_description
    external_report = next(
        item for item in payload["sources"] if item["source"] == "gotquestions"
    )
    assert external_report["truncated"] is True
    assert external_report["queries"]
    assert all(len(query) <= 50 for query in external_report["queries"])
    # Внешний источник получил только короткий запрос.
    sent = [request.url.params.get("search", "") for request in transport.requests]
    assert sent
    assert all(len(query) <= 50 for query in sent)


async def test_end_to_end_ui_shows_matches_and_answer(
    session_factory: async_sessionmaker[AsyncSession],
    seeded: Metrics,
) -> None:
    deps = AppDeps(
        session_factory=session_factory,
        embedding_provider=_KeywordEmbedder(),
        settings=Settings(_env_file=None),
        external_source=_external_source(
            FixtureTransport(_fixture("search_ok.html")), Metrics(CollectorRegistry())
        ),
        chat_provider=FakeChatProvider(),
    )
    app = create_app(deps, with_lifespan=False)
    api = SearchApiClient(app)

    view = await api.search("доза порошка в пороховнице", limit=5)

    assert view.state is SearchState.SUCCESS
    assert view.has_matches
    assert view.answer is not None and view.answer.available
    assert any(match.is_external for match in view.matches)
