"""Тесты HTTP API: поиск, импорт, служебные проверки и ошибки."""


import httpx
import pytest
from prometheus_client import CollectorRegistry

from chgk_agent.api.app import create_app
from chgk_agent.api.deps import AppDeps
from chgk_agent.config import Settings
from chgk_agent.external.base import (
    ExternalMatch,
    ExternalSearchResult,
    ExternalStatus,
)
from chgk_agent.ingestion.operations import ImportOperationRegistry, OperationStatus
from chgk_agent.ingestion.service import IngestionService
from chgk_agent.observability.metrics import Metrics
from chgk_agent.search.local import LocalSearchOutcome, LocalSearchResult


class FakeEmbeddingProvider:
    """Провайдер эмбеддингов с фиксированным вектором."""

    model = "fake"
    dimension = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeChatProvider:
    """Провайдер генерации с настраиваемым ответом."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self._fail = fail

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        if self._fail:
            raise RuntimeError("GigaChat недоступен")
        return "Сгенерированный ответ"


class FakeExternalSource:
    """Внешний источник для тестов API."""

    name = "gotquestions"

    def __init__(
        self,
        result: ExternalSearchResult | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._result = result
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def search(
        self,
        description: str,
        *,
        limit: int = 20,
        term_weights: dict[str, float] | None = None,
    ) -> ExternalSearchResult:
        if self._result is not None:
            return self._result
        return ExternalSearchResult(
            status=ExternalStatus.OK,
            query=description,
            matches=[
                ExternalMatch(
                    title="Вопрос",
                    question_text="Внешний вопрос про Тьюринга",
                    answer_text="Внешний ответ",
                    external_url="https://gotquestions.online/question/42",
                    external_id="42",
                    position=1,
                    score=1.0,
                )
            ],
        )


class _Session:
    """Пустая сессия: локальный поиск подменяется в тестах."""

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("сессия не настроена")


class _SessionFactory:
    def __call__(self) -> _Session:
        return _Session()


def _deps(
    *,
    external: FakeExternalSource | None = None,
    chat: FakeChatProvider | None = None,
    ingestion: IngestionService | None = None,
) -> AppDeps:
    metrics = Metrics(CollectorRegistry())
    return AppDeps(
        session_factory=_SessionFactory(),  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),
        settings=Settings(_env_file=None),
        external_source=external if external is not None else FakeExternalSource(),
        chat_provider=chat,
        ingestion=ingestion,
        metrics=metrics,
    )


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch):
    """Фабрика тестовых клиентов с подменённым локальным поиском."""

    def build(
        *,
        local: LocalSearchOutcome | None = None,
        external: FakeExternalSource | None = None,
        chat: FakeChatProvider | None = None,
        ingestion: IngestionService | None = None,
        database_ok: bool = True,
        external_ok: bool = True,
    ) -> httpx.AsyncClient:
        async def fake_local(self, query, *, limit=20, min_score=0.0):
            return local if local is not None else _local_outcome()

        monkeypatch.setattr(
            "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
            fake_local,
        )
        deps = _deps(external=external, chat=chat, ingestion=ingestion)

        async def fake_check_database(timeout_seconds: float = 3.0):
            return (database_ok, None if database_ok else "connection refused")

        async def fake_check_external(timeout_seconds: float = 3.0):
            return (external_ok, None if external_ok else "внешний источник недоступен")

        monkeypatch.setattr(AppDeps, "check_database", fake_check_database)
        monkeypatch.setattr(AppDeps, "check_external", fake_check_external)

        app = create_app(deps, with_lifespan=False)
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        )

    return build


def _local_outcome() -> LocalSearchOutcome:
    outcome = LocalSearchOutcome()
    outcome.matches = [
        LocalSearchResult(
            question_id=1,
            question_text="Локальный вопрос про Тьюринга",
            answer_text="Локальный ответ",
            comment=None,
            score=0.9,
            score_kind="semantic",
            source_location="fixture.html",
        )
    ]
    outcome.semantic_ranks = {1: 1}
    return outcome


async def test_search_returns_structured_result(client_factory) -> None:
    async with client_factory(chat=FakeChatProvider()) as client:
        response = await client.post("/search", json={"query": "Тьюринг", "limit": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "Тьюринг"
    assert payload["matches"]
    assert {source["source"] for source in payload["sources"]} == {
        "local",
        "gotquestions",
    }
    assert payload["request_id"]
    assert response.headers["X-Request-ID"] == payload["request_id"]


async def test_search_rejects_empty_query(client_factory) -> None:
    async with client_factory() as client:
        response = await client.post("/search", json={"query": "   "})

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["request_id"]


async def test_search_rejects_too_long_query(client_factory) -> None:
    async with client_factory() as client:
        response = await client.post("/search", json={"query": "а" * 2001})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_search_rejects_invalid_threshold(client_factory) -> None:
    async with client_factory() as client:
        response = await client.post(
            "/search", json={"query": "Тьюринг", "min_score": 1.5}
        )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_rejected_external_source_reported_as_rejected(client_factory) -> None:
    external = FakeExternalSource(
        ExternalSearchResult(status=ExternalStatus.REJECTED, query="Тьюринг")
    )
    async with client_factory(external=external, chat=FakeChatProvider()) as client:
        response = await client.post("/search", json={"query": "Тьюринг"})

    assert response.status_code == 200
    statuses = {item["source"]: item["status"] for item in response.json()["sources"]}
    assert statuses["gotquestions"] == "rejected"
    assert response.json()["partial"] is True


async def test_search_reports_shared_score_kind_for_both_sources(
    client_factory,
) -> None:
    """`score_kind` описывает состав оценки, а не позицию сайта."""

    async with client_factory(chat=FakeChatProvider()) as client:
        response = await client.post("/search", json={"query": "Тьюринг"})

    payload = response.json()
    kinds = {match["score_kind"] for match in payload["matches"]}

    assert kinds
    assert kinds <= {"semantic", "semantic+lexical", "lexical", "none"}
    assert all(match["score_kind"] != "external_rank" for match in payload["matches"])
    assert all(
        ref["score_kind"] != "external_rank"
        for match in payload["matches"]
        for ref in match["sources"]
    )


async def test_search_handles_internal_error_in_unified_envelope(
    client_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with client_factory() as client:
        monkeypatch.setattr(
            "chgk_agent.api.deps.build_search_graph",
            lambda deps: _ExplodingGraph(),  # type: ignore[arg-type]
        )
        response = await client.post("/search", json={"query": "Тьюринг"})

    assert response.status_code == 500
    payload = response.json()
    assert payload["code"] == "internal_error"
    assert payload["message"] == "внутренняя ошибка сервиса"
    assert "Traceback" not in response.text
    assert payload["request_id"]


class _ExplodingGraph:
    async def run(self, *args: object, **kwargs: object):
        raise RuntimeError("внутренняя поломка с деталями")


async def test_healthz_does_not_touch_dependencies(client_factory) -> None:
    async with client_factory(database_ok=False, external_ok=False) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readyz_ok_when_dependencies_available(client_factory) -> None:
    async with client_factory() as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert {item["name"] for item in payload["dependencies"]} == {
        "database",
        "external_source",
    }


async def test_readyz_503_when_database_unavailable(client_factory) -> None:
    async with client_factory(database_ok=False) as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unavailable"
    database = next(
        item for item in payload["dependencies"] if item["name"] == "database"
    )
    assert database["status"] == "unavailable"
    assert database["detail"]


async def test_readyz_503_when_external_unavailable(client_factory) -> None:
    async with client_factory(external_ok=False) as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    external = next(
        item
        for item in response.json()["dependencies"]
        if item["name"] == "external_source"
    )
    assert external["status"] == "unavailable"


async def test_import_lifecycle(client_factory, tmp_path) -> None:
    html = tmp_path / "source.html"
    html.write_text(
        "<html><body><div>Вопрос 1. Что такое ЧГК?</div>"
        "<div>Ответ: игра.</div></body></html>",
        encoding="utf-8",
    )
    registry = ImportOperationRegistry()
    ingestion = IngestionService(
        _SessionFactory(),  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),
        registry=registry,
    )
    await registry.register("op-1", locations=[str(html)])
    await registry.finish(
        "op-1",
        type(
            "R",
            (),
            {
                "operation_id": "op-1",
                "locations": [str(html)],
                "started_at": None,
                "finished_at": None,
                "processed": 3,
                "added": 2,
                "updated": 1,
                "unchanged": 0,
                "skipped": 0,
                "unembedded": 0,
                "issues": [],
                "is_partial": False,
            },
        )(),
    )

    async with client_factory(ingestion=ingestion) as client:
        response = await client.get("/imports/op-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["operation_id"] == "op-1"
    assert payload["status"] == OperationStatus.COMPLETED.value
    assert payload["processed"] == 3


async def test_import_unknown_operation_returns_404(client_factory) -> None:
    registry = ImportOperationRegistry()
    ingestion = IngestionService(
        _SessionFactory(),  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),
        registry=registry,
    )

    async with client_factory(ingestion=ingestion) as client:
        response = await client.get("/imports/unknown")

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "not_found"
    assert payload["request_id"]


async def test_metrics_endpoint_exposes_search_counters(client_factory) -> None:
    async with client_factory(chat=FakeChatProvider()) as client:
        await client.post("/search", json={"query": "Тьюринг"})
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert "chgk_search_requests_total" in response.text
    assert "chgk_external_query_truncated_total" in response.text
