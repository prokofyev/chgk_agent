"""Тесты веб-интерфейса: форма, состояния, результаты и генерация."""

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
from chgk_agent.observability.metrics import Metrics
from chgk_agent.search.local import LocalSearchOutcome, LocalSearchResult
from chgk_agent.ui.app import MOUNT_PATH, mount_ui
from chgk_agent.ui.client import SearchApiClient
from chgk_agent.ui.view import (
    MIN_QUERY_CHARS,
    SCORE_KIND_LABELS,
    SearchState,
    build_view,
    error_view,
    running_view,
    validate_form,
)


class FakeEmbeddingProvider:
    """Провайдер эмбеддингов-заглушка."""

    model = "fake"
    dimension = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeChatProvider:
    """Провайдер генерации-заглушка."""

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        return "Сгенерированный ответ"


class FakeExternalSource:
    """Внешний источник для проверки статусов в UI."""

    name = "gotquestions"
    enabled = True

    def __init__(self, result: ExternalSearchResult | None = None) -> None:
        self._result = result

    async def search(self, description: str, *, limit: int = 20) -> ExternalSearchResult:
        if self._result is not None:
            return self._result
        return ExternalSearchResult(
            status=ExternalStatus.OK,
            query=description,
            matches=[
                ExternalMatch(
                    title="В",
                    question_text="Внешний вопрос",
                    answer_text="Внешний ответ",
                    external_url="https://gotquestions.online/question/7",
                    external_id="7",
                    position=1,
                    score=0.5,
                )
            ],
        )


class _Session:
    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _SessionFactory:
    def __call__(self) -> _Session:
        return _Session()


def _local_outcome() -> LocalSearchOutcome:
    outcome = LocalSearchOutcome()
    outcome.matches = [
        LocalSearchResult(
            question_id=1,
            question_text="Локальный вопрос",
            answer_text="Локальный ответ",
            comment="Комментарий",
            score=0.8,
            score_kind="semantic",
            source_location="fixture.html",
        )
    ]
    outcome.semantic_ranks = {1: 1}
    return outcome


def _build_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    local: LocalSearchOutcome | None = None,
    external: FakeExternalSource | None = None,
    chat: FakeChatProvider | None = None,
    local_failure: bool = False,
) -> tuple[object, object]:
    async def fake_local(self, query, *, limit=20, min_score=0.0):
        if local_failure:
            raise RuntimeError("база недоступна")
        return local if local is not None else _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics", fake_local
    )
    deps = AppDeps(
        session_factory=_SessionFactory(),  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),
        settings=Settings(_env_file=None),
        external_source=external if external is not None else FakeExternalSource(),
        chat_provider=chat,
        metrics=Metrics(CollectorRegistry()),
    )
    app = create_app(deps, with_lifespan=False)
    return app, SearchApiClient(app)


def _build_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    local: LocalSearchOutcome | None = None,
    external: FakeExternalSource | None = None,
    chat: FakeChatProvider | None = None,
    local_failure: bool = False,
) -> httpx.AsyncClient:
    app, client = _build_app(
        monkeypatch,
        local=local,
        external=external,
        chat=chat,
        local_failure=local_failure,
    )
    mount_ui(app, client)  # type: ignore[arg-type]
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


def test_validate_form_blocks_empty_query() -> None:
    errors = validate_form("   ")

    assert errors.has_errors
    assert errors.query == "введите описание вопроса"


def test_validate_form_blocks_short_query() -> None:
    errors = validate_form("ок")

    assert errors.query is not None
    assert str(MIN_QUERY_CHARS) in errors.query


def test_validate_form_blocks_invalid_params() -> None:
    errors = validate_form("нормальное описание", limit=0, min_score=2.0)

    assert errors.limit is not None
    assert errors.min_score is not None


def test_validate_form_accepts_correct_input() -> None:
    errors = validate_form("нормальное описание", limit=10, min_score=0.5)

    assert errors.has_errors is False


def test_running_view_shows_progress_state() -> None:
    view = running_view()

    assert view.state is SearchState.RUNNING
    assert view.is_running
    assert view.message


def test_error_view_carries_request_id() -> None:
    view = error_view("внутренняя ошибка", request_id="abc123")

    assert view.state is SearchState.ERROR
    assert view.request_id == "abc123"
    assert view.error


def test_build_view_success_with_answer() -> None:
    payload = {
        "query": "Тьюринг",
        "matches": [
            {
                "question_text": "Вопрос про Тьюринга",
                "answer_text": "Ответ",
                "comment": None,
                "score": 0.87,
                "score_kind": "semantic",
                "sources": [
                    {
                        "name": "local",
                        "location": "f.html",
                        "score": 0.87,
                        "score_kind": "semantic",
                    },
                    {
                        "name": "gotquestions",
                        "external_url": "https://gotquestions.online/question/1",
                        "score": 0.5,
                        "score_kind": "semantic",
                    },
                ],
                "external_url": "https://gotquestions.online/question/1",
            }
        ],
        "sources": [
            {"source": "local", "status": "ok", "matches": 1, "truncated": False},
            {
                "source": "gotquestions",
                "status": "ok",
                "matches": 1,
                "truncated": True,
                "query": "Тьюринг",
            },
        ],
        "answer": {
            "text": "Ответ модели",
            "available": True,
            "used_matches": ["Вопрос про Тьюринга"],
        },
        "request_id": "req-1",
        "partial": False,
        "empty": False,
    }

    view = build_view(payload)

    assert view.state is SearchState.SUCCESS
    assert view.has_matches
    match = view.matches[0]
    assert match.score_percent == 87
    assert match.score_label == "семантическая близость"
    assert set(match.source_names) == {"local", "gotquestions"}
    assert match.is_external is True
    assert match.external_url
    assert view.sources[1].truncated is True
    assert view.answer is not None and view.answer.available
    assert view.answer.used_matches


def test_build_view_partial_results() -> None:
    payload = {
        "matches": [
            {
                "question_text": "Локальный",
                "answer_text": "Ответ",
                "score": 0.5,
                "score_kind": "lexical",
                "sources": [{"name": "local", "score": 0.5, "score_kind": "lexical"}],
            }
        ],
        "sources": [
            {"source": "local", "status": "ok", "matches": 1},
            {
                "source": "gotquestions",
                "status": "unavailable",
                "matches": 0,
                "error": "таймаут",
            },
        ],
        "partial": True,
        "empty": False,
    }

    view = build_view(payload)

    assert view.state is SearchState.PARTIAL
    assert view.sources[1].is_problem is True
    assert view.sources[1].status_label == "источник недоступен"
    assert view.message


def test_score_labels_describe_unified_value() -> None:
    """Подписи оценки не выдают позицию сайта за семантическую близость."""

    assert SCORE_KIND_LABELS["semantic"] == "семантическая близость"
    assert SCORE_KIND_LABELS["semantic+lexical"] == (
        "семантическая близость и точные слова"
    )
    assert SCORE_KIND_LABELS["lexical"] == "точные слова описания"
    assert "external_rank" not in SCORE_KIND_LABELS


def test_deduplicated_match_shows_both_sources_and_one_score() -> None:
    """Вопрос из двух источников показывается один раз с общей оценкой."""

    view = build_view(
        {
            "matches": [
                {
                    "question_text": "Один и тот же вопрос",
                    "answer_text": "Ответ",
                    "score": 0.83,
                    "score_kind": "semantic+lexical",
                    "external_url": "https://gotquestions.online/question/7",
                    "sources": [
                        {
                            "name": "local",
                            "location": "fixture.html",
                            "score": 0.83,
                            "score_kind": "semantic+lexical",
                        },
                        {
                            "name": "gotquestions",
                            "external_url": "https://gotquestions.online/question/7",
                            "position": 3,
                            "score": 0.83,
                            "score_kind": "semantic+lexical",
                        },
                    ],
                }
            ],
            "sources": [],
            "partial": False,
            "empty": False,
        }
    )

    assert len(view.matches) == 1
    match = view.matches[0]
    assert set(match.source_names) == {"local", "gotquestions"}
    assert match.score == 0.83
    assert match.score_percent == 83
    assert match.score_label == "семантическая близость и точные слова"


def test_build_view_distinguishes_rejected_from_empty() -> None:
    rejected = build_view(
        {
            "matches": [],
            "sources": [
                {"source": "gotquestions", "status": "rejected", "matches": 0},
            ],
            "partial": True,
            "empty": True,
        }
    )
    empty = build_view(
        {
            "matches": [],
            "sources": [{"source": "gotquestions", "status": "empty", "matches": 0}],
            "partial": False,
            "empty": True,
        }
    )

    assert rejected.sources[0].status_label == "запрос отвергнут источником"
    assert rejected.sources[0].is_problem is True
    assert empty.sources[0].status_label == "ничего не найдено"
    assert empty.sources[0].is_problem is False
    assert rejected.state is SearchState.EMPTY


def test_build_view_without_matches_suggests_reformulation() -> None:
    view = build_view({"matches": [], "sources": [], "partial": False, "empty": True})

    assert view.state is SearchState.EMPTY
    assert "Измените" in (view.message or "") or "измените" in (view.message or "")


def test_build_view_without_confirming_matches_hides_answer() -> None:
    payload = {
        "matches": [],
        "sources": [],
        "answer": {
            "text": None,
            "available": False,
            "error": "нет подтверждающих совпадений",
        },
        "empty": True,
    }

    view = build_view(payload)

    assert view.answer is not None
    assert view.answer.available is False
    assert "нельзя" in (view.answer.message or "")


async def test_ui_page_is_mounted(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _build_client(monkeypatch) as client:
        redirect = await client.get(MOUNT_PATH)
        page = await client.get(f"{MOUNT_PATH}/")

    assert redirect.status_code == 307
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]


async def test_ui_client_returns_success_view(monkeypatch: pytest.MonkeyPatch) -> None:
    _, api = _build_app(monkeypatch, chat=FakeChatProvider())
    view = await api.search("Тьюринг")

    assert view.state is SearchState.SUCCESS
    assert view.has_matches
    assert view.answer is not None and view.answer.available


async def test_ui_client_returns_partial_view(monkeypatch: pytest.MonkeyPatch) -> None:
    external = FakeExternalSource(
        ExternalSearchResult(
            status=ExternalStatus.UNAVAILABLE, query="Тьюринг", error="таймаут"
        )
    )
    _, api = _build_app(monkeypatch, external=external, chat=FakeChatProvider())
    view = await api.search("Тьюринг")

    assert view.state is SearchState.PARTIAL
    assert view.has_matches
    assert view.sources[1].status == "unavailable"


async def test_ui_client_shows_truncated_external_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = FakeExternalSource(
        ExternalSearchResult(
            status=ExternalStatus.OK,
            query="Ангус Бейтмен",
            queries=["Ангус Бейтмен"],
            truncated=True,
            matches=[
                ExternalMatch(
                    title="В",
                    question_text="Про Бейтмена",
                    answer_text="Ответ",
                    position=1,
                    score=1.0,
                )
            ],
        )
    )
    _, api = _build_app(monkeypatch, external=external)
    view = await api.search(
        "Английский учёный прошлого века Ангус Бейтмен давал ИМ клички"
    )

    assert view.truncated_query == "Ангус Бейтмен"
    assert view.sources[1].truncated is True
    assert view.sources[1].query == "Ангус Бейтмен"


async def test_ui_client_degrades_when_local_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, api = _build_app(monkeypatch, local_failure=True)
    view = await api.search("Тьюринг")

    assert view.state is SearchState.PARTIAL
    local = view.sources[0]
    assert local.status == "unavailable"
    assert local.is_problem is True


async def test_ui_client_handles_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, api = _build_app(monkeypatch)
    view = await api.search("   ")

    assert view.state is SearchState.ERROR
    assert view.request_id
