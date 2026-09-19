"""Тесты веб-интерфейса: поле вопроса, кнопка «Ответить» и ответ."""

from collections.abc import Iterator

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
from chgk_agent.ui.app import (
    BUTTON_LABEL,
    MOUNT_PATH,
    TITLE,
    answer_for,
    is_submittable,
    mount_ui,
)
from chgk_agent.ui.client import SearchApiClient
from chgk_agent.ui.view import (
    MIN_QUERY_CHARS,
    UNKNOWN_ANSWER,
    SearchView,
    build_view,
    unknown_view,
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

    def __init__(self, text: str = "Сгенерированный ответ", *, fail: bool = False) -> None:
        self._text = text
        self._fail = fail
        self.calls: list[str] = []

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(prompt)
        if self._fail:
            raise RuntimeError("генерация недоступна")
        return self._text


class FakeExternalSource:
    """Внешний источник для проверки выдачи."""

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


def _empty_outcome() -> LocalSearchOutcome:
    outcome = LocalSearchOutcome()
    outcome.matches = []
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


@pytest.fixture(scope="module")
def mounted_client() -> Iterator[httpx.AsyncClient]:
    """Смонтировать интерфейс с заглушками один раз на модуль.

    NiceGUI монтируется в глобальное приложение и после первого запроса
    запрещает добавлять middleware, поэтому повторное монтирование в
    рамках процесса невозможно.
    """

    patcher = pytest.MonkeyPatch()
    client = _build_client(patcher)
    yield client
    patcher.undo()


def test_validate_form_blocks_empty_query() -> None:
    errors = validate_form("   ")

    assert errors.has_errors
    assert errors.query == "введите описание вопроса"


def test_validate_form_blocks_short_query() -> None:
    errors = validate_form("ок")

    assert errors.query is not None
    assert str(MIN_QUERY_CHARS) in errors.query


def test_validate_form_accepts_correct_input() -> None:
    errors = validate_form("нормальное описание")

    assert errors.has_errors is False


def test_is_submittable_matches_validation() -> None:
    assert is_submittable("галстук") is True
    assert is_submittable("  ") is False
    assert is_submittable("ок") is False


async def test_answer_for_rejects_short_query_without_search() -> None:
    """Обход блокировки кнопки не приводит к обращению к поиску."""

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def search(
            self, query: str, *, limit: int = 20, min_score: float | None = None
        ) -> SearchView:
            self.calls.append(query)
            return SearchView(answer_text="Ответ модели")

    client = RecordingClient()

    view = await answer_for("ок", client)  # type: ignore[arg-type]

    assert view.answer_text == UNKNOWN_ANSWER
    assert client.calls == []


async def test_answer_for_passes_valid_query_to_search() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, float | None]] = []

        async def search(
            self, query: str, *, limit: int = 20, min_score: float | None = None
        ) -> SearchView:
            self.calls.append((query, limit, min_score))
            return SearchView(answer_text="Ответ модели")

    client = RecordingClient()

    view = await answer_for("галстук", client, limit=5, min_score=0.85)  # type: ignore[arg-type]

    assert view.answer_text == "Ответ модели"
    assert client.calls == [("галстук", 5, 0.85)]


def test_unknown_view_is_unknown() -> None:
    view = unknown_view(request_id="abc123")

    assert view.answer_text == UNKNOWN_ANSWER
    assert view.is_unknown
    assert view.request_id == "abc123"


def test_build_view_returns_generated_answer() -> None:
    view = build_view(
        {
            "query": "Тьюринг",
            "matches": [{"question_text": "Вопрос", "answer_text": "Ответ"}],
            "sources": [{"source": "local", "status": "ok", "matches": 1}],
            "answer": {
                "text": "Ответ модели",
                "available": True,
                "used_matches": ["Вопрос"],
            },
            "request_id": "req-1",
        }
    )

    assert view.answer_text == "Ответ модели"
    assert view.is_unknown is False


def test_build_view_ignores_matches_without_answer() -> None:
    """Выдача без сгенерированного ответа не показывается пользователю."""

    view = build_view(
        {
            "matches": [{"question_text": "Вопрос", "answer_text": "Ответ"}],
            "sources": [],
            "answer": {"text": None, "available": False},
            "empty": True,
        }
    )

    assert view.answer_text == UNKNOWN_ANSWER
    assert view.is_unknown


def test_build_view_without_answer_block_returns_unknown() -> None:
    view = build_view({"matches": [], "sources": []})

    assert view.is_unknown


def test_build_view_ignores_blank_answer_text() -> None:
    view = build_view({"answer": {"text": "   ", "available": True}})

    assert view.is_unknown


def test_view_does_not_expose_matches_or_sources() -> None:
    """На экране нет ни вопросов, ни оценок, ни источников."""

    view = build_view(
        {
            "matches": [
                {
                    "question_text": "Похожий вопрос",
                    "answer_text": "Ответ",
                    "score": 0.9,
                    "score_kind": "semantic",
                    "external_url": "https://gotquestions.online/question/7",
                    "sources": [{"name": "local", "score": 0.9, "score_kind": "semantic"}],
                }
            ],
            "sources": [{"source": "local", "status": "ok", "matches": 1}],
            "answer": {"text": "Ответ модели", "available": True},
        }
    )

    assert not hasattr(view, "matches")
    assert not hasattr(view, "sources")
    assert SearchView.__slots__ == ("answer_text", "request_id")


async def test_ui_page_is_mounted(mounted_client: httpx.AsyncClient) -> None:
    redirect = await mounted_client.get(MOUNT_PATH)
    page = await mounted_client.get(f"{MOUNT_PATH}/")

    assert redirect.status_code == 307
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]


async def test_ui_page_contains_only_field_and_button(
    mounted_client: httpx.AsyncClient,
) -> None:
    page = await mounted_client.get(f"{MOUNT_PATH}/")

    html = page.text
    assert "Описание вопроса" in html
    assert BUTTON_LABEL in html
    for removed in (
        "Число результатов",
        "Минимальный порог",
        "Сгенерировать ответ",
        "Поиск выполняется",
        "внешний источник",
    ):
        assert removed not in html


def test_title_and_button_label() -> None:
    assert TITLE == "ЧГК знаток"
    assert BUTTON_LABEL == "Ответить"


async def test_ui_client_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _, api = _build_app(monkeypatch, chat=FakeChatProvider())
    view = await api.search("Тьюринг")

    assert view.answer_text == "Сгенерированный ответ"
    assert view.is_unknown is False


async def test_ui_client_answers_without_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Даже без похожих вопросов пользователь получает ответ."""

    chat = FakeChatProvider(text="Ответ без контекста")
    _, api = _build_app(monkeypatch, local=_empty_outcome(), chat=chat)
    view = await api.search("Столица Австралии")

    assert view.answer_text == "Ответ без контекста"
    assert chat.calls and "Столица Австралии" in chat.calls[0]


async def test_ui_client_returns_unknown_when_chat_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, api = _build_app(monkeypatch, chat=FakeChatProvider(fail=True))
    view = await api.search("Тьюринг")

    assert view.answer_text == UNKNOWN_ANSWER


async def test_ui_client_returns_unknown_without_chat_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, api = _build_app(monkeypatch, chat=None)
    view = await api.search("Тьюринг")

    assert view.answer_text == UNKNOWN_ANSWER


async def test_ui_client_returns_unknown_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, api = _build_app(monkeypatch)
    view = await api.search("   ")

    assert view.answer_text == UNKNOWN_ANSWER


async def test_ui_client_degrades_when_local_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сбой локального поиска не лишает пользователя ответа."""

    _, api = _build_app(
        monkeypatch, local_failure=True, chat=FakeChatProvider(text="Ответ есть")
    )
    view = await api.search("Тьюринг")

    assert view.answer_text == "Ответ есть"


async def test_ui_client_keeps_unknown_request_id_off_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Идентификатор запроса хранится, но на странице не показывается."""

    _, api = _build_app(monkeypatch, chat=FakeChatProvider(fail=True))
    view = await api.search("Тьюринг")

    assert view.answer_text == UNKNOWN_ANSWER
    assert view.request_id
