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
    _render_view,
    answer_for,
    form_enabled,
    is_submittable,
    mount_ui,
    run_answer,
)
from chgk_agent.ui.client import SearchApiClient
from chgk_agent.ui.view import (
    MIN_QUERY_CHARS,
    PENDING_ANSWER,
    UNKNOWN_ANSWER,
    WITH_CONTEXT_CAPTION,
    WITHOUT_CONTEXT_CAPTION,
    SearchView,
    build_view,
    pending_view,
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


def test_form_enabled_blocks_field_and_button_while_busy() -> None:
    """Во время запроса недоступны и поле, и кнопка."""

    assert form_enabled("галстук", busy=True) == (False, False)
    assert form_enabled("", busy=True) == (False, False)


def test_form_enabled_keeps_field_editable_when_idle() -> None:
    """В покое поле доступно даже пустым, иначе в него не ввести текст."""

    assert form_enabled("", busy=False) == (True, False)
    assert form_enabled("галстук", busy=False) == (True, True)


class _RecordingClient:
    """Клиент, фиксирующий состояние формы на момент обращения."""

    def __init__(
        self,
        answer: str = "Ответ модели",
        *,
        answer_with_context: str | None = "Ответ с базой",
    ) -> None:
        self.answer = answer
        self.answer_with_context = answer_with_context
        self.calls: list[tuple[str, bool]] = []
        self.fail = False
        self.busy = False

    async def search(
        self, query: str, *, limit: int = 20, min_score: float | None = None
    ) -> SearchView:
        self.calls.append((query, self.busy))
        if self.fail:
            raise RuntimeError("сбой клиента")
        return SearchView(
            answer_without_context=self.answer,
            answer_with_context=self.answer_with_context,
        )


class _Recorder:
    """Клиент-заглушка, показанные тексты и история блокировок формы."""

    def __init__(self) -> None:
        self.client = _RecordingClient()
        self.shown: list[SearchView] = []
        self.busy_states: list[bool] = []

    def set_busy(self, value: bool) -> None:
        """Запомнить состояние формы на момент вызова."""

        self.client.busy = value
        self.busy_states.append(value)

    def show(self, view: SearchView) -> None:
        """Запомнить показанный пользователю текст."""

        self.shown.append(view)

    @property
    def texts(self) -> list[tuple[str | None, str | None]]:
        """Показанные пары ответов по порядку."""

        return [
            (view.answer_without_context, view.answer_with_context)
            for view in self.shown
        ]


async def test_run_answer_shows_status_before_answer() -> None:
    recorder = _Recorder()

    await run_answer(
        "галстук",
        recorder.client,  # type: ignore[arg-type]
        set_busy=recorder.set_busy,
        show=recorder.show,
    )

    assert recorder.texts == [
        (PENDING_ANSWER, PENDING_ANSWER),
        ("Ответ модели", "Ответ с базой"),
    ]
    assert recorder.client.calls == [("галстук", True)]


async def test_run_answer_replaces_previous_answer() -> None:
    """Ответ на предыдущий вопрос исчезает до появления нового."""

    recorder = _Recorder()
    recorder.show(
        SearchView(
            answer_without_context="Прошлый ответ",
            answer_with_context="Прошлый ответ с базой",
        )
    )

    await run_answer(
        "новый вопрос",
        recorder.client,  # type: ignore[arg-type]
        set_busy=recorder.set_busy,
        show=recorder.show,
    )

    assert recorder.texts[1] == (PENDING_ANSWER, PENDING_ANSWER)
    assert all(
        "Прошлый ответ" not in (without or "") and "Прошлый ответ" not in (with_ or "")
        for without, with_ in recorder.texts[1:]
    )


async def test_run_answer_unblocks_form_after_failure() -> None:
    """Неожиданный сбой не оставляет форму заблокированной."""

    recorder = _Recorder()
    recorder.client.fail = True

    await run_answer(
        "галстук",
        recorder.client,  # type: ignore[arg-type]
        set_busy=recorder.set_busy,
        show=recorder.show,
    )

    assert recorder.busy_states == [True, False]
    assert recorder.texts[-1] == (UNKNOWN_ANSWER, UNKNOWN_ANSWER)


async def test_run_answer_skips_status_for_short_query() -> None:
    """Короткое описание не показывает статус и не идёт в поиск."""

    recorder = _Recorder()

    await run_answer(
        "ок",
        recorder.client,  # type: ignore[arg-type]
        set_busy=recorder.set_busy,
        show=recorder.show,
    )

    assert recorder.texts == [(UNKNOWN_ANSWER, UNKNOWN_ANSWER)]
    assert recorder.busy_states == []
    assert recorder.client.calls == []


async def test_answer_for_rejects_short_query_without_search() -> None:
    """Обход блокировки кнопки не приводит к обращению к поиску."""

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def search(
            self, query: str, *, limit: int = 20, min_score: float | None = None
        ) -> SearchView:
            self.calls.append(query)
            return SearchView(answer_without_context="Ответ модели")

    client = RecordingClient()

    view = await answer_for("ок", client)  # type: ignore[arg-type]

    assert view.is_unknown
    assert client.calls == []


async def test_answer_for_passes_valid_query_to_search() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, float | None]] = []

        async def search(
            self, query: str, *, limit: int = 20, min_score: float | None = None
        ) -> SearchView:
            self.calls.append((query, limit, min_score))
            return SearchView(answer_without_context="Ответ модели")

    client = RecordingClient()

    view = await answer_for("галстук", client, limit=5, min_score=0.85)  # type: ignore[arg-type]

    assert view.answer_without_context == "Ответ модели"
    assert client.calls == [("галстук", 5, 0.85)]


def test_unknown_view_is_unknown() -> None:
    view = unknown_view(request_id="abc123")

    assert view.answer_without_context == UNKNOWN_ANSWER
    assert view.answer_with_context == UNKNOWN_ANSWER
    assert view.is_unknown
    assert view.request_id == "abc123"


def test_pending_view_shows_status() -> None:
    """Статус выполнения отличим от ответа «Не знаю»."""

    view = pending_view()

    assert view.answer_without_context == PENDING_ANSWER
    assert view.answer_with_context == PENDING_ANSWER
    assert view.is_pending
    assert view.is_unknown is False


def test_build_view_never_returns_pending_status() -> None:
    """Ответ API не может привести к статусу выполнения."""

    view = build_view(
        {
            "matches": [],
            "sources": [],
            "answer_without_context": {"text": None, "available": False},
        }
    )

    assert view.answer_without_context != PENDING_ANSWER
    assert view.is_pending is False


def test_build_view_returns_both_answers() -> None:
    view = build_view(
        {
            "query": "Тьюринг",
            "matches": [{"question_text": "Вопрос", "answer_text": "Ответ"}],
            "sources": [{"source": "local", "status": "ok", "matches": 1}],
            "answer_without_context": {
                "text": "Ответ без базы",
                "available": True,
                "used_matches": [],
            },
            "answer_with_context": {
                "text": "Ответ с базой",
                "available": True,
                "used_matches": ["Вопрос"],
            },
            "request_id": "req-1",
        }
    )

    assert view.answer_without_context == "Ответ без базы"
    assert view.answer_with_context == "Ответ с базой"
    assert view.is_unknown is False


def test_build_view_leaves_absent_context_answer_empty() -> None:
    """Прогона с подгрузкой не было: его место остаётся пустым."""

    view = build_view(
        {
            "matches": [],
            "sources": [],
            "answer_without_context": {"text": "Ответ без базы", "available": True},
            "answer_with_context": None,
        }
    )

    assert view.answer_without_context == "Ответ без базы"
    assert view.answer_with_context is None
    assert view.is_unknown is False


def test_build_view_marks_failed_context_run_as_unknown() -> None:
    """Прогон с подгрузкой был и не удался: на его месте «Не знаю»."""

    view = build_view(
        {
            "answer_without_context": {"text": "Ответ без базы", "available": True},
            "answer_with_context": {"text": None, "available": False},
        }
    )

    assert view.answer_with_context == UNKNOWN_ANSWER


def test_build_view_ignores_matches_without_answer() -> None:
    """Выдача без сгенерированного ответа не показывается пользователю."""

    view = build_view(
        {
            "matches": [{"question_text": "Вопрос", "answer_text": "Ответ"}],
            "sources": [],
            "answer_without_context": {"text": None, "available": False},
            "answer_with_context": None,
            "empty": True,
        }
    )

    assert view.answer_without_context == UNKNOWN_ANSWER
    assert view.answer_with_context is None


def test_build_view_without_answer_block_returns_unknown() -> None:
    view = build_view({"matches": [], "sources": []})

    assert view.answer_without_context == UNKNOWN_ANSWER
    assert view.answer_with_context is None


def test_build_view_ignores_blank_answer_text() -> None:
    view = build_view(
        {"answer_without_context": {"text": "   ", "available": True}}
    )

    assert view.answer_without_context == UNKNOWN_ANSWER


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
            "answer_without_context": {"text": "Ответ модели", "available": True},
        }
    )

    assert not hasattr(view, "matches")
    assert not hasattr(view, "sources")
    assert SearchView.__slots__ == (
        "answer_without_context",
        "answer_with_context",
        "request_id",
    )


def test_answer_captions_distinguish_runs() -> None:
    """Подписи различают прогон без базы и прогон с учётом базы."""

    assert WITHOUT_CONTEXT_CAPTION == "Без базы вопросов"
    assert WITH_CONTEXT_CAPTION == "С учётом базы вопросов"


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
    assert WITHOUT_CONTEXT_CAPTION in html
    assert WITH_CONTEXT_CAPTION in html
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


def test_render_view_shows_both_answers() -> None:
    """Оба ответа попадают на страницу, а отсутствующий остаётся пустым."""

    class Label:
        def __init__(self) -> None:
            self.text = "прежний текст"

    without = Label()
    with_context = Label()

    _render_view(
        SearchView(
            answer_without_context="Ответ без базы",
            answer_with_context=None,
        ),
        without,
        with_context,
    )

    assert without.text == "Ответ без базы"
    assert with_context.text == ""

    _render_view(
        SearchView(
            answer_without_context=UNKNOWN_ANSWER,
            answer_with_context="Ответ с базой",
        ),
        without,
        with_context,
    )

    assert without.text == UNKNOWN_ANSWER
    assert with_context.text == "Ответ с базой"


def test_render_view_clears_answers_for_pending_status() -> None:
    """Статус выполнения занимает оба места, прежние ответы не остаются."""

    class Label:
        def __init__(self) -> None:
            self.text = "прошлый ответ"

    without = Label()
    with_context = Label()

    _render_view(pending_view(), without, with_context)

    assert without.text == PENDING_ANSWER
    assert with_context.text == PENDING_ANSWER


async def test_ui_client_returns_both_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Клиент возвращает оба прогона генерации."""

    _, api = _build_app(monkeypatch, chat=FakeChatProvider())
    view = await api.search("Тьюринг")

    assert view.answer_without_context == "Сгенерированный ответ"
    assert view.answer_with_context == "Сгенерированный ответ"
    assert view.is_unknown is False


async def test_ui_client_answers_without_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без похожих вопросов пользователь получает ответ без подсказок."""

    chat = FakeChatProvider(text="Ответ без контекста")
    _, api = _build_app(
        monkeypatch,
        local=_empty_outcome(),
        external=FakeExternalSource(
            ExternalSearchResult(status=ExternalStatus.EMPTY, query="Столица Австралии")
        ),
        chat=chat,
    )
    view = await api.search("Столица Австралии")

    assert view.answer_without_context == "Ответ без контекста"
    assert view.answer_with_context is None
    assert chat.calls and "Столица Австралии" in chat.calls[0]
    # Второго прогона не было: модель вызвана один раз.
    assert len(chat.calls) == 1


async def test_ui_client_marks_unknown_when_chat_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, api = _build_app(monkeypatch, chat=FakeChatProvider(fail=True))
    view = await api.search("Тьюринг")

    assert view.answer_without_context == UNKNOWN_ANSWER


async def test_ui_client_returns_unknown_without_chat_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, api = _build_app(monkeypatch, chat=None)
    view = await api.search("Тьюринг")

    assert view.answer_without_context == UNKNOWN_ANSWER


async def test_ui_client_returns_unknown_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, api = _build_app(monkeypatch)
    view = await api.search("   ")

    assert view.is_unknown


async def test_ui_client_degrades_when_local_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сбой локального поиска не лишает пользователя ответа."""

    _, api = _build_app(
        monkeypatch, local_failure=True, chat=FakeChatProvider(text="Ответ есть")
    )
    view = await api.search("Тьюринг")

    assert view.answer_without_context == "Ответ есть"


async def test_ui_client_keeps_unknown_request_id_off_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Идентификатор запроса хранится, но на странице не показывается."""

    _, api = _build_app(monkeypatch, chat=FakeChatProvider(fail=True))
    view = await api.search("Тьюринг")

    assert view.answer_without_context == UNKNOWN_ANSWER
    assert view.request_id
