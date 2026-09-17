"""Тесты узлов и графа поиска на фейковых зависимостях."""

import asyncio
import time

import pytest

from chgk_agent.config import Settings
from chgk_agent.external.base import (
    ExternalErrorKind,
    ExternalMatch,
    ExternalSearchResult,
    ExternalStatus,
)
from chgk_agent.search import nodes
from chgk_agent.search.graph import build_search_graph
from chgk_agent.search.local import LocalSearchOutcome, LocalSearchResult
from chgk_agent.search.models import SearchMatch, SourceRef, SourceStatus
from chgk_agent.search.nodes import SearchDeps


class FakeEmbeddingProvider:
    """Провайдер эмбеддингов с фиксированным вектором."""

    model = "fake-embedding"
    dimension = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeChatProvider:
    """Провайдер генерации, считающий вызовы."""

    def __init__(self, *, fail: bool = False, text: str = "Ответ модели") -> None:
        self.calls: list[str] = []
        self._fail = fail
        self._text = text

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(prompt)
        if self._fail:
            raise RuntimeError("GigaChat недоступен")
        return self._text


class FakeExternalSource:
    """Внешний источник с настраиваемым поведением."""

    name = "gotquestions"
    enabled = True

    def __init__(
        self,
        result: ExternalSearchResult | None = None,
        *,
        delay: float = 0.0,
    ) -> None:
        self.calls: list[str] = []
        self.delay = delay
        self._result = result

    def _default_result(self) -> ExternalSearchResult:
        """Результат по умолчанию с одним внешним совпадением."""

        return ExternalSearchResult(
            status=ExternalStatus.OK,
            query="вопрос",
            matches=[
                ExternalMatch(
                    title="Вопрос",
                    question_text="Внешний вопрос",
                    answer_text="Внешний ответ",
                    external_url="https://gotquestions.online/question/1",
                    external_id="1",
                    position=1,
                    score=1.0,
                )
            ],
        )

    async def search(self, description: str, *, limit: int = 20) -> ExternalSearchResult:
        self.calls.append(description)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self._result is not None:
            return self._result
        return self._default_result()


def _local_outcome(text: str = "Локальный вопрос") -> LocalSearchOutcome:
    outcome = LocalSearchOutcome()
    outcome.matches = [
        LocalSearchResult(
            question_id=1,
            question_text=text,
            answer_text="Локальный ответ",
            comment=None,
            score=0.8,
            score_kind="semantic",
            source_location="fixture.html",
        )
    ]
    outcome.semantic_ranks = {1: 1}
    return outcome


class _Session:
    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _deps(
    *,
    local: LocalSearchOutcome | None = None,
    external: FakeExternalSource | None = None,
    chat: FakeChatProvider | None = None,
    settings: Settings | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> SearchDeps:
    if monkeypatch is not None and local is not None:
        async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
            return local

        monkeypatch.setattr(
            "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
            fake_search_with_diagnostics,
        )

    return SearchDeps(
        session_factory=_SessionFactory(),  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),
        external_source=external,
        chat_provider=chat,
        settings=settings or Settings(_env_file=None),
    )


class _SessionFactory:
    def __call__(self) -> _Session:
        return _Session()


async def test_parse_request_normalizes_query() -> None:
    state = await nodes.parse_request({"query": "  тест   Тьюринга ", "limit": 5})

    assert state["query"] == "тест Тьюринга"
    assert state["limit"] == 5
    assert state["request_id"]


async def test_parse_request_defaults() -> None:
    state = await nodes.parse_request({"query": "вопрос"})

    assert state["limit"] == 20
    assert state["min_score"] == 0.0
    assert state["generate_answer"] is True


async def test_local_node_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = _deps(local=_local_outcome(), monkeypatch=monkeypatch)

    update = await nodes.local_search(
        {"query": "вопрос", "limit": 5}, deps
    )

    assert update["local"].matches
    assert update["local_seconds"] >= 0.0


async def test_local_node_handles_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(self, query, *, limit=20, min_score=0.0):
        raise RuntimeError("база недоступна")

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics", broken
    )
    deps = _deps()

    update = await nodes.local_search({"query": "вопрос", "limit": 5}, deps)

    assert update["local"].degraded is True
    assert "база недоступна" in update["local"].error


async def test_external_node_isolated() -> None:
    source = FakeExternalSource()
    deps = _deps(external=source)

    update = await nodes.external_search({"query": "вопрос", "limit": 5}, deps)

    assert update["external"].status is ExternalStatus.OK
    assert source.calls == ["вопрос"]


async def test_external_node_times_out() -> None:
    settings = Settings(_env_file=None)
    settings.search.external_timeout_seconds = 0.01
    source = FakeExternalSource(delay=0.5)
    deps = _deps(external=source, settings=settings)

    update = await nodes.external_search({"query": "вопрос", "limit": 5}, deps)

    assert update["external"].status is ExternalStatus.UNAVAILABLE
    assert "таймаут" in update["external"].error


def test_merge_deduplicates_by_canonical_question() -> None:
    external = FakeExternalSource(
        ExternalSearchResult(
            status=ExternalStatus.OK,
            query="вопрос",
            matches=[
                ExternalMatch(
                    title="В",
                    question_text="Локальный вопрос",
                    answer_text="Другой ответ",
                    external_url="https://gotquestions.online/question/9",
                    external_id="9",
                    position=1,
                    score=0.5,
                )
            ],
        )
    )
    state = {
        "local": _local_outcome(),
        "external": external._result,
        "local_seconds": 0.1,
    }

    update = nodes.merge_and_dedupe(state)

    assert len(update["matches"]) == 1
    assert set(update["matches"][0].source_names) == {"local", "gotquestions"}
    assert update["matches"][0].score == 0.8


def test_merge_keeps_distinct_questions() -> None:
    state = {"local": _local_outcome(), "external": FakeExternalSource()._default_result()}

    update = nodes.merge_and_dedupe(state)

    assert len(update["matches"]) == 2


def test_rerank_applies_threshold_and_limit() -> None:
    matches = [
        SearchMatch("Вопрос 1", "Ответ", None, 0.9, "semantic", [SourceRef("local")]),
        SearchMatch("Вопрос 2", "Ответ", None, 0.4, "semantic", [SourceRef("local")]),
        SearchMatch("Вопрос 3", "Ответ", None, 0.2, "semantic", [SourceRef("local")]),
    ]

    update = nodes.rerank({"matches": matches, "min_score": 0.4, "limit": 1})

    assert len(update["matches"]) == 1
    assert update["matches"][0].score == 0.9


def test_rerank_threshold_by_score_kind() -> None:
    matches = [
        SearchMatch("Локальный", "Ответ", None, 0.4, "semantic", [SourceRef("local")]),
        SearchMatch("Внешний", "Ответ", None, 0.4, "external_rank", [SourceRef("ext")]),
    ]

    update = nodes.rerank(
        {
            "matches": matches,
            "min_score": 0.5,
            "thresholds": {"external_rank": 0.3},
        }
    )

    assert [match.question_text for match in update["matches"]] == ["Внешний"]


def test_has_results_skips_rerank_without_matches() -> None:
    assert nodes.has_results({"matches": []}) == "skip"


def test_has_results_reranks_with_matches() -> None:
    assert nodes.has_results({"matches": [object()]}) == "rerank"


def test_needs_generation_skips_when_disabled() -> None:
    assert nodes.needs_generation({"generate_answer": False, "matches": [object()]}) == "skip"


def test_needs_generation_skips_without_matches() -> None:
    assert nodes.needs_generation({"generate_answer": True, "matches": []}) == "skip"


def test_needs_generation_runs_with_matches() -> None:
    assert nodes.needs_generation({"generate_answer": True, "matches": [object()]}) == "generate"


async def test_generate_uses_matches_and_reports_sources() -> None:
    chat = FakeChatProvider(text="Сгенерированный ответ")
    deps = _deps(chat=chat)
    matches = [
        SearchMatch("Вопрос 1", "Ответ 1", None, 0.9, "semantic", [SourceRef("local")])
    ]

    update = await nodes.generate(
        {"query": "описание", "matches": matches, "generate_answer": True}, deps
    )

    assert update["answer"].text == "Сгенерированный ответ"
    assert update["answer"].used_matches == ["Вопрос 1"]
    assert "Вопрос 1" in chat.calls[0]


async def test_generate_degrades_on_provider_error() -> None:
    chat = FakeChatProvider(fail=True)
    deps = _deps(chat=chat)
    matches = [
        SearchMatch("Вопрос 1", "Ответ 1", None, 0.9, "semantic", [SourceRef("local")])
    ]

    update = await nodes.generate(
        {"query": "описание", "matches": matches, "generate_answer": True}, deps
    )

    assert update["answer"].available is False
    assert update["answer"].text is None


async def test_graph_runs_both_branches_in_parallel_and_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered: list[str] = []
    started: list[float] = []
    finished: list[float] = []

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        started.append(time.perf_counter())
        entered.append("local")
        await asyncio.sleep(0.1)
        finished.append(time.perf_counter())
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )

    class SlowExternal(FakeExternalSource):
        async def search(self, description: str, *, limit: int = 20):
            started.append(time.perf_counter())
            entered.append("external")
            await asyncio.sleep(0.1)
            finished.append(time.perf_counter())
            return await super().search(description, limit=limit)

    deps = _deps(external=SlowExternal(), chat=FakeChatProvider(text="Ответ"))
    graph = build_search_graph(deps)

    wall_start = time.perf_counter()
    outcome = await graph.run("описание вопроса", limit=5, generate_answer=True)
    wall = time.perf_counter() - wall_start

    assert "local" in entered
    assert "external" in entered
    # Ветки пересекаются во времени и укладываются в один бюджет ожидания.
    assert min(finished) > max(started)
    assert wall < 0.2
    assert outcome.matches
    assert outcome.answer is not None and outcome.answer.text == "Ответ"
    assert {report.source for report in outcome.sources} == {"local", "gotquestions"}
    assert outcome.request_id


async def test_graph_marks_partial_when_external_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    deps = _deps(
        external=FakeExternalSource(
            ExternalSearchResult(
                status=ExternalStatus.UNAVAILABLE,
                query="описание",
                error="таймаут",
                error_kind=ExternalErrorKind.TIMEOUT,
            )
        )
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("описание", generate_answer=False)

    assert outcome.is_partial is True
    assert outcome.status_of("gotquestions") is SourceStatus.UNAVAILABLE
    assert outcome.status_of("local") is SourceStatus.OK
    assert outcome.matches
    assert outcome.answer is None


async def test_graph_reports_empty_and_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def empty_local(self, query, *, limit=20, min_score=0.0):
        return LocalSearchOutcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics", empty_local
    )
    deps = _deps(
        external=FakeExternalSource(
            ExternalSearchResult(status=ExternalStatus.REJECTED, query="описание")
        ),
        chat=FakeChatProvider(),
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("описание", generate_answer=True)

    assert outcome.is_empty is True
    assert outcome.status_of("gotquestions") is SourceStatus.REJECTED
    assert outcome.status_of("local") is SourceStatus.EMPTY
    assert outcome.answer is not None and outcome.answer.available is False


async def test_graph_exposes_truncated_external_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    deps = _deps(
        external=FakeExternalSource(
            ExternalSearchResult(
                status=ExternalStatus.OK,
                query="Ангус Бейтмен",
                queries=["Ангус Бейтмен"],
                truncated=True,
            )
        ),
        chat=FakeChatProvider(),
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("длинное описание " * 5, generate_answer=False)

    assert outcome.truncated_query == "Ангус Бейтмен"
    assert outcome.sources[1].truncated is True


async def test_graph_applies_external_threshold_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Порог для внешних оценок берётся из настроек и применяется в графе."""

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )

    settings = Settings(_env_file=None)
    settings.search.external_min_score = 0.3
    deps = _deps(
        external=FakeExternalSource(
            ExternalSearchResult(
                status=ExternalStatus.OK,
                query="вопрос",
                matches=[
                    ExternalMatch(
                        title="Слабый",
                        question_text="Слабое внешнее совпадение",
                        answer_text="Ответ",
                        external_id="weak",
                        position=9,
                        score=0.1,
                        score_kind="external_rank",
                    )
                ],
            )
        ),
        settings=settings,
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("описание", generate_answer=False)

    assert outcome.status_of("gotquestions") is SourceStatus.OK
    assert all(match.score_kind != "external_rank" for match in outcome.matches)


async def test_graph_keeps_external_matches_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Внешние совпадения выше порога остаются в выдаче."""

    async def empty_local(self, query, *, limit=20, min_score=0.0):
        return LocalSearchOutcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics", empty_local
    )

    settings = Settings(_env_file=None)
    settings.search.external_min_score = 0.3
    deps = _deps(
        external=FakeExternalSource(
            ExternalSearchResult(
                status=ExternalStatus.OK,
                query="вопрос",
                matches=[
                    ExternalMatch(
                        title="Сильный",
                        question_text="Сильное внешнее совпадение",
                        answer_text="Ответ",
                        external_id="strong",
                        position=1,
                        score=0.9,
                        score_kind="external_rank",
                    )
                ],
            )
        ),
        settings=settings,
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("описание", generate_answer=False)

    assert [match.question_text for match in outcome.matches] == [
        "Сильное внешнее совпадение"
    ]
