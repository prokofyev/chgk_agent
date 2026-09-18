"""Тесты узлов и графа поиска на фейковых зависимостях."""

import asyncio
import time

import pytest
from prometheus_client import CollectorRegistry

from chgk_agent.config import Settings
from chgk_agent.external.base import (
    ExternalErrorKind,
    ExternalMatch,
    ExternalSearchResult,
    ExternalStatus,
)
from chgk_agent.observability.metrics import Metrics
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


class FailingEmbeddingProvider:
    """Провайдер эмбеддингов, всегда завершающийся ошибкой."""

    model = "fake-embedding"
    dimension = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("эмбеддинги недоступны")


class FixedEmbeddingProvider:
    """Провайдер эмбеддингов с одним и тем же вектором для любого текста."""

    model = "fixed-embedding"
    dimension = 4

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class CountingEmbeddingProvider:
    """Провайдер эмбеддингов, считающий вызовы и тексты."""

    model = "counting-embedding"
    dimension = 4

    def __init__(self) -> None:
        self.texts: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.append(list(texts))
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
            key="local:1",
            semantic_similarity=0.8,
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
    embedding_provider: object | None = None,
    metrics: Metrics | None = None,
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
        embedding_provider=(
            embedding_provider
            if embedding_provider is not None
            else FakeEmbeddingProvider()
        ),  # type: ignore[arg-type]
        external_source=external,
        chat_provider=chat,
        settings=settings or Settings(_env_file=None),
        metrics=metrics,
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


def test_rerank_applies_one_threshold_to_both_sources() -> None:
    """Порог применяется к единой величине, без разбиения по типу оценки."""

    matches = [
        SearchMatch("Локальный", "Ответ", None, 0.4, "semantic", [SourceRef("local")]),
        SearchMatch("Внешний", "Ответ", None, 0.4, "gotquestions", [SourceRef("ext")]),
        SearchMatch("Высокий", "Ответ", None, 0.6, "gotquestions", [SourceRef("ext")]),
    ]

    update = nodes.rerank({"matches": matches, "min_score": 0.5})

    assert [match.question_text for match in update["matches"]] == ["Высокий"]


def test_lowering_threshold_adds_matches_without_reordering() -> None:
    """Понижение порога добавляет совпадения, не меняя порядок остальных."""

    matches = [
        SearchMatch("Высокий", "Ответ", None, 0.9, "semantic", [SourceRef("local")]),
        SearchMatch("Средний", "Ответ", None, 0.6, "semantic", [SourceRef("local")]),
        SearchMatch("Низкий", "Ответ", None, 0.3, "semantic", [SourceRef("local")]),
    ]

    strict = nodes.rerank({"matches": list(matches), "min_score": 0.6})
    loose = nodes.rerank({"matches": list(matches), "min_score": 0.2})

    assert [match.question_text for match in strict["matches"]] == ["Высокий", "Средний"]
    assert [match.question_text for match in loose["matches"]] == [
        "Высокий",
        "Средний",
        "Низкий",
    ]
    assert [match.question_text for match in loose["matches"]][:2] == [
        match.question_text for match in strict["matches"]
    ]


def test_score_without_lexical_index_keeps_semantic_value() -> None:
    """Недоступный индекс не обнуляет оценку и не роняет поиск."""

    matches = [
        SearchMatch(
            "Вопрос про галстук",
            "Ответ",
            None,
            0.0,
            "semantic",
            [SourceRef("local")],
            key="local:1",
            semantic_similarity=0.72,
        )
    ]

    class _EmptyCorpus:
        index = None

    original = nodes.get_corpus_index
    nodes.get_corpus_index = lambda: _EmptyCorpus()  # type: ignore[assignment]
    try:
        update = nodes.score_matches(
            {"query": "галстук", "matches": matches}, _deps()
        )
    finally:
        nodes.get_corpus_index = original  # type: ignore[assignment]

    match = update["matches"][0]
    assert match.score == pytest.approx(0.72)
    assert match.semantic_similarity == pytest.approx(0.72)
    assert match.lexical_score == 0.0


def test_score_matches_handles_matches_without_key() -> None:
    """Совпадение без ключа всё равно проходит через единый узел оценки.

    Иначе поиск по пустому ключу возвращал бы `None`, и совпадение молча
    оставалось бы с одной семантикой и признаком `none`.
    """

    matches = [
        SearchMatch(
            "Вопрос про галстук",
            "Ответ",
            None,
            0.0,
            "semantic",
            [SourceRef("local")],
            semantic_similarity=0.6,
        )
    ]

    update = nodes.score_matches({"query": "галстук", "matches": matches}, _deps())

    match = update["matches"][0]
    assert match.score_kind != "none"
    assert match.score >= match.semantic_similarity


def test_external_match_with_rare_word_beats_equal_local() -> None:
    """При равной близости побеждает сторона с точным словом из описания.

    Исходная жалоба: локальный результат вытеснял с сайта карточку, где
    встречается искомое слово. Теперь обе стороны считаются от одной коллекции,
    поэтому точное слово даёт внешней карточке перевес независимо от источника.
    """

    from chgk_agent.search.lexical import Bm25Index

    class _Corpus:
        """Кэш локального корпуса с документами без искомого слова."""

        def __init__(self) -> None:
            index = Bm25Index()
            for number in range(30):
                index.add(f"local:{number}", "и в что он на и")
            self.index = index.finalize()

    original = nodes.get_corpus_index
    nodes.get_corpus_index = _Corpus  # type: ignore[assignment]
    try:
        matches = [
            SearchMatch(
                "Локальный без нужного слова",
                "Ответ",
                None,
                0.0,
                "semantic",
                [SourceRef("local")],
                key="local:1",
                semantic_similarity=0.60,
            ),
            SearchMatch(
                "Карточка с сайта про галстук",
                "Ответ",
                None,
                0.0,
                "semantic",
                [SourceRef("gotquestions")],
                key="external:1",
                semantic_similarity=0.60,
            ),
        ]

        update = nodes.score_matches(
            {"query": "галстук", "matches": matches}, _deps()
        )
    finally:
        nodes.get_corpus_index = original  # type: ignore[assignment]

    ranked = nodes.rerank({"matches": update["matches"], "min_score": 0.0})

    assert [match.question_text for match in ranked["matches"]] == [
        "Карточка с сайта про галстук",
        "Локальный без нужного слова",
    ]
    assert ranked["matches"][0].lexical_score > 0.0


def test_lexical_candidates_are_never_derived_from_truncated_semantic_list() -> None:
    """Лексика добавляет кандидатов, а не переупорядочивает готовую выдачу."""

    from chgk_agent.search.local import LocalSearch, _Candidate

    semantic = [
        _Candidate(
            id=index,
            question_text=f"Семантический {index}",
            answer_text="ответ",
            comment=None,
            semantic_similarity=0.9 - index * 0.01,
        )
        for index in range(1, 6)
    ]
    lexical_only = _Candidate(
        id=99,
        question_text="Редкий термин",
        answer_text="ответ",
        comment=None,
        semantic_similarity=0.05,
    )

    search = LocalSearch(
        _Session(),  # type: ignore[arg-type]
        FakeEmbeddingProvider(),
        query_embedding=[1.0, 0.0, 0.0, 0.0],
    )

    async def fake_semantic(query, *, limit):
        return list(semantic)

    async def fake_lexical(query, outcome):
        return [99]

    async def fake_load(question_ids):
        return [lexical_only] if 99 in question_ids else []

    search._semantic_branch = fake_semantic  # type: ignore[method-assign]
    search._lexical_candidates = fake_lexical  # type: ignore[method-assign]
    search._load_candidates = fake_load  # type: ignore[method-assign]

    import asyncio

    outcome = asyncio.run(search.search_with_diagnostics("галстук", limit=1))

    texts = [match.question_text for match in outcome.matches]
    assert "Редкий термин" in texts
    assert len(texts) > 1


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


async def test_graph_scores_both_sources_on_one_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Оценки обеих сторон считаются от одной величины близости.

    Раньше локальные совпадения несли нормированный ранговый балл, а внешние —
    позицию сайта, поэтому их нельзя было сравнивать. Здесь обе стороны
    проходят через один узел, и оценка перестаёт зависеть от того, откуда
    пришло совпадение.
    """

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome("Локальный вопрос")

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )

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
                    )
                ],
            )
        ),
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("Локальный вопрос", generate_answer=False)

    assert len(outcome.matches) == 2
    assert {match.sources[0].name for match in outcome.matches} == {
        "local",
        "gotquestions",
    }
    assert all(match.score_kind != "external_rank" for match in outcome.matches)
    assert all(0.0 <= match.score <= 1.0 for match in outcome.matches)


async def test_external_position_is_diagnostics_not_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Позиция сайта сохраняется в отчёте, но не влияет на оценку."""

    async def empty_local(self, query, *, limit=20, min_score=0.0):
        return LocalSearchOutcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics", empty_local
    )

    def card(external_id: str, position: int) -> ExternalMatch:
        return ExternalMatch(
            title="Карточка",
            question_text=f"Карточка {external_id}",
            answer_text="Ответ",
            external_id=external_id,
            position=position,
        )

    deps = _deps(
        external=FakeExternalSource(
            ExternalSearchResult(
                status=ExternalStatus.OK,
                query="вопрос",
                matches=[card("first", 1), card("tenth", 10)],
            )
        )
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("описание", generate_answer=False)

    by_text = {match.question_text: match for match in outcome.matches}
    assert by_text["Карточка first"].sources[0].position == 1
    assert by_text["Карточка tenth"].sources[0].position == 10
    assert by_text["Карточка first"].score == by_text["Карточка tenth"].score


async def test_graph_exposes_unavailable_source_with_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Недоступность источника отдаётся статусом с причиной, а не пустой выдачей."""

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
                query="вопрос",
                error="эмбеддинг карточек недоступен",
                error_kind=ExternalErrorKind.EMBEDDING_FAILED,
                matches=[],
            )
        ),
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("описание", generate_answer=False)

    assert outcome.status_of("gotquestions") is SourceStatus.UNAVAILABLE
    assert outcome.is_partial is True
    assert outcome.is_empty is False
    assert outcome.matches


async def test_score_external_computes_cosine_against_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Внешние карточки получают косинусную близость, а не ноль.

    Раньше карточки эмбеддились, но близость не считалась, поэтому внешняя
    сторона молча получала нулевую семантику и проигрывала локальной.
    """

    provider = FixedEmbeddingProvider([1.0, 1.0, 0.0, 0.0])
    source = FakeExternalSource()
    deps = _deps(
        external=source,
        embedding_provider=provider,
    )
    state = {
        "query_embedding": [1.0, 1.0, 0.0, 0.0],
        "external": source._default_result(),
    }

    update = await nodes.score_external(state, deps)

    matches = update["external"].matches
    assert matches
    assert matches[0].embedding is not None
    assert matches[0].semantic_similarity == pytest.approx(1.0)


async def test_graph_embeds_description_once_per_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Описание эмбеддится один раз и переиспользуется обеими ветками."""

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    provider = CountingEmbeddingProvider()
    deps = _deps(external=FakeExternalSource(), embedding_provider=provider)
    graph = build_search_graph(deps)

    await graph.run("  Галстук", generate_answer=False)

    # Запрос нормализуется перед эмбеддингом: пробелы схлопнуты, регистр снят.
    assert provider.texts[0] == ["галстук"]
    assert len(provider.texts) == 2  # описание и карточки, но не по ветке
    assert len(provider.texts[1]) == 1


async def test_local_branch_does_not_retry_failed_description_embedding() -> None:
    """Сбой эмбеддинга описания не повторяется локальной веткой."""

    from chgk_agent.search.local import LocalSearch

    calls = 0

    class CountingFailure:
        model = "failing"
        dimension = 4

        async def embed(self, texts: list[str]) -> list[list[float]]:
            nonlocal calls
            calls += 1
            raise RuntimeError("эмбеддинги недоступны")

    search = LocalSearch(
        _Session(),  # type: ignore[arg-type]
        CountingFailure(),
        use_lexical=False,
        query_embedding_failed=True,
    )

    outcome = await search.search_with_diagnostics("описание", limit=5)

    assert calls == 0
    assert outcome.matches == []


async def test_score_external_drops_cards_when_embedding_fails() -> None:
    """Сбой эмбеддинга карточек убирает внешние совпадения и метит источник."""

    source = FakeExternalSource()
    deps = _deps(
        external=source,
        embedding_provider=FailingEmbeddingProvider(),
        metrics=Metrics(CollectorRegistry()),
    )
    state = {
        "query_embedding": [1.0, 0.0, 0.0, 0.0],
        "external": source._default_result(),
    }

    update = await nodes.score_external(state, deps)

    result = update["external"]
    assert result.matches == []
    assert result.status is ExternalStatus.UNAVAILABLE
    assert result.error_kind is ExternalErrorKind.EMBEDDING_FAILED
    assert (
        deps.current_metrics.search.external_embedding_failed.labels(
            reason="provider_error"
        )._value.get()
        == 1
    )


async def test_graph_keeps_local_matches_when_card_embedding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При сбое эмбеддинга внешних карточек локальные результаты сохраняются."""

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    deps = _deps(
        external=FakeExternalSource(),
        embedding_provider=FailingEmbeddingProvider(),
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("описание", generate_answer=False)

    assert outcome.status_of("gotquestions") is SourceStatus.UNAVAILABLE
    assert outcome.is_partial is True
    assert [match.question_text for match in outcome.matches] == ["Локальный вопрос"]
