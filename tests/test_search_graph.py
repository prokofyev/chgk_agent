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
        self.term_weights: list[dict[str, float] | None] = []
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

    async def search(
        self,
        description: str,
        *,
        limit: int = 20,
        term_weights: dict[str, float] | None = None,
    ) -> ExternalSearchResult:
        self.calls.append(description)
        self.term_weights.append(term_weights)
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


def _corpus_index() -> object:
    """Готовый индекс локального корпуса для проверок подготовки."""

    from chgk_agent.search.lexical import Bm25Index

    index = Bm25Index()
    index.add("local:1", "галстук и шляпа")
    for number in range(20):
        index.add(f"local:{number + 2}", "и шляпа")
    return index.finalize()


class _ReadyCorpus:
    """Кэш с готовым индексом: база не нужна."""

    def __init__(self, index: object) -> None:
        self.index = index
        self.is_ready = True


class _StoredCorpus:
    """Кэш без индекса: сборка должна записать его сюда."""

    def __init__(self) -> None:
        self.index: object | None = None
        self.is_ready = False

    def build(self, documents: object) -> object:
        self.index = _corpus_index()
        self.is_ready = True
        return self.index


async def test_parse_request_normalizes_query() -> None:
    state = await nodes.parse_request({"query": "  тест   Тьюринга ", "limit": 5})

    assert state["query"] == "тест Тьюринга"
    assert state["limit"] == 5
    assert state["request_id"]


async def test_parse_request_defaults() -> None:
    state = await nodes.parse_request({"query": "вопрос"})

    assert state["limit"] == 20
    assert state["min_score"] == 0.0
    assert "generate_answer" not in state


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


async def test_index_node_fills_state_from_ready_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Готовый индекс даёт информативность терминов без обращения к базе."""

    class UnusableSessionFactory:
        def __call__(self) -> object:
            raise AssertionError("база не должна опрашиваться при готовом индексе")

    metrics = Metrics(CollectorRegistry())
    deps = SearchDeps(
        session_factory=UnusableSessionFactory(),  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),
        metrics=metrics,
    )
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _ReadyCorpus(_corpus_index()))

    update = await nodes.ensure_corpus_index({"query": "жираф и шляпа"}, deps)

    weights = update["term_weights"]
    assert weights is not None
    assert weights["жираф"] > weights["шляп"]
    assert update["term_weights_error"] is None
    assert metrics.search.corpus_index.labels(outcome="ready")._value.get() == 1
    assert metrics.search.term_weights_unavailable.labels(reason="unavailable")._value.get() == 0


async def test_index_node_builds_index_when_cache_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустой кэш заставляет узел построить индекс и посчитать веса."""

    cache = _StoredCorpus()
    metrics = Metrics(CollectorRegistry())
    deps = _deps(metrics=metrics)
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: cache)

    async def fake_build(session: object, *, cache: object = None) -> object:
        return _corpus_index()

    monkeypatch.setattr(nodes, "build_corpus_index", fake_build)

    update = await nodes.ensure_corpus_index({"query": "галстук"}, deps)

    assert update["term_weights"] is not None
    assert update["term_weights"] is not None and update["term_weights"]["галстук"] > 0
    assert update["term_weights_error"] is None
    assert metrics.search.corpus_index.labels(outcome="built")._value.get() == 1


async def test_index_node_survives_database_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Недоступная база не роняет узел: поиск пойдёт без учёта редкости."""

    metrics = Metrics(CollectorRegistry())
    deps = _deps(metrics=metrics)
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _StoredCorpus())

    async def broken_build(session: object, *, cache: object = None) -> object:
        raise RuntimeError("база недоступна")

    monkeypatch.setattr(nodes, "build_corpus_index", broken_build)

    update = await nodes.ensure_corpus_index({"query": "галстук"}, deps)

    assert update["term_weights"] is None
    assert "база недоступна" in (update["term_weights_error"] or "")
    assert metrics.search.corpus_index.labels(outcome="unavailable")._value.get() == 1
    assert metrics.search.term_weights_unavailable.labels(reason="unavailable")._value.get() == 1


async def test_index_node_times_out_with_own_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """У узла собственный бюджет ожидания, отдельный от ветвей поиска."""

    settings = Settings(_env_file=None)
    settings.search.corpus_index_timeout_seconds = 0.01
    metrics = Metrics(CollectorRegistry())
    deps = _deps(settings=settings, metrics=metrics)
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _StoredCorpus())

    async def slow_build(session: object, *, cache: object = None) -> object:
        await asyncio.sleep(0.5)
        return _corpus_index()

    monkeypatch.setattr(nodes, "build_corpus_index", slow_build)

    update = await nodes.ensure_corpus_index({"query": "галстук"}, deps)

    assert update["term_weights"] is None
    assert "таймаут" in (update["term_weights_error"] or "")
    assert metrics.search.corpus_index.labels(outcome="timeout")._value.get() == 1
    assert metrics.search.term_weights_unavailable.labels(reason="timeout")._value.get() == 1


async def test_index_node_runs_with_lexical_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отключённая лексика не мешает посчитать информативность терминов."""

    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _ReadyCorpus(_corpus_index()))
    deps = _deps()

    update = await nodes.ensure_corpus_index(
        {"query": "жираф", "disable_lexical": True}, deps
    )

    assert update["term_weights"] is not None
    assert update["term_weights_error"] is None


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


def test_generation_is_not_routed_around() -> None:
    """Ветки, пропускающей генерацию, больше нет."""

    assert not hasattr(nodes, "needs_generation")
    assert not hasattr(nodes, "has_results")


async def test_generate_without_context_ignores_matches() -> None:
    """Ответ без подсказок не зависит от найденных совпадений."""

    chat = FakeChatProvider(text="Ответ без подсказок")
    deps = _deps(chat=chat)

    update = await nodes.generate_without_context(
        {"query": "Столица Австралии"}, deps
    )

    assert update["answer_without_context"].text == "Ответ без подсказок"
    assert update["answer_without_context"].used_matches == []
    prompt = chat.calls[0]
    assert "Столица Австралии" in prompt
    assert "Раньше ты встречал такие похожие вопросы" not in prompt


async def test_generate_without_context_degrades_on_provider_error() -> None:
    chat = FakeChatProvider(fail=True)
    deps = _deps(chat=chat)

    update = await nodes.generate_without_context({"query": "описание"}, deps)

    assert update["answer_without_context"].available is False
    assert update["answer_without_context"].text is None


async def test_generate_without_context_degrades_without_provider() -> None:
    deps = _deps(chat=None)

    update = await nodes.generate_without_context({"query": "описание"}, deps)

    assert update["answer_without_context"].available is False


async def test_generate_with_context_uses_matches_and_reports_sources() -> None:
    chat = FakeChatProvider(text="Сгенерированный ответ")
    deps = _deps(chat=chat)
    matches = [
        SearchMatch("Вопрос 1", "Ответ 1", None, 0.9, "semantic", [SourceRef("local")])
    ]

    update = await nodes.generate_with_context(
        {"query": "описание", "matches": matches}, deps
    )

    assert update["answer_with_context"].text == "Сгенерированный ответ"
    assert update["answer_with_context"].used_matches == ["Вопрос 1"]
    assert "Вопрос 1" in chat.calls[0]
    assert "Раньше ты встречал такие похожие вопросы" in chat.calls[0]


async def test_generate_with_context_degrades_on_provider_error() -> None:
    chat = FakeChatProvider(fail=True)
    deps = _deps(chat=chat)
    matches = [
        SearchMatch("Вопрос 1", "Ответ 1", None, 0.9, "semantic", [SourceRef("local")])
    ]

    update = await nodes.generate_with_context(
        {"query": "описание", "matches": matches}, deps
    )

    assert update["answer_with_context"].available is False
    assert update["answer_with_context"].text is None


async def test_generate_with_context_skips_model_without_matches() -> None:
    """Без совпадений второго прогона нет: модель не вызывается вовсе."""

    chat = FakeChatProvider(text="Ответ модели")
    deps = _deps(chat=chat)

    update = await nodes.generate_with_context({"query": "описание", "matches": []}, deps)

    assert update["answer_with_context"] is None
    assert chat.calls == []

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
        async def search(
            self,
            description: str,
            *,
            limit: int = 20,
            term_weights: dict[str, float] | None = None,
        ):
            started.append(time.perf_counter())
            entered.append("external")
            await asyncio.sleep(0.1)
            finished.append(time.perf_counter())
            return await super().search(
                description, limit=limit, term_weights=term_weights
            )

    deps = _deps(external=SlowExternal(), chat=FakeChatProvider(text="Ответ"))
    graph = build_search_graph(deps)

    wall_start = time.perf_counter()
    outcome = await graph.run("описание вопроса", limit=5)
    wall = time.perf_counter() - wall_start

    assert "local" in entered
    assert "external" in entered
    # Ветки пересекаются во времени и укладываются в один бюджет ожидания.
    assert min(finished) > max(started)
    assert wall < 0.2
    assert outcome.matches
    assert outcome.answer_without_context is not None
    assert outcome.answer_without_context.text == "Ответ"
    assert outcome.answer_with_context is not None
    assert outcome.answer_with_context.text == "Ответ"
    assert {report.source for report in outcome.sources} == {"local", "gotquestions"}
    assert outcome.request_id


async def test_generation_without_context_overlaps_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Первый прогон генерации идёт в одном шаге с подготовкой к поиску.

    LangGraph выполняет граф супершагами, поэтому генерация без подсказок
    перекрывается с подготовкой, а не складывается с ней последовательно.
    Здесь это видно по пересечению интервалов: оба шага начались до того,
    как завершился любой из них.
    """

    delay = 0.2
    spans: dict[str, list[float]] = {"plain": [], "with_context": [], "index": []}

    class SlowChatProvider:
        async def complete(self, prompt: str, *, system: str | None = None) -> str:
            is_plain = "Раньше ты встречал такие похожие вопросы" not in prompt
            spans["plain" if is_plain else "with_context"].append(time.perf_counter())
            await asyncio.sleep(delay)
            spans["plain" if is_plain else "with_context"].append(time.perf_counter())
            return "Ответ без подсказок"

    async def slow_build(session: object, *, cache: object = None) -> object:
        spans["index"].append(time.perf_counter())
        await asyncio.sleep(delay)
        spans["index"].append(time.perf_counter())
        return _corpus_index()

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _StoredCorpus())
    monkeypatch.setattr(nodes, "build_corpus_index", slow_build)

    deps = _deps(chat=SlowChatProvider())
    graph = build_search_graph(deps)

    outcome = await graph.run("описание вопроса")

    plain_start, plain_finish = spans["plain"]
    index_start, index_finish = spans["index"]
    assert plain_start < index_finish
    assert index_start < plain_finish
    assert outcome.answer_without_context is not None
    assert outcome.answer_without_context.text == "Ответ без подсказок"


async def test_graph_formats_once_with_and_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Join собирает обе ветки и вызывает сборку ответа ровно один раз."""

    calls: list[str] = []
    original = nodes.format_response

    def counting_format(state: dict, deps: SearchDeps | None = None) -> dict:
        calls.append("format")
        return original(state, deps=deps)

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _ReadyCorpus(_corpus_index()))
    monkeypatch.setattr("chgk_agent.search.graph.format_response", counting_format)
    deps = _deps(chat=FakeChatProvider(text="Ответ"))
    graph = build_search_graph(deps)

    outcome = await graph.run("описание")

    assert calls == ["format"]
    assert outcome.answer_without_context is not None
    assert outcome.answer_with_context is not None


async def test_graph_skips_context_run_on_empty_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без совпадений выше порога второго прогона нет."""

    async def empty_local(self, query, *, limit=20, min_score=0.0):
        return LocalSearchOutcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics", empty_local
    )
    chat = FakeChatProvider(text="Ответ без подсказок")
    deps = _deps(
        external=FakeExternalSource(
            ExternalSearchResult(status=ExternalStatus.EMPTY, query="описание")
        ),
        chat=chat,
    )
    graph = build_search_graph(deps)

    outcome = await graph.run("описание")

    assert outcome.answer_without_context is not None
    assert outcome.answer_without_context.text == "Ответ без подсказок"
    assert outcome.answer_with_context is None
    # Модель вызвана один раз: только прогон без подсказок.
    assert len(chat.calls) == 1


async def test_graph_reports_failed_run_without_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отказ генерации виден в ответе и не делает поиск частичным."""

    class FlakyChatProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, prompt: str, *, system: str | None = None) -> str:
            self.calls += 1
            if "Раньше ты встречал такие похожие вопросы" in prompt:
                raise RuntimeError("GigaChat недоступен")
            return "Ответ без подсказок"

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _ReadyCorpus(_corpus_index()))
    deps = _deps(chat=FlakyChatProvider())
    graph = build_search_graph(deps)

    outcome = await graph.run("описание")

    assert outcome.answer_without_context is not None
    assert outcome.answer_without_context.available is True
    assert outcome.answer_with_context is not None
    assert outcome.answer_with_context.available is False
    assert outcome.is_partial is False


async def test_graph_prepares_index_and_embedding_before_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обе подготовительные ноды завершаются до старта ветвей поиска."""

    order: list[str] = []

    async def slow_embed(texts: list[str]) -> list[list[float]]:
        order.append("embed")
        await asyncio.sleep(0.05)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    class SlowEmbeddingProvider:
        model = "slow-embedding"
        dimension = 4
        embed = staticmethod(slow_embed)

    async def slow_build(session: object, *, cache: object = None) -> object:
        order.append("index")
        await asyncio.sleep(0.05)
        return _corpus_index()

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        order.append("local")
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _StoredCorpus())
    monkeypatch.setattr(nodes, "build_corpus_index", slow_build)

    class RecordingExternal(FakeExternalSource):
        async def search(self, description, *, limit=20, term_weights=None):
            order.append("external")
            return await super().search(
                description, limit=limit, term_weights=term_weights
            )

    deps = _deps(
        external=RecordingExternal(),
        embedding_provider=SlowEmbeddingProvider(),
    )
    graph = build_search_graph(deps)

    await graph.run("описание вопроса")

    branch_start = min(order.index("local"), order.index("external"))
    assert order.index("embed") < branch_start
    assert order.index("index") < branch_start


async def test_index_preparation_does_not_serialize_behind_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сборка индекса идёт параллельно эмбеддингу, а не после него."""

    calls = 0

    async def slow_embed(texts: list[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        # Замедляется только эмбеддинг описания: эмбеддинг карточек внешнего
        # источника добавляет собственную задержку и замаскировал бы замер.
        if calls > 1:
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
        await asyncio.sleep(0.15)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    class SlowEmbeddingProvider:
        model = "slow-embedding"
        dimension = 4
        embed = staticmethod(slow_embed)

    async def slow_build(session: object, *, cache: object = None) -> object:
        await asyncio.sleep(0.15)
        return _corpus_index()

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _StoredCorpus())
    monkeypatch.setattr(nodes, "build_corpus_index", slow_build)

    deps = _deps(
        external=FakeExternalSource(),
        embedding_provider=SlowEmbeddingProvider(),
    )
    graph = build_search_graph(deps)

    started = time.perf_counter()
    await graph.run("описание вопроса")
    wall = time.perf_counter() - started

    # Последовательное выполнение заняло бы не меньше 0.30 с.
    assert wall < 0.28


async def test_index_failure_does_not_cancel_external_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отказ индекса не отменяет внешний поиск и помечает ответ частичным."""

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    async def broken_build(session: object, *, cache: object = None) -> object:
        raise RuntimeError("база недоступна")

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _StoredCorpus())
    monkeypatch.setattr(nodes, "build_corpus_index", broken_build)
    source = FakeExternalSource()
    deps = _deps(external=source)
    graph = build_search_graph(deps)

    outcome = await graph.run("описание")

    assert source.calls == ["описание"]
    assert outcome.status_of("gotquestions") is SourceStatus.OK
    assert outcome.is_partial is True
    assert outcome.matches


async def test_graph_passes_term_weights_when_index_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Источник получает веса, когда индекс доступен."""

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _ReadyCorpus(_corpus_index()))
    source = FakeExternalSource()
    deps = _deps(external=source)
    graph = build_search_graph(deps)

    outcome = await graph.run("жираф и шляпа")

    weights = source.term_weights[0]
    assert weights is not None
    assert weights["жираф"] > weights["шляп"]
    assert outcome.is_partial is False


async def test_graph_passes_no_weights_when_index_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без индекса источник получает `None` и строит запрос прежним правилом."""

    async def fake_search_with_diagnostics(self, query, *, limit=20, min_score=0.0):
        return _local_outcome()

    async def broken_build(session: object, *, cache: object = None) -> object:
        raise RuntimeError("база недоступна")

    monkeypatch.setattr(
        "chgk_agent.search.local.LocalSearch.search_with_diagnostics",
        fake_search_with_diagnostics,
    )
    monkeypatch.setattr(nodes, "get_corpus_index", lambda: _StoredCorpus())
    monkeypatch.setattr(nodes, "build_corpus_index", broken_build)
    source = FakeExternalSource()
    deps = _deps(external=source)
    graph = build_search_graph(deps)

    outcome = await graph.run("описание")

    assert source.term_weights == [None]
    assert outcome.is_partial is True


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

    outcome = await graph.run("описание")

    assert outcome.is_partial is True
    assert outcome.status_of("gotquestions") is SourceStatus.UNAVAILABLE
    assert outcome.status_of("local") is SourceStatus.OK
    assert outcome.matches
    # Генерация теперь безусловна: её отсутствие означает недоступность.
    assert outcome.answer_without_context is not None
    assert outcome.answer_without_context.available is False


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

    outcome = await graph.run("описание")

    assert outcome.is_empty is True
    assert outcome.status_of("gotquestions") is SourceStatus.REJECTED
    assert outcome.status_of("local") is SourceStatus.EMPTY
    assert outcome.answer_without_context is not None
    assert outcome.answer_without_context.available is True
    assert outcome.answer_without_context.used_matches == []
    # Совпадений выше порога нет, поэтому второго прогона не было.
    assert outcome.answer_with_context is None


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

    outcome = await graph.run("длинное описание " * 5)

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

    outcome = await graph.run("Локальный вопрос")

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

    outcome = await graph.run("описание")

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

    outcome = await graph.run("описание")

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

    await graph.run("  Галстук")

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

    outcome = await graph.run("описание")

    assert outcome.status_of("gotquestions") is SourceStatus.UNAVAILABLE
    assert outcome.is_partial is True
    assert [match.question_text for match in outcome.matches] == ["Локальный вопрос"]
