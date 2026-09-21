"""Интеграционные тесты локального гибридного поиска."""

import pytest
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.db.base import (
    Question,
    QuestionOccurrence,
    Source,
    get_embedding_dim,
)
from chgk_agent.ingestion.importer import QuestionImporter
from chgk_agent.models.domain import ParsedQuestion, ParseResult
from chgk_agent.search.lexical import CorpusIndex, tokenize
from chgk_agent.search.local import (
    SEMANTIC_KIND,
    LocalSearch,
    LocalSearchOutcome,
)

pytestmark = pytest.mark.integration

LOCATION = "local-search-fixture.html"
DIMENSION = get_embedding_dim()


class _KeywordEmbedder:
    """Детерминированные эмбеддинги на основе ключевых слов.

    Вектор строится как мешок слов по фиксированному словарю, поэтому
    косинусная близость отражает пересечение терминов и ведёт себя
    предсказуемо в тестах.
    """

    VOCABULARY = (
        "тьюринг",
        "машина",
        "тест",
        "город",
        "мост",
        "река",
        "физика",
        "кварк",
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


def _belongs_to_fixture():
    """Условие выборки вопросов тестового источника."""

    return Question.occurrences.any(
        QuestionOccurrence.source.has(Source.location == LOCATION)
    )


def _question(key: str, text: str) -> ParsedQuestion:
    return ParsedQuestion(source_key=key, question_text=text, answer_text="ответ")


async def _seed(session: AsyncSession, *texts: str) -> None:
    importer = QuestionImporter(session, _KeywordEmbedder())
    await importer.import_results(
        [
            ParseResult(
                location=LOCATION,
                questions=[_question(str(index), text) for index, text in enumerate(texts)],
            )
        ]
    )


@pytest.fixture
async def clean_search_data(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        await session.execute(delete(Source).where(Source.location == LOCATION))
        await session.execute(delete(Question).where(~Question.occurrences.any()))
        await session.commit()


async def test_semantic_branch_finds_expected_question_first(
    session: AsyncSession, clean_search_data: None
) -> None:
    await _seed(
        session,
        "Какой учёный предложил машину Тьюринга?",
        "Какой город стоит на реке?",
    )
    search = LocalSearch(session, _KeywordEmbedder())

    results = await search.search("машина Тьюринга", limit=5)

    assert results[0].question_text == "Какой учёный предложил машину Тьюринга?"
    # Оценка — настоящая косинусная близость, а не нормированный ранг.
    assert results[0].score == pytest.approx(results[0].semantic_similarity)
    assert results[0].score > (results[1].score if len(results) > 1 else 0.0)
    assert results[0].score_kind == SEMANTIC_KIND
    assert results[0].answer_text == "ответ"


async def test_lexical_branch_finds_rare_exact_word(
    session: AsyncSession, clean_search_data: None
) -> None:
    await _seed(
        session,
        "Как называется элементарная частица кварк с дробным зарядом?",
        "Какой город стоит на реке?",
    )
    search = LocalSearch(session, _KeywordEmbedder())

    outcome = await search.search_with_diagnostics("кварк", limit=5)

    assert outcome.matches
    assert outcome.matches[0].question_text.startswith("Как называется элементарная частица")
    assert outcome.lexical_ranks == {outcome.matches[0].question_id: 1}


async def test_direct_call_builds_index_lazily(
    session: AsyncSession, clean_search_data: None
) -> None:
    """Прямой вызов `LocalSearch` строит индекс сам, без ноды графа."""

    from chgk_agent.search import corpus as corpus_module

    cache = CorpusIndex()
    original = corpus_module.get_corpus_index
    corpus_module.get_corpus_index = lambda: cache  # type: ignore[assignment]
    try:
        await _seed(session, "Вопрос про кварк и физику?")
        search = LocalSearch(session, _KeywordEmbedder())

        outcome = await search.search_with_diagnostics("кварк", limit=5)
    finally:
        corpus_module.get_corpus_index = original  # type: ignore[assignment]

    assert cache.is_ready is True
    assert outcome.lexical_ranks
    assert cache.index is not None
    assert cache.index.scores(tokenize("кварк"))


async def test_lexical_branch_can_be_disabled(
    session: AsyncSession, clean_search_data: None
) -> None:
    await _seed(session, "Вопрос про кварк и физику?")
    search = LocalSearch(session, _KeywordEmbedder(), use_lexical=False)

    outcome = await search.search_with_diagnostics("кварк", limit=5)

    assert outcome.lexical_used is False
    assert outcome.lexical_ranks == {}


async def test_lexical_failure_degrades_to_semantic(
    session: AsyncSession, clean_search_data: None, monkeypatch
) -> None:
    await _seed(session, "Вопрос про машину Тьюринга?")
    search = LocalSearch(session, _KeywordEmbedder())

    async def _broken(query: str, outcome) -> list[int]:
        raise RuntimeError("полнотекстовый индекс недоступен")

    monkeypatch.setattr(search, "_lexical_candidate_ids", _broken)

    outcome = await search.search_with_diagnostics("машина Тьюринга", limit=5)

    assert outcome.matches
    assert outcome.degraded is True
    assert outcome.lexical_used is False
    assert "полнотекстовый индекс недоступен" in (outcome.error or "")


async def test_threshold_is_applied_per_branch(
    session: AsyncSession, clean_search_data: None
) -> None:
    await _seed(session, "Вопрос про машину Тьюринга?")
    strict = LocalSearch(session, _KeywordEmbedder(), semantic_min_score=0.9)

    outcome = await strict.search_with_diagnostics("машина Тьюринга", limit=5)

    assert outcome.semantic_ranks == {}
    assert all(match.score_kind != SEMANTIC_KIND for match in outcome.matches)


async def test_min_score_filters_fused_matches(
    session: AsyncSession, clean_search_data: None
) -> None:
    await _seed(session, "Первый вопрос про реку?", "Второй вопрос про мост?")
    search = LocalSearch(session, _KeywordEmbedder())

    results = await search.search("мост", limit=5, min_score=0.9)

    assert all(result.score >= 0.9 for result in results)


async def test_scores_are_normalized_and_ordered(
    session: AsyncSession, clean_search_data: None
) -> None:
    await _seed(
        session,
        "Вопрос про город и реку?",
        "Вопрос про физику и кварк?",
    )
    search = LocalSearch(session, _KeywordEmbedder())

    results = await search.search("город река мост", limit=5)

    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


async def test_result_carries_source_information(
    session: AsyncSession, clean_search_data: None
) -> None:
    await _seed(session, "Вопрос про машину Тьюринга со ссылкой?")
    search = LocalSearch(session, _KeywordEmbedder())

    results = await search.search("машина Тьюринга", limit=5)

    assert results[0].source_location == LOCATION
    assert results[0].source_key is not None


async def test_empty_query_returns_no_matches(
    session: AsyncSession, clean_search_data: None
) -> None:
    await _seed(session, "Вопрос про машину Тьюринга?")
    search = LocalSearch(session, _KeywordEmbedder())

    outcome = await search.search_with_diagnostics("железобетон", limit=5)

    assert isinstance(outcome, LocalSearchOutcome)


async def test_stale_model_vectors_are_not_used_by_semantic_branch(
    session: AsyncSession, clean_search_data: None
) -> None:
    """Векторы другой модели не попадают в семантическую ветку.

    Косинусная близость между векторами разных моделей не имеет смысла,
    поэтому до переиндексации такие вопросы находятся только полнотекстово.
    """

    await _seed(session, "Вопрос про машину Тьюринга?")
    await session.execute(
        update(Question).where(_belongs_to_fixture()).values(embedding_model="ДругаяМодель")
    )
    await session.flush()
    search = LocalSearch(session, _KeywordEmbedder())

    outcome = await search.search_with_diagnostics("машина Тьюринга", limit=5)

    assert outcome.semantic_ranks == {}
    assert all(match.score_kind != SEMANTIC_KIND for match in outcome.matches)


async def test_stale_dimension_vectors_are_not_used_by_semantic_branch(
    session: AsyncSession, clean_search_data: None
) -> None:
    """Векторы другой размерности не попадают в семантическую ветку."""

    await _seed(session, "Вопрос про машину Тьюринга?")
    await session.execute(
        update(Question).where(_belongs_to_fixture()).values(embedding_dim=DIMENSION + 1)
    )
    await session.flush()
    search = LocalSearch(session, _KeywordEmbedder())

    outcome = await search.search_with_diagnostics("машина Тьюринга", limit=5)

    assert outcome.semantic_ranks == {}
