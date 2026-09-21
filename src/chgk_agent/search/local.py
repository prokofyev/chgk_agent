"""Локальный поиск: векторные соседи по pgvector и лексические кандидаты BM25.

Ветки независимы и больше не сливаются через ранги. Векторная ветка даёт
семантическую близость, лексическая — кандидатов с точными терминами описания.
Итоговая оценка собирается выше по графу: там же, где считаются карточки
внешнего источника, чтобы обе стороны получили сигнал от одной коллекции.
"""

from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from chgk_agent.db.base import Question, QuestionOccurrence, Source
from chgk_agent.embeddings.base import EmbeddingProvider
from chgk_agent.embeddings.text import embedding_query_text
from chgk_agent.logging_setup import get_logger
from chgk_agent.search.corpus import LOCAL_KEY_PREFIX, ensure_corpus_index, local_key
from chgk_agent.search.lexical import tokenize
from chgk_agent.search.scoring import LEXICAL_KIND, SEMANTIC_KIND

logger = get_logger(__name__)


@dataclass(slots=True)
class LocalSearchResult:
    """Совпадение из локальной базы.

    `score` — семантическая близость, `score_kind` — происхождение кандидата.
    Лексический вклад и итоговая оценка считаются выше по графу, поэтому здесь
    их нет: один узел должен видеть кандидатов обоих источников.
    """

    question_id: int
    question_text: str
    answer_text: str
    comment: str | None
    score: float
    score_kind: str
    key: str = ""
    source_location: str | None = None
    external_url: str | None = None
    source_key: str | None = None
    semantic_similarity: float | None = None


@dataclass(slots=True)
class LocalSearchOutcome:
    """Результат локального поиска вместе с диагностикой веток."""

    matches: list[LocalSearchResult] = field(default_factory=list)
    semantic_ranks: dict[int, int] = field(default_factory=dict)
    lexical_ranks: dict[int, int] = field(default_factory=dict)
    semantic_used: bool = True
    lexical_used: bool = True
    degraded: bool = False
    error: str | None = None

    @property
    def is_partial(self) -> bool:
        """Потеряли ли мы одну из веток поиска."""

        return self.degraded or not self.semantic_used or not self.lexical_used


def _occurrence_subquery():
    """Подзапрос с первым вхождением вопроса для ссылки на источник."""

    return (
        select(
            QuestionOccurrence.question_id.label("question_id"),
            func.min(Source.location).label("location"),
            func.min(QuestionOccurrence.external_url).label("external_url"),
            func.min(QuestionOccurrence.source_key).label("source_key"),
        )
        .join(Source, Source.id == QuestionOccurrence.source_id)
        .group_by(QuestionOccurrence.question_id)
        .subquery()
    )


def _question_id(key: str) -> int:
    """Извлечь идентификатор вопроса из ключа лексического индекса."""

    return int(key.removeprefix(LOCAL_KEY_PREFIX))


@dataclass(slots=True)
class _Candidate:
    """Строка выдачи одной из веток поиска."""

    id: int
    question_text: str
    answer_text: str
    comment: str | None
    source_location: str | None = None
    external_url: str | None = None
    source_key: str | None = None
    semantic_similarity: float | None = None


class LocalSearch:
    """Поиск по локальной базе: векторные соседи и лексические кандидаты."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        *,
        query_embedding: list[float] | None = None,
        use_lexical: bool = True,
        lexical_candidate_limit: int = 200,
        semantic_candidate_limit: int = 80,
        semantic_min_score: float = 0.0,
        query_embedding_failed: bool = False,
    ) -> None:
        self._session = session
        self._provider = embedding_provider
        self._query_embedding = query_embedding
        self._query_embedding_failed = query_embedding_failed
        self._use_lexical = use_lexical
        self._lexical_candidate_limit = lexical_candidate_limit
        self._semantic_candidate_limit = semantic_candidate_limit
        self._semantic_min_score = semantic_min_score

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        min_score: float = 0.0,
    ) -> list[LocalSearchResult]:
        """Найти вопросы, похожие на описание."""

        outcome = await self.search_with_diagnostics(query, limit=limit)
        matches = [match for match in outcome.matches if match.score >= min_score]
        return matches[:limit]

    async def search_with_diagnostics(
        self,
        query: str,
        *,
        limit: int = 20,
        min_score: float = 0.0,
    ) -> LocalSearchOutcome:
        """Выполнить поиск и вернуть диагностику веток."""

        outcome = LocalSearchOutcome()
        candidate_limit = max(limit * 4, self._semantic_candidate_limit)

        semantic = [
            candidate
            for candidate in await self._semantic_candidates(
                query, candidate_limit, outcome
            )
            if (candidate.semantic_similarity or 0.0) >= self._semantic_min_score
        ]
        if not semantic and not outcome.semantic_used:
            outcome.matches = []
            return outcome

        outcome.semantic_ranks = {
            candidate.id: position
            for position, candidate in enumerate(semantic, start=1)
        }

        candidates = {candidate.id: candidate for candidate in semantic}
        lexical_ids = await self._lexical_candidates(query, outcome)
        outcome.lexical_ranks = {
            question_id: position
            for position, question_id in enumerate(lexical_ids, start=1)
        }

        missing = [qid for qid in lexical_ids if qid not in candidates]
        if missing:
            for candidate in await self._load_candidates(missing):
                candidates.setdefault(candidate.id, candidate)

        # Кандидаты не обрезаются здесь по `limit`: итоговая оценка считается
        # выше по графу, а лексика должна успеть добавить вопросы, которых нет
        # в семантической выдаче. Обрезает выдачу общий узел ранжирования.
        ordered = sorted(
            candidates.values(),
            key=lambda item: (-(item.semantic_similarity or 0.0), item.id),
        )

        matches = [
            LocalSearchResult(
                question_id=candidate.id,
                question_text=candidate.question_text,
                answer_text=candidate.answer_text,
                comment=candidate.comment,
                score=round(candidate.semantic_similarity or 0.0, 6),
                score_kind=(
                    SEMANTIC_KIND
                    if candidate.id in outcome.semantic_ranks
                    else LEXICAL_KIND
                ),
                key=local_key(candidate.id),
                source_location=candidate.source_location,
                external_url=candidate.external_url,
                source_key=candidate.source_key,
                semantic_similarity=candidate.semantic_similarity,
            )
            for candidate in ordered
        ]

        outcome.matches = matches
        return outcome

    async def _semantic_candidates(
        self, query: str, limit: int, outcome: LocalSearchOutcome
    ) -> list[_Candidate]:
        """Выполнить векторную ветку, переживая её недоступность."""

        try:
            return await self._semantic_branch(query, limit=limit)
        except Exception as error:
            outcome.semantic_used = False
            outcome.degraded = True
            outcome.error = str(error)
            logger.warning("векторная ветка недоступна", error=str(error))
            return []

    async def _lexical_candidate_ids(
        self, query: str, outcome: LocalSearchOutcome
    ) -> list[int]:
        """Отобрать кандидатов по лексическому индексу BM25.

        Лексика работает независимо от векторной ветки: без этого она не
        добавляла бы полноту, а лишь переупорядочивала уже найденное.
        """

        if not self._use_lexical:
            outcome.lexical_used = False
            return []

        terms = tokenize(query)
        if not terms:
            return []

        try:
            current = await ensure_corpus_index(self._session)
        except Exception as error:
            outcome.lexical_used = False
            outcome.degraded = True
            outcome.error = str(error)
            logger.warning("лексический индекс недоступен", error=str(error))
            return []

        scored = current.scores(terms)
        best = sorted(
            scored.items(),
            key=lambda pair: (-pair[1], _question_id(pair[0])),
        )
        return [_question_id(key) for key, _ in best[: self._lexical_candidate_limit]]

    async def _lexical_candidates(
        self, query: str, outcome: LocalSearchOutcome
    ) -> list[int]:
        """Выполнить лексическую ветку, переживая её недоступность.

        Лексика — добавка к семантике, поэтому её падение не должно отменять
        поиск: он продолжается по векторным соседям и помечается частичным.
        """

        try:
            return await self._lexical_candidate_ids(query, outcome)
        except Exception as error:
            outcome.lexical_used = False
            outcome.degraded = True
            outcome.error = str(error)
            logger.warning("лексическая ветка недоступна", error=str(error))
            return []

    async def _load_candidates(self, question_ids: list[int]) -> list[_Candidate]:
        """Загрузить лексических кандидатов вместе с их семантической близостью.

        Вопросы, найденные только лексикой, не попали в векторную выдачу, но их
        векторы в базе есть. Близость для них считается здесь: без неё итоговая
        оценка сравнивала бы семантику одной стороны с её отсутствием у другой,
        то есть вернула бы несопоставимость, которую мы устраняем.
        """

        if not question_ids:
            return []

        embedding = self._query_embedding
        occurrence = _occurrence_subquery()
        columns = [
            Question.id,
            Question.question_text,
            Question.answer_text,
            Question.comment,
            occurrence.c.location,
            occurrence.c.external_url,
            occurrence.c.source_key,
        ]
        if embedding:
            columns.append(
                Question.embedding.cosine_distance(embedding).label("distance")
            )

        statement = (
            select(*columns)
            .outerjoin(occurrence, occurrence.c.question_id == Question.id)
            .where(Question.id.in_(list(question_ids)))
        )
        rows = (await self._session.execute(statement)).all()
        return [_to_candidate(row) for row in rows]

    async def _semantic_branch(self, query: str, *, limit: int) -> list[_Candidate]:
        """Найти ближайшие вопросы по косинусной близости эмбеддингов.

        Сравнивать близость можно только между векторами одной модели и одной
        размерности, поэтому вопросы с векторами другой модели в семантическую
        ветку не попадают: до переиндексации они остаются доступными только
        лексическому поиску.
        """

        embedding = self._query_embedding
        if embedding is None:
            if self._query_embedding_failed:
                # Эмбеддинг описания уже не удался выше по графу: повторный
                # вызов не поможет, а только удвоит ожидание и счётчик ошибок.
                raise RuntimeError("эмбеддинг описания недоступен")
            vectors = await self._provider.embed([embedding_query_text(query)])
            embedding = list(vectors[0]) if vectors and vectors[0] else None
        if not embedding:
            return []

        distance = Question.embedding.cosine_distance(embedding).label("distance")
        occurrence = _occurrence_subquery()

        statement = (
            select(
                Question.id,
                Question.question_text,
                Question.answer_text,
                Question.comment,
                occurrence.c.location,
                occurrence.c.external_url,
                occurrence.c.source_key,
                distance,
            )
            .outerjoin(occurrence, occurrence.c.question_id == Question.id)
            .where(
                Question.embedding.is_not(None),
                Question.embedding_model == self._provider.model,
                Question.embedding_dim == self._provider.dimension,
            )
            .order_by(distance, Question.id)
            .limit(limit)
        )

        rows = (await self._session.execute(statement)).all()
        return [
            _to_candidate(row, semantic_similarity=max(0.0, 1.0 - float(row.distance)))
            for row in rows
        ]


def _to_candidate(row: object, **metrics: float) -> _Candidate:
    """Преобразовать строку выборки в кандидата поиска."""

    distance = getattr(row, "distance", None)
    semantic = metrics.get("semantic_similarity")
    if semantic is None and distance is not None:
        semantic = max(0.0, 1.0 - float(distance))

    return _Candidate(
        id=int(row.id),
        question_text=row.question_text,
        answer_text=row.answer_text,
        comment=row.comment,
        source_location=row.location,
        external_url=row.external_url,
        source_key=row.source_key,
        semantic_similarity=semantic,
    )


__all__ = [
    "SEMANTIC_KIND",
    "LocalSearch",
    "LocalSearchOutcome",
    "LocalSearchResult",
]
