"""Локальный гибридный поиск: kNN по pgvector и полнотекстовый поиск.

Две ветки выполняются независимо и объединяются reciprocal rank fusion.
Если полнотекстовая ветка отключена или падает, поиск деградирует до
чистого векторного поиска, а не отдаёт ошибку.
"""

from dataclasses import dataclass, field

from sqlalchemy import Float, func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from chgk_agent.db.base import Question, QuestionOccurrence, Source
from chgk_agent.embeddings.base import EmbeddingProvider
from chgk_agent.logging_setup import get_logger
from chgk_agent.search.rrf import RankedItem, reciprocal_rank_fusion

logger = get_logger(__name__)

SEMANTIC_KIND = "semantic"
LEXICAL_KIND = "lexical"

FTS_CONFIG = "russian"
_FTS_LITERAL = literal_column(f"'{FTS_CONFIG}'")


@dataclass(slots=True)
class LocalSearchResult:
    """Совпадение из локальной базы.

    `score` — нормированная оценка RRF в `[0, 1]`, по ней же строится порядок
    выдачи. `score_kind` показывает, какая ветка подтвердила вопрос: `semantic`
    — вопрос есть в векторной выдаче, `lexical` — только в полнотекстовой.
    Пороги применяются к метрике ветки до объединения, иначе они теряли бы
    смысл: RRF зависит от рангов, а не от близости.
    """

    question_id: int
    question_text: str
    answer_text: str
    comment: str | None
    score: float
    score_kind: str
    source_location: str | None = None
    external_url: str | None = None
    source_key: str | None = None
    semantic_similarity: float | None = None
    lexical_rank: float | None = None


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
    lexical_rank: float | None = None


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


class LocalSearch:
    """Гибридный поиск по локальной базе."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        *,
        use_lexical: bool = True,
        semantic_min_score: float = 0.0,
        lexical_min_score: float = 0.0,
    ) -> None:
        self._session = session
        self._provider = embedding_provider
        self._use_lexical = use_lexical
        self._semantic_min_score = semantic_min_score
        self._lexical_min_score = lexical_min_score

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        min_score: float = 0.0,
    ) -> list[LocalSearchResult]:
        """Найти вопросы, похожие на описание, и отфильтровать по порогу."""

        outcome = await self.search_with_diagnostics(
            query, limit=limit, min_score=min_score
        )
        return outcome.matches

    async def search_with_diagnostics(
        self,
        query: str,
        *,
        limit: int = 20,
        min_score: float = 0.0,
    ) -> LocalSearchOutcome:
        """Выполнить поиск и вернуть ранги веток для наблюдаемости."""

        outcome = LocalSearchOutcome()
        candidate_limit = max(limit * 4, limit)

        semantic = await self._run_semantic(query, candidate_limit, outcome)
        lexical = await self._run_lexical(query, candidate_limit, outcome)

        outcome.semantic_ranks = {
            candidate.id: position for position, candidate in enumerate(semantic, start=1)
        }
        outcome.lexical_ranks = {
            candidate.id: position for position, candidate in enumerate(lexical, start=1)
        }

        branches = [_as_ranked_items(semantic)]
        if lexical:
            branches.append(_as_ranked_items(lexical))

        fused = reciprocal_rank_fusion(branches)
        candidates = {candidate.id: candidate for candidate in [*semantic, *lexical]}

        matches: list[LocalSearchResult] = []
        for key, score in sorted(fused.items(), key=lambda pair: -pair[1]):
            if score < min_score:
                continue
            question_id = int(key)
            candidate = candidates.get(question_id)
            if candidate is None:
                continue

            matches.append(
                LocalSearchResult(
                    question_id=question_id,
                    question_text=candidate.question_text,
                    answer_text=candidate.answer_text,
                    comment=candidate.comment,
                    score=round(score, 6),
                    score_kind=(
                        SEMANTIC_KIND
                        if question_id in outcome.semantic_ranks
                        else LEXICAL_KIND
                    ),
                    source_location=candidate.source_location,
                    external_url=candidate.external_url,
                    source_key=candidate.source_key,
                    semantic_similarity=candidate.semantic_similarity,
                    lexical_rank=candidate.lexical_rank,
                )
            )
            if len(matches) >= limit:
                break

        outcome.matches = matches
        return outcome

    async def _run_semantic(
        self, query: str, limit: int, outcome: LocalSearchOutcome
    ) -> list[_Candidate]:
        """Выполнить векторную ветку, переживая её недоступность."""

        try:
            candidates = await self._semantic_branch(query, limit=limit)
        except Exception as error:
            outcome.semantic_used = False
            outcome.degraded = True
            outcome.error = str(error)
            logger.warning("векторная ветка недоступна", error=str(error))
            return []

        return [
            candidate
            for candidate in candidates
            if (candidate.semantic_similarity or 0.0) >= self._semantic_min_score
        ]

    async def _run_lexical(
        self, query: str, limit: int, outcome: LocalSearchOutcome
    ) -> list[_Candidate]:
        """Выполнить полнотекстовую ветку, если она включена."""

        if not self._use_lexical:
            outcome.lexical_used = False
            return []

        try:
            candidates = await self._lexical_branch(query, limit=limit)
        except Exception as error:
            outcome.lexical_used = False
            outcome.degraded = True
            outcome.error = str(error)
            logger.warning(
                "полнотекстовая ветка недоступна, деградируем до векторного поиска",
                error=str(error),
            )
            return []

        return [
            candidate
            for candidate in candidates
            if (candidate.lexical_rank or 0.0) >= self._lexical_min_score
        ]

    async def _semantic_branch(self, query: str, *, limit: int) -> list[_Candidate]:
        """Найти ближайшие вопросы по косинусной близости эмбеддингов.

        Сравнивать косинусную близость можно только между векторами одной
        модели и одной размерности, поэтому вопросы с векторами другой модели
        в семантическую ветку не попадают: до переиндексации они остаются
        доступными только полнотекстовому поиску.
        """

        vectors = await self._provider.embed([query])
        if not vectors or not vectors[0]:
            return []

        embedding = list(vectors[0])
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

    async def _lexical_branch(self, query: str, *, limit: int) -> list[_Candidate]:
        """Найти вопросы по полнотекстовому совпадению."""

        tsquery = func.websearch_to_tsquery(_FTS_LITERAL, query)
        document = func.to_tsvector(_FTS_LITERAL, Question.normalized_text)
        rank = func.ts_rank_cd(document, tsquery).cast(Float).label("rank")
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
                rank,
            )
            .outerjoin(occurrence, occurrence.c.question_id == Question.id)
            .where(document.op("@@")(tsquery))
            .order_by(rank.desc(), Question.id)
            .limit(limit)
        )

        rows = (await self._session.execute(statement)).all()
        return [_to_candidate(row, lexical_rank=_normalize_rank(row.rank)) for row in rows]


def _as_ranked_items(candidates: list[_Candidate]) -> list[RankedItem]:
    """Преобразовать выдачу ветки в ранжированный список для RRF."""

    return [
        RankedItem(key=str(candidate.id), rank=position)
        for position, candidate in enumerate(candidates, start=1)
    ]


def _to_candidate(row: object, **metrics: float) -> _Candidate:
    """Преобразовать строку выборки в кандидата поиска."""

    return _Candidate(
        id=int(row.id),
        question_text=row.question_text,
        answer_text=row.answer_text,
        comment=row.comment,
        source_location=row.location,
        external_url=row.external_url,
        source_key=row.source_key,
        **metrics,
    )


def _normalize_rank(rank: float) -> float:
    """Привести `ts_rank_cd` к монотонной шкале `[0, 1)`.

    Полнотекстовый ранг не нормирован сверху, поэтому используется сжатие
    `rank / (rank + 1)`: порядок сохраняется, а значение всегда в границах.
    """

    value = float(rank or 0.0)
    if value <= 0:
        return 0.0
    return value / (value + 1.0)


__all__ = [
    "LEXICAL_KIND",
    "SEMANTIC_KIND",
    "LocalSearch",
    "LocalSearchOutcome",
    "LocalSearchResult",
]
