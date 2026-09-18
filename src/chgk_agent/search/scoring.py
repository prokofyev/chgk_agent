"""Единая оценка совпадений из локальной базы и внешнего источника.

Оценка собирается здесь, а не в ветках поиска, потому что лексический сигнал
BM25 — функция от коллекции. Локальная ветка не видит карточки внешнего
источника, поэтому, считай она BM25 сама, стороны получили бы статистику от
разных коллекций и оценки снова стали бы несопоставимыми.

Итог: `score = семантическая близость + вес * нормированный BM25`, где оба
слагаемых считаются от текста «вопрос + ответ». Сумма обрезается сверху
единицей: порог и шкала близости заданы в `[0, 1]`. Когда лексического вклада
нет, оценка в точности равна семантической близости — подпись в интерфейсе
остаётся правдой, а не пересчитанной величиной.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from chgk_agent.logging_setup import get_logger
from chgk_agent.search.lexical import Bm25Index, normalize_bm25, tokenize

logger = get_logger(__name__)

SEMANTIC_KIND = "semantic"
SEMANTIC_LEXICAL_KIND = "semantic+lexical"
LEXICAL_KIND = "lexical"
NONE_KIND = "none"

_CONTRIBUTION_EPSILON = 1e-9


@dataclass(slots=True)
class ScoredCandidate:
    """Кандидат с составляющими единой оценки."""

    key: str
    text: str
    semantic_similarity: float
    lexical_score: float = 0.0
    lexical_normalized: float = 0.0

    @property
    def score(self) -> float:
        """Семантическая близость без лексического вклада."""

        return self.semantic_similarity


@dataclass(slots=True)
class CandidateScore:
    """Итог оценки одного кандидата."""

    score: float
    lexical_score: float
    lexical_normalized: float
    score_kind: str


def describe_score_kind(semantic_similarity: float, lexical_normalized: float) -> str:
    """Назвать состав итоговой оценки кандидата.

    Происхождение совпадения несёт список источников, а `score_kind` говорит,
    из чего собрана величина: только семантика, только лексика или обе части.
    """

    semantic = semantic_similarity > _CONTRIBUTION_EPSILON
    lexical = lexical_normalized > _CONTRIBUTION_EPSILON
    if semantic and lexical:
        return SEMANTIC_LEXICAL_KIND
    if semantic:
        return SEMANTIC_KIND
    if lexical:
        return LEXICAL_KIND
    return NONE_KIND


def build_index(
    documents: list[tuple[str, str]],
    *,
    base: Bm25Index | None = None,
) -> Bm25Index:
    """Построить индекс BM25 по документам, при необходимости поверх базового.

    Базовый индекс — кэш локального корпуса. Карточки внешнего источника
    добавляются копией, чтобы состав коллекции одного запроса не влиял на
    статистику следующего.
    """

    if base is not None:
        return base.copy_with(documents)
    index = Bm25Index()
    index.extend(documents)
    return index.finalize()


def score_candidates(
    description: str,
    candidates: list[ScoredCandidate],
    *,
    base_index: Bm25Index | None = None,
    lexical_weight: float = 1.0,
    lexical_saturation: float = 10.0,
) -> dict[str, CandidateScore]:
    """Посчитать единые оценки кандидатов обоих источников.

    Возвращает карту «ключ -> составляющие оценки». Кандидаты, для которых не
    нашлось ни семантики, ни лексики, получают ноль. Лексика считается только
    при готовом базовом индексе: индекс по одной горстке кандидатов дал бы
    статистику, несовместимую с коллекцией, ради которой всё затевалось.
    """

    if not candidates:
        return {}

    index = (
        build_index(
            [(candidate.key, candidate.text) for candidate in candidates],
            base=base_index,
        )
        if base_index is not None
        else None
    )
    terms = tokenize(description) if index is not None else []
    scores: dict[str, CandidateScore] = {}

    for candidate in candidates:
        lexical = index.score(terms, candidate.key) if index is not None and terms else 0.0
        normalized = normalize_bm25(lexical, saturation=lexical_saturation)
        total = candidate.semantic_similarity + lexical_weight * normalized
        scores[candidate.key] = CandidateScore(
            score=min(1.0, max(0.0, total)),
            lexical_score=lexical,
            lexical_normalized=normalized,
            score_kind=describe_score_kind(candidate.semantic_similarity, normalized),
        )

    return scores


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Косинусная близость двух векторов в диапазоне `[0, 1]`.

    Отрицательные значения обрезаются: они означают противоположные векторы,
    а не «лучше, чем ничего», и в оценке им места нет.
    """

    if not left or not right:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


__all__ = [
    "LEXICAL_KIND",
    "NONE_KIND",
    "SEMANTIC_KIND",
    "SEMANTIC_LEXICAL_KIND",
    "CandidateScore",
    "ScoredCandidate",
    "build_index",
    "cosine_similarity",
    "describe_score_kind",
    "score_candidates",
]
