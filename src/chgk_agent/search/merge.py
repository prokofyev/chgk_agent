"""Объединение, дедупликация и ранжирование совпадений из источников.

Совпадения из локальной базы и внешнего источника приводятся к одной
модели `SearchMatch`. Дедупликация идёт по каноническому тексту вопроса:
если вопрос найден в нескольких источниках, остаётся одно совпадение с
лучшей оценкой и ссылками на все источники.
"""

from collections.abc import Iterable

from chgk_agent.external.base import ExternalSearchResult, ExternalStatus
from chgk_agent.external.gotquestions import SOURCE_NAME as EXTERNAL_SOURCE_NAME
from chgk_agent.external.gotquestions import deduplicate_matches
from chgk_agent.search.local import LocalSearchOutcome
from chgk_agent.search.models import (
    LOCAL_SOURCE,
    SearchMatch,
    SourceRef,
    SourceReport,
    SourceStatus,
)

EXTERNAL_STATUS_MAP: dict[ExternalStatus, SourceStatus] = {
    ExternalStatus.OK: SourceStatus.OK,
    ExternalStatus.EMPTY: SourceStatus.EMPTY,
    ExternalStatus.REJECTED: SourceStatus.REJECTED,
    ExternalStatus.UNAVAILABLE: SourceStatus.UNAVAILABLE,
    ExternalStatus.DISABLED: SourceStatus.DISABLED,
}


EXTERNAL_KEY_PREFIX = "external:"


def external_key(item: object) -> str:
    """Ключ внешнего совпадения для единой оценки."""

    identifier = getattr(item, "external_id", None) or getattr(item, "external_url", None)
    return f"{EXTERNAL_KEY_PREFIX}{identifier}"


def local_matches(outcome: LocalSearchOutcome) -> list[SearchMatch]:
    """Преобразовать результат локального поиска в общий формат."""

    return [
        SearchMatch(
            question_text=item.question_text,
            answer_text=item.answer_text,
            comment=item.comment,
            score=item.score,
            score_kind=item.score_kind,
            key=item.key,
            semantic_similarity=item.semantic_similarity or 0.0,
            sources=[
                SourceRef(
                    name=LOCAL_SOURCE,
                    location=item.source_location,
                    external_url=item.external_url,
                    score=item.score,
                    score_kind=item.score_kind,
                )
            ],
            external_url=item.external_url,
        )
        for item in outcome.matches
    ]


def external_matches(result: ExternalSearchResult) -> list[SearchMatch]:
    """Преобразовать результат внешнего поиска в общий формат."""

    matches: list[SearchMatch] = []
    for item in deduplicate_matches(result.matches):
        matches.append(
            SearchMatch(
                question_text=item.question_text,
                answer_text=item.answer_text or "",
                comment=item.comment,
                score=item.score,
                score_kind=item.score_kind,
                key=external_key(item),
                semantic_similarity=item.semantic_similarity,
                sources=[
                    SourceRef(
                        name=EXTERNAL_SOURCE_NAME,
                        external_url=item.external_url,
                        score=item.score,
                        score_kind=item.score_kind,
                        position=item.position or None,
                    )
                ],
                external_url=item.external_url,
            )
        )
    return matches


def local_report(
    outcome: LocalSearchOutcome,
    *,
    matches: int,
    duration_seconds: float = 0.0,
) -> SourceReport:
    """Описать локальный источник для итогового ответа."""

    if outcome.degraded and not outcome.matches:
        status = SourceStatus.UNAVAILABLE
    elif matches:
        status = SourceStatus.OK
    else:
        status = SourceStatus.EMPTY

    return SourceReport(
        source=LOCAL_SOURCE,
        status=status,
        matches=matches,
        error=outcome.error if status is SourceStatus.UNAVAILABLE else None,
        error_kind="local_failure" if status is SourceStatus.UNAVAILABLE else None,
        degraded=outcome.is_partial and status is not SourceStatus.UNAVAILABLE,
        duration_seconds=duration_seconds,
    )


def external_report(
    result: ExternalSearchResult,
    *,
    matches: int,
) -> SourceReport:
    """Описать внешний источник для итогового ответа."""

    return SourceReport(
        source=EXTERNAL_SOURCE_NAME,
        status=EXTERNAL_STATUS_MAP[result.status],
        matches=matches,
        error=result.error,
        error_kind=result.error_kind.value if result.error_kind else None,
        query=result.query or None,
        queries=list(result.queries),
        truncated=result.truncated,
        duration_seconds=result.duration_seconds,
    )


def merge_matches(branches: Iterable[Iterable[SearchMatch]]) -> list[SearchMatch]:
    """Объединить совпадения, схлопнув дубликаты по каноническому тексту."""

    merged: dict[str, SearchMatch] = {}
    for branch in branches:
        for match in branch:
            key = match.dedupe_key
            current = merged.get(key)
            if current is None:
                merged[key] = match
                continue
            merged[key] = _combine(current, match)

    return sorted(merged.values(), key=lambda item: (-item.score, item.question_text))


def _combine(left: SearchMatch, right: SearchMatch) -> SearchMatch:
    """Слить два совпадения одного вопроса из разных источников."""

    best = left if left.score >= right.score else right
    sources = list(left.sources)
    for ref in right.sources:
        if not any(
            existing.name == ref.name and existing.external_url == ref.external_url
            for existing in sources
        ):
            sources.append(ref)

    return SearchMatch(
        question_text=best.question_text,
        answer_text=best.answer_text or left.answer_text or right.answer_text,
        comment=best.comment or left.comment or right.comment,
        score=max(left.score, right.score),
        score_kind=best.score_kind,
        sources=sources,
        external_url=best.external_url or left.external_url or right.external_url,
        semantic_similarity=max(left.semantic_similarity, right.semantic_similarity),
        lexical_score=max(left.lexical_score, right.lexical_score),
        key=best.key or left.key or right.key,
    )


def apply_threshold(
    matches: Iterable[SearchMatch],
    *,
    min_score: float,
) -> list[SearchMatch]:
    """Отфильтровать совпадения по единому порогу.

    Порог применяется к той же величине, по которой ранжируются оба источника.
    Прежнее разделение порогов по типу оценки потеряло смысл: оценка теперь
    одна, а `score_kind` обозначает происхождение совпадения.
    """

    return [match for match in matches if match.score >= min_score]


def limit_matches(matches: list[SearchMatch], limit: int) -> list[SearchMatch]:
    """Ограничить число совпадений, сохранив порядок по убыванию оценки."""

    if limit <= 0:
        return []
    return matches[:limit]


__all__ = [
    "EXTERNAL_STATUS_MAP",
    "apply_threshold",
    "external_matches",
    "external_report",
    "limit_matches",
    "local_matches",
    "local_report",
    "merge_matches",
]
