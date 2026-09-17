"""Узлы графа поиска как обычные async-функции.

Каждый узел принимает состояние и зависимости и возвращает часть
состояния. Узлы не знают про LangGraph, поэтому тестируются изолированно
с фейковыми зависимостями.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.config import Settings
from chgk_agent.embeddings.base import ChatProvider, EmbeddingProvider
from chgk_agent.external.base import (
    ExternalQuestionSource,
    ExternalSearchResult,
    ExternalStatus,
)
from chgk_agent.logging_setup import get_logger, get_request_id, new_request_id
from chgk_agent.observability.metrics import Metrics, get_metrics
from chgk_agent.search.local import LocalSearch, LocalSearchOutcome
from chgk_agent.search.merge import (
    apply_threshold,
    external_matches,
    external_report,
    limit_matches,
    local_matches,
    local_report,
    merge_matches,
)
from chgk_agent.search.models import (
    GeneratedAnswer,
    SearchMatch,
    SearchOutcome,
    SourceReport,
)

logger = get_logger(__name__)

GENERATION_SYSTEM_PROMPT = (
    "Ты помощник редактора «Что? Где? Когда?». Отвечай кратко и опирайся "
    "только на приведённые вопросы и ответы. Не придумывай факты."
)


@dataclass(slots=True)
class SearchDeps:
    """Зависимости узлов графа поиска."""

    session_factory: async_sessionmaker[AsyncSession]
    embedding_provider: EmbeddingProvider
    external_source: ExternalQuestionSource | None = None
    chat_provider: ChatProvider | None = None
    settings: Settings = field(default_factory=Settings)
    metrics: Metrics | None = None
    clock: Callable[[], float] = time.perf_counter

    @property
    def current_metrics(self) -> Metrics:
        """Метрики приложения."""

        return self.metrics or get_metrics()


async def parse_request(state: dict) -> dict:
    """Нормализовать описание вопроса и параметры поиска."""

    query = " ".join(str(state.get("query") or "").split())
    limit = max(int(state.get("limit") or 20), 1)
    min_score = float(state.get("min_score") or 0.0)
    generate = bool(state.get("generate_answer", True))
    request_id = state.get("request_id") or get_request_id() or new_request_id()

    return {
        "query": query,
        "started_at": time.perf_counter(),
        "limit": limit,
        "min_score": min_score,
        "generate_answer": generate,
        "request_id": request_id,
    }


async def local_search(state: dict, deps: SearchDeps) -> dict:
    """Выполнить локальный гибридный поиск с собственным таймаутом."""

    started = deps.clock()
    timeout = deps.settings.search.local_timeout_seconds
    query = state["query"]
    limit = state["limit"]

    try:
        async with asyncio.timeout(timeout):
            async with deps.session_factory() as session:
                search = LocalSearch(
                    session,
                    deps.embedding_provider,
                    use_lexical=not state.get("disable_lexical", False),
                    semantic_min_score=deps.settings.search.semantic_min_score,
                    lexical_min_score=deps.settings.search.lexical_min_score,
                )
                outcome = await search.search_with_diagnostics(query, limit=limit)
    except TimeoutError:
        outcome = LocalSearchOutcome(
            degraded=True, semantic_used=False, error="таймаут локального поиска"
        )
    except Exception as error:  # сеть, схема, недоступная база
        outcome = LocalSearchOutcome(
            degraded=True, semantic_used=False, error=str(error)
        )
        logger.warning("локальный поиск недоступен", error=str(error))

    return {"local": outcome, "local_seconds": deps.clock() - started}


async def external_search(state: dict, deps: SearchDeps) -> dict:
    """Выполнить поиск во внешнем источнике с собственным таймаутом."""

    started = deps.clock()
    source = deps.external_source
    if source is None:
        result = ExternalSearchResult(
            status=ExternalStatus.DISABLED, query=state["query"], error="источник не настроен"
        )
        return {"external": result, "external_seconds": deps.clock() - started}

    timeout = deps.settings.search.external_timeout_seconds
    try:
        async with asyncio.timeout(timeout):
            result = await source.search(state["query"], limit=state["limit"])
    except TimeoutError:
        result = ExternalSearchResult(
            status=ExternalStatus.UNAVAILABLE,
            query=state["query"],
            error="таймаут внешнего поиска",
        )
    except Exception as error:
        result = ExternalSearchResult(
            status=ExternalStatus.UNAVAILABLE, query=state["query"], error=str(error)
        )
        logger.warning("внешний поиск недоступен", error=str(error))

    return {"external": result, "external_seconds": deps.clock() - started}


def merge_and_dedupe(state: dict) -> dict:
    """Объединить совпадения источников и схлопнуть дубликаты."""

    local: LocalSearchOutcome | None = state.get("local")
    external: ExternalSearchResult | None = state.get("external")

    local_items = local_matches(local) if local is not None else []
    external_items = external_matches(external) if external is not None else []
    matches = merge_matches([local_items, external_items])

    reports: list[SourceReport] = []
    if local is not None:
        reports.append(
            local_report(
                local,
                matches=len(local_items),
                duration_seconds=state.get("local_seconds", 0.0),
            )
        )
    if external is not None:
        reports.append(external_report(external, matches=len(external_items)))

    return {"matches": matches, "sources": reports}


def rerank(state: dict) -> dict:
    """Применить порог и ограничение числа результатов."""

    matches = apply_threshold(
        state.get("matches", []),
        min_score=state.get("min_score", 0.0),
        thresholds_by_kind=state.get("thresholds"),
    )
    matches = limit_matches(matches, state.get("limit", 20))
    return {"matches": matches}


def has_results(state: dict) -> str:
    """Выбрать ветку после объединения совпадений.

    Если ни один источник не дал совпадений, ранжировать нечего: граф
    сразу формирует ответ, минуя переранжирование и генерацию.
    """

    return "rerank" if state.get("matches") else "skip"


def needs_generation(state: dict) -> str:
    """Выбрать ветку после ранжирования.

    Генерация вызывается только тогда, когда она запрошена и есть хотя бы
    одно подтверждающее совпадение; иначе граф сразу формирует ответ.
    """

    if not state.get("generate_answer", True):
        return "skip"
    if not state.get("matches"):
        return "skip"
    return "generate"


async def generate(state: dict, deps: SearchDeps) -> dict:
    """Сгенерировать ответ по найденным совпадениям."""

    matches: list[SearchMatch] = state["matches"]
    provider = deps.chat_provider
    if provider is None:
        return {
            "answer": GeneratedAnswer(
                text=None, available=False, error="провайдер генерации не настроен"
            )
        }

    prompt = _generation_prompt(state.get("query", ""), matches)
    try:
        text = await provider.complete(prompt, system=GENERATION_SYSTEM_PROMPT)
    except Exception as error:
        logger.warning("генерация ответа недоступна", error=str(error))
        return {
            "answer": GeneratedAnswer(text=None, available=False, error=str(error))
        }

    return {
        "answer": GeneratedAnswer(
            text=text.strip() or None,
            available=bool(text.strip()),
            used_matches=[match.question_text for match in matches],
        )
    }


def format_response(state: dict, deps: SearchDeps | None = None) -> dict:
    """Собрать итоговый ответ поиска и записать метрики."""

    matches: list[SearchMatch] = state.get("matches", [])
    external = state.get("external")
    answer: GeneratedAnswer | None = state.get("answer")
    if answer is None and state.get("generate_answer", True) and not matches:
        answer = GeneratedAnswer(
            text=None,
            available=False,
            error="нет подтверждающих совпадений",
        )
    truncated_query = (
        external.query if external is not None and external.truncated else None
    )

    outcome = SearchOutcome(
        query=state.get("query", ""),
        matches=matches,
        sources=state.get("sources", []),
        answer=answer,
        request_id=state.get("request_id"),
        truncated_query=truncated_query,
    )

    started_at = state.get("started_at")
    duration = time.perf_counter() - started_at if started_at else 0.0
    status = "partial" if outcome.is_partial else ("empty" if outcome.is_empty else "ok")

    _record_metrics(deps=deps, outcome=outcome, duration=duration)

    logger.info(
        "поиск завершён",
        operation="search",
        status=status,
        duration_seconds=round(duration, 4),
        sources=[report.source for report in outcome.sources],
        matches=len(matches),
        request_id=outcome.request_id,
    )
    return {"outcome": outcome}


def _record_metrics(
    *,
    deps: SearchDeps | None,
    outcome: SearchOutcome,
    duration: float,
) -> None:
    """Записать метрики завершённого поиска по источникам и статусам."""

    from chgk_agent.observability.metrics import get_metrics

    metrics = deps.current_metrics if deps is not None else get_metrics()

    for report in outcome.sources:
        metrics.record_search_request(report.source, report.status.value)
        metrics.search.duration.labels(source=report.source).observe(duration)
        metrics.record_matches(report.source, report.matches)
        if report.status.is_failure:
            metrics.record_partial_search(report.source)
            metrics.record_search_error(
                report.source, report.error_kind or report.status.value
            )

    if outcome.is_partial:
        logger.warning(
            "поиск завершён с частичными результатами",
            operation="search",
            status="partial",
            duration_seconds=round(duration, 4),
            sources=outcome.failed_sources,
            request_id=outcome.request_id,
        )


def _generation_prompt(query: str, matches: list[SearchMatch]) -> str:
    """Собрать промпт генерации из описания и найденных вопросов."""

    lines = [f"Описание вопроса: {query}", "", "Найденные вопросы и ответы:"]
    for index, match in enumerate(matches, start=1):
        answer = match.answer_text or "не указан"
        lines.append(f"{index}. Вопрос: {match.question_text}\n   Ответ: {answer}")
    lines.append("")
    lines.append("Сформулируй ответ на описание, опираясь только на эти данные.")
    return "\n".join(lines)


__all__ = [
    "SearchDeps",
    "external_search",
    "format_response",
    "generate",
    "has_results",
    "local_search",
    "merge_and_dedupe",
    "needs_generation",
    "parse_request",
    "rerank",
]
