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
from chgk_agent.embeddings.text import embedding_document_text, embedding_query_text
from chgk_agent.external.base import (
    ExternalErrorKind,
    ExternalQuestionSource,
    ExternalSearchResult,
    ExternalStatus,
)
from chgk_agent.logging_setup import get_logger, get_request_id, new_request_id
from chgk_agent.observability.metrics import Metrics, get_metrics
from chgk_agent.search.corpus import (
    ensure_corpus_index as build_corpus_index,
)
from chgk_agent.search.corpus import term_informativeness
from chgk_agent.search.lexical import get_corpus_index
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
from chgk_agent.search.scoring import (
    NONE_KIND,
    CandidateScore,
    ScoredCandidate,
    cosine_similarity,
    score_candidates,
)

logger = get_logger(__name__)

GENERATION_SYSTEM_PROMPT = (
    "Ты игрок «Что? Где? Когда?». Тебе дают вопрос, и ты отвечаешь на него "
    "кратко и по существу. Если тебе приводят похожие вопросы, которые ты "
    "встречал раньше, используй их как подсказку, но отвечай на заданный "
    "вопрос, а не пересказывай подсказки."
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


async def embed_query(state: dict, deps: SearchDeps) -> dict:
    """Построить единый эмбеддинг описания для обеих веток поиска.

    Описание нормализуется тем же правилом, что и сохраняемые вопросы, а
    вектор считается один раз: семантическая близость локальных вопросов и
    карточек внешнего источника измеряется одной и той же величиной, поэтому
    второй вызов провайдера был бы и лишним, и источником расхождения.
    """

    text = embedding_query_text(state["query"])
    try:
        vectors = await deps.embedding_provider.embed([text])
    except Exception as error:
        logger.warning("эмбеддинг описания недоступен", error=str(error))
        return {"query_text": text, "query_embedding": None, "query_embedding_error": str(error)}

    embedding = list(vectors[0]) if vectors and vectors[0] else None
    return {
        "query_text": text,
        "query_embedding": embedding,
        "query_embedding_error": None if embedding else "провайдер вернул пустой эмбеддинг",
    }


async def ensure_corpus_index(state: dict, deps: SearchDeps) -> dict:
    """Подготовить лексический индекс корпуса до ветвления поиска.

    Индекс нужен обеим веткам: локальной — как источник кандидатов, внешней —
    как источник редкости терминов описания. Узел стоит параллельно вычислению
    эмбеддинга, поэтому его время скрыто за обращением к провайдеру, а не
    добавляется к задержке поиска.

    Отказ подготовки не отменяет поиск: состояние получает причину, метрики —
    счётчик, а внешний запрос строится без учёта редкости терминов.
    """

    timeout = deps.settings.search.corpus_index_timeout_seconds
    metrics = deps.current_metrics.search
    corpus = get_corpus_index()
    if corpus.is_ready:
        metrics.corpus_index.labels(outcome="ready").inc()
        return {
            "term_weights": term_informativeness(
                corpus.index, state.get("query", "")
            ),
            "term_weights_error": None,
        }

    try:
        async with asyncio.timeout(timeout):
            async with deps.session_factory() as session:
                index = await build_corpus_index(session, cache=corpus)
    except TimeoutError:
        return _fail_term_weights(
            deps,
            outcome="timeout",
            reason="timeout",
            error="таймаут подготовки лексического индекса",
        )
    except Exception as error:  # недоступная база, схема, ошибка сборки
        logger.warning("лексический индекс недоступен", error=str(error))
        return _fail_term_weights(
            deps,
            outcome="unavailable",
            reason="unavailable",
            error=str(error),
        )

    metrics.corpus_index.labels(outcome="built").inc()
    return {
        "term_weights": term_informativeness(index, state.get("query", "")),
        "term_weights_error": None,
    }


def _fail_term_weights(
    deps: SearchDeps,
    *,
    outcome: str,
    reason: str,
    error: str,
) -> dict:
    """Отметить недоступность информативности терминов без срыва поиска."""

    deps.current_metrics.search.corpus_index.labels(outcome=outcome).inc()
    deps.current_metrics.search.term_weights_unavailable.labels(reason=reason).inc()
    return {"term_weights": None, "term_weights_error": error}


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
                    query_embedding=state.get("query_embedding"),
                    use_lexical=not state.get("disable_lexical", False),
                    lexical_candidate_limit=(
                        deps.settings.search.lexical_candidate_limit
                    ),
                    semantic_candidate_limit=(
                        deps.settings.search.semantic_fetch_limit
                    ),
                    semantic_min_score=deps.settings.search.semantic_min_score,
                    query_embedding_failed=bool(
                        state.get("query_embedding_error")
                    ) and not state.get("query_embedding"),
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

    if not outcome.lexical_used:
        reason = "disabled" if state.get("disable_lexical") else "unavailable"
        deps.current_metrics.search.lexical_degraded.labels(reason=reason).inc()

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
    # Внешняя выдача берётся шире запрошенной: страница сайта содержит
    # `page_limit` карточек, и все они должны пройти единую оценку, иначе
    # сравнение шло бы только по тем, что случайно попали в короткий список.
    limit = deps.settings.external.page_limit
    term_weights = state.get("term_weights")
    try:
        async with asyncio.timeout(timeout):
            result = await source.search(
                state["query"], limit=limit, term_weights=term_weights
            )
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


async def score_external(state: dict, deps: SearchDeps) -> dict:
    """Посчитать семантическую близость карточек внешнего источника.

    Карточки приходят без векторов, а единая оценка требует одной и той же
    величины для обоих источников. Если эмбеддинг получить не удалось, внешние
    совпадения не показываются: подставлять вместо близости позицию сайта
    означало бы вернуть несопоставимую шкалу в выдачу.
    """

    result: ExternalSearchResult | None = state.get("external")
    if result is None or not result.matches:
        return {}

    embedding = state.get("query_embedding")
    if not embedding:
        return _fail_external_embedding(
            result,
            deps,
            error=state.get("query_embedding_error") or "эмбеддинг описания недоступен",
            reason="description_embedding_failed",
        )

    texts = [
        embedding_document_text(match.question_text, match.answer_text)
        for match in result.matches
    ]
    try:
        vectors = await deps.embedding_provider.embed(texts)
    except Exception as error:
        logger.warning("эмбеддинг внешних карточек недоступен", error=str(error))
        return _fail_external_embedding(
            result, deps, error=str(error), reason="provider_error"
        )

    if len(vectors) != len(result.matches):
        return _fail_external_embedding(
            result,
            deps,
            error="провайдер вернул неверное число эмбеддингов карточек",
            reason="unexpected_count",
        )

    for match, vector in zip(result.matches, vectors, strict=True):
        match.embedding = list(vector) if vector else None
        match.semantic_similarity = (
            cosine_similarity(embedding, match.embedding) if match.embedding else 0.0
        )

    if any(match.embedding is None for match in result.matches):
        return _fail_external_embedding(
            result,
            deps,
            error="провайдер вернул пустой эмбеддинг карточки",
            reason="empty_embedding",
        )

    return {"external": result}


def _fail_external_embedding(
    result: ExternalSearchResult,
    deps: SearchDeps,
    *,
    error: str,
    reason: str = "provider_error",
) -> dict:
    """Пометить внешний источник недоступным из-за сбоя эмбеддинга."""

    result.status = ExternalStatus.UNAVAILABLE
    result.error = error
    result.error_kind = ExternalErrorKind.EMBEDDING_FAILED
    result.matches = []
    deps.current_metrics.search.external_embedding_failed.labels(
        reason=reason
    ).inc()
    logger.warning("внешние совпадения исключены из выдачи", error=error)
    return {"external": result}


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


def score_matches(state: dict, deps: SearchDeps) -> dict:
    """Посчитать единую оценку для совпадений обоих источников.

    Лексический сигнал и итоговая оценка считаются здесь, а не в ветках: BM25 —
    функция от коллекции, и только общий узел видит кандидатов обеих сторон
    одновременно. Нормировка насыщением выбрана вместо деления на максимум:
    максимум зависел бы от состава выдачи, то есть от того, сколько результатов
    вернул каждый источник.
    """

    matches: list[SearchMatch] = state.get("matches", [])
    if not matches:
        return {}

    description = state.get("query_text") or state.get("query", "")
    settings = deps.settings.search
    candidates: list[ScoredCandidate] = []
    keys: list[str] = []
    for index, match in enumerate(matches):
        key = match.key or f"match:{index}"
        keys.append(key)
        candidates.append(
            ScoredCandidate(
                key=key,
                text=embedding_document_text(match.question_text, match.answer_text),
                semantic_similarity=match.semantic_similarity,
            )
        )

    base_index = get_corpus_index().index
    scores = score_candidates(
        description,
        candidates,
        base_index=base_index,
        lexical_weight=settings.lexical_weight,
        lexical_saturation=settings.lexical_saturation,
    )

    for match, key in zip(matches, keys, strict=True):
        score = scores.get(key)
        if score is None:
            score = CandidateScore(
                score=match.semantic_similarity,
                lexical_score=0.0,
                lexical_normalized=0.0,
                score_kind=NONE_KIND,
            )
        match.score = round(score.score, 6)
        match.lexical_score = round(score.lexical_score, 6)
        match.score_kind = score.score_kind
        for ref in match.sources:
            ref.score = match.score
            ref.score_kind = match.score_kind

    return {"matches": matches}


def rerank(state: dict) -> dict:
    """Применить порог и ограничение числа результатов."""

    matches = apply_threshold(
        state.get("matches", []),
        min_score=state.get("min_score", 0.0),
    )
    # Потолок единицы может сблизить оценки, поэтому ничьи разрешаются
    # составляющими: сначала семантика, затем лексика, и лишь потом текст.
    matches.sort(
        key=lambda item: (
            -item.score,
            -item.semantic_similarity,
            -item.lexical_score,
            item.question_text,
        )
    )
    matches = limit_matches(matches, state.get("limit", 20))
    return {"matches": matches}


def has_results(state: dict) -> str:
    """Выбрать ветку после объединения совпадений.

    Если ни один источник не дал совпадений, ранжировать нечего, но
    генерация всё равно должна получить управление: ответ на исходный
    вопрос формируется и с пустым списком похожих вопросов.
    """

    return "rerank" if state.get("matches") else "generate"


def needs_generation(state: dict) -> str:
    """Выбрать ветку после ранжирования.

    Генерация вызывается всегда, когда она запрошена: совпадения нужны
    только как необязательный контекст, а не как условие ответа.
    """

    return "generate" if state.get("generate_answer", True) else "skip"


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
    if answer is None and state.get("generate_answer", True):
        answer = GeneratedAnswer(
            text=None,
            available=False,
            error="генерация ответа недоступна",
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
        degraded=bool(state.get("term_weights_error")),
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
        failed = list(outcome.failed_sources)
        if outcome.degraded:
            failed.append("corpus_index")
        logger.warning(
            "поиск завершён с частичными результатами",
            operation="search",
            status="partial",
            duration_seconds=round(duration, 4),
            sources=failed,
            request_id=outcome.request_id,
        )


def _generation_prompt(query: str, matches: list[SearchMatch]) -> str:
    """Собрать промпт генерации из вопроса и похожих вопросов.

    Список похожих вопросов необязателен: когда ничего не нашлось, модель
    отвечает на исходный вопрос без подсказок.
    """

    lines = [f"Вопрос: {query}", ""]
    if matches:
        lines.append("Раньше ты встречал такие похожие вопросы:")
        for index, match in enumerate(matches, start=1):
            answer = match.answer_text or "не указан"
            lines.append(f"{index}. Вопрос: {match.question_text}\n   Ответ: {answer}")
        lines.append("")

    lines.append("Ответь на вопрос.")
    return "\n".join(lines)


__all__ = [
    "SearchDeps",
    "ensure_corpus_index",
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
