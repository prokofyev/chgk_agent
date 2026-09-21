"""Адаптер внешнего источника gotquestions.online.

Поиск выполняется обычным `GET /search` без сессий и кук. Сайт принимает
запрос не длиннее 50 символов, требует осмысленный `User-Agent` и может
молча отдать страницу без блока результатов, поэтому адаптер различает
успешную непустую выдачу, пустую выдачу, отвергнутый запрос и недоступность.
"""

import time
from collections.abc import Iterable
from urllib.parse import urlencode

import httpx

from chgk_agent.config import ExternalSourceSettings
from chgk_agent.external.base import (
    ExternalErrorKind,
    ExternalMatch,
    ExternalQuestionSource,
    ExternalSearchResult,
    ExternalStatus,
)
from chgk_agent.external.parser import (
    ParsedExternalMatch,
    ParsedSearchPage,
    parse_search_page,
)
from chgk_agent.external.query import ShortQueryBuilder
from chgk_agent.external.ratelimit import CircuitBreaker, RateLimiter
from chgk_agent.external.robots import RobotsPolicy
from chgk_agent.logging_setup import get_logger
from chgk_agent.observability.metrics import Metrics, get_metrics

logger = get_logger(__name__)

SOURCE_NAME = "gotquestions"
SEARCH_PATH = "/search"
SEARCH_TYPE = "questions"

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class GotQuestionsSource(ExternalQuestionSource):
    """Внешний источник вопросов на базе gotquestions.online."""

    def __init__(
        self,
        settings: ExternalSourceSettings,
        *,
        client: httpx.AsyncClient | None = None,
        metrics: Metrics | None = None,
        query_builder: ShortQueryBuilder | None = None,
        rate_limiter: RateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        robots: RobotsPolicy | None = None,
    ) -> None:
        self._settings = settings
        self._metrics = metrics or get_metrics()
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"User-Agent": settings.user_agent},
            timeout=settings.timeout_seconds,
            follow_redirects=True,
        )
        self._owns_client = client is None
        self._query_builder = query_builder or ShortQueryBuilder(
            max_chars=settings.max_query_chars,
            max_queries=settings.max_query_terms,
        )
        self._rate_limiter = rate_limiter or RateLimiter(settings.min_interval_seconds)
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            threshold=settings.circuit_breaker_threshold,
            reset_seconds=settings.circuit_breaker_reset_seconds,
        )
        self._robots = robots or RobotsPolicy(
            self._client,
            user_agent=settings.user_agent,
            enabled=settings.respect_robots,
        )

    @property
    def name(self) -> str:
        """Имя источника."""

        return SOURCE_NAME

    @property
    def enabled(self) -> bool:
        """Включён ли источник настройкой."""

        return self._settings.enabled

    @property
    def user_agent(self) -> str:
        """Настроенный `User-Agent`."""

        return self._settings.user_agent

    async def aclose(self) -> None:
        """Закрыть собственный HTTP-клиент."""

        if self._owns_client:
            await self._client.aclose()

    async def search(
        self,
        description: str,
        *,
        limit: int = 20,
        term_weights: dict[str, float] | None = None,
    ) -> ExternalSearchResult:
        """Найти вопросы по описанию."""

        started = time.perf_counter()
        if not self._settings.enabled:
            return ExternalSearchResult(
                status=ExternalStatus.DISABLED,
                query=description,
                error="внешний источник отключён настройкой",
            )

        plan = self._query_builder.build(description, term_weights=term_weights)
        if plan.truncated:
            self._metrics.search.external_truncated.inc()

        result = ExternalSearchResult(
            status=ExternalStatus.OK,
            query=plan.primary,
            queries=list(plan.queries),
            truncated=plan.truncated,
        )

        if not plan.queries:
            result.status = ExternalStatus.EMPTY
            return result

        if not self._circuit_breaker.allow():
            result.status = ExternalStatus.UNAVAILABLE
            result.error_kind = ExternalErrorKind.CIRCUIT_OPEN
            result.error = "предохранитель внешнего источника разомкнут"
            self._record(result, started)
            return result

        seen: dict[str, ExternalMatch] = {}
        statuses: list[ExternalStatus] = []

        for query in plan.queries:
            page_number = 1
            while page_number <= self._settings.max_pages:
                page = await self._search_single(
                    query, limit=limit, page=page_number
                )
                result.pages_fetched += 1
                if page.status is not ExternalStatus.UNAVAILABLE:
                    statuses.append(page.status)
                if page.status is ExternalStatus.UNAVAILABLE:
                    result.status = ExternalStatus.UNAVAILABLE
                    result.error = page.error
                    result.error_kind = page.error_kind
                    result.matches = []
                    self._record(result, started)
                    return result

                for match in page.matches:
                    key = match.external_id or match.question_text
                    if key not in seen:
                        seen[key] = match
                result.has_more = page.has_more
                if len(seen) >= limit or not page.has_more:
                    break
                page_number += 1
            if len(seen) >= limit:
                break

        matches = list(seen.values())[:limit]
        for position, match in enumerate(matches, start=1):
            match.position = position

        result.matches = matches
        if matches:
            result.status = ExternalStatus.OK
        elif ExternalStatus.REJECTED in statuses:
            result.status = ExternalStatus.REJECTED
        elif ExternalStatus.UNAVAILABLE in statuses:
            result.status = ExternalStatus.UNAVAILABLE
        else:
            result.status = ExternalStatus.EMPTY

        self._record(result, started)
        return result

    async def _search_single(
        self,
        query: str,
        *,
        limit: int,
        page: int = 1,
    ) -> ExternalSearchResult:
        """Выполнить поиск по одному короткому запросу и странице выдачи."""

        self._metrics.search.external_queries.observe(len(query))
        result = ExternalSearchResult(status=ExternalStatus.OK, query=query, page=page)

        if not await self._robots.allows(_search_url(self._settings.base_url, query)):
            return self._unavailable(
                result,
                ExternalErrorKind.ROBOTS_DISALLOWED,
                "robots.txt запрещает страницу поиска",
            )

        last_error: Exception | None = None
        for attempt in range(1, self._settings.max_attempts + 1):
            result.attempts = attempt
            await self._rate_limiter.acquire()
            try:
                response = await self._client.get(
                    SEARCH_PATH,
                    params={
                        "search": query,
                        "type": SEARCH_TYPE,
                        "page": max(page, 1),
                        "limit": max(limit, 1),
                    },
                )
            except httpx.TimeoutException as error:
                last_error = error
                if attempt < self._settings.max_attempts:
                    continue
                return self._unavailable(
                    result, ExternalErrorKind.TIMEOUT, "таймаут внешнего источника"
                )
            except httpx.HTTPError as error:
                last_error = error
                if attempt < self._settings.max_attempts:
                    continue
                return self._unavailable(
                    result, ExternalErrorKind.TRANSPORT, str(error)
                )

            if response.status_code in RETRYABLE_STATUS:
                last_error = httpx.HTTPStatusError(
                    "retryable", request=response.request, response=response
                )
                if attempt < self._settings.max_attempts:
                    continue
                kind = (
                    ExternalErrorKind.RATE_LIMITED
                    if response.status_code == 429
                    else ExternalErrorKind.SERVER
                )
                return self._unavailable(
                    result, kind, f"внешний источник вернул {response.status_code}"
                )

            if response.status_code == 403:
                return self._unavailable(
                    result,
                    ExternalErrorKind.FORBIDDEN,
                    "внешний источник отклонил User-Agent (403)",
                )

            if response.status_code >= 400:
                return self._unavailable(
                    result,
                    ExternalErrorKind.SERVER,
                    f"внешний источник вернул {response.status_code}",
                )

            parsed = parse_search_page(response.text)
            break
        else:  # pragma: no cover - защита от неожиданного выхода из цикла
            return self._unavailable(
                result, ExternalErrorKind.TRANSPORT, str(last_error or "неизвестная ошибка")
            )

        self._circuit_breaker.record_success()
        return self._from_parsed(result, parsed)

    def _from_parsed(
        self, result: ExternalSearchResult, parsed: ParsedSearchPage
    ) -> ExternalSearchResult:
        """Преобразовать разобранную страницу в результат поиска."""

        if not parsed.recognized:
            return self._unavailable(
                result,
                ExternalErrorKind.UNRECOGNIZED,
                "разметка выдачи внешнего источника не распознана",
            )

        if parsed.has_empty_marker:
            result.status = ExternalStatus.EMPTY
            return result

        if not parsed.matches:
            if parsed.has_results_block:
                return self._unavailable(
                    result,
                    ExternalErrorKind.UNRECOGNIZED,
                    "страница сообщает о найденных вопросах, но карточки не разобраны",
                )
            result.status = ExternalStatus.REJECTED
            result.error = "внешний источник отверг запрос без блока результатов"
            return result

        result.status = ExternalStatus.OK
        result.matches = [_to_match(item) for item in parsed.matches]
        result.has_more = parsed.has_more
        return result

    def _unavailable(
        self,
        result: ExternalSearchResult,
        kind: ExternalErrorKind,
        message: str,
    ) -> ExternalSearchResult:
        """Пометить результат как недоступность источника."""

        self._circuit_breaker.record_failure()
        result.status = ExternalStatus.UNAVAILABLE
        result.error_kind = kind
        result.error = message
        result.matches = []
        return result

    def _record(self, result: ExternalSearchResult, started: float) -> None:
        """Опубликовать метрики обращения к внешнему источнику."""

        duration = time.perf_counter() - started
        result.duration_seconds = round(duration, 4)

        metrics = self._metrics
        # Общие метрики операции поиска (счётчик запросов, длительность,
        # ошибки, частичные результаты) публикует граф поиска: он знает
        # итоговый статус каждой ветки. Здесь остаются только метрики,
        # специфичные для внешнего HTTP-источника.
        metrics.search.external_duration.observe(duration)

        if result.status is ExternalStatus.EMPTY:
            metrics.search.external_empty.inc()
        elif result.status is ExternalStatus.REJECTED:
            metrics.search.external_rejected.inc()

        logger.info(
            "поиск во внешнем источнике",
            source=SOURCE_NAME,
            status=result.status.value,
            duration_seconds=result.duration_seconds,
            matches=len(result.matches),
            truncated=result.truncated,
            queries=len(result.queries),
            pages=result.pages_fetched,
        )


def _search_url(base_url: str, query: str) -> str:
    """Абсолютный URL страницы поиска для проверки правил `robots.txt`."""

    base = base_url.rstrip("/")
    return f"{base}{SEARCH_PATH}?{urlencode({'search': query, 'type': SEARCH_TYPE})}"


def _to_match(item: ParsedExternalMatch) -> ExternalMatch:
    """Преобразовать разобранную карточку в совпадение."""

    return ExternalMatch(
        title=item.title,
        question_text=item.question_text,
        answer_text=item.answer_text,
        comment=item.comment,
        authors=list(item.authors),
        pack=item.pack,
        external_url=item.external_url,
        external_id=item.external_id,
        position=item.position,
        score=0.0,
        score_kind="",
    )


def deduplicate_matches(matches: Iterable[ExternalMatch]) -> list[ExternalMatch]:
    """Убрать дубликаты по идентификатору вопроса, сохранив порядок."""

    seen: set[str] = set()
    unique: list[ExternalMatch] = []
    for match in matches:
        key = match.external_id or match.question_text
        if key in seen:
            continue
        seen.add(key)
        unique.append(match)
    return unique


__all__ = [
    "GotQuestionsSource",
    "SOURCE_NAME",
    "deduplicate_matches",
]
