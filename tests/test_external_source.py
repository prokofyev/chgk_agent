"""Тесты адаптера gotquestions.online: исходы, лимиты, деградация."""

from pathlib import Path

import httpx
import respx
from prometheus_client import CollectorRegistry

from chgk_agent.config import ExternalSourceSettings
from chgk_agent.external.base import (
    ExternalErrorKind,
    ExternalQuestionSource,
    ExternalStatus,
)
from chgk_agent.external.gotquestions import SOURCE_NAME, GotQuestionsSource
from chgk_agent.external.ratelimit import CircuitBreaker, RateLimiter
from chgk_agent.observability.metrics import Metrics

FIXTURES = Path(__file__).parent / "fixtures" / "external"
BASE_URL = "https://gotquestions.online"
USER_AGENT = "chgk-agent/0.1 (+https://localhost/chgk-agent)"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _settings(**overrides: object) -> ExternalSourceSettings:
    return ExternalSourceSettings(
        _env_file=None,
        base_url=BASE_URL,
        user_agent=USER_AGENT,
        min_interval_seconds=0.0,
        **overrides,
    )


def _metrics() -> Metrics:
    return Metrics(CollectorRegistry())


def _source(**overrides: object) -> GotQuestionsSource:
    settings = _settings(**overrides)
    client = httpx.AsyncClient(
        base_url=settings.base_url,
        headers={"User-Agent": settings.user_agent},
        timeout=settings.timeout_seconds,
    )
    return GotQuestionsSource(settings, client=client, metrics=_metrics())


async def test_source_satisfies_protocol() -> None:
    source = _source()

    assert isinstance(source, ExternalQuestionSource)
    assert source.name == SOURCE_NAME
    assert source.enabled is True


@respx.mock
async def test_non_empty_search_returns_matches() -> None:
    respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    source = _source(page_limit=5)

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.OK
    assert result.matches
    assert result.matches[0].external_url
    assert result.matches[0].answer_text
    # Позиция сайта остаётся диагностикой и в оценку не попадает.
    assert result.matches[0].score_kind == ""
    assert result.matches[0].score == 0.0
    assert result.query == "Тьюринг"
    assert result.truncated is False


@respx.mock
async def test_empty_result_is_not_an_error() -> None:
    respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_empty.html"))
    )
    source = _source()

    result = await source.search("ззззззззнесуществует", limit=5)

    assert result.status is ExternalStatus.EMPTY
    assert result.matches == []
    assert result.error is None


@respx.mock
async def test_rejected_query_gets_own_status() -> None:
    respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_rejected.html"))
    )
    source = _source()

    rejected_query = "предложил использовать две комнаты шахматы операто"
    assert len(rejected_query) <= 50

    result = await source.search(rejected_query, limit=5)

    assert result.status is ExternalStatus.REJECTED
    assert result.matches == []


@respx.mock
async def test_changed_markup_is_unavailable_not_empty() -> None:
    respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_changed_markup.html"))
    )
    source = _source()

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.UNAVAILABLE
    assert result.error_kind is ExternalErrorKind.UNRECOGNIZED


@respx.mock
async def test_forbidden_user_agent_is_unavailable() -> None:
    respx.get(f"{BASE_URL}/search").mock(return_value=httpx.Response(403))
    source = _source()

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.UNAVAILABLE
    assert result.error_kind is ExternalErrorKind.FORBIDDEN


@respx.mock
async def test_server_error_is_unavailable_after_retries() -> None:
    route = respx.get(f"{BASE_URL}/search").mock(return_value=httpx.Response(503))
    source = _source(max_attempts=2)

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.UNAVAILABLE
    assert result.error_kind is ExternalErrorKind.SERVER
    assert route.call_count == 2


@respx.mock
async def test_timeout_is_unavailable() -> None:
    respx.get(f"{BASE_URL}/search").mock(side_effect=httpx.ConnectTimeout("таймаут"))
    source = _source()

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.UNAVAILABLE
    assert result.error_kind is ExternalErrorKind.TIMEOUT


@respx.mock
async def test_retry_succeeds_after_transient_error() -> None:
    responses = [httpx.Response(503), httpx.Response(200, text=_fixture("search_ok.html"))]
    respx.get(f"{BASE_URL}/search").mock(side_effect=responses)
    source = _source(max_attempts=3)

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.OK
    assert result.matches


async def test_disabled_source_does_not_send_requests() -> None:
    settings = _settings(enabled=False)
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, text=_fixture("search_ok.html"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    source = GotQuestionsSource(settings, client=client, metrics=_metrics())

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.DISABLED
    assert captured == []
    assert source.enabled is False


@respx.mock
async def test_request_sends_user_agent_and_short_query() -> None:
    route = respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    source = _source()

    await source.search(
        "В докладе 1947 года этот человек предложил использовать две комнаты, "
        "двух не очень сильных игроков в шахматы и оператора. Назовите этого человека.",
        limit=5,
    )

    request = route.calls[0].request
    assert request.headers["User-Agent"] == USER_AGENT
    assert len(request.url.params.get("search", "")) <= 50
    assert request.url.params.get("type") == "questions"
    assert request.url.params.get("limit") == "5"


@respx.mock
async def test_circuit_breaker_opens_after_threshold() -> None:
    respx.get(f"{BASE_URL}/search").mock(return_value=httpx.Response(503))
    settings = _settings(max_attempts=1, circuit_breaker_threshold=2)
    source = GotQuestionsSource(
        settings,
        client=httpx.AsyncClient(base_url=BASE_URL, headers={"User-Agent": USER_AGENT}),
        metrics=_metrics(),
    )

    first = await source.search("Тьюринг", limit=5)
    second = await source.search("Тьюринг", limit=5)
    third = await source.search("Тьюринг", limit=5)

    assert first.status is ExternalStatus.UNAVAILABLE
    assert second.status is ExternalStatus.UNAVAILABLE
    assert third.status is ExternalStatus.UNAVAILABLE
    assert third.error_kind is ExternalErrorKind.CIRCUIT_OPEN


@respx.mock
async def test_circuit_breaker_recovers_after_reset() -> None:
    now = [0.0]
    breaker = CircuitBreaker(threshold=1, reset_seconds=0.05, clock=lambda: now[0])
    respx.get(f"{BASE_URL}/search").mock(return_value=httpx.Response(503))
    settings = _settings(max_attempts=1)
    source = GotQuestionsSource(
        settings,
        client=httpx.AsyncClient(base_url=BASE_URL, headers={"User-Agent": USER_AGENT}),
        metrics=_metrics(),
        circuit_breaker=breaker,
    )

    await source.search("Тьюринг", limit=5)
    assert breaker.is_open

    now[0] = 1.0
    assert breaker.allow()
    assert breaker.failures == 0


async def test_rate_limiter_enforces_minimum_interval() -> None:
    slept: list[float] = []
    timestamps = iter([0.0, 0.2, 1.2])

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    limiter = RateLimiter(
        min_interval_seconds=1.0,
        clock=lambda: next(timestamps),
        sleep=fake_sleep,
    )

    first_wait = await limiter.acquire()
    second_wait = await limiter.acquire()

    assert first_wait == 0.0
    assert second_wait > 0.0
    assert slept


async def test_rate_limiter_with_zero_interval_does_not_sleep() -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    limiter = RateLimiter(
        0.0, clock=lambda: 0.0, sleep=fake_sleep
    )

    first = await limiter.acquire()
    second = await limiter.acquire()

    assert first == 0.0
    assert second == 0.0
    assert slept == []


@respx.mock
async def test_external_metrics_are_recorded() -> None:
    respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    metrics = _metrics()
    settings = _settings()
    source = GotQuestionsSource(
        settings,
        client=httpx.AsyncClient(base_url=BASE_URL, headers={"User-Agent": USER_AGENT}),
        metrics=metrics,
    )

    await source.search("Тьюринг", limit=5)

    assert metrics.search.external_duration._sum.get() > 0
    assert metrics.search.external_queries._sum.get() > 0


@respx.mock
async def test_truncated_query_increments_metric() -> None:
    respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    metrics = _metrics()
    settings = _settings()
    source = GotQuestionsSource(
        settings,
        client=httpx.AsyncClient(base_url=BASE_URL, headers={"User-Agent": USER_AGENT}),
        metrics=metrics,
    )

    result = await source.search(
        "Английский учёный прошлого века Ангус Бейтмен во время экспериментов давал ИМ "
        "клички: Щетинка, Лысый, Волосатое крыло. Назовите ИХ.",
        limit=5,
    )

    assert result.truncated is True
    assert metrics.search.external_truncated._value.get() == 1


@respx.mock
async def test_multiple_pages_are_merged_without_duplicates() -> None:
    route = respx.get(f"{BASE_URL}/search").mock(
        side_effect=[
            httpx.Response(200, text=_fixture("search_ok.html")),
            httpx.Response(200, text=_fixture("search_ok_page2.html")),
        ]
    )
    source = _source(max_pages=2)

    result = await source.search("Тьюринг", limit=100)

    assert result.status is ExternalStatus.OK
    ids = [match.external_id for match in result.matches]
    assert len(ids) == len(set(ids))
    assert route.call_count >= 1


@respx.mock
async def test_source_does_not_score_by_position() -> None:
    """Источник не выставляет оценку: её считает граф по единой мере.

    Позиция в выдаче остаётся диагностикой, но не участвует в ранжировании:
    раньше она давала внешним результатам шкалу, несопоставимую с локальной.
    """

    respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    source = _source(page_limit=3)

    result = await source.search("Тьюринг", limit=3)

    assert [match.position for match in result.matches] == [1, 2, 3]
    assert all(match.score == 0.0 for match in result.matches)
    assert all(match.semantic_similarity == 0.0 for match in result.matches)


@respx.mock
async def test_page_parameter_is_sent_for_pagination() -> None:
    route = respx.get(f"{BASE_URL}/search").mock(
        side_effect=[
            httpx.Response(200, text=_fixture("search_ok.html")),
            httpx.Response(200, text=_fixture("search_ok_page2.html")),
        ]
    )
    source = _source(max_pages=2)

    result = await source.search("Тьюринг", limit=100)

    pages = [call.request.url.params.get("page") for call in route.calls]
    assert pages == ["1", "2"]
    assert result.pages_fetched == 2
    assert result.has_more is True


@respx.mock
async def test_pagination_stops_at_configured_limit() -> None:
    route = respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    source = _source(max_pages=1)

    result = await source.search("Тьюринг", limit=100)

    assert route.call_count == 1
    assert result.pages_fetched == 1
    assert result.has_more is True


@respx.mock
async def test_pagination_stops_when_limit_reached() -> None:
    route = respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    source = _source(max_pages=5)

    result = await source.search("Тьюринг", limit=3)

    assert route.call_count == 1
    assert len(result.matches) == 3


@respx.mock
async def test_empty_and_rejected_metrics_are_counted() -> None:
    route = respx.get(f"{BASE_URL}/search")
    metrics = _metrics()
    settings = _settings()
    source = GotQuestionsSource(
        settings,
        client=httpx.AsyncClient(base_url=BASE_URL, headers={"User-Agent": USER_AGENT}),
        metrics=metrics,
    )

    route.mock(return_value=httpx.Response(200, text=_fixture("search_empty.html")))
    empty = await source.search("нетсовпадений", limit=5)

    route.mock(return_value=httpx.Response(200, text=_fixture("search_rejected.html")))
    rejected = await source.search("отвергнутый", limit=5)

    assert empty.status is ExternalStatus.EMPTY
    assert rejected.status is ExternalStatus.REJECTED
    assert metrics.search.external_empty._value.get() == 1
    assert metrics.search.external_rejected._value.get() == 1


@respx.mock
async def test_robots_disallow_blocks_request_when_enabled() -> None:
    search_route = respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    respx.get(f"{BASE_URL}/robots.txt").mock(
        return_value=httpx.Response(
            200,
            text="User-agent: *\nDisallow: /search\nAllow: /\n",
        )
    )
    source = _source(respect_robots=True)

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.UNAVAILABLE
    assert result.error_kind is ExternalErrorKind.ROBOTS_DISALLOWED
    assert search_route.call_count == 0


@respx.mock
async def test_robots_rules_are_ignored_by_default() -> None:
    search_route = respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    respx.get(f"{BASE_URL}/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /search\n")
    )
    source = _source()

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.OK
    assert search_route.call_count == 1


@respx.mock
async def test_missing_robots_file_allows_search() -> None:
    search_route = respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, text=_fixture("search_ok.html"))
    )
    respx.get(f"{BASE_URL}/robots.txt").mock(return_value=httpx.Response(404))
    source = _source(respect_robots=True)

    result = await source.search("Тьюринг", limit=5)

    assert result.status is ExternalStatus.OK
    assert search_route.call_count == 1
