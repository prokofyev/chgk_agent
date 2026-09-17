"""Метрики Prometheus для поиска, индексации и обращений к GigaChat.

Метки ограничены значениями с низкой кардинальностью: имена источников,
итоговые статусы, стадии и типы операций. Тексты запросов и URL в метки
не попадают.
"""

from functools import lru_cache

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

SEARCH_DURATION_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

EXTERNAL_DURATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0)


class SearchMetrics:
    """Метрики операций поиска."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "chgk_search_requests_total",
            "Число запросов поиска по источнику и итоговому статусу",
            ["source", "status"],
            registry=registry,
        )
        self.duration = Histogram(
            "chgk_search_duration_seconds",
            "Длительность операции поиска по источнику",
            ["source"],
            buckets=SEARCH_DURATION_BUCKETS,
            registry=registry,
        )
        self.errors = Counter(
            "chgk_search_errors_total",
            "Ошибки поиска по источнику и типу ошибки",
            ["source", "error_type"],
            registry=registry,
        )
        self.partial = Counter(
            "chgk_search_partial_total",
            "Поиски, завершившиеся с частичными результатами",
            ["source"],
            registry=registry,
        )
        self.queries = Counter(
            "chgk_search_matches_total",
            "Число совпадений, возвращённых источником",
            ["source"],
            registry=registry,
        )
        self.external_truncated = Counter(
            "chgk_external_query_truncated_total",
            "Внешние поисковые запросы, сокращённые до допустимой длины",
            registry=registry,
        )
        self.external_queries = Histogram(
            "chgk_external_query_chars",
            "Длина поискового запроса к внешнему источнику в символах",
            buckets=(10, 20, 30, 40, 50),
            registry=registry,
        )
        self.external_duration = Histogram(
            "chgk_external_request_duration_seconds",
            "Длительность обращения к внешнему источнику",
            buckets=EXTERNAL_DURATION_BUCKETS,
            registry=registry,
        )
        self.external_empty = Counter(
            "chgk_external_empty_total",
            "Успешные обращения к внешнему источнику без совпадений",
            registry=registry,
        )
        self.external_rejected = Counter(
            "chgk_external_rejected_total",
            "Запросы, отвергнутые внешним источником без блока результатов",
            registry=registry,
        )


class IngestionMetrics:
    """Метрики операций индексации."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.questions = Counter(
            "chgk_ingestion_questions_total",
            "Вопросы, обработанные импортом, по итоговой категории",
            ["status"],
            registry=registry,
        )
        self.errors = Counter(
            "chgk_ingestion_errors_total",
            "Ошибки индексации по стадии",
            ["stage"],
            registry=registry,
        )
        self.operations = Counter(
            "chgk_ingestion_operations_total",
            "Операции импорта по итоговому статусу",
            ["status"],
            registry=registry,
        )
        self.unembedded = Gauge(
            "chgk_unembedded_questions",
            "Число вопросов, ожидающих векторизации",
            registry=registry,
        )


class GigaChatMetrics:
    """Метрики использования GigaChat через единый слой интеграции."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.calls = Counter(
            "chgk_gigachat_calls_total",
            "Вызовы GigaChat по типу операции",
            ["operation"],
            registry=registry,
        )
        self.errors = Counter(
            "chgk_gigachat_errors_total",
            "Ошибки GigaChat по типу операции",
            ["operation"],
            registry=registry,
        )
        self.tokens = Counter(
            "chgk_gigachat_tokens_total",
            "Израсходованные токены GigaChat по типу операции и виду",
            ["operation", "kind"],
            registry=registry,
        )


class Metrics:
    """Единый реестр метрик сервиса."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else REGISTRY
        self.search = SearchMetrics(self.registry)
        self.ingestion = IngestionMetrics(self.registry)
        self.gigachat = GigaChatMetrics(self.registry)

    def record_search_request(self, source: str, status: str) -> None:
        """Учесть обращение к источнику с итоговым статусом."""

        self.search.requests.labels(source=source, status=status).inc()

    def record_search_error(self, source: str, error_type: str) -> None:
        """Учесть ошибку источника поиска."""

        self.search.errors.labels(source=source, error_type=error_type).inc()

    def record_partial_search(self, source: str) -> None:
        """Учесть поиск с частичным результатом."""

        self.search.partial.labels(source=source).inc()

    def record_matches(self, source: str, count: int) -> None:
        """Учесть число возвращённых источником совпадений."""

        if count > 0:
            self.search.queries.labels(source=source).inc(count)


@lru_cache(maxsize=1)
def get_metrics() -> Metrics:
    """Вернуть общий набор метрик приложения."""

    return Metrics()
