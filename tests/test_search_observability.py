"""Тесты наблюдаемости поиска: метрики, структурные логи, сквозной идентификатор."""

import json

import pytest
from prometheus_client import CollectorRegistry

from chgk_agent.config import Settings
from chgk_agent.external.base import (
    ExternalErrorKind,
    ExternalSearchResult,
    ExternalStatus,
)
from chgk_agent.logging_setup import configure_logging
from chgk_agent.observability.metrics import Metrics
from chgk_agent.search import nodes
from chgk_agent.search.local import LocalSearchOutcome
from chgk_agent.search.models import SearchOutcome, SourceReport, SourceStatus
from chgk_agent.search.nodes import SearchDeps


class FakeEmbeddingProvider:
    """Провайдер эмбеддингов-заглушка."""

    model = "fake"
    dimension = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _Session:
    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _SessionFactory:
    def __call__(self) -> _Session:
        return _Session()


def _deps(metrics: Metrics) -> SearchDeps:
    return SearchDeps(
        session_factory=_SessionFactory(),  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),
        settings=Settings(_env_file=None),
        metrics=metrics,
    )


def _value(metric, **labels: str) -> float:
    return metric.labels(**labels)._value.get()


def _outcome(*, partial: bool, empty: bool = False) -> SearchOutcome:
    return SearchOutcome(
        query="описание",
        matches=[] if empty else [object()],  # type: ignore[list-item]
        sources=[
            SourceReport(source="local", status=SourceStatus.OK, matches=1),
            SourceReport(
                source="gotquestions",
                status=SourceStatus.UNAVAILABLE if partial else SourceStatus.OK,
                matches=0,
                error="таймаут" if partial else None,
                error_kind="timeout" if partial else None,
            ),
        ],
        request_id="req-7",
    )


def test_successful_search_records_counters_and_duration() -> None:
    metrics = Metrics(CollectorRegistry())

    nodes.format_response(
        {"query": "описание", "matches": [], "sources": [], "started_at": 0.0},
        deps=_deps(metrics),
    )
    metrics.record_search_request("local", "ok")
    metrics.search.duration.labels(source="local").observe(0.25)
    metrics.record_matches("local", 3)

    assert _value(metrics.search.requests, source="local", status="ok") == 1
    assert metrics.search.duration.labels(source="local")._sum.get() == 0.25
    assert metrics.search.queries.labels(source="local")._value.get() == 3


def test_partial_search_records_partial_counter_and_error() -> None:
    metrics = Metrics(CollectorRegistry())

    nodes.format_response(
        {
            "query": "описание",
            "matches": [],
            "sources": _outcome(partial=True).sources,
            "started_at": 0.0,
            "request_id": "req-7",
        },
        deps=_deps(metrics),
    )

    assert _value(metrics.search.partial, source="gotquestions") == 1
    assert (
        _value(metrics.search.errors, source="gotquestions", error_type="timeout") == 1
    )
    assert (
        _value(metrics.search.requests, source="gotquestions", status="unavailable") == 1
    )
    assert _value(metrics.search.requests, source="local", status="ok") == 1


def test_empty_search_records_empty_sources() -> None:
    metrics = Metrics(CollectorRegistry())

    nodes.format_response(
        {
            "query": "описание",
            "matches": [],
            "sources": [
                SourceReport(source="local", status=SourceStatus.EMPTY),
                SourceReport(source="gotquestions", status=SourceStatus.REJECTED),
            ],
            "started_at": 0.0,
        },
        deps=_deps(metrics),
    )

    assert _value(metrics.search.requests, source="local", status="empty") == 1
    assert _value(metrics.search.requests, source="gotquestions", status="rejected") == 1
    assert _value(metrics.search.partial, source="gotquestions") == 1


def test_search_log_contains_request_id_status_duration_and_sources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(json_logs=True, level="INFO")
    metrics = Metrics(CollectorRegistry())

    nodes.format_response(
        {
            "query": "описание",
            "matches": [],
            "sources": _outcome(partial=True).sources,
            "started_at": 0.0,
            "request_id": "req-42",
        },
        deps=_deps(metrics),
    )

    records = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    search_records = [item for item in records if item.get("operation") == "search"]

    assert search_records
    record = search_records[-1]
    assert record["request_id"] == "req-42"
    assert record["status"] == "partial"
    assert "duration_seconds" in record
    assert record["sources"] == ["local", "gotquestions"]


def test_search_logs_do_not_contain_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    secret = "super-secret-token-value"
    metrics = Metrics(CollectorRegistry())

    nodes.format_response(
        {
            "query": "описание",
            "matches": [],
            "sources": _outcome(partial=False).sources,
            "started_at": 0.0,
            "request_id": "req-1",
        },
        deps=_deps(metrics),
    )
    logger = nodes.logger
    logger.info("вызов провайдера", credentials=secret, access_token=secret)

    output = capsys.readouterr().out
    assert secret not in output


def test_local_failure_records_unavailable_with_kind() -> None:
    outcome = LocalSearchOutcome(degraded=True, semantic_used=False, error="база недоступна")

    report = nodes.merge_and_dedupe(
        {"local": outcome, "local_seconds": 0.5}
    )["sources"][0]

    assert report.status is SourceStatus.UNAVAILABLE
    assert report.error_kind == "local_failure"


def test_external_report_carries_error_kind() -> None:
    report = nodes.merge_and_dedupe(
        {
            "external": ExternalSearchResult(
                status=ExternalStatus.UNAVAILABLE,
                query="описание",
                error="таймаут",
                error_kind=ExternalErrorKind.TIMEOUT,
            )
        }
    )["sources"][0]

    assert report.error_kind == "timeout"
    assert report.status is SourceStatus.UNAVAILABLE


def test_import_operation_id_is_shared_between_report_and_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from chgk_agent.logging_setup import request_context

    configure_logging(json_logs=True, level="INFO")
    metrics = Metrics(CollectorRegistry())

    with request_context("op-shared"):
        nodes.format_response(
            {
                "query": "описание",
                "matches": [],
                "sources": [SourceReport(source="local", status=SourceStatus.OK)],
                "started_at": 0.0,
                "request_id": "op-shared",
            },
            deps=_deps(metrics),
        )

    records = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    assert any(item.get("request_id") == "op-shared" for item in records)
