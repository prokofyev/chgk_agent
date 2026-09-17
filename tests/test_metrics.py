"""Тесты реестра метрик Prometheus."""

from prometheus_client import CollectorRegistry

from chgk_agent.observability.metrics import Metrics


def _metrics() -> Metrics:
    return Metrics(CollectorRegistry())


def _value(metric, **labels: str) -> float:
    return metric.labels(**labels)._value.get()


def test_search_request_counter_is_split_by_source_and_status() -> None:
    metrics = _metrics()

    metrics.record_search_request("local", "ok")
    metrics.record_search_request("local", "ok")
    metrics.record_search_request("external", "empty")

    assert _value(metrics.search.requests, source="local", status="ok") == 2
    assert _value(metrics.search.requests, source="external", status="empty") == 1


def test_search_errors_and_partial_results_are_counted() -> None:
    metrics = _metrics()

    metrics.record_search_error("external", "timeout")
    metrics.record_partial_search("external")

    assert _value(metrics.search.errors, source="external", error_type="timeout") == 1
    assert _value(metrics.search.partial, source="external") == 1


def test_duration_histogram_records_observation() -> None:
    metrics = _metrics()

    metrics.search.duration.labels(source="local").observe(0.42)

    assert metrics.search.duration.labels(source="local")._sum.get() == 0.42


def test_empty_result_does_not_increment_matches() -> None:
    metrics = _metrics()

    metrics.record_matches("external", 0)
    metrics.record_matches("local", 3)

    assert metrics.search.queries.labels(source="external")._value.get() == 0
    assert metrics.search.queries.labels(source="local")._value.get() == 3


def test_ingestion_and_unembedded_queue_metrics() -> None:
    metrics = _metrics()

    metrics.ingestion.questions.labels(status="added").inc(5)
    metrics.ingestion.errors.labels(stage="parsing").inc()
    metrics.ingestion.operations.labels(status="completed").inc()
    metrics.ingestion.unembedded.set(7)

    assert metrics.ingestion.questions.labels(status="added")._value.get() == 5
    assert metrics.ingestion.errors.labels(stage="parsing")._value.get() == 1
    assert metrics.ingestion.operations.labels(status="completed")._value.get() == 1
    assert metrics.ingestion.unembedded._value.get() == 7


def test_gigachat_counters_by_operation() -> None:
    metrics = _metrics()

    metrics.gigachat.calls.labels(operation="generation").inc()
    metrics.gigachat.errors.labels(operation="embedding").inc()
    metrics.gigachat.tokens.labels(operation="generation", kind="total_tokens").inc(120)

    assert metrics.gigachat.calls.labels(operation="generation")._value.get() == 1
    assert metrics.gigachat.errors.labels(operation="embedding")._value.get() == 1
    assert (
        metrics.gigachat.tokens.labels(
            operation="generation", kind="total_tokens"
        )._value.get()
        == 120
    )


def test_metrics_share_one_registry() -> None:
    metrics = _metrics()

    names = {collector.name for collector in metrics.registry.collect()}

    assert "chgk_search_requests" in names
    assert "chgk_unembedded_questions" in names
    assert "chgk_gigachat_tokens" in names
