"""Тесты учёта использования GigaChat через единый слой."""

from prometheus_client import CollectorRegistry

from chgk_agent.embeddings.usage import (
    EMBEDDING_OPERATION,
    GENERATION_OPERATION,
    record_call_error,
    record_call_success,
)
from chgk_agent.observability.metrics import Metrics


def _metrics() -> Metrics:
    return Metrics(CollectorRegistry())


def test_successful_call_increments_counter() -> None:
    metrics = _metrics()

    record_call_success(GENERATION_OPERATION, metrics=metrics)

    assert metrics.gigachat.calls.labels(operation="generation")._value.get() == 1
    assert metrics.gigachat.errors.labels(operation="generation")._value.get() == 0


def test_tokens_are_counted_by_kind() -> None:
    metrics = _metrics()

    record_call_success(
        GENERATION_OPERATION,
        usage={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
        metrics=metrics,
    )

    assert (
        metrics.gigachat.tokens.labels(
            operation="generation", kind="input_tokens"
        )._value.get()
        == 30
    )
    assert (
        metrics.gigachat.tokens.labels(
            operation="generation", kind="output_tokens"
        )._value.get()
        == 12
    )
    assert (
        metrics.gigachat.tokens.labels(
            operation="generation", kind="total_tokens"
        )._value.get()
        == 42
    )


def test_unknown_usage_shapes_are_ignored() -> None:
    metrics = _metrics()

    record_call_success(
        EMBEDDING_OPERATION,
        usage={"input_tokens": "много", "total_tokens": -5},
        metrics=metrics,
    )

    assert (
        metrics.gigachat.tokens.labels(
            operation="embedding", kind="total_tokens"
        )._value.get()
        == 0
    )


def test_failed_call_increments_calls_and_errors() -> None:
    metrics = _metrics()

    record_call_error(EMBEDDING_OPERATION, RuntimeError("нет связи"), metrics=metrics)

    assert metrics.gigachat.calls.labels(operation="embedding")._value.get() == 1
    assert metrics.gigachat.errors.labels(operation="embedding")._value.get() == 1


def test_operations_are_counted_separately() -> None:
    metrics = _metrics()

    record_call_success(EMBEDDING_OPERATION, metrics=metrics)
    record_call_error(GENERATION_OPERATION, metrics=metrics)

    assert metrics.gigachat.calls.labels(operation="embedding")._value.get() == 1
    assert metrics.gigachat.calls.labels(operation="generation")._value.get() == 1
    assert metrics.gigachat.errors.labels(operation="embedding")._value.get() == 0
    assert metrics.gigachat.errors.labels(operation="generation")._value.get() == 1
