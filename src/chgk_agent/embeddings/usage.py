"""Учёт обращений к GigaChat через единый слой интеграции.

Счётчики вызовов, ошибок и токенов живут здесь, а не в адаптерах, чтобы
метрики оставались одинаковыми для эмбеддингов и генерации.
"""

from chgk_agent.observability.metrics import Metrics, get_metrics

EMBEDDING_OPERATION = "embedding"
GENERATION_OPERATION = "generation"

TOKEN_KINDS = ("input_tokens", "output_tokens", "total_tokens")


def record_call_success(
    operation: str,
    *,
    usage: dict[str, object] | None = None,
    metrics: Metrics | None = None,
) -> None:
    """Учесть успешный вызов GigaChat и израсходованные токены."""

    current = metrics or get_metrics()
    current.gigachat.calls.labels(operation=operation).inc()

    if not usage:
        return

    for kind in TOKEN_KINDS:
        value = usage.get(kind)
        if isinstance(value, int) and value > 0:
            current.gigachat.tokens.labels(operation=operation, kind=kind).inc(value)


def record_call_error(
    operation: str,
    error: Exception | None = None,
    *,
    metrics: Metrics | None = None,
) -> None:
    """Учесть неуспешный вызов GigaChat."""

    current = metrics or get_metrics()
    current.gigachat.calls.labels(operation=operation).inc()
    current.gigachat.errors.labels(operation=operation).inc()
