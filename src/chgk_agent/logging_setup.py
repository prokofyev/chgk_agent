"""Структурированное логирование с идентификатором запроса и маскированием секретов."""

import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import structlog

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

SENSITIVE_KEYS = frozenset(
    {
        "credentials",
        "authorization",
        "access_token",
        "password",
        "token",
        "api_key",
        "secret",
    }
)

MASK = "***"


def new_request_id() -> str:
    """Сгенерировать новый идентификатор запроса."""

    return uuid.uuid4().hex


def get_request_id() -> str | None:
    """Вернуть идентификатор текущего запроса, если он задан."""

    return request_id_var.get()


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Привязать идентификатор запроса к текущему контексту выполнения."""

    value = request_id or new_request_id()
    token = request_id_var.set(value)
    try:
        yield value
    finally:
        request_id_var.reset(token)


def _mask_secrets(
    _logger: object, _method_name: str, event_dict: dict[str, object]
) -> dict[str, object]:
    """Заменить значения чувствительных полей на маску."""

    for key in list(event_dict):
        if key.lower() in SENSITIVE_KEYS:
            event_dict[key] = MASK
    return event_dict


def _add_request_id(
    _logger: object, _method_name: str, event_dict: dict[str, object]
) -> dict[str, object]:
    """Добавить идентификатор запроса в каждую запись."""

    request_id = request_id_var.get()
    if request_id is not None:
        event_dict.setdefault("request_id", request_id)
    return event_dict


def configure_logging(*, json_logs: bool = False, level: str = "INFO") -> None:
    """Настроить structlog и стандартный logging."""

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    shared_processors: list[object] = [
        structlog.contextvars.merge_contextvars,
        _add_request_id,
        _mask_secrets,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: object = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Вернуть structlog-логгер."""

    return structlog.get_logger(name)
