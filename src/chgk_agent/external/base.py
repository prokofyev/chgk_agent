"""Интерфейс внешнего источника вопросов и его результат."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class ExternalStatus(StrEnum):
    """Исход обращения к внешнему источнику."""

    OK = "ok"
    EMPTY = "empty"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"

    @property
    def is_success(self) -> bool:
        """Ответ получен и разобран без технической ошибки."""

        return self in {ExternalStatus.OK, ExternalStatus.EMPTY, ExternalStatus.REJECTED}


class ExternalErrorKind(StrEnum):
    """Тип технической ошибки внешнего источника."""

    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    FORBIDDEN = "forbidden"
    SERVER = "server"
    RATE_LIMITED = "rate_limited"
    UNRECOGNIZED = "unrecognized"
    CIRCUIT_OPEN = "circuit_open"
    ROBOTS_DISALLOWED = "robots_disallowed"
    EMBEDDING_FAILED = "embedding_failed"


@dataclass(slots=True)
class ExternalMatch:
    """Совпадение, полученное из внешнего источника."""

    title: str
    question_text: str
    answer_text: str | None
    comment: str | None = None
    authors: list[str] = field(default_factory=list)
    pack: str | None = None
    external_url: str | None = None
    external_id: str | None = None
    position: int = 0
    score: float = 0.0
    score_kind: str = ""
    embedding: list[float] | None = None
    semantic_similarity: float = 0.0


@dataclass(slots=True)
class ExternalSearchResult:
    """Результат поиска во внешнем источнике."""

    status: ExternalStatus
    matches: list[ExternalMatch] = field(default_factory=list)
    query: str = ""
    queries: list[str] = field(default_factory=list)
    error: str | None = None
    error_kind: ExternalErrorKind | None = None
    truncated: bool = False
    duration_seconds: float = 0.0
    attempts: int = 0
    page: int = 1
    pages_fetched: int = 0
    has_more: bool = False


@runtime_checkable
class ExternalQuestionSource(Protocol):
    """Внешний источник вопросов ЧГК."""

    @property
    def name(self) -> str:
        """Имя источника для метрик и статусов."""

    @property
    def enabled(self) -> bool:
        """Включён ли источник настройкой."""

    async def search(
        self,
        description: str,
        *,
        limit: int = 20,
    ) -> ExternalSearchResult:
        """Найти вопросы по описанию."""
