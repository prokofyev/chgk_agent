"""Доменные модели результата поиска, общие для графа, API и UI."""

from dataclasses import dataclass, field
from enum import StrEnum

from chgk_agent.models.normalization import normalize_text

LOCAL_SOURCE = "local"


class SourceStatus(StrEnum):
    """Итоговый статус источника в рамках одного поиска."""

    OK = "ok"
    EMPTY = "empty"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"

    @property
    def is_failure(self) -> bool:
        """Считается ли статус проблемой источника, а не результатом поиска."""

        return self in {SourceStatus.UNAVAILABLE, SourceStatus.REJECTED}


@dataclass(slots=True)
class SourceReport:
    """Диагностика обращения к одному источнику поиска."""

    source: str
    status: SourceStatus
    matches: int = 0
    error: str | None = None
    error_kind: str | None = None
    query: str | None = None
    queries: list[str] = field(default_factory=list)
    truncated: bool = False
    duration_seconds: float = 0.0
    degraded: bool = False
    """Источник отработал, но без части сигнала, например без лексики."""


@dataclass(slots=True)
class SourceRef:
    """Источник, в котором найден вопрос."""

    name: str
    location: str | None = None
    external_url: str | None = None
    score: float = 0.0
    score_kind: str = ""
    position: int | None = None
    """Позиция во внешней выдаче — только диагностика, в оценку не входит."""


@dataclass(slots=True)
class SearchMatch:
    """Единое совпадение после объединения и дедупликации."""

    question_text: str
    answer_text: str
    comment: str | None
    score: float
    score_kind: str
    sources: list[SourceRef] = field(default_factory=list)
    external_url: str | None = None
    semantic_similarity: float = 0.0
    lexical_score: float = 0.0
    key: str = ""

    @property
    def dedupe_key(self) -> str:
        """Ключ дедупликации по каноническому тексту вопроса."""

        return normalize_text(self.question_text)

    @property
    def source_names(self) -> list[str]:
        """Имена источников, где найден вопрос."""

        return [ref.name for ref in self.sources]


@dataclass(slots=True)
class GeneratedAnswer:
    """Результат генерации ответа поверх найденных совпадений."""

    text: str | None = None
    available: bool = True
    used_matches: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def is_empty(self) -> bool:
        """Есть ли содержательный ответ."""

        return bool(self.text and self.text.strip())


@dataclass(slots=True)
class SearchOutcome:
    """Итог поиска по описанию вопроса."""

    query: str
    matches: list[SearchMatch] = field(default_factory=list)
    sources: list[SourceReport] = field(default_factory=list)
    answer: GeneratedAnswer | None = None
    request_id: str | None = None
    truncated_query: str | None = None

    @property
    def is_partial(self) -> bool:
        """Есть ли источники, которые не отработали штатно."""

        return any(
            report.status.is_failure or report.degraded for report in self.sources
        )

    @property
    def is_empty(self) -> bool:
        """Уверены ли мы, что совпадений нет нигде.

        Если источник был недоступен, пустой список его совпадений не означает
        отсутствия результатов: выдавать такую деградацию за пустую выдачу
        значило бы сообщать пользователю неправду. Отвергнутый запрос — это
        ответ источника, поэтому он пустой выдачи не отменяет.
        """

        return not self.matches and not any(
            report.status is SourceStatus.UNAVAILABLE for report in self.sources
        )

    @property
    def failed_sources(self) -> list[str]:
        """Имена источников с проблемами."""

        return [
            report.source
            for report in self.sources
            if report.status.is_failure or report.degraded
        ]

    def status_of(self, source: str) -> SourceStatus | None:
        """Статус источника по имени, если он опрашивался."""

        for report in self.sources:
            if report.source == source:
                return report.status
        return None


__all__ = [
    "LOCAL_SOURCE",
    "GeneratedAnswer",
    "SearchMatch",
    "SearchOutcome",
    "SourceRef",
    "SourceReport",
    "SourceStatus",
]
