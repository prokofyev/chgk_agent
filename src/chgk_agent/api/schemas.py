"""Pydantic-схемы запросов и ответов HTTP API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chgk_agent.search.models import (
    GeneratedAnswer,
    SearchMatch,
    SearchOutcome,
    SourceRef,
    SourceReport,
    SourceStatus,
)


class SearchRequest(BaseModel):
    """Запрос поиска по описанию вопроса."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=20, ge=1, le=100)
    min_score: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("описание вопроса не может быть пустым")
        return stripped


class SourceRefSchema(BaseModel):
    """Источник, в котором найден вопрос."""

    name: str
    location: str | None = None
    external_url: str | None = None
    score: float
    score_kind: str
    position: int | None = None
    """Позиция во внешней выдаче: диагностика, а не часть оценки."""

    @classmethod
    def from_domain(cls, ref: SourceRef) -> "SourceRefSchema":
        """Построить схему по доменной модели."""

        return cls(
            name=ref.name,
            location=ref.location,
            external_url=ref.external_url,
            score=ref.score,
            score_kind=ref.score_kind,
            position=ref.position,
        )


class MatchSchema(BaseModel):
    """Совпадение в ответе поиска."""

    question_text: str
    answer_text: str
    comment: str | None = None
    score: float
    score_kind: str
    sources: list[SourceRefSchema] = Field(default_factory=list)
    external_url: str | None = None

    @classmethod
    def from_domain(cls, match: SearchMatch) -> "MatchSchema":
        """Построить схему по доменной модели."""

        return cls(
            question_text=match.question_text,
            answer_text=match.answer_text,
            comment=match.comment,
            score=match.score,
            score_kind=match.score_kind,
            sources=[SourceRefSchema.from_domain(ref) for ref in match.sources],
            external_url=match.external_url,
        )


class SourceReportSchema(BaseModel):
    """Статус одного источника поиска."""

    source: str
    status: SourceStatus
    matches: int
    error: str | None = None
    error_kind: str | None = None
    query: str | None = None
    queries: list[str] = Field(default_factory=list)
    truncated: bool = False
    duration_seconds: float = 0.0
    degraded: bool = False

    @classmethod
    def from_domain(cls, report: SourceReport) -> "SourceReportSchema":
        """Построить схему по доменной модели."""

        return cls(
            source=report.source,
            status=report.status,
            matches=report.matches,
            error=report.error,
            error_kind=report.error_kind,
            query=report.query,
            queries=list(report.queries),
            truncated=report.truncated,
            duration_seconds=report.duration_seconds,
            degraded=report.degraded,
        )


class AnswerSchema(BaseModel):
    """Один результат генерации и его доступность."""

    text: str | None = None
    available: bool
    used_matches: list[str] = Field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_domain(cls, answer: GeneratedAnswer) -> "AnswerSchema":
        """Построить схему по доменной модели."""

        return cls(
            text=answer.text,
            available=answer.available,
            used_matches=list(answer.used_matches),
            error=answer.error,
        )


class SearchResponse(BaseModel):
    """Ответ поиска.

    Генерация даёт два независимых результата: без подгрузки похожих
    вопросов и с подгрузкой. Поле с единственным ответом не сохраняется,
    чтобы клиент не мог принять один прогон за другой.
    """

    query: str
    matches: list[MatchSchema]
    sources: list[SourceReportSchema]
    answer_without_context: AnswerSchema | None = None
    answer_with_context: AnswerSchema | None = None
    request_id: str | None = None
    truncated_query: str | None = None
    partial: bool = False
    empty: bool = False

    @classmethod
    def from_domain(cls, outcome: SearchOutcome) -> "SearchResponse":
        """Построить схему по результату графа поиска."""

        return cls(
            query=outcome.query,
            matches=[MatchSchema.from_domain(match) for match in outcome.matches],
            sources=[SourceReportSchema.from_domain(report) for report in outcome.sources],
            answer_without_context=(
                AnswerSchema.from_domain(outcome.answer_without_context)
                if outcome.answer_without_context is not None
                else None
            ),
            answer_with_context=(
                AnswerSchema.from_domain(outcome.answer_with_context)
                if outcome.answer_with_context is not None
                else None
            ),
            request_id=outcome.request_id,
            truncated_query=outcome.truncated_query,
            partial=outcome.is_partial,
            empty=outcome.is_empty,
        )


class ImportRequest(BaseModel):
    """Запрос на запуск импорта HTML-источников."""

    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(min_length=1)


class ImportIssueSchema(BaseModel):
    """Проблема, возникшая при импорте."""

    location: str
    message: str


class ImportResponse(BaseModel):
    """Состояние операции импорта."""

    operation_id: str
    status: str
    locations: list[str] = Field(default_factory=list)
    processed: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    unembedded: int = 0
    issues: list[ImportIssueSchema] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class HealthResponse(BaseModel):
    """Ответ проверки жизнеспособности."""

    status: str = "ok"


class ReadinessDependency(BaseModel):
    """Состояние одной зависимости сервиса."""

    name: str
    status: str
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Ответ проверки готовности."""

    status: str
    dependencies: list[ReadinessDependency] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Единый конверт ошибки."""

    code: str
    message: str
    request_id: str | None = None
    details: list[dict[str, object]] | None = None


__all__ = [
    "AnswerSchema",
    "ErrorResponse",
    "HealthResponse",
    "ImportIssueSchema",
    "ImportRequest",
    "ImportResponse",
    "MatchSchema",
    "ReadinessDependency",
    "ReadinessResponse",
    "SearchRequest",
    "SearchResponse",
    "SourceRefSchema",
    "SourceReportSchema",
]
