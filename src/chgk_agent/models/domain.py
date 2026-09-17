"""Доменные модели вопросов, источников и отчётов об импорте."""

from dataclasses import dataclass, field
from datetime import datetime

from chgk_agent.models.normalization import normalize_text, text_hash


@dataclass(frozen=True, slots=True)
class ParsedQuestion:
    """Вопрос, извлечённый из источника до нормализации."""

    source_key: str
    question_text: str
    answer_text: str
    pass_criteria: str | None = None
    comment: str | None = None
    sources: str | None = None
    author: str | None = None
    external_url: str | None = None
    status_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalQuestion:
    """Канонический вопрос, независимый от источника."""

    normalized_text: str
    question_text: str
    answer_text: str
    text_hash: str
    comment: str | None = None

    @classmethod
    def from_parsed(cls, parsed: ParsedQuestion) -> "CanonicalQuestion":
        """Построить канонический вопрос из разобранного."""

        return cls(
            normalized_text=normalize_text(parsed.question_text),
            question_text=parsed.question_text,
            answer_text=parsed.answer_text,
            text_hash=text_hash(parsed.question_text),
            comment=parsed.comment,
        )


@dataclass(frozen=True, slots=True)
class ParseIssue:
    """Проблема, возникшая при разборе или импорте."""

    location: str
    message: str


@dataclass(slots=True)
class ParseResult:
    """Результат разбора одного источника."""

    location: str
    questions: list[ParsedQuestion] = field(default_factory=list)
    skipped: int = 0
    skipped_with_media: int = 0
    skipped_no_answer: int = 0
    skipped_not_a_question: int = 0
    issues: list[ParseIssue] = field(default_factory=list)


@dataclass(slots=True)
class ImportReport:
    """Отчёт об операции импорта."""

    operation_id: str
    locations: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    processed: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    unembedded: int = 0
    issues: list[ParseIssue] = field(default_factory=list)

    @property
    def is_partial(self) -> bool:
        """Были ли при импорте ошибки или пропуски."""

        return bool(self.issues) or self.skipped > 0 or self.unembedded > 0
