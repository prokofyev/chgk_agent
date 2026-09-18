"""Представление результатов поиска для веб-интерфейса.

Логика отображения отделена от NiceGUI: здесь только чистые функции,
превращающие ответ API в модель экрана, которую легко проверить тестами.
"""

from dataclasses import dataclass, field
from enum import StrEnum

MIN_QUERY_CHARS = 3
MAX_QUERY_CHARS = 2000

EXTERNAL_SOURCE = "gotquestions"
LOCAL_SOURCE = "local"

STATUS_LABELS: dict[str, str] = {
    "ok": "совпадения найдены",
    "empty": "ничего не найдено",
    "rejected": "запрос отвергнут источником",
    "unavailable": "источник недоступен",
    "disabled": "источник отключён",
}

SCORE_KIND_LABELS: dict[str, str] = {
    "semantic": "семантическая близость",
    "semantic+lexical": "семантическая близость и точные слова",
    "lexical": "точные слова описания",
    "none": "близость не определена",
}


class SearchState(StrEnum):
    """Состояние поиска, отображаемое пользователю."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY = "empty"
    ERROR = "error"


@dataclass(slots=True)
class FormErrors:
    """Ошибки формы поиска."""

    query: str | None = None
    limit: str | None = None
    min_score: str | None = None

    @property
    def has_errors(self) -> bool:
        """Есть ли хотя бы одна ошибка."""

        return any(
            value is not None for value in (self.query, self.limit, self.min_score)
        )


@dataclass(slots=True)
class MatchView:
    """Совпадение, подготовленное к отображению."""

    question_text: str
    answer_text: str
    score: float
    score_kind: str
    score_label: str
    score_percent: int
    source_names: list[str]
    source_labels: list[str]
    external_url: str | None = None
    comment: str | None = None
    origin: str = LOCAL_SOURCE
    location: str | None = None

    @property
    def is_external(self) -> bool:
        """Найдено ли совпадение во внешнем источнике."""

        return EXTERNAL_SOURCE in self.source_names


@dataclass(slots=True)
class SourceView:
    """Строка состояния источника на экране."""

    source: str
    label: str
    status: str
    status_label: str
    is_problem: bool
    matches: int
    query: str | None = None
    truncated: bool = False
    error: str | None = None


@dataclass(slots=True)
class AnswerView:
    """Блок сгенерированного ответа."""

    text: str | None
    available: bool
    used_matches: list[str] = field(default_factory=list)
    message: str | None = None


@dataclass(slots=True)
class SearchView:
    """Готовая модель экрана результатов."""

    state: SearchState
    message: str | None = None
    request_id: str | None = None
    matches: list[MatchView] = field(default_factory=list)
    sources: list[SourceView] = field(default_factory=list)
    answer: AnswerView | None = None
    truncated_query: str | None = None
    error: str | None = None

    @property
    def has_matches(self) -> bool:
        """Есть ли что показывать в списке совпадений."""

        return bool(self.matches)

    @property
    def is_running(self) -> bool:
        """Идёт ли выполнение поиска."""

        return self.state is SearchState.RUNNING


def validate_form(
    query: str,
    *,
    limit: int = 20,
    min_score: float = 0.0,
) -> FormErrors:
    """Проверить параметры формы перед отправкой запроса."""

    errors = FormErrors()
    stripped = query.strip()
    if not stripped:
        errors.query = "введите описание вопроса"
    elif len(stripped) < MIN_QUERY_CHARS:
        errors.query = f"описание должно быть не короче {MIN_QUERY_CHARS} символов"
    elif len(stripped) > MAX_QUERY_CHARS:
        errors.query = f"описание должно быть не длиннее {MAX_QUERY_CHARS} символов"

    if limit < 1 or limit > 100:
        errors.limit = "число результатов должно быть от 1 до 100"
    if min_score < 0.0 or min_score > 1.0:
        errors.min_score = "порог должен быть в диапазоне от 0 до 1"
    return errors


def running_view() -> SearchView:
    """Состояние выполнения поиска."""

    return SearchView(state=SearchState.RUNNING, message="Поиск выполняется…")


def error_view(message: str, *, request_id: str | None = None) -> SearchView:
    """Состояние ошибки поиска."""

    return SearchView(
        state=SearchState.ERROR,
        message=message,
        request_id=request_id,
        error=message,
    )


def build_view(payload: dict) -> SearchView:
    """Построить модель экрана по ответу API поиска."""

    matches = [_match_view(item) for item in payload.get("matches", [])]
    sources = [_source_view(item) for item in payload.get("sources", [])]
    answer = _answer_view(payload.get("answer"))

    partial = bool(payload.get("partial"))
    empty = bool(payload.get("empty")) or not matches
    unavailable = any(
        item.get("status") == "unavailable" for item in payload.get("sources", [])
    )

    if unavailable and not matches:
        state = SearchState.PARTIAL
        message = "Часть источников недоступна, поэтому выдача может быть неполной."
    elif empty:
        state = SearchState.EMPTY
        message = "По вашему описанию ничего не найдено — измените формулировку или порог."
    elif partial:
        state = SearchState.PARTIAL
        message = "Часть источников недоступна, показаны неполные результаты."
    else:
        state = SearchState.SUCCESS
        message = None

    return SearchView(
        state=state,
        message=message,
        request_id=payload.get("request_id"),
        matches=matches,
        sources=sources,
        answer=answer,
        truncated_query=payload.get("truncated_query"),
    )


def _match_view(item: dict) -> MatchView:
    """Построить представление одного совпадения."""

    sources = item.get("sources") or []
    names = [ref.get("name", "") for ref in sources if ref.get("name")]
    location = next(
        (ref.get("location") for ref in sources if ref.get("location")),
        None,
    )
    score = float(item.get("score", 0.0))
    score_kind = item.get("score_kind", "")

    return MatchView(
        question_text=item.get("question_text", ""),
        answer_text=item.get("answer_text", ""),
        score=score,
        score_kind=score_kind,
        score_label=SCORE_KIND_LABELS.get(score_kind, score_kind),
        score_percent=round(max(0.0, min(score, 1.0)) * 100),
        source_names=names,
        source_labels=[_source_label(name) for name in names],
        external_url=item.get("external_url"),
        comment=item.get("comment"),
        origin=EXTERNAL_SOURCE if EXTERNAL_SOURCE in names else LOCAL_SOURCE,
        location=location,
    )


def _source_view(item: dict) -> SourceView:
    """Построить представление статуса одного источника."""

    status = item.get("status", "empty")
    return SourceView(
        source=item.get("source", ""),
        label=_source_label(item.get("source", "")),
        status=status,
        status_label=STATUS_LABELS.get(status, status),
        is_problem=status in {"rejected", "unavailable"},
        matches=int(item.get("matches", 0)),
        query=item.get("query"),
        truncated=bool(item.get("truncated")),
        error=item.get("error"),
    )


def _answer_view(item: dict | None) -> AnswerView | None:
    """Построить представление блока сгенерированного ответа."""

    if item is None:
        return None

    available = bool(item.get("available"))
    text = item.get("text")
    message = None
    if not available:
        message = (
            "Надёжный ответ сформировать нельзя: нет подтверждающих совпадений."
            if item.get("error") == "нет подтверждающих совпадений"
            else "Сгенерированный ответ получить не удалось."
        )
    return AnswerView(
        text=text,
        available=available,
        used_matches=list(item.get("used_matches") or []),
        message=message,
    )


def _source_label(source: str) -> str:
    """Человекочитаемое имя источника."""

    if source == EXTERNAL_SOURCE:
        return "внешний источник gotquestions.online"
    if source == LOCAL_SOURCE:
        return "локальная база"
    return source or "неизвестный источник"


__all__ = [
    "EXTERNAL_SOURCE",
    "LOCAL_SOURCE",
    "MAX_QUERY_CHARS",
    "MIN_QUERY_CHARS",
    "SCORE_KIND_LABELS",
    "STATUS_LABELS",
    "AnswerView",
    "FormErrors",
    "MatchView",
    "SearchState",
    "SearchView",
    "SourceView",
    "build_view",
    "error_view",
    "running_view",
    "validate_form",
]
