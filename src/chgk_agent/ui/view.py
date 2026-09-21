"""Модель экрана веб-интерфейса: описание вопроса и два ответа.

Логика отображения отделена от NiceGUI: здесь только чистые функции,
превращающие ответ API в текст, который видит пользователь. Промежуточная
выдача поиска пользователю не показывается, поэтому модель экрана состоит из
двух сгенерированных ответов: без подгрузки похожих вопросов и с подгрузкой.
Каждый из них либо получен, либо равен «Не знаю», либо отсутствует, если
соответствующего прогона не было.
"""

from dataclasses import dataclass

MIN_QUERY_CHARS = 3
MAX_QUERY_CHARS = 2000

UNKNOWN_ANSWER = "Не знаю"
PENDING_ANSWER = "Думаю..."

WITHOUT_CONTEXT_CAPTION = "Без базы вопросов"
WITH_CONTEXT_CAPTION = "С учётом базы вопросов"


@dataclass(slots=True)
class FormErrors:
    """Ошибки поля описания вопроса."""

    query: str | None = None

    @property
    def has_errors(self) -> bool:
        """Есть ли ошибка в описании."""

        return self.query is not None


@dataclass(slots=True)
class SearchView:
    """Готовая модель экрана: два ответа на вопрос.

    `None` в поле ответа означает, что прогона не было: место остаётся
    пустым. Неудача прогона — это `UNKNOWN_ANSWER`, то есть «Не знаю».
    `request_id` хранится для журнала, а не для показа: на странице
    технических подробностей быть не должно.
    """

    answer_without_context: str | None = UNKNOWN_ANSWER
    answer_with_context: str | None = None
    request_id: str | None = None

    @property
    def is_unknown(self) -> bool:
        """Не удалось ли получить ни одного ответа."""

        return (
            self.answer_without_context == UNKNOWN_ANSWER
            and self.answer_with_context == UNKNOWN_ANSWER
        )

    @property
    def is_pending(self) -> bool:
        """Показывается ли статус выполнения."""

        return (
            self.answer_without_context == PENDING_ANSWER
            or self.answer_with_context == PENDING_ANSWER
        )

    def answer_for(self, *, with_context: bool) -> str | None:
        """Текст ответа для одного из двух мест на странице."""

        if with_context:
            return self.answer_with_context
        return self.answer_without_context


def validate_form(query: str) -> FormErrors:
    """Проверить описание вопроса перед отправкой запроса."""

    errors = FormErrors()
    stripped = query.strip()
    if not stripped:
        errors.query = "введите описание вопроса"
    elif len(stripped) < MIN_QUERY_CHARS:
        errors.query = f"описание должно быть не короче {MIN_QUERY_CHARS} символов"
    elif len(stripped) > MAX_QUERY_CHARS:
        errors.query = f"описание должно быть не длиннее {MAX_QUERY_CHARS} символов"
    return errors


def unknown_view(*, request_id: str | None = None) -> SearchView:
    """Ответ «Не знаю» на месте обоих ответов: поиск или генерация не удались."""

    return SearchView(
        answer_without_context=UNKNOWN_ANSWER,
        answer_with_context=UNKNOWN_ANSWER,
        request_id=request_id,
    )


def pending_view() -> SearchView:
    """Статус на время выполнения запроса."""

    return SearchView(
        answer_without_context=PENDING_ANSWER,
        answer_with_context=PENDING_ANSWER,
    )


def build_view(payload: dict) -> SearchView:
    """Построить модель экрана по ответу API поиска.

    Показываются только сгенерированные ответы: список похожих вопросов,
    оценки и статусы источников намеренно не переносятся на экран. Ответ без
    подгрузки формируется всегда, поэтому его отсутствие или неудача сводятся
    к «Не знаю». Ответ с подгрузкой может отсутствовать вовсе: тогда его место
    остаётся пустым, и выдумывать вместо него текст нельзя.
    """

    request_id = payload.get("request_id")
    without_context = _answer_text(payload.get("answer_without_context"))
    with_context = _answer_or_absent(payload.get("answer_with_context"))

    if without_context is None:
        without_context = UNKNOWN_ANSWER

    return SearchView(
        answer_without_context=without_context,
        answer_with_context=with_context,
        request_id=request_id,
    )


def _answer_or_absent(answer: object) -> str | None:
    """Ответ с подгрузкой: текст, «Не знаю» или пустое место."""

    if answer is None:
        return None
    return _answer_text(answer) or UNKNOWN_ANSWER


def _answer_text(answer: object) -> str | None:
    """Текст из блока ответа API, если он содержательный."""

    if not isinstance(answer, dict):
        return None
    text = answer.get("text")
    if answer.get("available") and isinstance(text, str) and text.strip():
        return text.strip()
    return None


__all__ = [
    "MAX_QUERY_CHARS",
    "MIN_QUERY_CHARS",
    "PENDING_ANSWER",
    "UNKNOWN_ANSWER",
    "WITHOUT_CONTEXT_CAPTION",
    "WITH_CONTEXT_CAPTION",
    "FormErrors",
    "SearchView",
    "build_view",
    "pending_view",
    "unknown_view",
    "validate_form",
]
