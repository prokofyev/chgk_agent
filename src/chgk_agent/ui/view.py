"""Модель экрана веб-интерфейса: описание вопроса и ответ.

Логика отображения отделена от NiceGUI: здесь только чистые функции,
превращающие ответ API в текст, который видит пользователь. Промежуточная
выдача поиска пользователю не показывается, поэтому модель экрана состоит
из одного ответа: он либо получен, либо равен «Не знаю».
"""

from dataclasses import dataclass

MIN_QUERY_CHARS = 3
MAX_QUERY_CHARS = 2000

UNKNOWN_ANSWER = "Не знаю"


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
    """Готовая модель экрана: только ответ на вопрос.

    `request_id` хранится для журнала, а не для показа: на странице
    технических подробностей быть не должно.
    """

    answer_text: str = UNKNOWN_ANSWER
    request_id: str | None = None

    @property
    def is_unknown(self) -> bool:
        """Не удалось ли получить ответ."""

        return self.answer_text == UNKNOWN_ANSWER


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
    """Ответ «Не знаю» для неудачи поиска или генерации."""

    return SearchView(answer_text=UNKNOWN_ANSWER, request_id=request_id)


def build_view(payload: dict) -> SearchView:
    """Построить модель экрана по ответу API поиска.

    Показывается только сгенерированный ответ: список похожих вопросов,
    оценки и статусы источников намеренно не переносятся на экран. Любой
    исход, кроме содержательного ответа, сводится к «Не знаю».
    """

    request_id = payload.get("request_id")
    answer = payload.get("answer")
    if not isinstance(answer, dict):
        return unknown_view(request_id=request_id)

    text = answer.get("text")
    if answer.get("available") and isinstance(text, str) and text.strip():
        return SearchView(answer_text=text.strip(), request_id=request_id)
    return unknown_view(request_id=request_id)


__all__ = [
    "MAX_QUERY_CHARS",
    "MIN_QUERY_CHARS",
    "UNKNOWN_ANSWER",
    "FormErrors",
    "SearchView",
    "build_view",
    "unknown_view",
    "validate_form",
]
