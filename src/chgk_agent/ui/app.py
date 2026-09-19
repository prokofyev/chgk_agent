"""Веб-интерфейс NiceGUI, смонтированный в то же ASGI-приложение.

Страница ходит в API внутрипроцессно через `SearchApiClient`, поэтому
контракт поиска ровно один, а интерфейс можно тестировать без запуска
сервера.

Пользователь видит только вопрос и ответ: похожие вопросы служат модели
контекстом, а не выдачей. Всё, что не удалось получить, показывается как
«Не знаю», без технических подробностей.
"""

from fastapi import FastAPI

from chgk_agent.ui.client import SearchApiClient
from chgk_agent.ui.view import (
    SearchView,
    unknown_view,
    validate_form,
)

MOUNT_PATH = "/ui"
PAGE_PATH = "/"
TITLE = "ЧГК знаток"
BUTTON_LABEL = "Ответить"
DESCRIPTION_LABEL = "Описание вопроса"
DESCRIPTION_PLACEHOLDER = "Опишите вопрос своими словами"

DEFAULT_LIMIT = 20


def is_submittable(query: str) -> bool:
    """Достаточно ли описание, чтобы отправлять запрос."""

    return not validate_form(query).has_errors


async def answer_for(
    query: str,
    client: SearchApiClient,
    *,
    limit: int = DEFAULT_LIMIT,
    min_score: float | None = None,
) -> SearchView:
    """Получить ответ на описание вопроса.

    Недостаточное описание обрабатывается здесь, а не только блокировкой
    кнопки: обход блокировки не должен приводить к обращению к поиску и
    даёт «Не знаю».
    """

    if not is_submittable(query):
        return unknown_view()
    return await client.search(query, limit=limit, min_score=min_score)


def register_pages(
    client: SearchApiClient,
    *,
    limit: int = DEFAULT_LIMIT,
    min_score: float | None = None,
) -> None:
    """Зарегистрировать страницы интерфейса."""

    from nicegui import ui

    @ui.page(PAGE_PATH)
    def index() -> None:
        """Страница ответа на вопрос по описанию."""

        ui.page_title(TITLE)

        query_input = ui.textarea(
            label=DESCRIPTION_LABEL,
            placeholder=DESCRIPTION_PLACEHOLDER,
        ).classes("w-full")

        answer_label = ui.label().classes("text-lg")

        async def run_search() -> None:
            """Проверить описание и показать ответ."""

            view = await answer_for(
                query_input.value or "",
                client,
                limit=limit,
                min_score=min_score,
            )
            _render_view(view, answer_label)

        button = ui.button(BUTTON_LABEL, on_click=run_search).props("color=primary")
        button.bind_enabled_from(query_input, "value", backward=is_submittable)


def _render_view(view: SearchView, answer_label: object) -> None:
    """Показать ответ пользователю."""

    answer_label.text = view.answer_text


__all__ = [
    "BUTTON_LABEL",
    "DEFAULT_LIMIT",
    "DESCRIPTION_LABEL",
    "DESCRIPTION_PLACEHOLDER",
    "MOUNT_PATH",
    "PAGE_PATH",
    "TITLE",
    "answer_for",
    "is_submittable",
    "mount_ui",
    "register_pages",
]


def mount_ui(app: FastAPI, client: SearchApiClient, *, limit: int | None = None) -> None:
    """Смонтировать NiceGUI в существующее приложение FastAPI."""

    from nicegui import ui

    settings = getattr(app.state, "settings", None)
    search_settings = getattr(settings, "search", None)
    register_pages(
        client,
        limit=limit or getattr(search_settings, "top_k", DEFAULT_LIMIT),
        min_score=getattr(search_settings, "min_score", None),
    )
    ui.run_with(app, mount_path=MOUNT_PATH, title=TITLE, storage_secret="chgk-agent")
