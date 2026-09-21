"""Веб-интерфейс NiceGUI, смонтированный в то же ASGI-приложение.

Страница ходит в API внутрипроцессно через `SearchApiClient`, поэтому
контракт поиска ровно один, а интерфейс можно тестировать без запуска
сервера.

Пользователь видит два ответа: без подгрузки похожих вопросов и с
подгрузкой. Так видно, изменился ли ответ благодаря поиску по базам
вопросов. Похожие вопросы остаются контекстом модели, а не выдачей, а всё,
что не удалось получить, показывается как «Не знаю», без технических
подробностей.
"""

from collections.abc import Callable

from fastapi import FastAPI

from chgk_agent.logging_setup import get_logger
from chgk_agent.ui.client import SearchApiClient
from chgk_agent.ui.view import (
    WITH_CONTEXT_CAPTION,
    WITHOUT_CONTEXT_CAPTION,
    SearchView,
    pending_view,
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

logger = get_logger(__name__)


def is_submittable(query: str) -> bool:
    """Достаточно ли описание, чтобы отправлять запрос."""

    return not validate_form(query).has_errors


def form_enabled(query: str, *, busy: bool) -> tuple[bool, bool]:
    """Включены ли поле ввода и кнопка.

    Порядок значений — поле, кнопка. Пока идёт запрос, заблокировано и
    поле: иначе ответ пришёл бы на прежний текст, а в поле уже стоял бы
    другой вопрос. В покое поле доступно всегда, даже пустое, — иначе в
    него нельзя было бы ничего ввести; кнопка же требует описания
    достаточной длины.
    """

    if busy:
        return False, False
    return True, is_submittable(query)


async def run_answer(
    query: str,
    client: SearchApiClient,
    *,
    limit: int = DEFAULT_LIMIT,
    min_score: float | None = None,
    set_busy: Callable[[bool], None],
    show: Callable[[SearchView], None],
) -> None:
    """Провести один запрос: статус выполнения, затем ответ.

    Порядок показа задан здесь, а не в разметке: прежний ответ исчезает
    сразу, статус появляется до ожидания, а разблокировка выполняется в
    `finally`, чтобы неожиданный сбой не оставил форму заблокированной.
    """

    if not is_submittable(query):
        show(unknown_view())
        return

    set_busy(True)
    show(pending_view())
    try:
        view = await answer_for(query, client, limit=limit, min_score=min_score)
    except Exception:
        logger.warning("не удалось получить ответ", exc_info=True)
        view = unknown_view()
    finally:
        set_busy(False)
    show(view)


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

        ui.label(WITHOUT_CONTEXT_CAPTION).classes("text-sm text-gray-500")
        answer_without_context_label = ui.label().classes("text-lg")
        ui.label(WITH_CONTEXT_CAPTION).classes("text-sm text-gray-500")
        answer_with_context_label = ui.label().classes("text-lg")
        button = ui.button(BUTTON_LABEL, on_click=lambda: run_search()).props(
            "color=primary"
        )
        busy = False

        def sync_form() -> None:
            """Привести доступность поля и кнопки к текущему состоянию."""

            query_input.enabled, button.enabled = form_enabled(
                query_input.value or "", busy=busy
            )

        def set_busy(value: bool) -> None:
            """Запомнить, что запрос выполняется, и обновить форму."""

            nonlocal busy
            busy = value
            sync_form()

        async def run_search() -> None:
            """Проверить описание и показать ответ."""

            await run_answer(
                query_input.value or "",
                client,
                limit=limit,
                min_score=min_score,
                set_busy=set_busy,
                show=lambda view: _render_view(
                    view,
                    answer_without_context_label,
                    answer_with_context_label,
                ),
            )

        query_input.on_value_change(sync_form)
        sync_form()


def _render_view(
    view: SearchView,
    answer_without_context_label: object,
    answer_with_context_label: object,
) -> None:
    """Показать оба ответа пользователю.

    Пустое место, а не выдуманный текст: когда второго прогона не было,
    показывать на его месте нечего.
    """

    answer_without_context_label.text = view.answer_without_context or ""
    answer_with_context_label.text = view.answer_with_context or ""


__all__ = [
    "BUTTON_LABEL",
    "DEFAULT_LIMIT",
    "DESCRIPTION_LABEL",
    "DESCRIPTION_PLACEHOLDER",
    "MOUNT_PATH",
    "PAGE_PATH",
    "TITLE",
    "answer_for",
    "form_enabled",
    "is_submittable",
    "mount_ui",
    "register_pages",
    "run_answer",
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
