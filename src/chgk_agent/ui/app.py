"""Веб-интерфейс NiceGUI, смонтированный в то же ASGI-приложение.

Страница ходит в API внутрипроцессно через `SearchApiClient`, поэтому
контракт поиска ровно один, а интерфейс можно тестировать без запуска
сервера.
"""

from fastapi import FastAPI

from chgk_agent.ui.client import SearchApiClient
from chgk_agent.ui.view import (
    MatchView,
    SearchState,
    SearchView,
    SourceView,
    running_view,
    validate_form,
)

MOUNT_PATH = "/ui"
PAGE_PATH = "/"
TITLE = "Поиск вопросов ЧГК"

STATE_COLORS: dict[SearchState, str] = {
    SearchState.IDLE: "#616161",
    SearchState.RUNNING: "#1565c0",
    SearchState.SUCCESS: "#2e7d32",
    SearchState.PARTIAL: "#ef6c00",
    SearchState.EMPTY: "#616161",
    SearchState.ERROR: "#c62828",
}

STATE_LABELS: dict[SearchState, str] = {
    SearchState.IDLE: "ожидание",
    SearchState.RUNNING: "поиск выполняется",
    SearchState.SUCCESS: "успешно",
    SearchState.PARTIAL: "частичные результаты",
    SearchState.EMPTY: "совпадений нет",
    SearchState.ERROR: "ошибка",
}

STATUS_COLORS: dict[str, str] = {
    "ok": "green",
    "empty": "grey",
    "rejected": "orange",
    "unavailable": "red",
    "disabled": "grey",
}


def register_pages(client: SearchApiClient) -> None:
    """Зарегистрировать страницы интерфейса."""

    from nicegui import ui

    @ui.page(PAGE_PATH)
    def index() -> None:
        """Страница поиска по описанию вопроса."""

        ui.page_title(TITLE)
        ui.label(TITLE).classes("text-2xl font-bold")

        query_input = ui.textarea(
            label="Описание вопроса",
            placeholder="Опишите вопрос своими словами",
        ).classes("w-full")

        with ui.row():
            limit_input = ui.number(label="Число результатов", value=20, min=1, max=100)
            score_input = ui.number(
                label="Минимальный порог", value=0.0, min=0.0, max=1.0, step=0.05
            )
            generate_input = ui.checkbox("Сгенерировать ответ", value=True)

        error_label = ui.label().classes("text-red-600")
        state_label = ui.label().classes("text-sm")

        status_container = ui.column().classes("w-full")
        result_container = ui.column().classes("w-full")
        answer_container = ui.column().classes("w-full")

        async def run_search() -> None:
            """Проверить форму и выполнить поиск."""

            errors = validate_form(
                query_input.value or "",
                limit=int(limit_input.value or 0),
                min_score=float(score_input.value or 0.0),
            )
            if errors.has_errors:
                error_label.text = (
                    errors.query or errors.limit or errors.min_score or ""
                )
                return

            error_label.text = ""
            render_view(
                running_view(),
                state_label,
                status_container,
                result_container,
                answer_container,
            )
            view = await client.search(
                query_input.value or "",
                limit=int(limit_input.value or 20),
                min_score=float(score_input.value or 0.0),
                generate_answer=bool(generate_input.value),
            )
            render_view(
                view,
                state_label,
                status_container,
                result_container,
                answer_container,
            )

        ui.button("Найти", on_click=run_search).props("color=primary")


def render_view(
    view: SearchView,
    state_label: object,
    status_container: object,
    result_container: object,
    answer_container: object,
) -> None:
    """Отрисовать модель экрана в контейнерах NiceGUI."""

    _render_state(view, state_label)
    _render_sources(view.sources, status_container)
    _render_matches(view, result_container)
    _render_answer(view, answer_container)


def _render_state(view: SearchView, state_label: object) -> None:
    """Показать состояние поиска и идентификатор запроса."""

    parts = [STATE_LABELS[view.state]]
    if view.message:
        parts.append(view.message)
    if view.request_id:
        parts.append(f"request_id: {view.request_id}")
    state_label.text = " · ".join(parts)
    state_label.style(f"color: {STATE_COLORS[view.state]}")


def _render_sources(sources: list[SourceView], container: object) -> None:
    """Показать статусы источников, включая отвергнутый и недоступный."""

    container.clear()
    with container:
        from nicegui import ui

        for source in sources:
            with ui.row().classes("items-center gap-2"):
                ui.label(source.label)
                ui.badge(
                    source.status_label,
                    color=STATUS_COLORS.get(source.status, "grey"),
                )
                if source.truncated and source.query:
                    ui.label(
                        f'внешний поиск шёл по короткому запросу: «{source.query}»'
                    ).classes("text-orange-600 text-sm")
                if source.is_problem and source.error:
                    ui.label(source.error).classes("text-red-600 text-sm")


def _render_matches(view: SearchView, container: object) -> None:
    """Показать список совпадений с источником, оценкой и ссылкой."""

    container.clear()
    with container:
        from nicegui import ui

        if not view.has_matches:
            ui.label(
                "Совпадений нет. Измените формулировку описания или понизьте порог."
            )
            return

        for match in view.matches:
            _render_match(match)


def _render_match(match: MatchView) -> None:
    """Показать одно совпадение."""

    from nicegui import ui

    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-2"):
            ui.badge(
                "внешний источник" if match.is_external else "локальная база",
                color="purple" if match.is_external else "blue",
            )
            ui.label(f"{match.score_percent}% — {match.score_label}")
            for label in match.source_labels:
                ui.label(label).classes("text-sm text-gray-500")
        ui.label(match.question_text).classes("text-lg")
        if match.answer_text:
            ui.label(f"Ответ: {match.answer_text}")
        if match.comment:
            ui.label(f"Комментарий: {match.comment}").classes("text-sm")
        if match.external_url:
            ui.link("Открыть на gotquestions.online", match.external_url)


def _render_answer(view: SearchView, container: object) -> None:
    """Показать сгенерированный ответ отдельно от совпадений."""

    container.clear()
    answer = view.answer
    if answer is None:
        return

    with container:
        from nicegui import ui

        ui.separator()
        ui.label("Сгенерированный ответ").classes("text-lg font-bold")
        if answer.available and answer.text:
            ui.label(answer.text)
            if answer.used_matches:
                ui.label("Опирается на найденные вопросы:").classes("text-sm")
                for text in answer.used_matches:
                    ui.label(f"— {text}").classes("text-sm text-gray-600")
        else:
            ui.label(
                answer.message
                or "Сгенерированный ответ получить не удалось."
            ).classes("text-orange-600")


def mount_ui(app: FastAPI, client: SearchApiClient) -> None:
    """Смонтировать NiceGUI в существующее приложение FastAPI."""

    from nicegui import ui

    register_pages(client)
    ui.run_with(app, mount_path=MOUNT_PATH, title=TITLE, storage_secret="chgk-agent")


__all__ = [
    "MOUNT_PATH",
    "PAGE_PATH",
    "STATE_COLORS",
    "STATE_LABELS",
    "TITLE",
    "mount_ui",
    "register_pages",
    "render_view",
]
