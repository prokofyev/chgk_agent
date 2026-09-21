"""Внутрипроцессный клиент API для веб-интерфейса.

UI ходит в API через `ASGITransport`: без сети и порта, но по тому же
контракту, что и внешние клиенты. Это позволяет тестировать интерфейс на
том же приложении FastAPI.
"""

import httpx

from chgk_agent.logging_setup import get_logger
from chgk_agent.ui.view import SearchView, build_view, unknown_view

logger = get_logger(__name__)


class SearchApiError(RuntimeError):
    """Ошибка обращения к API поиска."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class SearchApiClient:
    """Клиент эндпоинта поиска поверх ASGI-транспорта."""

    def __init__(self, app: object) -> None:
        self._transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        min_score: float | None = None,
    ) -> SearchView:
        """Выполнить поиск и вернуть модель экрана."""

        payload = {"query": query, "limit": limit}
        if min_score is not None:
            payload["min_score"] = min_score
        try:
            async with httpx.AsyncClient(
                transport=self._transport, base_url="http://ui"
            ) as client:
                response = await client.post("/search", json=payload)
        except httpx.HTTPError as error:  # pragma: no cover - защита от сети
            logger.warning("сервис поиска недоступен", error=str(error))
            return unknown_view()

        if response.status_code != 200:
            return self._error_view(response)
        return build_view(response.json())

    @staticmethod
    def _error_view(response: httpx.Response) -> SearchView:
        """Превратить ответ об ошибке в модель экрана."""

        request_id = None
        try:
            request_id = response.json().get("request_id")
        except ValueError:  # pragma: no cover - не-JSON ответ
            request_id = None
        return unknown_view(request_id=request_id)


__all__ = ["SearchApiClient", "SearchApiError"]
