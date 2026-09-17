"""Внутрипроцессный клиент API для веб-интерфейса.

UI ходит в API через `ASGITransport`: без сети и порта, но по тому же
контракту, что и внешние клиенты. Это позволяет тестировать интерфейс на
том же приложении FastAPI.
"""

import httpx

from chgk_agent.ui.view import SearchView, build_view, error_view


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
        min_score: float = 0.0,
        generate_answer: bool = True,
    ) -> SearchView:
        """Выполнить поиск и вернуть модель экрана."""

        payload = {
            "query": query,
            "limit": limit,
            "min_score": min_score,
            "generate_answer": generate_answer,
        }
        try:
            async with httpx.AsyncClient(
                transport=self._transport, base_url="http://ui"
            ) as client:
                response = await client.post("/search", json=payload)
        except httpx.HTTPError as error:  # pragma: no cover - защита от сети
            return error_view(f"Сервис поиска недоступен: {error}")

        if response.status_code != 200:
            return self._error_view(response)
        return build_view(response.json())

    @staticmethod
    def _error_view(response: httpx.Response) -> SearchView:
        """Превратить ответ об ошибке в модель экрана."""

        try:
            payload = response.json()
        except ValueError:  # pragma: no cover - не-JSON ответ
            return error_view(f"Поиск завершился ошибкой ({response.status_code})")

        message = payload.get("message") or f"Поиск завершился ошибкой ({response.status_code})"
        if response.status_code == 422:
            message = "Проверьте параметры запроса: " + message
        return error_view(message, request_id=payload.get("request_id"))


__all__ = ["SearchApiClient", "SearchApiError"]
