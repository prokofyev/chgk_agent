"""Middleware сквозного идентификатора запроса."""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from chgk_agent.api.errors import REQUEST_ID_HEADER
from chgk_agent.logging_setup import request_context


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Привязать идентификатор запроса к контексту и вернуть его в ответе."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        with request_context(incoming) as request_id:
            request.state.request_id = request_id
            response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


__all__ = ["RequestIdMiddleware"]
