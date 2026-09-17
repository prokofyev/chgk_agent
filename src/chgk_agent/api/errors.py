"""Единый конверт ошибок HTTP API и обработчики исключений."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from chgk_agent.api.schemas import ErrorResponse
from chgk_agent.logging_setup import get_logger, get_request_id

logger = get_logger(__name__)

GENERIC_ERROR_CODE = "internal_error"
GENERIC_ERROR_MESSAGE = "внутренняя ошибка сервиса"
VALIDATION_ERROR_CODE = "validation_error"
NOT_FOUND_CODE = "not_found"

REQUEST_ID_HEADER = "X-Request-ID"


def error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, object]] | None = None,
    request: Request | None = None,
) -> JSONResponse:
    """Сформировать ответ об ошибке в едином формате.

    Идентификатор берётся из контекста, а если обработчик вызван за его
    пределами (например, для необработанного исключения) — из состояния
    запроса, куда его положил middleware.
    """

    request_id = get_request_id()
    if request_id is None and request is not None:
        request_id = getattr(request.state, "request_id", None)

    payload = ErrorResponse(
        code=code,
        message=message,
        request_id=request_id,
        details=details,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def install_error_handlers(app: FastAPI) -> None:
    """Зарегистрировать обработчики ошибок в едином формате."""

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException):
        is_not_found = exc.status_code == 404
        code = "not_found" if is_not_found else "http_error"
        message = exc.detail if isinstance(exc.detail, str) else "ошибка запроса"
        return error_response(
            status_code=exc.status_code,
            code=code,
            message=message,
            request=request,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())),
                "message": error.get("msg", ""),
            }
            for error in exc.errors()
        ]
        return error_response(
            status_code=422,
            code=VALIDATION_ERROR_CODE,
            message="запрос не прошёл валидацию",
            details=details,
            request=request,
        )

    @app.exception_handler(Exception)
    async def _unexpected_error(request: Request, exc: Exception):
        logger.exception("необработанная ошибка запроса", error=str(exc))
        return error_response(
            status_code=500,
            code=GENERIC_ERROR_CODE,
            message=GENERIC_ERROR_MESSAGE,
            request=request,
        )


__all__ = [
    "GENERIC_ERROR_CODE",
    "GENERIC_ERROR_MESSAGE",
    "REQUEST_ID_HEADER",
    "VALIDATION_ERROR_CODE",
    "error_response",
    "install_error_handlers",
]
