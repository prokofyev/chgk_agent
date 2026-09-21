"""Эндпоинты поиска, индексации и служебных проверок."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from chgk_agent.api.deps import EXTERNAL_DEPENDENCY, AppDeps
from chgk_agent.api.schemas import (
    HealthResponse,
    ImportRequest,
    ImportResponse,
    ReadinessDependency,
    ReadinessResponse,
    SearchRequest,
    SearchResponse,
)
from chgk_agent.ingestion.operations import ImportOperation
from chgk_agent.logging_setup import get_request_id

router = APIRouter()


def get_deps(request: Request) -> AppDeps:
    """Достать зависимости приложения из состояния запроса."""

    return request.app.state.deps


Deps = Annotated[AppDeps, Depends(get_deps)]


def _import_response(operation: ImportOperation) -> ImportResponse:
    """Преобразовать состояние операции импорта в схему ответа."""

    return ImportResponse(
        operation_id=operation.operation_id,
        status=operation.status.value,
        locations=list(operation.locations),
        processed=operation.processed,
        added=operation.added,
        updated=operation.updated,
        unchanged=operation.unchanged,
        skipped=operation.skipped,
        unembedded=operation.unembedded,
        issues=[
            {"location": issue.location, "message": issue.message}
            for issue in operation.issues
        ],
        error=operation.error,
        started_at=operation.started_at,
        finished_at=operation.finished_at,
    )


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, deps: Deps) -> SearchResponse:
    """Найти похожие вопросы по описанию."""

    outcome = await deps.search_graph().run(
        payload.query,
        limit=payload.limit,
        min_score=payload.min_score,
        disable_lexical=not deps.settings.search.use_lexical,
        request_id=get_request_id(),
    )
    return SearchResponse.from_domain(outcome)


@router.post("/imports", response_model=ImportResponse, status_code=202)
async def start_import(payload: ImportRequest, deps: Deps) -> ImportResponse:
    """Запустить операцию импорта HTML-источников."""

    operation = await deps.start_import([Path(path) for path in payload.paths])
    return _import_response(operation)


@router.get("/imports/{operation_id}", response_model=ImportResponse)
async def get_import(operation_id: str, deps: Deps):
    """Вернуть состояние операции импорта."""

    if deps.ingestion is None:
        return JSONResponse(
            status_code=404,
            content={
                "code": "not_found",
                "message": "сервис импорта не настроен",
                "request_id": get_request_id(),
            },
        )

    operation = await deps.ingestion.registry.get(operation_id)
    if operation is None:
        return JSONResponse(
            status_code=404,
            content={
                "code": "not_found",
                "message": f"операция {operation_id} не найдена",
                "request_id": get_request_id(),
            },
        )
    return _import_response(operation)


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Проверка жизнеспособности без обращения к внешним системам."""

    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadinessResponse)
async def readyz(deps: Deps):
    """Проверка готовности с учётом базы данных и внешнего источника."""

    dependencies: list[ReadinessDependency] = []

    database_ok, database_error = await deps.check_database()
    dependencies.append(
        ReadinessDependency(
            name="database",
            status="ok" if database_ok else "unavailable",
            detail=database_error,
        )
    )

    external_ok, external_error = True, None
    if deps.external_source is not None and deps.external_source.enabled:
        external_ok, external_error = await deps.check_external()
        dependencies.append(
            ReadinessDependency(
                name=EXTERNAL_DEPENDENCY,
                status="ok" if external_ok else "unavailable",
                detail=external_error,
            )
        )
    else:
        dependencies.append(
            ReadinessDependency(name=EXTERNAL_DEPENDENCY, status="disabled")
        )

    ready = database_ok and external_ok
    payload = ReadinessResponse(
        status="ok" if ready else "unavailable", dependencies=dependencies
    )
    if not ready:
        return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))
    return payload


__all__ = ["get_deps", "router"]
