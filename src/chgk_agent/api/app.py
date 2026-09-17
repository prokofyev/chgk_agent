"""Сборка ASGI-приложения FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from chgk_agent.api.deps import AppDeps
from chgk_agent.api.errors import install_error_handlers
from chgk_agent.api.middleware import RequestIdMiddleware
from chgk_agent.api.routes import router
from chgk_agent.config import Settings, get_settings
from chgk_agent.embeddings.validation import verify_schema_embedding_dimension
from chgk_agent.ingestion.service import IngestionService
from chgk_agent.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(
    deps: AppDeps | None = None,
    *,
    settings: Settings | None = None,
    with_lifespan: bool = True,
) -> FastAPI:
    """Собрать приложение с роутерами, middleware и обработчиками ошибок."""

    current_settings = settings or (deps.settings if deps else get_settings())
    application = FastAPI(
        title="chgk-agent",
        version="0.1.0",
        lifespan=lifespan if with_lifespan else None,
    )
    application.state.deps = deps
    application.state.settings = current_settings

    application.add_middleware(RequestIdMiddleware)
    install_error_handlers(application)
    application.include_router(router)

    @application.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        registry = deps.current_metrics.registry if deps is not None else None
        payload = generate_latest(registry) if registry is not None else generate_latest()
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    return application


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Настроить логирование и корректно освободить зависимости."""

    settings: Settings = application.state.settings
    configure_logging(
        json_logs=settings.observability.json_logs,
        level=settings.observability.log_level,
    )
    deps: AppDeps | None = application.state.deps
    logger.info("сервис запускается")

    if deps is not None:
        await verify_schema_embedding_dimension(deps.session_factory, settings)

    if deps is not None and deps.ingestion is None:
        deps.ingestion = IngestionService(
            deps.session_factory,
            embedding_provider=deps.embedding_provider,
            metrics=deps.metrics,
        )

    try:
        yield
    finally:
        source = deps.external_source if deps is not None else None
        close = getattr(source, "aclose", None)
        if close is not None:
            await close()
        logger.info("сервис остановлен")


def build_default_app(*, with_ui: bool = True) -> FastAPI:
    """Собрать приложение с зависимостями из настроек окружения.

    Это точка входа для `uvicorn --factory`: приложение получает реальные
    провайдеры GigaChat, клиент внешнего источника и смонтированный
    веб-интерфейс.
    """

    settings = get_settings()
    from chgk_agent.db.session import create_engine, create_session_factory
    from chgk_agent.embeddings.gigachat import (
        GigaChatChatProvider,
        GigaChatEmbeddingProvider,
    )
    from chgk_agent.external.gotquestions import GotQuestionsSource

    engine = create_engine(settings)
    deps = AppDeps(
        session_factory=create_session_factory(engine),
        embedding_provider=GigaChatEmbeddingProvider(settings.gigachat),
        settings=settings,
        external_source=GotQuestionsSource(settings.external),
        chat_provider=GigaChatChatProvider(settings.gigachat),
    )
    application = create_app(deps, settings=settings)

    if with_ui:
        from chgk_agent.ui.app import mount_ui
        from chgk_agent.ui.client import SearchApiClient

        mount_ui(application, SearchApiClient(application))

    return application


__all__ = ["build_default_app", "create_app", "lifespan"]
