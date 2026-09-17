"""Контейнер зависимостей HTTP-слоя."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.config import Settings, get_settings
from chgk_agent.embeddings.base import ChatProvider, EmbeddingProvider
from chgk_agent.external.base import ExternalQuestionSource
from chgk_agent.ingestion.service import IngestionService
from chgk_agent.observability.metrics import Metrics, get_metrics
from chgk_agent.search.graph import SearchGraph, build_search_graph
from chgk_agent.search.nodes import SearchDeps

EXTERNAL_DEPENDENCY = "external_source"


@dataclass(slots=True)
class AppDeps:
    """Зависимости приложения, собранные на старте."""

    session_factory: async_sessionmaker[AsyncSession]
    embedding_provider: EmbeddingProvider
    settings: Settings = field(default_factory=get_settings)
    external_source: ExternalQuestionSource | None = None
    chat_provider: ChatProvider | None = None
    ingestion: IngestionService | None = None
    metrics: Metrics | None = None

    @property
    def current_metrics(self) -> Metrics:
        """Метрики приложения."""

        return self.metrics or get_metrics()

    def search_graph(self) -> SearchGraph:
        """Собрать граф поиска с текущими зависимостями."""

        return build_search_graph(
            SearchDeps(
                session_factory=self.session_factory,
                embedding_provider=self.embedding_provider,
                external_source=self.external_source,
                chat_provider=self.chat_provider,
                settings=self.settings,
                metrics=self.metrics,
            )
        )

    async def check_database(self, timeout_seconds: float = 3.0) -> tuple[bool, str | None]:
        """Проверить доступность базы данных."""

        try:
            async with asyncio.timeout(timeout_seconds):
                async with self.session_factory() as session:
                    await session.execute(text("SELECT 1"))
        except Exception as error:  # pragma: no cover - зависит от окружения
            return False, str(error)
        return True, None

    async def check_external(self, timeout_seconds: float = 3.0) -> tuple[bool, str | None]:
        """Проверить доступность внешнего источника."""

        source = self.external_source
        if source is None:
            return False, "внешний источник не настроен"
        if not source.enabled:
            return True, "источник отключён настройкой"

        try:
            async with asyncio.timeout(timeout_seconds):
                result = await source.search("проверка готовности", limit=1)
        except Exception as error:  # pragma: no cover - зависит от сети
            return False, str(error)

        if result.status.value == "unavailable":
            return False, result.error or "внешний источник недоступен"
        return True, None

    async def start_import(self, paths: list[Path], *, operation_id: str | None = None):
        """Запустить операцию импорта через сервис индексации."""

        if self.ingestion is None:
            raise RuntimeError("сервис импорта не настроен")
        return await self.ingestion.start_import(paths, operation_id=operation_id)


__all__ = ["EXTERNAL_DEPENDENCY", "AppDeps"]
