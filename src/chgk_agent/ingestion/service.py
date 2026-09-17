"""Фоновый запуск операций импорта с отслеживанием состояния.

HTTP-слой должен отвечать быстро, поэтому импорт выполняется отдельной
задачей: эндпоинт возвращает идентификатор операции, а клиент опрашивает
её состояние. Счётчики обновляются по ходу импорта, а не только в конце.
"""

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.db import repository
from chgk_agent.embeddings.base import EmbeddingProvider
from chgk_agent.ingestion.discovery import discover_files, parse_all
from chgk_agent.ingestion.importer import QuestionImporter
from chgk_agent.ingestion.operations import (
    ImportOperation,
    ImportOperationRegistry,
    OperationStatus,
)
from chgk_agent.logging_setup import get_logger, request_context
from chgk_agent.models.domain import ImportReport, ParseResult
from chgk_agent.models.reporting import summarizing_issue
from chgk_agent.observability.metrics import Metrics

logger = get_logger(__name__)


@dataclass(slots=True)
class ImportRequest:
    """Запрос на импорт набора путей."""

    paths: list[Path]


class IngestionService:
    """Оркестратор импорта HTML-источников."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        embedding_provider: EmbeddingProvider | None = None,
        registry: ImportOperationRegistry | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_provider = embedding_provider
        self._registry = registry or ImportOperationRegistry()
        self._metrics = metrics
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def registry(self) -> ImportOperationRegistry:
        """Реестр операций импорта."""

        return self._registry

    async def start_import(
        self, paths: list[Path], *, operation_id: str | None = None
    ) -> ImportOperation:
        """Запустить импорт в фоне и вернуть зарегистрированную операцию."""

        files = discover_files(paths)
        if not files:
            raise FileNotFoundError("не найдено ни одного HTML-файла по указанным путям")

        identifier = operation_id or uuid.uuid4().hex
        operation = await self._registry.register(
            identifier, locations=[str(path) for path in files]
        )
        task = asyncio.create_task(
            self._run(identifier, files), name=f"chgk-import-{identifier}"
        )
        self._tasks[identifier] = task
        task.add_done_callback(lambda _: self._tasks.pop(identifier, None))
        return operation

    async def run_import(
        self, paths: list[Path], *, operation_id: str | None = None
    ) -> ImportReport:
        """Выполнить импорт синхронно (для CLI и тестов)."""

        files = discover_files(paths)
        if not files:
            raise FileNotFoundError("не найдено ни одного HTML-файла по указанным путям")

        identifier = operation_id or uuid.uuid4().hex
        await self._registry.register(identifier, locations=[str(path) for path in files])
        return await self._execute(identifier, files)

    async def wait(self, operation_id: str) -> None:
        """Дождаться завершения фоновой операции, если она запущена."""

        task = self._tasks.get(operation_id)
        if task is not None:
            await asyncio.shield(task)

    async def _run(self, operation_id: str, files: list[Path]) -> None:
        try:
            await self._execute(operation_id, files)
        except Exception as error:  # pragma: no cover - защита фоновой задачи
            logger.exception("импорт прерван ошибкой", operation_id=operation_id)
            await self._registry.fail(operation_id, str(error))

    async def _execute(self, operation_id: str, files: list[Path]) -> ImportReport:
        with request_context(operation_id):
            results: list[ParseResult] = [
                result for _, result in parse_all(files)
            ]
            summary = summarizing_issue(results)
            if summary is not None:
                for result in results:
                    result.issues.append(summary)

            async with self._session_factory() as session:
                importer = QuestionImporter(
                    session,
                    self._embedding_provider,
                    metrics=self._metrics,
                    on_progress=lambda report: self._on_progress(operation_id, report),
                )
                report = await importer.import_results(
                    results, operation_id=operation_id
                )
                await session.commit()

        await self._registry.finish(operation_id, report)
        await self._refresh_queue_metric()
        return report

    async def _on_progress(self, operation_id: str, report: ImportReport) -> None:
        await self._registry.update(operation_id, report)

    async def _refresh_queue_metric(self) -> None:
        """Обновить метрику очереди невекторизованных вопросов."""

        if self._metrics is None:
            return

        async with self._session_factory() as session:
            remaining = await repository.count_questions_without_embedding(session)
        self._metrics.ingestion.unembedded.set(remaining)


__all__ = [
    "ImportOperation",
    "IngestionService",
    "OperationStatus",
]
