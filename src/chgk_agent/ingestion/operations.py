"""Реестр операций импорта для мониторинга их состояния.

Операции хранятся в памяти процесса: сервис рассчитан на одного
пользователя, а отчёт нужен только на время импорта. Каждое обновление
публикует снимок счётчиков, поэтому длительный импорт можно наблюдать
до его завершения.
"""

import asyncio
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum

from chgk_agent.models.domain import ImportReport, ParseIssue


class OperationStatus(StrEnum):
    """Статус операции импорта."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass(slots=True)
class ImportOperation:
    """Состояние одной операции импорта."""

    operation_id: str
    status: OperationStatus = OperationStatus.PENDING
    locations: list[str] = field(default_factory=list)
    processed: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    unembedded: int = 0
    issues: list[ParseIssue] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @property
    def is_finished(self) -> bool:
        """Завершена ли операция."""

        return self.status in {
            OperationStatus.COMPLETED,
            OperationStatus.FAILED,
            OperationStatus.PARTIAL,
        }

    def to_report(self) -> ImportReport:
        """Представить состояние операции отчётом импорта."""

        report = ImportReport(operation_id=self.operation_id)
        report.locations = list(self.locations)
        report.started_at = self.started_at
        report.finished_at = self.finished_at
        report.processed = self.processed
        report.added = self.added
        report.updated = self.updated
        report.unchanged = self.unchanged
        report.skipped = self.skipped
        report.unembedded = self.unembedded
        report.issues = list(self.issues)
        return report

    @classmethod
    def from_report(
        cls,
        report: ImportReport,
        *,
        status: OperationStatus,
        error: str | None = None,
    ) -> "ImportOperation":
        """Построить состояние операции по готовому отчёту."""

        return cls(
            operation_id=report.operation_id,
            status=status,
            locations=list(report.locations),
            processed=report.processed,
            added=report.added,
            updated=report.updated,
            unchanged=report.unchanged,
            skipped=report.skipped,
            unembedded=report.unembedded,
            issues=list(report.issues),
            started_at=report.started_at,
            finished_at=report.finished_at,
            error=error,
        )


class ImportOperationRegistry:
    """Хранилище операций импорта с асинхронным доступом."""

    def __init__(self, *, max_operations: int = 50) -> None:
        self._operations: dict[str, ImportOperation] = {}
        self._lock = asyncio.Lock()
        self._max_operations = max(max_operations, 1)

    async def register(self, operation_id: str, *, locations: list[str]) -> ImportOperation:
        """Зарегистрировать операцию в статусе ожидания."""

        operation = ImportOperation(
            operation_id=operation_id,
            status=OperationStatus.PENDING,
            locations=list(locations),
            started_at=datetime.now(UTC),
        )
        async with self._lock:
            self._operations[operation_id] = operation
            self._evict_locked()
        return operation

    async def update(self, operation_id: str, report: ImportReport) -> ImportOperation:
        """Обновить счётчики операции по промежуточному отчёту."""

        async with self._lock:
            current = self._operations.get(operation_id)
            if current is None:
                raise KeyError(operation_id)
            updated = replace(
                current,
                status=OperationStatus.RUNNING,
                locations=list(report.locations) or current.locations,
                processed=report.processed,
                added=report.added,
                updated=report.updated,
                unchanged=report.unchanged,
                skipped=report.skipped,
                unembedded=report.unembedded,
                issues=list(report.issues),
                started_at=report.started_at or current.started_at,
            )
            self._operations[operation_id] = updated
            return updated

    async def finish(self, operation_id: str, report: ImportReport) -> ImportOperation:
        """Завершить операцию, выбрав статус по признаку частичного результата."""

        status = OperationStatus.PARTIAL if report.is_partial else OperationStatus.COMPLETED
        return await self._finalize(operation_id, report, status=status)

    async def fail(self, operation_id: str, error: str) -> ImportOperation:
        """Пометить операцию как завершившуюся ошибкой."""

        async with self._lock:
            current = self._operations.get(operation_id)
            if current is None:
                raise KeyError(operation_id)
            failed = replace(
                current,
                status=OperationStatus.FAILED,
                error=error,
                finished_at=datetime.now(UTC),
            )
            self._operations[operation_id] = failed
            return failed

    async def get(self, operation_id: str) -> ImportOperation | None:
        """Вернуть состояние операции или `None`, если она неизвестна."""

        async with self._lock:
            return self._operations.get(operation_id)

    async def snapshot(self) -> list[ImportOperation]:
        """Вернуть все известные операции, свежие первыми."""

        async with self._lock:
            return list(reversed(list(self._operations.values())))

    async def clear(self) -> None:
        """Очистить реестр (используется в тестах)."""

        async with self._lock:
            self._operations.clear()

    async def _finalize(
        self,
        operation_id: str,
        report: ImportReport,
        *,
        status: OperationStatus,
    ) -> ImportOperation:
        async with self._lock:
            current = self._operations.get(operation_id)
            if current is None:
                raise KeyError(operation_id)
            finished = replace(
                ImportOperation.from_report(report, status=status),
                started_at=report.started_at or current.started_at,
            )
            self._operations[operation_id] = finished
            return finished

    def _evict_locked(self) -> None:
        """Удалить самые старые завершённые операции при переполнении."""

        if len(self._operations) <= self._max_operations:
            return
        for operation_id, operation in list(self._operations.items()):
            if len(self._operations) <= self._max_operations:
                break
            if operation.is_finished:
                del self._operations[operation_id]
