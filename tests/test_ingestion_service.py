"""Тесты фонового сервиса импорта."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.db import repository
from chgk_agent.db.base import Question, Source
from chgk_agent.ingestion.operations import OperationStatus
from chgk_agent.ingestion.service import IngestionService
from chgk_agent.observability.metrics import Metrics

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "telegram_export.html"
LOCATION = str(FIXTURE)


def _metrics() -> Metrics:
    return Metrics(CollectorRegistry())


async def _purge(session_factory: async_sessionmaker[AsyncSession]) -> None:
    """Удалить данные фикстуры: сервис коммитит их в общую базу."""

    async with session_factory() as session:
        await session.execute(delete(Source).where(Source.location == LOCATION))
        await session.execute(delete(Question).where(~Question.occurrences.any()))
        await session.commit()


@pytest.fixture
async def clean_fixture(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    """Гарантировать, что импорт фикстуры начинается с чистого состояния."""

    await _purge(session_factory)
    try:
        yield
    finally:
        await _purge(session_factory)


async def test_run_import_returns_full_report(
    session_factory: async_sessionmaker[AsyncSession],
    clean_fixture: None,
) -> None:
    service = IngestionService(session_factory, metrics=_metrics())

    report = await service.run_import([FIXTURE])

    assert report.processed > 0
    assert report.added + report.unchanged == report.processed
    assert report.added > 0
    assert report.locations == [LOCATION]


async def test_repeat_import_is_idempotent_through_service(
    session_factory: async_sessionmaker[AsyncSession],
    clean_fixture: None,
) -> None:
    service = IngestionService(session_factory)

    first = await service.run_import([FIXTURE])
    second = await service.run_import([FIXTURE])

    assert first.added > 0
    assert second.added == 0
    assert second.unchanged == first.processed


async def test_run_import_registers_finished_operation(
    session_factory: async_sessionmaker[AsyncSession],
    clean_fixture: None,
) -> None:
    service = IngestionService(session_factory)
    operation_id = "sync-op"

    await service.run_import([FIXTURE], operation_id=operation_id)
    operation = await service.registry.get(operation_id)

    assert operation is not None
    assert operation.is_finished
    assert operation.status in {OperationStatus.COMPLETED, OperationStatus.PARTIAL}
    assert operation.processed > 0


async def test_background_import_is_registered_and_observable(
    session_factory: async_sessionmaker[AsyncSession],
    clean_fixture: None,
) -> None:
    service = IngestionService(session_factory)

    operation = await service.start_import([FIXTURE], operation_id="background-op")

    assert operation.status is OperationStatus.PENDING
    assert operation.locations == [LOCATION]

    await service.wait("background-op")
    finished = await service.registry.get("background-op")

    assert finished is not None
    assert finished.is_finished
    assert finished.processed > 0


async def test_unknown_paths_are_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    service = IngestionService(session_factory)

    with pytest.raises(FileNotFoundError):
        await service.start_import([Path("no-such-dir")])


async def test_queue_metric_is_refreshed_after_import(
    session_factory: async_sessionmaker[AsyncSession],
    clean_fixture: None,
) -> None:
    metrics = _metrics()
    service = IngestionService(session_factory, metrics=metrics)

    report = await service.run_import([FIXTURE])

    async with session_factory() as session:
        pending = await repository.count_questions_without_embedding(session)

    assert report.added > 0
    assert metrics.ingestion.unembedded._value.get() == pending
