"""Тесты реестра операций импорта."""

import pytest

from chgk_agent.ingestion.operations import (
    ImportOperationRegistry,
    OperationStatus,
)
from chgk_agent.models.domain import ImportReport, ParseIssue


def _report(operation_id: str, **overrides: object) -> ImportReport:
    report = ImportReport(operation_id=operation_id)
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


async def test_register_creates_pending_operation() -> None:
    registry = ImportOperationRegistry()

    operation = await registry.register("op-1", locations=["file-1.html"])

    assert operation.status is OperationStatus.PENDING
    assert operation.locations == ["file-1.html"]
    assert not operation.is_finished


async def test_update_exposes_intermediate_counters() -> None:
    registry = ImportOperationRegistry()
    await registry.register("op-1", locations=["file-1.html"])

    updated = await registry.update(
        "op-1", _report("op-1", processed=10, added=8, locations=["file-1.html"])
    )

    assert updated.status is OperationStatus.RUNNING
    assert updated.processed == 10
    assert updated.added == 8
    assert not updated.is_finished


async def test_finish_marks_completed_operation() -> None:
    registry = ImportOperationRegistry()
    await registry.register("op-1", locations=["file-1.html"])

    finished = await registry.finish(
        "op-1", _report("op-1", processed=3, added=3, locations=["file-1.html"])
    )

    assert finished.status is OperationStatus.COMPLETED
    assert finished.is_finished


async def test_finish_marks_partial_when_skips_exist() -> None:
    registry = ImportOperationRegistry()
    await registry.register("op-1", locations=["file-1.html"])

    finished = await registry.finish(
        "op-1", _report("op-1", processed=3, skipped=2, locations=["file-1.html"])
    )

    assert finished.status is OperationStatus.PARTIAL
    assert finished.is_finished


async def test_failed_operation_keeps_error() -> None:
    registry = ImportOperationRegistry()
    await registry.register("op-1", locations=[])

    failed = await registry.fail("op-1", "диск недоступен")

    assert failed.status is OperationStatus.FAILED
    assert failed.error == "диск недоступен"
    assert failed.finished_at is not None


async def test_unknown_operation_returns_none() -> None:
    registry = ImportOperationRegistry()

    assert await registry.get("missing") is None


async def test_update_unknown_operation_raises() -> None:
    registry = ImportOperationRegistry()

    with pytest.raises(KeyError):
        await registry.update("missing", _report("missing"))


async def test_snapshot_returns_latest_first() -> None:
    registry = ImportOperationRegistry()
    await registry.register("op-1", locations=[])
    await registry.register("op-2", locations=[])

    snapshot = await registry.snapshot()

    assert [operation.operation_id for operation in snapshot] == ["op-2", "op-1"]


async def test_registry_evicts_finished_operations() -> None:
    registry = ImportOperationRegistry(max_operations=2)

    for index in range(4):
        operation_id = f"op-{index}"
        await registry.register(operation_id, locations=[])
        await registry.finish(operation_id, _report(operation_id))

    snapshot = await registry.snapshot()

    assert len(snapshot) <= 2


async def test_to_report_round_trip() -> None:
    registry = ImportOperationRegistry()
    await registry.register("op-1", locations=["file-1.html"])
    report = _report(
        "op-1",
        processed=5,
        added=4,
        unembedded=1,
        locations=["file-1.html"],
        issues=[ParseIssue(location="file-1.html", message="пропущено")],
    )

    finished = await registry.finish("op-1", report)
    restored = finished.to_report()

    assert restored.operation_id == "op-1"
    assert restored.processed == 5
    assert restored.added == 4
    assert restored.unembedded == 1
    assert restored.is_partial is True
