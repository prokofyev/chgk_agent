"""Тесты наблюдаемости импорта: метрики, промежуточный прогресс, идентификатор."""

import pytest
from prometheus_client import CollectorRegistry
from sqlalchemy.ext.asyncio import AsyncSession

from chgk_agent.ingestion.importer import QuestionImporter
from chgk_agent.ingestion.operations import ImportOperationRegistry, OperationStatus
from chgk_agent.models.domain import ImportReport, ParsedQuestion, ParseResult
from chgk_agent.observability.metrics import Metrics

pytestmark = pytest.mark.integration


def _question(key: str, text: str) -> ParsedQuestion:
    return ParsedQuestion(source_key=key, question_text=text, answer_text="ответ")


def _metrics() -> Metrics:
    return Metrics(CollectorRegistry())


async def test_import_records_ingestion_metrics(session: AsyncSession) -> None:
    metrics = _metrics()
    importer = QuestionImporter(session, metrics=metrics)

    await importer.import_results(
        [
            ParseResult(
                location="metrics-fixture.html",
                questions=[_question("1", "Первый метрический вопрос?")],
                skipped=2,
            )
        ]
    )

    assert metrics.ingestion.questions.labels(status="added")._value.get() == 1
    assert metrics.ingestion.questions.labels(status="skipped")._value.get() == 2
    assert metrics.ingestion.errors.labels(stage="parsing")._value.get() == 2
    assert metrics.ingestion.operations.labels(status="partial")._value.get() == 1


async def test_import_records_embedding_errors(session: AsyncSession) -> None:
    class _BrokenEmbedder:
        @property
        def model(self) -> str:
            return "Broken"

        @property
        def dimension(self) -> int:
            from chgk_agent.db.base import get_embedding_dim

            return get_embedding_dim()

        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("GigaChat недоступен")

    metrics = _metrics()
    importer = QuestionImporter(session, _BrokenEmbedder(), metrics=metrics)

    report = await importer.import_results(
        [
            ParseResult(
                location="metrics-failure-fixture.html",
                questions=[_question("1", "Вопрос без вектора?")],
            )
        ]
    )

    assert report.unembedded == 1
    assert metrics.ingestion.errors.labels(stage="embedding")._value.get() == 1
    assert metrics.ingestion.questions.labels(status="unembedded")._value.get() == 1


async def test_progress_callback_receives_intermediate_counters(
    session: AsyncSession,
) -> None:
    snapshots: list[ImportReport] = []

    async def on_progress(report: ImportReport) -> None:
        snapshots.append(
            ImportReport(
                operation_id=report.operation_id,
                processed=report.processed,
            )
        )

    importer = QuestionImporter(session, on_progress=on_progress)

    await importer.import_results(
        [
            ParseResult(
                location="progress-fixture-1.html",
                questions=[_question("1", "Первый вопрос прогресса?")],
            ),
            ParseResult(
                location="progress-fixture-2.html",
                questions=[_question("1", "Второй вопрос прогресса?")],
            ),
        ]
    )

    assert [snapshot.processed for snapshot in snapshots] == [1, 2]


async def test_registry_operation_id_matches_report_and_logs(
    session: AsyncSession,
) -> None:
    registry = ImportOperationRegistry()
    operation_id = "op-shared-id"
    await registry.register(operation_id, locations=["shared-id-fixture.html"])

    async def on_progress(report: ImportReport) -> None:
        await registry.update(operation_id, report)

    importer = QuestionImporter(session, on_progress=on_progress)
    report = await importer.import_results(
        [
            ParseResult(
                location="shared-id-fixture.html",
                questions=[_question("1", "Вопрос со сквозным идентификатором?")],
            )
        ],
        operation_id=operation_id,
    )

    finished = await registry.finish(operation_id, report)

    assert report.operation_id == operation_id
    assert finished.operation_id == report.operation_id
    assert finished.status is OperationStatus.COMPLETED
    assert finished.processed == 1
