"""Идемпотентный импорт разобранных вопросов в базу."""

import asyncio
import hashlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from chgk_agent.db import repository
from chgk_agent.embeddings.base import EmbeddingProvider
from chgk_agent.embeddings.indexer import embed_pending
from chgk_agent.logging_setup import get_logger
from chgk_agent.models.domain import (
    CanonicalQuestion,
    ImportReport,
    ParseResult,
)
from chgk_agent.observability.metrics import Metrics
from chgk_agent.search.lexical import get_corpus_index

logger = get_logger(__name__)

EXTERNAL_SOURCE_KIND = "html"

ProgressCallback = Callable[[ImportReport], Awaitable[None]]


def file_content_hash(path: Path) -> str:
    """Вернуть sha256 содержимого файла."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_source_file(path: Path) -> str | None:
    """Посчитать хеш файла-источника, если он существует.

    Выполняется в отдельном потоке: чтение файла — блокирующая операция.
    """

    return file_content_hash(path) if path.is_file() else None


class QuestionImporter:
    """Импорт разобранных вопросов с дедупликацией и векторизацией."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider | None = None,
        *,
        metrics: Metrics | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._session = session
        self._embedding_provider = embedding_provider
        self._metrics = metrics
        self._on_progress = on_progress

    async def import_results(
        self,
        results: list[ParseResult],
        *,
        operation_id: str | None = None,
    ) -> ImportReport:
        """Импортировать результаты разбора и вернуть отчёт."""

        report = ImportReport(operation_id=operation_id or uuid.uuid4().hex)
        report.started_at = datetime.now(UTC)
        report.locations = [result.location for result in results]

        for result in results:
            report.issues.extend(result.issues)
            report.skipped += result.skipped
            await self._import_one(result, report)
            await self._publish(report)

        if self._embedding_provider is not None:
            report.unembedded = await self._embed_pending()
            await self._publish(report)

        report.finished_at = datetime.now(UTC)
        self._record_metrics(report)
        logger.info(
            "импорт завершён",
            operation_id=report.operation_id,
            processed=report.processed,
            added=report.added,
            updated=report.updated,
            unchanged=report.unchanged,
            skipped=report.skipped,
            unembedded=report.unembedded,
            partial=report.is_partial,
        )
        return report

    async def _publish(self, report: ImportReport) -> None:
        """Отдать промежуточный снимок счётчиков наблюдателю."""

        if self._on_progress is None:
            return
        await self._on_progress(report)

    def _record_metrics(self, report: ImportReport) -> None:
        """Обновить метрики индексации по итоговому отчёту."""

        metrics = self._metrics
        if metrics is None:
            return

        for status, count in (
            ("added", report.added),
            ("updated", report.updated),
            ("unchanged", report.unchanged),
            ("skipped", report.skipped),
            ("unembedded", report.unembedded),
        ):
            if count:
                metrics.ingestion.questions.labels(status=status).inc(count)

        if report.skipped:
            metrics.ingestion.errors.labels(stage="parsing").inc(report.skipped)
        if report.unembedded:
            metrics.ingestion.errors.labels(stage="embedding").inc(report.unembedded)

        metrics.ingestion.operations.labels(
            status="partial" if report.is_partial else "completed"
        ).inc()

    async def _import_one(self, result: ParseResult, report: ImportReport) -> None:
        content_hash = await asyncio.to_thread(_hash_source_file, Path(result.location))
        source = await repository.upsert_source(
            self._session,
            kind=EXTERNAL_SOURCE_KIND,
            location=result.location,
            content_hash=content_hash,
            status="imported",
        )

        existing = await repository.existing_occurrence_hashes(self._session, source.id)

        for parsed in result.questions:
            canonical = CanonicalQuestion.from_parsed(parsed)
            report.processed += 1

            previous_hash = existing.get(parsed.source_key)
            if previous_hash == canonical.text_hash:
                report.unchanged += 1
                continue

            question_id = await repository.upsert_canonical_question(
                self._session, canonical
            )
            await repository.upsert_occurrence(
                self._session,
                question_id=question_id,
                source_id=source.id,
                parsed=replace(parsed, source_key=parsed.source_key),
            )

            if previous_hash is None:
                report.added += 1
            else:
                report.updated += 1

        await repository.delete_stale_occurrences(
            self._session,
            source_id=source.id,
            keep_keys=[parsed.source_key for parsed in result.questions],
        )
        await repository.delete_orphan_questions(self._session)

        # Тексты вопросов изменились, поэтому кэш лексического индекса
        # устарел: BM25 считает статистику по составу корпуса.
        get_corpus_index().invalidate()

    async def _embed_pending(self) -> int:
        """Векторизовать вопросы без эмбеддинга.

        Возвращает число вопросов, которые остались без эмбеддинга из-за ошибки.
        """

        provider = self._embedding_provider
        if provider is None:
            return 0

        outcome = await embed_pending(self._session, provider)
        return outcome.remaining
