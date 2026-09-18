"""Фоновый доиндексатор очереди невекторизованных вопросов.

Импорт не должен падать из-за недоступного GigaChat: вопросы без
эмбеддинга остаются в очереди, а фоновый воркер постепенно их
векторизует и обновляет метрику размера очереди.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.db import repository
from chgk_agent.embeddings.base import EmbeddingProvider
from chgk_agent.embeddings.indexer import DEFAULT_BATCH_SIZE, embed_pending
from chgk_agent.logging_setup import get_logger
from chgk_agent.observability.metrics import Metrics, get_metrics
from chgk_agent.search.lexical import get_corpus_index

logger = get_logger(__name__)

DEFAULT_INTERVAL_SECONDS = 60.0


@dataclass(slots=True)
class ReindexRun:
    """Итог одного прохода доиндексатора."""

    embedded: int = 0
    failed: int = 0
    remaining: int = 0
    errors: list[str] = field(default_factory=list)


async def reindex_once(
    session: AsyncSession,
    provider: EmbeddingProvider,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    force: bool = False,
    metrics: Metrics | None = None,
) -> ReindexRun:
    """Выполнить один проход векторизации очереди и обновить метрики."""

    current = metrics or get_metrics()
    outcome = await embed_pending(
        session,
        provider,
        batch_size=batch_size,
        limit=limit,
        force=force,
    )
    if force:
        remaining = outcome.failed
    else:
        remaining = await repository.count_questions_for_reindex(
            session,
            model=provider.model,
            dimension=provider.dimension,
        )

    current.ingestion.unembedded.set(remaining)
    if outcome.embedded:
        current.ingestion.questions.labels(status="embedded").inc(outcome.embedded)
    if outcome.failed:
        current.ingestion.errors.labels(stage="embedding").inc(outcome.failed)
        current.ingestion.questions.labels(status="unembedded").inc(outcome.failed)

    if outcome.embedded:
        # Состав и тексты вопросов могли измениться вместе с переиндексацией,
        # поэтому лексический индекс перестраивается следующим запросом.
        get_corpus_index().invalidate()

    if outcome.embedded or outcome.failed:
        logger.info(
            "доиндексация завершена",
            embedded=outcome.embedded,
            failed=outcome.failed,
            remaining=remaining,
        )

    return ReindexRun(
        embedded=outcome.embedded,
        failed=outcome.failed,
        remaining=remaining,
        errors=[issue.message for issue in outcome.issues],
    )


class BackgroundReindexer:
    """Периодический воркер, разбирающий очередь невекторизованных вопросов."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        provider: EmbeddingProvider,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        metrics: Metrics | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        self._interval = interval_seconds
        self._batch_size = batch_size
        self._metrics = metrics
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def is_running(self) -> bool:
        """Запущен ли воркер."""

        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Запустить фоновый цикл, если он ещё не запущен."""

        if self.is_running:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="chgk-reindexer")
        logger.info("доиндексатор запущен", interval_seconds=self._interval)

    async def stop(self) -> None:
        """Остановить фоновый цикл и дождаться его завершения."""

        if self._task is None:
            return
        self._stop.set()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
        logger.info("доиндексатор остановлен")

    async def run_once(
        self,
        *,
        force: bool = False,
    ) -> ReindexRun:
        """Выполнить один проход с собственной сессией."""

        async with self._session_factory() as session:
            run = await reindex_once(
                session,
                self._provider,
                batch_size=self._batch_size,
                force=force,
                metrics=self._metrics,
            )
            await session.commit()
            return run

    async def _run(self) -> None:
        """Цикл периодической доиндексации."""

        while not self._stop.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("проход доиндексации завершился ошибкой", error=str(error))

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)


async def drain_queue(
    session: AsyncSession,
    provider: EmbeddingProvider,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    metrics: Metrics | None = None,
) -> ReindexRun:
    """Привести векторы в порядок за один или несколько проходов.

    Без `force` проходы повторяются, пока есть прогресс: очередь
    отсутствующих и устаревших векторов уменьшается, поэтому цикл
    завершается. При `force=True` выполняется ровно один проход по всем
    вопросам: выборка не зависит от состояния строк, и повторный проход
    пересчитывал бы те же вопросы бесконечно.
    """

    if force:
        return await reindex_once(
            session,
            provider,
            batch_size=batch_size,
            force=True,
            metrics=metrics,
        )

    run = ReindexRun()
    while True:
        current = await reindex_once(
            session,
            provider,
            batch_size=batch_size,
            metrics=metrics,
        )
        run.embedded += current.embedded
        run.failed += current.failed
        run.remaining = current.remaining
        run.errors.extend(current.errors)
        if current.embedded == 0:
            break
    return run
