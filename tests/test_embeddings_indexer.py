"""Тесты векторизации очереди вопросов и фонового доиндексатора."""

import pytest
from prometheus_client import CollectorRegistry
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.db import repository
from chgk_agent.db.base import Question, Source, get_embedding_dim
from chgk_agent.embeddings.indexer import embed_pending
from chgk_agent.embeddings.reindexer import (
    BackgroundReindexer,
    drain_queue,
    reindex_once,
)
from chgk_agent.ingestion.importer import QuestionImporter
from chgk_agent.models.domain import ParsedQuestion, ParseResult
from chgk_agent.observability.metrics import Metrics

pytestmark = pytest.mark.integration


class _StubEmbedder:
    """Провайдер эмбеддингов с управляемыми сбоями.

    `fail_on` ломает и батч, и одиночный вызов — так проверяется изоляция
    проблемной записи. `batch_only_fail_on` ломает только батч, то есть
    провайдер, который восстанавливается при разбиении на одиночные вызовы.
    """

    def __init__(
        self,
        *,
        fail_on: set[str] | None = None,
        batch_only_fail_on: set[str] | None = None,
        empty_on: set[str] | None = None,
    ) -> None:
        self._fail_on = fail_on or set()
        self._batch_only_fail_on = batch_only_fail_on or set()
        self._empty_on = empty_on or set()
        self.batch_calls = 0
        self.single_calls = 0

    @property
    def model(self) -> str:
        return "StubEmbeddings"

    @property
    def dimension(self) -> int:
        return get_embedding_dim()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > 1:
            self.batch_calls += 1
            if any(
                needle in text
                for text in texts
                for needle in self._fail_on | self._batch_only_fail_on
            ):
                raise RuntimeError("батч недоступен")
        else:
            self.single_calls += 1
            if any(needle in texts[0] for needle in self._fail_on):
                raise RuntimeError("провайдер недоступен")

        vectors: list[list[float]] = []
        for text in texts:
            if any(needle in text for needle in self._empty_on):
                vectors.append([])
            else:
                vectors.append([float(len(text))] * self.dimension)
        return vectors


LOCATION = "indexer-fixture.html"


def _question(key: str, text: str) -> ParsedQuestion:
    return ParsedQuestion(source_key=key, question_text=text, answer_text="ответ")


def _metrics() -> Metrics:
    return Metrics(CollectorRegistry())


async def _seed(session: AsyncSession, *texts: str) -> None:
    importer = QuestionImporter(session)
    await importer.import_results(
        [
            ParseResult(
                location=LOCATION,
                questions=[_question(str(index), text) for index, text in enumerate(texts)],
            )
        ]
    )


async def test_partial_batch_failure_isolates_broken_questions(
    session: AsyncSession,
) -> None:
    await _seed(session, "Хороший вопрос про эмбеддинги?", "Сломанный вопрос?", "Ещё один хороший?")
    embedder = _StubEmbedder(fail_on={"сломанный"})

    outcome = await embed_pending(session, embedder, location=LOCATION)

    assert outcome.embedded == 2
    assert outcome.failed == 1
    assert embedder.single_calls == 3

    rows = (
        await session.execute(select(Question.normalized_text, Question.embedding))
    ).all()
    by_text = {row.normalized_text: row.embedding for row in rows}
    assert by_text["сломанный вопрос?"] is None
    assert by_text["хороший вопрос про эмбеддинги?"] is not None


async def test_empty_vector_marks_question_as_failed(session: AsyncSession) -> None:
    await _seed(session, "Пустой эмбеддинг?")
    embedder = _StubEmbedder(empty_on={"пустой"})

    outcome = await embed_pending(session, embedder, location=LOCATION)

    assert outcome.embedded == 0
    assert outcome.failed == 1
    assert outcome.issues[0].message.startswith("провайдер вернул пустой")


async def test_no_pending_questions_makes_no_calls(session: AsyncSession) -> None:
    embedder = _StubEmbedder()

    outcome = await embed_pending(session, embedder, location=LOCATION)

    assert outcome.embedded == 0
    assert embedder.batch_calls == 0
    assert embedder.single_calls == 0


async def test_disabled_provider_leaves_questions_in_queue(
    session: AsyncSession,
) -> None:
    await _seed(session, "Вопрос без вектора?")

    assert (
        await repository.count_questions_without_embedding(session, location=LOCATION)
        == 1
    )

    outcome = await embed_pending(session, _StubEmbedder(), location=LOCATION)

    assert outcome.embedded == 1
    assert (
        await repository.count_questions_without_embedding(session, location=LOCATION)
        == 0
    )


async def test_reindex_once_updates_queue_metric(session: AsyncSession) -> None:
    await _seed(session, "Метрика очереди?")
    metrics = _metrics()

    run = await reindex_once(
        session, _StubEmbedder(), location=LOCATION, metrics=metrics
    )

    assert run.embedded == 1
    assert run.remaining == 0
    assert metrics.ingestion.unembedded._value.get() == 0
    assert metrics.ingestion.questions.labels(status="embedded")._value.get() == 1


async def test_reindex_once_records_failures(session: AsyncSession) -> None:
    await _seed(session, "Сломанный вопрос про сбой?")
    metrics = _metrics()

    run = await reindex_once(
        session,
        _StubEmbedder(fail_on={"сбой"}),
        location=LOCATION,
        metrics=metrics,
    )

    assert run.failed == 1
    assert run.remaining == 1
    assert metrics.ingestion.unembedded._value.get() == 1
    assert metrics.ingestion.errors.labels(stage="embedding")._value.get() == 1


async def test_drain_queue_reports_remaining(session: AsyncSession) -> None:
    await _seed(session, "Первый вопрос?", "Сломанный вопрос?")

    run = await drain_queue(
        session, _StubEmbedder(fail_on={"сломанный"}), location=LOCATION
    )

    assert run.embedded == 1
    assert run.remaining == 1


async def _cleanup_location(
    session_factory: async_sessionmaker[AsyncSession], location: str
) -> None:
    """Удалить данные источника, закоммиченные тестом про фоновый воркер."""

    async with session_factory() as cleanup_session:
        await cleanup_session.execute(delete(Source).where(Source.location == location))
        await cleanup_session.execute(delete(Question).where(~Question.occurrences.any()))
        await cleanup_session.commit()


async def test_background_reindexer_processes_queue(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Воркер работает со своей сессией, поэтому данные коммитятся и удаляются явно."""

    location = LOCATION
    await _cleanup_location(session_factory, location)
    try:
        async with session_factory() as seed_session:
            await _seed(seed_session, "Фоновый вопрос?")
            await seed_session.commit()

        reindexer = BackgroundReindexer(
            session_factory, _StubEmbedder(), interval_seconds=3600, metrics=_metrics()
        )
        reindexer.start()
        assert reindexer.is_running

        run = await reindexer.run_once(location=location)
        await reindexer.stop()

        assert run.embedded == 1
        assert run.remaining == 0
        assert not reindexer.is_running
    finally:
        await _cleanup_location(session_factory, location)


async def test_background_reindexer_start_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    reindexer = BackgroundReindexer(
        session_factory, _StubEmbedder(), interval_seconds=3600
    )

    reindexer.start()
    task = reindexer._task
    reindexer.start()

    assert reindexer._task is task
    await reindexer.stop()
    assert not reindexer.is_running
