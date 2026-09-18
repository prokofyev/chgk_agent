"""Тесты векторизации очереди вопросов и фонового доиндексатора."""

import pytest
from prometheus_client import CollectorRegistry
from sqlalchemy import delete, select, update
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
from chgk_agent.search.local import LocalSearch

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
        model: str = "StubEmbeddings",
        fail_on: set[str] | None = None,
        batch_only_fail_on: set[str] | None = None,
        empty_on: set[str] | None = None,
        max_single_calls: int | None = None,
    ) -> None:
        self._model = model
        self._fail_on = fail_on or set()
        self._batch_only_fail_on = batch_only_fail_on or set()
        self._empty_on = empty_on or set()
        self._max_single_calls = max_single_calls
        self.batch_calls = 0
        self.single_calls = 0

    @property
    def model(self) -> str:
        return self._model

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
            if (
                self._max_single_calls is not None
                and self.single_calls > self._max_single_calls
            ):
                raise RuntimeError("проход переиндексации повторился")
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

    outcome = await embed_pending(session, embedder)

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

    outcome = await embed_pending(session, embedder)

    assert outcome.embedded == 0
    assert outcome.failed == 1
    assert outcome.issues[0].message.startswith("провайдер вернул пустой")


async def test_no_pending_questions_makes_no_calls(session: AsyncSession) -> None:
    embedder = _StubEmbedder()

    outcome = await embed_pending(session, embedder)

    assert outcome.embedded == 0
    assert embedder.batch_calls == 0
    assert embedder.single_calls == 0


async def test_disabled_provider_leaves_questions_in_queue(
    session: AsyncSession,
) -> None:
    await _seed(session, "Вопрос без вектора?")

    assert (
        await repository.count_questions_without_embedding(session)
        == 1
    )

    outcome = await embed_pending(session, _StubEmbedder())

    assert outcome.embedded == 1
    assert (
        await repository.count_questions_without_embedding(session)
        == 0
    )


async def test_reindex_once_updates_queue_metric(session: AsyncSession) -> None:
    await _seed(session, "Метрика очереди?")
    metrics = _metrics()

    run = await reindex_once(
        session, _StubEmbedder(), metrics=metrics
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
        metrics=metrics,
    )

    assert run.failed == 1
    assert run.remaining == 1
    assert metrics.ingestion.unembedded._value.get() == 1
    assert metrics.ingestion.errors.labels(stage="embedding")._value.get() == 1


async def test_drain_queue_reports_remaining(session: AsyncSession) -> None:
    await _seed(session, "Первый вопрос?", "Сломанный вопрос?")

    run = await drain_queue(session, _StubEmbedder(fail_on={"сломанный"}))

    assert run.embedded == 1
    assert run.remaining == 1


async def test_default_run_skips_vectors_of_current_model_and_dimension(
    session: AsyncSession,
) -> None:
    """Обычный запуск не трогает векторы текущей модели и размерности."""

    await _seed(session, "Актуальный вопрос?")
    embedder = _StubEmbedder()
    await drain_queue(session, embedder)
    assert embedder.single_calls == 1

    repeat = _StubEmbedder()
    run = await drain_queue(session, repeat)

    assert run.embedded == 0
    assert repeat.batch_calls == 0
    assert repeat.single_calls == 0


async def test_default_run_rewrites_vectors_of_other_model(
    session: AsyncSession,
) -> None:
    """Обычный запуск исправляет векторы, посчитанные другой моделью."""

    await _seed(session, "Вопрос прежней модели?")
    await drain_queue(session, _StubEmbedder(model="OldEmbeddings"))

    run = await drain_queue(session, _StubEmbedder(model="StubEmbeddings"))

    assert run.embedded == 1
    assert run.remaining == 0
    models = (
        await session.execute(select(Question.embedding_model))
    ).scalars().all()
    assert models == ["StubEmbeddings"]


async def test_question_with_wrong_dimension_is_treated_as_stale(
    session: AsyncSession,
) -> None:
    """Вектор с верной моделью и неверной размерностью подлежит переиндексации."""

    await _seed(session, "Вопрос с чужой размерностью?")
    await drain_queue(session, _StubEmbedder())
    await session.execute(
        update(Question).values(embedding_dim=get_embedding_dim() + 1)
    )

    run = await drain_queue(session, _StubEmbedder())

    assert run.embedded == 1
    stored = (
        await session.execute(select(Question.embedding_dim))
    ).scalars().all()
    assert stored == [get_embedding_dim()]


async def test_force_reindex_processes_all_questions_in_single_pass(
    session: AsyncSession,
) -> None:
    """Принудительный пересчёт обрабатывает все вопросы ровно за один проход."""

    await _seed(session, "Первый актуальный?", "Второй актуальный?")
    await drain_queue(session, _StubEmbedder())

    embedder = _StubEmbedder(max_single_calls=1)
    run = await drain_queue(session, embedder, force=True)

    assert run.embedded == 2
    assert run.remaining == 0
    assert embedder.batch_calls == 1


async def test_force_reindex_reports_failed_questions_as_remaining(
    session: AsyncSession,
) -> None:
    """В принудительном режиме остаток равен числу неуспешных за проход."""

    await _seed(session, "Сломанный актуальный?")
    await drain_queue(session, _StubEmbedder())

    run = await drain_queue(session, _StubEmbedder(fail_on={"сломанный"}), force=True)

    assert run.embedded == 0
    assert run.failed == 1
    assert run.remaining == 1


async def test_search_sees_question_only_after_reindex(
    session: AsyncSession,
) -> None:
    """Критерий переиндексации согласован с условиями семантической ветки."""

    await _seed(session, "Тьюринг и его машина?")
    await drain_queue(session, _StubEmbedder(model="OldEmbeddings"))

    search = LocalSearch(session, _StubEmbedder(), use_lexical=False)
    before = await search.search("Тьюринг", limit=5)
    assert before == []

    await drain_queue(session, _StubEmbedder())

    after = await search.search("Тьюринг", limit=5)
    assert [result.question_text for result in after] == ["Тьюринг и его машина?"]


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

        run = await reindexer.run_once()
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
