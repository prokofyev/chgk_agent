"""Интеграционные тесты импорта вопросов в базу."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.db.base import (
    Question,
    QuestionOccurrence,
    Source,
    get_embedding_dim,
)
from chgk_agent.ingestion.importer import QuestionImporter, file_content_hash
from chgk_agent.models.domain import ParsedQuestion, ParseResult

pytestmark = pytest.mark.integration

TEST_LOCATIONS = ("file-1.html", "file-2.html")


@pytest.fixture(autouse=True)
async def isolated_sources(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    """Изолировать данные теста от уже импортированных в базу.

    База разработки может содержать реальные вопросы, поэтому тесты
    работают только со своими источниками и считают строки по ним.
    """

    async def purge() -> None:
        async with session_factory() as session:
            await session.execute(
                delete(Source).where(Source.location.in_(TEST_LOCATIONS))
            )
            await session.execute(delete(Question).where(~Question.occurrences.any()))
            await session.commit()

    await purge()
    try:
        yield
    finally:
        await purge()


def _questions_at(session: AsyncSession, *locations: str):
    """Выборка вопросов, входящих в указанные источники."""

    return (
        select(func.count(func.distinct(Question.id)))
        .select_from(Question)
        .join(QuestionOccurrence, QuestionOccurrence.question_id == Question.id)
        .join(Source, Source.id == QuestionOccurrence.source_id)
        .where(Source.location.in_(locations))
    )


def _occurrences_at(session: AsyncSession, *locations: str):
    """Выборка вхождений в указанные источники."""

    return (
        select(func.count())
        .select_from(QuestionOccurrence)
        .join(Source, Source.id == QuestionOccurrence.source_id)
        .where(Source.location.in_(locations))
    )


class _FakeEmbedder:
    """Детерминированный провайдер эмбеддингов."""

    def __init__(
        self,
        *,
        dimension: int | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self._dimension = dimension if dimension is not None else get_embedding_dim()
        self._fail_on = fail_on or set()
        self.calls = 0

    @property
    def model(self) -> str:
        return "FakeEmbeddings"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if any(needle in text for text in texts for needle in self._fail_on):
            raise RuntimeError("провайдер недоступен")
        return [[float(len(text))] * self._dimension for text in texts]


def _result(location: str, *questions: ParsedQuestion) -> ParseResult:
    return ParseResult(location=location, questions=list(questions))


def _question(key: str, question: str, answer: str = "ответ") -> ParsedQuestion:
    return ParsedQuestion(source_key=key, question_text=question, answer_text=answer)


async def test_counts_questions_without_embedding(session: AsyncSession) -> None:
    from chgk_agent.db import repository

    assert (
        await repository.count_questions_without_embedding(
            session, location="file-1.html"
        )
        == 0
    )


async def test_first_import_adds_questions(session: AsyncSession) -> None:
    importer = QuestionImporter(session)

    report = await importer.import_results(
        [_result("file-1.html", _question("1", "Первый вопрос?"))]
    )

    assert report.added == 1
    assert report.updated == 0
    assert report.unchanged == 0
    assert report.processed == 1

    count = await session.execute(_questions_at(session, "file-1.html"))
    assert count.scalar_one() == 1


async def test_repeat_import_is_idempotent(session: AsyncSession) -> None:
    importer = QuestionImporter(session)
    payload = [_result("file-1.html", _question("1", "Один и тот же вопрос?"))]

    await importer.import_results(payload)
    report = await importer.import_results(payload)

    assert report.unchanged == 1
    assert report.added == 0

    count = await session.execute(_questions_at(session, "file-1.html"))
    assert count.scalar_one() == 1


async def test_changed_question_updates_occurrence(session: AsyncSession) -> None:
    importer = QuestionImporter(session)
    await importer.import_results(
        [_result("file-1.html", _question("1", "Исходная формулировка?"))]
    )

    report = await importer.import_results(
        [_result("file-1.html", _question("1", "Изменённая формулировка?"))]
    )

    assert report.updated == 1
    assert report.added == 0

    occurrences = await session.execute(_occurrences_at(session, "file-1.html"))
    assert occurrences.scalar_one() == 1

    questions = await session.execute(_questions_at(session, "file-1.html"))
    assert questions.scalar_one() == 1


async def test_same_question_from_two_sources(session: AsyncSession) -> None:
    importer = QuestionImporter(session)

    await importer.import_results([_result("file-1.html", _question("1", "Общий вопрос?"))])
    report = await importer.import_results(
        [_result("file-2.html", _question("1", "Общий вопрос?"))]
    )

    assert report.added == 1

    questions = await session.execute(
        _questions_at(session, "file-1.html", "file-2.html")
    )
    assert questions.scalar_one() == 1

    occurrences = await session.execute(
        _occurrences_at(session, "file-1.html", "file-2.html")
    )
    assert occurrences.scalar_one() == 2


async def test_removed_question_is_deleted_as_orphan(session: AsyncSession) -> None:
    importer = QuestionImporter(session)
    await importer.import_results(
        [
            _result(
                "file-1.html",
                _question("1", "Первый вопрос?"),
                _question("2", "Второй вопрос?"),
            )
        ]
    )

    await importer.import_results([_result("file-1.html", _question("1", "Первый вопрос?"))])

    questions = await session.execute(_questions_at(session, "file-1.html"))
    assert questions.scalar_one() == 1


async def test_embeddings_are_stored(session: AsyncSession) -> None:
    importer = QuestionImporter(session, _FakeEmbedder())

    report = await importer.import_results(
        [_result("file-1.html", _question("1", "Вопрос для векторизации?"))]
    )

    assert report.unembedded == 0
    row = (
        await session.execute(
            select(Question.embedding_model, Question.embedding_dim, Question.embedding)
            .join(QuestionOccurrence, QuestionOccurrence.question_id == Question.id)
            .join(Source, Source.id == QuestionOccurrence.source_id)
            .where(Source.location == "file-1.html")
        )
    ).one()
    assert row.embedding_model == "FakeEmbeddings"
    assert row.embedding_dim == get_embedding_dim()
    assert row.embedding is not None


async def test_failed_embedding_is_reported_without_partial_vector(
    session: AsyncSession,
) -> None:
    embedder = _FakeEmbedder(fail_on={"сломанный"})
    importer = QuestionImporter(session, embedder)

    report = await importer.import_results(
        [
            _result(
                "file-1.html",
                _question("1", "Хороший вопрос?"),
                _question("2", "Сломанный вопрос?"),
            )
        ]
    )

    assert report.unembedded == 1
    assert report.is_partial is True

    rows = (
        await session.execute(select(Question.normalized_text, Question.embedding))
    ).all()
    by_text = {row.normalized_text: row.embedding for row in rows}
    assert by_text["хороший вопрос?"] is not None
    assert by_text["сломанный вопрос?"] is None


async def test_pending_questions_are_picked_up_later(session: AsyncSession) -> None:
    importer = QuestionImporter(session)
    await importer.import_results([_result("file-1.html", _question("1", "Без вектора?"))])

    from chgk_agent.db import repository

    assert await repository.count_questions_without_embedding(session) == 1

    importer_with_embeddings = QuestionImporter(session, _FakeEmbedder())
    report = await importer_with_embeddings.import_results([])

    assert report.unembedded == 0
    assert await repository.count_questions_without_embedding(session) == 0


async def test_file_content_hash_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "sample.html"
    target.write_text("<html>стабильно</html>", encoding="utf-8")

    original = file_content_hash(target)
    assert original == file_content_hash(target)

    target.write_text("<html>изменилось</html>", encoding="utf-8")
    changed = file_content_hash(target)

    assert changed != original
    assert changed == file_content_hash(target)


async def test_import_report_aggregates_skips(session: AsyncSession) -> None:
    result = ParseResult(location="file-1.html", questions=[], skipped=3)
    importer = QuestionImporter(session)

    report = await importer.import_results([result])

    assert report.skipped == 3
    assert report.is_partial is True
