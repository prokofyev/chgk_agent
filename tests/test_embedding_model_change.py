"""Смена модели эмбеддингов: размерность схемы и переиндексация без парсинга."""

import asyncio
import os
import shutil
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chgk_agent.config import Settings
from chgk_agent.db.base import (
    Question,
    QuestionOccurrence,
    Source,
    get_embedding_dim,
)
from chgk_agent.embeddings.reindexer import drain_queue
from chgk_agent.embeddings.validation import (
    EmbeddingDimensionMismatchError,
    verify_schema_embedding_dimension,
)
from chgk_agent.ingestion.service import IngestionService
from chgk_agent.search.local import LocalSearch
from db_setup import DatabaseSetupUnavailable, disposable_database

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "telegram_export.html"
ALTERNATIVE_DIM = 384
DIMENSION = get_embedding_dim()


class _VocabularyEmbedder:
    """Детерминированные эмбеддинги фиксированной размерности."""

    VOCABULARY = ("порошок", "пороховница", "гоголь", "жираф", "бульба")

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

    @property
    def model(self) -> str:
        return "VocabularyEmbeddings"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        lowered = text.casefold()
        vector = [0.01] * self._dimension
        for index, term in enumerate(self.VOCABULARY):
            if index >= self._dimension:
                break
            if term in lowered:
                vector[index] = 1.0
        return vector


def test_migration_uses_configured_embedding_dimension() -> None:
    """Схема создаётся под размерность из конфигурации, а не под зашитую."""

    settings = Settings(_env_file=None)
    shutil.which("psql") or pytest.skip("psql недоступен")

    try:
        with disposable_database(
            settings.database.test_dsn, embedding_dim=ALTERNATIVE_DIM
        ) as plan:
            column = subprocess.run(
                [
                    "psql",
                    "-h",
                    "localhost",
                    "-p",
                    "5432",
                    "-U",
                    os.environ.get("USER", "postgres"),
                    "-d",
                    plan.database,
                    "-tAc",
                    "SELECT format_type(a.atttypid, a.atttypmod) "
                    "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
                    "WHERE c.relname = 'questions' AND a.attname = 'embedding'",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            assert column.stdout.strip() == f"vector({ALTERNATIVE_DIM})"
            created = plan.database
    except DatabaseSetupUnavailable as error:
        pytest.skip(str(error))

    remaining = subprocess.run(
        [
            "psql",
            "-h",
            "localhost",
            "-p",
            "5432",
            "-U",
            os.environ.get("USER", "postgres"),
            "-d",
            "postgres",
            "-tAc",
            f"SELECT count(*) FROM pg_database WHERE datname = '{created}'",
        ],
        capture_output=True,
        text=True,
    )
    assert remaining.stdout.strip() == "0"


@pytest.fixture
async def imported_fixture(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> AsyncIterator[dict[str, object]]:
    """Импортировать фикстуру из временного файла и вернуть контекст."""

    source_file = tmp_path / "telegram_export.html"
    shutil.copyfile(FIXTURE, source_file)
    location = str(source_file)

    await _purge(session_factory, location)

    service = IngestionService(session_factory, embedding_provider=_VocabularyEmbedder())
    report = await service.run_import([source_file])
    assert report.unembedded == 0

    context = {"location": location, "source_file": source_file, "report": report}
    try:
        yield context
    finally:
        await _purge(session_factory, location)


def _belongs_to(location: str):
    """Условие «вопрос входит в указанный источник»."""

    return Question.occurrences.any(
        QuestionOccurrence.source.has(Source.location == location)
    )


async def _purge(
    session_factory: async_sessionmaker[AsyncSession], location: str
) -> None:
    async with session_factory() as session:
        await session.execute(delete(Source).where(Source.location == location))
        await session.execute(delete(Question).where(~Question.occurrences.any()))
        await session.commit()


async def test_reindex_without_source_file_keeps_search_working(
    session_factory: async_sessionmaker[AsyncSession],
    imported_fixture: dict[str, object],
) -> None:
    """Переиндексация работает без повторного разбора HTML."""

    source_file = Path(imported_fixture["source_file"])
    location = str(imported_fixture["location"])

    async with session_factory() as session:
        # Зафиксировать состав вопросов и источников до переиндексации.
        questions_before = (
            await session.execute(
                select(Question.id, Question.question_text).where(_belongs_to(location))
            )
        ).all()
        sources_before = (
            await session.execute(
                select(Source.id, Source.location, Source.content_hash).where(
                    Source.location == location
                )
            )
        ).all()

        # Эмулируем смену модели: сбрасываем векторы только у вопросов этого
        # источника. База разработки может содержать реальные импортированные
        # вопросы, и трогать их тест не должен.
        await session.execute(
            update(Question)
            .where(_belongs_to(location))
            .values(embedding=None, embedding_model=None, embedding_dim=None)
        )
        await session.commit()

    # Удаляем исходный HTML: переиндексация не должна его разбирать.
    await asyncio.to_thread(source_file.unlink)

    async with session_factory() as session:
        run = await drain_queue(session, _VocabularyEmbedder(), location=location)
        await session.commit()

        assert run.embedded > 0
        assert run.remaining == 0
        assert run.failed == 0

        reindexed = (
            await session.execute(
                select(Question.id, Question.question_text).where(_belongs_to(location))
            )
        ).all()
        assert reindexed == questions_before

        sources_after = (
            await session.execute(
                select(Source.id, Source.location, Source.content_hash).where(
                    Source.location == location
                )
            )
        ).all()
        assert sources_after == sources_before

        embedded = (
            await session.execute(
                select(Question.id).where(_belongs_to(location))
            )
        ).all()
        assert len(embedded) == len(questions_before)

        search = LocalSearch(session, _VocabularyEmbedder())
        results = await search.search("порошок в пороховнице", limit=5)

    assert results
    assert any("порош" in result.question_text.casefold() for result in results)


async def test_reindexed_questions_report_current_model_and_dimension(
    session_factory: async_sessionmaker[AsyncSession],
    imported_fixture: dict[str, object],
) -> None:
    """После переиндексации размерность согласована с настройкой."""

    location = str(imported_fixture["location"])

    async with session_factory() as session:
        await session.execute(
            update(Question)
            .where(_belongs_to(location))
            .values(embedding=None, embedding_model=None, embedding_dim=None)
        )
        await session.commit()

    async with session_factory() as session:
        await drain_queue(session, _VocabularyEmbedder(), location=location)
        await session.commit()

        stored = (
            await session.execute(
                select(Question.embedding_dim).where(_belongs_to(location))
            )
        ).scalars().all()
        dimension_in_schema = (
            await session.execute(
                text(
                    "SELECT format_type(a.atttypid, a.atttypmod) "
                    "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
                    "WHERE c.relname = 'questions' AND a.attname = 'embedding'"
                )
            )
        ).scalar_one()

    assert stored
    assert all(dimension == DIMENSION for dimension in stored)
    assert dimension_in_schema == f"vector({DIMENSION})"


async def test_startup_check_matches_real_schema(session_factory) -> None:
    """Проверка при старте сверяется с настоящим типом колонки в базе."""

    settings = Settings(_env_file=None)
    settings.gigachat.embedding_dim = DIMENSION

    actual = await verify_schema_embedding_dimension(session_factory, settings)

    assert actual == DIMENSION


async def test_startup_check_catches_wrong_configured_dimension(session_factory) -> None:
    """Забытая миграция при смене размерности ловится при старте."""

    settings = Settings(_env_file=None)
    settings.gigachat.embedding_dim = DIMENSION + 1

    with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
        await verify_schema_embedding_dimension(session_factory, settings)

    assert f"vector({DIMENSION})" in str(excinfo.value)
