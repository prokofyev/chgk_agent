"""Интеграционные тесты схемы и сессии."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from chgk_agent.config import Settings
from chgk_agent.db.base import Question, Source

pytestmark = pytest.mark.integration


async def test_session_executes_query_and_closes(engine: AsyncEngine) -> None:
    from chgk_agent.db.session import create_session_factory, session_scope

    factory = create_session_factory(engine)

    async with session_scope(factory) as scope_session:
        result = await scope_session.execute(text("SELECT 1"))
        assert result.scalar_one() == 1

    async with factory() as session:
        assert isinstance(session, AsyncSession)
        result = await session.execute(text("SELECT 1"))
        assert result.scalar_one() == 1


async def test_session_scope_rolls_back_on_error(
    engine: AsyncEngine, session: AsyncSession
) -> None:
    from chgk_agent.db.session import create_session_factory, session_scope

    factory = create_session_factory(engine)
    location = "rollback-check.html"

    with pytest.raises(RuntimeError):
        async with session_scope(factory) as scope_session:
            scope_session.add(Source(kind="html", location=location))
            await scope_session.flush()
            raise RuntimeError("сбой во время импорта")

    remaining = await session.execute(
        text("SELECT count(*) FROM sources WHERE location = :location"),
        {"location": location},
    )
    assert remaining.scalar_one() == 0


async def test_pgvector_extension_is_installed(session: AsyncSession) -> None:
    result = await session.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    )
    assert result.scalar_one()


async def test_vector_column_dimension_matches_config(session: AsyncSession) -> None:
    dim = Settings(_env_file=None).gigachat.embedding_dim
    result = await session.execute(
        text(
            "SELECT format_type(a.atttypid, a.atttypmod) "
            "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.relname = 'questions' AND a.attname = 'embedding'"
        )
    )
    assert result.scalar_one() == f"vector({dim})"


async def test_full_text_and_hnsw_indexes_exist(session: AsyncSession) -> None:
    result = await session.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'questions'")
    )
    indexes = {row.indexname: row.indexdef for row in result}

    assert "ix_questions_normalized_text_fts" in indexes
    assert "USING gin" in indexes["ix_questions_normalized_text_fts"]
    assert "russian" in indexes["ix_questions_normalized_text_fts"]
    assert "ix_questions_embedding_hnsw" in indexes
    assert "hnsw" in indexes["ix_questions_embedding_hnsw"]
    assert "vector_cosine_ops" in indexes["ix_questions_embedding_hnsw"]


async def test_full_text_search_query_uses_gin_index(session: AsyncSession) -> None:
    await session.execute(text("SET enable_seqscan = off"))
    result = await session.execute(
        text(
            "EXPLAIN SELECT id FROM questions "
            "WHERE to_tsvector('russian', normalized_text) @@ plainto_tsquery('russian', :q)"
        ),
        {"q": "космонавт"},
    )
    plan = "\n".join(row[0] for row in result)

    assert "ix_questions_normalized_text_fts" in plan


async def test_cascade_delete_removes_occurrences(session: AsyncSession) -> None:
    source = Source(kind="html", location="fixture-1.html")
    question = Question(
        normalized_text="вопрос",
        question_text="Вопрос?",
        answer_text="Ответ",
        text_hash="hash-1",
    )
    session.add_all([source, question])
    await session.flush()

    from chgk_agent.db.base import QuestionOccurrence

    occurrence = QuestionOccurrence(
        question_id=question.id,
        source_id=source.id,
        source_key="1",
        raw_question_text="Вопрос?",
        raw_answer_text="Ответ",
    )
    session.add(occurrence)
    await session.flush()

    await session.delete(question)
    await session.flush()

    remaining = await session.execute(
        text("SELECT count(*) FROM question_occurrences WHERE id = :id"),
        {"id": occurrence.id},
    )
    assert remaining.scalar_one() == 0
