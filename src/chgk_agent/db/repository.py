"""Запросы к базе для импорта вопросов и работы с эмбеддингами."""

from collections.abc import Iterable, Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from chgk_agent.db.base import Question, QuestionOccurrence, Source
from chgk_agent.models.domain import CanonicalQuestion, ParsedQuestion


async def upsert_source(
    session: AsyncSession,
    *,
    kind: str,
    location: str,
    content_hash: str | None = None,
    status: str = "imported",
) -> Source:
    """Создать или обновить источник и вернуть его."""

    statement = (
        insert(Source)
        .values(kind=kind, location=location, content_hash=content_hash, status=status)
        .on_conflict_do_update(
            constraint="uq_sources_kind_location",
            set_={
                "content_hash": content_hash,
                "status": status,
                "updated_at": func.now(),
            },
        )
        .returning(Source.id)
    )
    source_id = (await session.execute(statement)).scalar_one()
    await session.flush()
    return await session.get_one(Source, source_id)


async def existing_occurrence_hashes(
    session: AsyncSession,
    source_id: int,
) -> dict[str, str]:
    """Вернуть карту `source_key -> text_hash` для источника."""

    statement = (
        select(QuestionOccurrence.source_key, Question.text_hash)
        .join(Question, Question.id == QuestionOccurrence.question_id)
        .where(QuestionOccurrence.source_id == source_id)
    )
    rows = (await session.execute(statement)).all()
    return {row.source_key: row.text_hash for row in rows}


async def upsert_canonical_question(
    session: AsyncSession,
    canonical: CanonicalQuestion,
) -> int:
    """Создать или найти канонический вопрос и вернуть его идентификатор."""

    statement = (
        insert(Question)
        .values(
            normalized_text=canonical.normalized_text,
            question_text=canonical.question_text,
            answer_text=canonical.answer_text,
            comment=canonical.comment,
            text_hash=canonical.text_hash,
        )
        .on_conflict_do_nothing(constraint="questions_text_hash_key")
        .returning(Question.id)
    )
    inserted = (await session.execute(statement)).scalar_one_or_none()
    if inserted is not None:
        return int(inserted)

    found = await session.execute(
        select(Question.id).where(Question.text_hash == canonical.text_hash)
    )
    return int(found.scalar_one())


async def upsert_occurrence(
    session: AsyncSession,
    *,
    question_id: int,
    source_id: int,
    parsed: ParsedQuestion,
) -> None:
    """Создать или обновить вхождение вопроса в источник."""

    statement = (
        insert(QuestionOccurrence)
        .values(
            question_id=question_id,
            source_id=source_id,
            source_key=parsed.source_key,
            raw_question_text=parsed.question_text,
            raw_answer_text=parsed.answer_text,
            external_url=parsed.external_url,
        )
        .on_conflict_do_update(
            constraint="uq_question_occurrences_source_key",
            set_={
                "question_id": question_id,
                "raw_question_text": parsed.question_text,
                "raw_answer_text": parsed.answer_text,
                "external_url": parsed.external_url,
                "updated_at": func.now(),
            },
        )
    )
    await session.execute(statement)
    await session.flush()


async def delete_stale_occurrences(
    session: AsyncSession,
    *,
    source_id: int,
    keep_keys: Sequence[str],
) -> int:
    """Удалить вхождения источника, исчезнувшие из свежего разбора."""

    statement = delete(QuestionOccurrence).where(
        QuestionOccurrence.source_id == source_id
    )
    if keep_keys:
        statement = statement.where(QuestionOccurrence.source_key.not_in(keep_keys))
    result = await session.execute(statement)
    await session.flush()
    return int(result.rowcount or 0)


async def delete_orphan_questions(session: AsyncSession) -> int:
    """Удалить вопросы, у которых не осталось ни одного вхождения."""

    orphan_ids = (
        select(Question.id)
        .outerjoin(QuestionOccurrence, QuestionOccurrence.question_id == Question.id)
        .where(QuestionOccurrence.id.is_(None))
    )
    result = await session.execute(
        delete(Question).where(Question.id.in_(orphan_ids.scalar_subquery()))
    )
    await session.flush()
    return int(result.rowcount or 0)


def _pending_embedding_statement():
    """Построить выборку вопросов без эмбеддинга."""

    return select(Question).where(Question.embedding.is_(None))


def _stale_embedding_statement(
    *,
    model: str | None = None,
    dimension: int | None = None,
):
    """Построить выборку вопросов для переиндексации.

    Без `model` возвращаются все вопросы: это полный пересчёт, нужный,
    когда векторы формально есть, но их требуется посчитать заново.
    С `model` и `dimension` — только вопросы без вектора и вопросы с
    устаревшим вектором. Критерий устаревания обратен условию
    семантической ветки поиска: там требуется совпадение и модели, и
    размерности, поэтому вектор с верной моделью и неверной
    размерностью выпадал бы из поиска, не попадая в переиндексацию.
    """

    statement = select(Question)
    if model is None:
        return statement
    if dimension is None:
        raise ValueError("Для выборки устаревших векторов нужна размерность")

    return statement.where(
        Question.embedding.is_(None)
        | Question.embedding_model.is_distinct_from(model)
        | Question.embedding_dim.is_distinct_from(dimension)
    )


async def questions_for_reindex(
    session: AsyncSession,
    *,
    model: str | None = None,
    dimension: int | None = None,
    limit: int | None = None,
) -> list[Question]:
    """Вернуть вопросы для переиндексации, включая уже векторизованные.

    Без `model` возвращаются все вопросы, с моделью и размерностью — только
    вопросы без вектора и вопросы с устаревшим вектором.
    """

    statement = _stale_embedding_statement(model=model, dimension=dimension).order_by(
        Question.id
    )
    if limit is not None:
        statement = statement.limit(limit)
    result = await session.execute(statement)
    return list(result.scalars())


async def count_questions_for_reindex(
    session: AsyncSession,
    *,
    model: str | None = None,
    dimension: int | None = None,
) -> int:
    """Сколько вопросов подлежит переиндексации."""

    statement = select(func.count()).select_from(
        _stale_embedding_statement(model=model, dimension=dimension).subquery()
    )
    result = await session.execute(statement)
    return int(result.scalar_one())


async def count_questions_without_embedding(
    session: AsyncSession,
) -> int:
    """Сколько вопросов ожидают векторизации."""

    statement = select(func.count()).select_from(
        _pending_embedding_statement().subquery()
    )
    result = await session.execute(statement)
    return int(result.scalar_one())


async def store_embeddings(
    session: AsyncSession,
    vectors: Iterable[tuple[int, Sequence[float]]],
    *,
    model: str,
    dimension: int,
) -> int:
    """Сохранить эмбеддинги, не затрагивая вопросы без вектора."""

    stored = 0
    for question_id, vector in vectors:
        await session.execute(
            update(Question)
            .where(Question.id == question_id)
            .values(
                embedding=list(vector),
                embedding_model=model,
                embedding_dim=dimension,
                updated_at=func.now(),
            )
        )
        stored += 1
    await session.flush()
    return stored


async def all_question_texts(session: AsyncSession) -> list[tuple[int, str, str]]:
    """Вернуть идентификаторы и тексты всех вопросов для лексического индекса."""

    statement = select(Question.id, Question.question_text, Question.answer_text).order_by(
        Question.id
    )
    rows = (await session.execute(statement)).all()
    return [(int(row.id), row.question_text, row.answer_text) for row in rows]
