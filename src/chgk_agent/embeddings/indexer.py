"""Векторизация очереди вопросов без эмбеддинга.

Модуль общий для импорта и фонового доиндексатора: сначала вопросы
векторизуются батчами, а упавший батч разбирается по одному вопросу,
чтобы одна сломанная запись не блокировала остальные.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from chgk_agent.db import repository
from chgk_agent.db.base import Question
from chgk_agent.embeddings.base import EmbeddingProvider
from chgk_agent.logging_setup import get_logger
from chgk_agent.models.domain import ParseIssue

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 16


@dataclass(slots=True)
class EmbeddingOutcome:
    """Результат векторизации очереди вопросов."""

    embedded: int = 0
    failed: int = 0
    issues: list[ParseIssue] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        """Число вопросов, которые остались без эмбеддинга."""

        return self.failed


def _chunk(items: Sequence[Question], size: int) -> list[list[Question]]:
    """Разбить список на части заданного размера."""

    return [list(items[index : index + size]) for index in range(0, len(items), size)]


async def embed_pending(
    session: AsyncSession,
    provider: EmbeddingProvider,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    force: bool = False,
) -> EmbeddingOutcome:
    """Векторизовать вопросы очереди.

    Без `force` обрабатываются вопросы без вектора и вопросы с устаревшим
    вектором: критерий совпадает с условием семантической ветки поиска,
    поэтому после смены модели эмбеддингов достаточно обычного запуска.
    При `force=True` пересчитываются все вопросы подряд — это полный
    пересчёт по явному требованию.
    """

    outcome = EmbeddingOutcome()
    if force:
        pending = await repository.questions_for_reindex(
            session,
            limit=limit,
        )
    else:
        pending = await repository.questions_for_reindex(
            session,
            model=provider.model,
            dimension=provider.dimension,
            limit=limit,
        )
    if not pending:
        return outcome

    size = max(batch_size, 1)
    for batch in _chunk(pending, size):
        await _embed_batch(session, provider, batch, outcome)

    return outcome


async def _embed_batch(
    session: AsyncSession,
    provider: EmbeddingProvider,
    batch: Sequence[Question],
    outcome: EmbeddingOutcome,
) -> None:
    """Векторизовать один батч с изоляцией частичных сбоев."""

    texts = [question.normalized_text for question in batch]
    try:
        vectors = await provider.embed(texts)
    except Exception as error:
        logger.warning(
            "батч эмбеддингов упал целиком, разбираем по одному вопросу",
            batch_size=len(batch),
            error=str(error),
        )
        await _embed_individually(session, provider, batch, outcome)
        return

    if len(vectors) != len(batch):
        logger.warning(
            "провайдер вернул неверное число эмбеддингов",
            expected=len(batch),
            received=len(vectors),
        )
        await _embed_individually(session, provider, batch, outcome)
        return

    stored: list[tuple[int, list[float]]] = []
    for question, vector in zip(batch, vectors, strict=True):
        if not vector:
            _mark_failed(question, outcome, "провайдер вернул пустой эмбеддинг")
            continue
        stored.append((question.id, list(vector)))

    if stored:
        await repository.store_embeddings(
            session,
            stored,
            model=provider.model,
            dimension=provider.dimension,
        )
        outcome.embedded += len(stored)


async def _embed_individually(
    session: AsyncSession,
    provider: EmbeddingProvider,
    batch: Sequence[Question],
    outcome: EmbeddingOutcome,
) -> None:
    """Векторизовать вопросы по одному, изолируя проблемные записи."""

    for question in batch:
        try:
            vectors = await provider.embed([question.normalized_text])
        except Exception as error:
            _mark_failed(question, outcome, str(error))
            continue

        if not vectors or not vectors[0]:
            _mark_failed(question, outcome, "провайдер вернул пустой эмбеддинг")
            continue

        await repository.store_embeddings(
            session,
            [(question.id, list(vectors[0]))],
            model=provider.model,
            dimension=provider.dimension,
        )
        outcome.embedded += 1
        await asyncio.sleep(0)


def _mark_failed(
    question: Question, outcome: EmbeddingOutcome, message: str
) -> None:
    """Зафиксировать невекторизованный вопрос в отчёте и логах."""

    outcome.failed += 1
    outcome.issues.append(
        ParseIssue(location=f"question:{question.id}", message=message)
    )
    logger.warning(
        "не удалось векторизовать вопрос",
        question_id=question.id,
        error=message,
    )
