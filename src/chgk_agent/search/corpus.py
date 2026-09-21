"""Сборка лексического индекса локального корпуса из базы.

Индекс нужен и локальной ветке поиска, и подготовке короткого запроса к
внешнему источнику, поэтому построение живёт в одном месте: два владельца
индекса разошлись бы в правилах сборки, а сравнивать оценки стало бы нельзя.
Кэш (`CorpusIndex`) общий и переживает запросы; инвалидация происходит после
импорта и переиндексации.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from chgk_agent.db import repository
from chgk_agent.embeddings.text import embedding_document_text
from chgk_agent.search.lexical import Bm25Index, CorpusIndex, get_corpus_index, tokenize

LOCAL_KEY_PREFIX = "local:"


def local_key(question_id: int) -> str:
    """Ключ вопроса в лексическом индексе."""

    return f"{LOCAL_KEY_PREFIX}{question_id}"


async def ensure_corpus_index(
    session: AsyncSession,
    *,
    cache: CorpusIndex | None = None,
) -> Bm25Index:
    """Вернуть готовый индекс корпуса, построив его при необходимости."""

    corpus = cache if cache is not None else get_corpus_index()
    current = corpus.index
    if current is not None:
        return current

    documents = await repository.all_question_texts(session)
    return corpus.build(
        (
            local_key(question_id),
            embedding_document_text(question_text, answer_text),
        )
        for question_id, question_text, answer_text in documents
    )


def term_informativeness(index: Bm25Index, description: str) -> dict[str, float]:
    """Посчитать информативность основ описания по локальному корпусу.

    Ключ — основа слова: описание токенизируется тем же стеммером, что и
    корпус, поэтому «галстуки» и «галстук» получают одну величину.
    """

    return {stem: index.idf(stem) for stem in dict.fromkeys(tokenize(description))}


__all__ = [
    "LOCAL_KEY_PREFIX",
    "ensure_corpus_index",
    "local_key",
    "term_informativeness",
]
