"""Единое правило построения текста для эмбеддингов.

Вектор вопроса считается по тексту «вопрос + ответ», а не только по
вопросу: искомое слово часто стоит в ответе, поэтому близость описания к
ответу важна так же, как к вопросу. Правило применяется одинаково к
сохраняемым вопросам и к описанию при поиске — иначе близость описания и
документа считалась бы по разным текстам.
"""

from chgk_agent.models.normalization import normalize_text

ANSWER_SEPARATOR = "\n"


def embedding_document_text(question_text: str, answer_text: str | None) -> str:
    """Собрать и нормализовать текст вопроса и ответа для эмбеддинга."""

    parts = [question_text or ""]
    if answer_text:
        parts.append(answer_text)
    return normalize_text(ANSWER_SEPARATOR.join(parts))


def embedding_query_text(description: str) -> str:
    """Собрать и нормализовать текст описания для эмбеддинга."""

    return normalize_text(description)


__all__ = ["embedding_document_text", "embedding_query_text"]
