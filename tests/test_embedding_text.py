"""Тесты единого правила построения текста для эмбеддингов."""

from chgk_agent.embeddings.text import embedding_document_text, embedding_query_text
from chgk_agent.models.normalization import normalize_text


def test_document_text_combines_question_and_answer() -> None:
    text = embedding_document_text("Кто придумал галстук?", "яхтсмен")

    assert "кто придумал галстук?" in text
    assert "яхтсмен" in text


def test_document_text_is_normalized() -> None:
    text = embedding_document_text("ТЕСТ\u00a0 Тьюринга", "Ответ…  тут")

    assert text == normalize_text("ТЕСТ Тьюринга Ответ... тут")


def test_document_text_without_answer_is_question_only() -> None:
    assert embedding_document_text("Вопрос?", None) == normalize_text("Вопрос?")
    assert embedding_document_text("Вопрос?", "") == normalize_text("Вопрос?")


def test_query_text_is_normalized_like_documents() -> None:
    description = "ТЕСТ\u00a0 Тьюринга"

    assert embedding_query_text(description) == normalize_text(description)


def test_question_and_answer_are_normalized_by_one_rule() -> None:
    """Одна и та же пара даёт одинаковый текст независимо от вызывающей стороны."""

    from_indexer = embedding_document_text("Вопрос про мост?", "река")
    from_search = embedding_document_text("Вопрос про мост?", "река")

    assert from_indexer == from_search


def test_indexer_builds_text_from_question_and_answer() -> None:
    """Векторизация берёт склеенный текст, а не только текст вопроса."""

    from types import SimpleNamespace

    from chgk_agent.embeddings.indexer import embedding_texts

    question = SimpleNamespace(
        question_text="Что ищут на клипсах?",
        answer_text="галстук",
    )

    texts = embedding_texts([question])

    assert texts == [embedding_document_text("Что ищут на клипсах?", "галстук")]
    assert "галстук" in texts[0]


def test_indexer_text_applies_stemming_free_normalization_consistently() -> None:
    """Текст документа не зависит от того, откуда вызвана функция."""

    from types import SimpleNamespace

    from chgk_agent.embeddings.indexer import embedding_texts

    question = SimpleNamespace(question_text="Вопрос?", answer_text="Ответ")

    assert embedding_texts([question])[0] == embedding_document_text("Вопрос?", "Ответ")
