"""Тесты нормализации текста вопросов."""

from chgk_agent.models.domain import CanonicalQuestion, ImportReport, ParsedQuestion
from chgk_agent.models.normalization import normalize_text, text_hash


def test_normalization_is_idempotent() -> None:
    source = "  Вопрос\u00a0 «Тьюринг»\u2014 тест\u2026  "

    once = normalize_text(source)

    assert normalize_text(once) == once


def test_normalization_unifies_case_and_whitespace() -> None:
    assert normalize_text("ТЕСТ   Тьюринга") == "тест тьюринга"
    assert normalize_text("Тест\n\tТьюринга") == "тест тьюринга"


def test_normalization_unifies_typography() -> None:
    assert normalize_text("«кавычки» — тире… многоточие") == normalize_text(
        "\"кавычки\" - тире... многоточие"
    )


def test_normalization_removes_space_before_punctuation() -> None:
    assert normalize_text("Кто ? Он , она .") == "кто? он, она."


def test_normalization_of_empty_text() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_text_hash_ignores_case_and_whitespace() -> None:
    assert text_hash("ТЕСТ  Тьюринга") == text_hash("тест тьюринга ")


def test_text_hash_differs_for_different_texts() -> None:
    assert text_hash("первый вопрос") != text_hash("второй вопрос")


def test_canonical_question_is_built_from_parsed() -> None:
    parsed = ParsedQuestion(
        source_key="3",
        question_text="Кто предложил тест?",
        answer_text="Тьюринг",
        comment="известный тест",
    )

    canonical = CanonicalQuestion.from_parsed(parsed)

    assert canonical.normalized_text == normalize_text(parsed.question_text)
    assert canonical.text_hash == text_hash(parsed.question_text)
    assert canonical.question_text == parsed.question_text
    assert canonical.answer_text == parsed.answer_text
    assert canonical.comment == parsed.comment


def test_import_report_marks_partial_results() -> None:
    report = ImportReport(operation_id="op-1")
    assert report.is_partial is False

    report.skipped = 1
    assert report.is_partial is True

    clean = ImportReport(operation_id="op-2", unembedded=2)
    assert clean.is_partial is True
