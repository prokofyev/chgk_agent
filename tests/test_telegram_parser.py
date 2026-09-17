"""Тесты разбора HTML-экспорта Telegram с вопросами."""

from pathlib import Path

import pytest

from chgk_agent.ingestion.telegram_html import TelegramHtmlParser

FIXTURE = Path(__file__).parent / "fixtures" / "telegram_export.html"


@pytest.fixture
def parsed():
    return TelegramHtmlParser().parse(FIXTURE)


def test_supports_telegram_export() -> None:
    parser = TelegramHtmlParser()

    assert parser.supports(FIXTURE) is True
    assert parser.supports(Path("pyproject.toml")) is False


def test_extracts_question_answer_pairs(parsed) -> None:
    assert len(parsed.questions) == 4


def test_extracts_all_fields(parsed) -> None:
    question = parsed.questions[0]

    assert question.source_key == "message2"
    assert question.question_text.startswith("В них хранилась доза порошка")
    assert question.question_text.endswith("остался ли у них ещё этот порошок?")
    assert question.answer_text == "Тарас Бульба"
    assert question.comment == "Спрашивал про порох в пороховницах."
    assert "Гоголь" in (question.sources or "")
    assert question.author == "Александр Прокофьев"


def test_captures_pass_criteria(parsed) -> None:
    question = parsed.questions[1]

    assert question.answer_text == "жираф"
    assert question.pass_criteria == "камелопард"


def test_captures_status_notes_before_question(parsed) -> None:
    assert parsed.questions[0].status_notes == ("❗️отдал Гарановичу❗️",)
    assert parsed.questions[1].status_notes == ()


def test_joins_multiline_fields(parsed) -> None:
    question = parsed.questions[3]

    assert question.question_text == "Многострочный вопрос:\nпервая строка\nвторая строка"
    assert question.answer_text == "многострочный ответ,\nпродолжение ответа"
    assert question.comment == "проверка многострочных полей."
    assert "1. https://example.org/one" in (question.sources or "")


def test_skips_messages_with_media(parsed) -> None:
    assert parsed.skipped_with_media == 1
    assert all("девочка" not in q.question_text for q in parsed.questions)


def test_skips_question_without_answer(parsed) -> None:
    assert parsed.skipped_no_answer == 1
    assert all("нет ответа" not in q.question_text for q in parsed.questions)


def test_total_skipped_counter(parsed) -> None:
    assert parsed.skipped == parsed.skipped_with_media + parsed.skipped_no_answer
    assert parsed.skipped > 0


def test_reports_media_and_answer_issues(parsed) -> None:
    messages = " ".join(issue.message for issue in parsed.issues)

    assert "медиа" in messages
    assert "без ответа" in messages


def test_no_questions_produces_issue(tmp_path: Path) -> None:
    empty = tmp_path / "empty.html"
    empty.write_text(
        '<div class="page_wrap"><div class="history"></div></div>',
        encoding="utf-8",
    )

    result = TelegramHtmlParser().parse(empty)

    assert result.questions == []
    assert result.issues
    assert "не найдено ни одного сообщения" in result.issues[0].message


def test_parses_real_resources_when_present() -> None:
    resources = Path("resources")
    files = sorted(resources.glob("messages*.html"))
    if not files:
        pytest.skip("файлы resources/messages*.html отсутствуют")

    parser = TelegramHtmlParser()
    total_questions = 0
    total_media = 0
    for path in files:
        assert parser.supports(path)
        result = parser.parse(path)
        assert result.questions
        assert len(result.questions) > result.skipped_with_media
        total_questions += len(result.questions)
        total_media += result.skipped_with_media

    assert total_questions > 1000
    assert total_media > 0
