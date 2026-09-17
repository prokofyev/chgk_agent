"""Контрактные тесты разбора страниц gotquestions.online на сохранённых фикстурах."""

from pathlib import Path

import pytest

from chgk_agent.external.parser import parse_search_page

FIXTURES = Path(__file__).parent / "fixtures" / "external"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_non_empty_page_returns_matches() -> None:
    page = parse_search_page(_load("search_ok.html"))

    assert page.matches
    assert page.recognized is True
    assert page.has_empty_marker is False


def test_match_contains_question_answer_comment_author_and_link() -> None:
    page = parse_search_page(_load("search_ok.html"))

    match = page.matches[0]

    assert match.question_text
    assert "Назовите" in match.question_text
    assert match.answer_text
    assert match.comment
    assert match.authors
    assert match.external_url == f"https://gotquestions.online/question/{match.external_id}"
    assert match.external_id.isdigit()
    assert match.pack


def test_matches_have_sequential_positions() -> None:
    page = parse_search_page(_load("search_ok.html"))

    assert [match.position for match in page.matches] == list(
        range(1, len(page.matches) + 1)
    )


def test_non_empty_page_reports_total_and_more_pages() -> None:
    page = parse_search_page(_load("search_ok.html"))

    assert page.total == 159
    assert page.has_more is True
    assert page.query == "Тьюринг"


def test_empty_page_is_recognized_as_empty() -> None:
    page = parse_search_page(_load("search_empty.html"))

    assert page.matches == []
    assert page.has_empty_marker is True
    assert page.recognized is True
    assert page.has_results_block is False


def test_rejected_page_has_no_results_and_no_empty_marker() -> None:
    page = parse_search_page(_load("search_rejected.html"))

    assert page.matches == []
    assert page.has_empty_marker is False
    assert page.has_results_block is False
    assert page.has_search_scaffold is True
    assert page.recognized is True


def test_changed_markup_is_not_silently_empty() -> None:
    page = parse_search_page(_load("search_changed_markup.html"))

    assert page.matches == []
    assert page.has_results_block is True
    assert page.total == 159


def test_second_page_is_parsed() -> None:
    page = parse_search_page(_load("search_ok_page2.html"))

    assert page.matches
    assert page.page_size == 5
    assert len(page.matches) == 5


def test_question_page_fixture_exists() -> None:
    html = _load("question_page.html")

    assert "gotquestions" in html
    assert len(html) > 1000


@pytest.mark.parametrize(
    "name",
    ["search_ok.html", "search_ok_page2.html", "search_empty.html", "search_rejected.html"],
)
def test_parser_does_not_raise_on_fixtures(name: str) -> None:
    page = parse_search_page(_load(name))

    assert page is not None
