"""Тесты CLI разбора HTML-источников."""

import io
import json
from pathlib import Path

from chgk_agent.ingestion.cli import build_parser, run
from chgk_agent.ingestion.discovery import discover_files, select_parser

FIXTURE = Path(__file__).parent / "fixtures" / "telegram_export.html"


def test_discover_files_expands_directories(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "one.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "two.htm").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")

    found = discover_files([tmp_path])

    assert [path.name for path in found] == ["one.html", "two.htm"]


def test_discover_files_keeps_explicit_files(tmp_path: Path) -> None:
    target = tmp_path / "single.html"
    target.write_text("<html></html>", encoding="utf-8")

    assert discover_files([target]) == [target]


def test_select_parser_returns_none_for_unknown_file(tmp_path: Path) -> None:
    unknown = tmp_path / "data.txt"
    unknown.write_text("plain", encoding="utf-8")

    assert select_parser(unknown) is None


def test_cli_reports_counts() -> None:
    args = build_parser().parse_args([str(FIXTURE)])
    out = io.StringIO()

    exit_code = run(args, out=out)

    text = out.getvalue()
    assert exit_code == 0
    assert "Разобрано вопросов: 4" in text
    assert "Пропущено: 2" in text


def test_cli_json_output_is_valid() -> None:
    args = build_parser().parse_args([str(FIXTURE), "--json"])
    out = io.StringIO()

    run(args, out=out)

    payload = json.loads(out.getvalue())
    assert payload["processed"] == 4
    assert payload["skipped"] == 2
    assert payload["locations"] == [str(FIXTURE)]
    assert any("медиа" in issue["message"] for issue in payload["issues"])


def test_cli_shows_questions() -> None:
    args = build_parser().parse_args([str(FIXTURE), "--show-questions", "2"])
    out = io.StringIO()

    run(args, out=out)

    text = out.getvalue()
    assert "[message2]" in text
    assert "Ответ: Тарас Бульба" in text
    assert "[message4]" not in text


def test_cli_fails_without_matching_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    args = build_parser().parse_args([str(empty)])

    exit_code = run(args, out=io.StringIO())

    assert exit_code == 1
