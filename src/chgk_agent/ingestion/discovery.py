"""Обнаружение файлов-источников и выбор подходящего разборщика."""

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from chgk_agent.ingestion.base import QuestionParser
from chgk_agent.ingestion.telegram_html import TelegramHtmlParser

SUPPORTED_SUFFIXES = frozenset({".html", ".htm"})


def default_parsers() -> list[QuestionParser]:
    """Вернуть разборщики по умолчанию."""

    return [TelegramHtmlParser()]


def discover_files(paths: Iterable[Path]) -> list[Path]:
    """Развернуть список путей в отсортированный список HTML-файлов."""

    found: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.update(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif path.is_file():
            found.add(path)
    return sorted(found)


def select_parser(
    path: Path,
    parsers: Sequence[QuestionParser] | None = None,
) -> QuestionParser | None:
    """Выбрать первый разборщик, поддерживающий файл."""

    for parser in parsers or default_parsers():
        if parser.supports(path):
            return parser
    return None


def parse_all(
    files: Sequence[Path],
    parsers: Sequence[QuestionParser] | None = None,
) -> Iterator[tuple[Path, object]]:
    """Разобрать файлы, возвращая пары «файл — результат разбора»."""

    for path in files:
        parser = select_parser(path, parsers)
        if parser is None:
            continue
        yield path, parser.parse(path)
