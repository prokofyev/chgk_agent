"""Интерфейсы разбора источников вопросов."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from chgk_agent.models.domain import ParseResult


@runtime_checkable
class QuestionParser(Protocol):
    """Разборщик источников вопросов."""

    def supports(self, path: Path) -> bool:
        """Умеет ли разборщик работать с этим файлом."""

    def parse(self, path: Path) -> ParseResult:
        """Разобрать файл и вернуть результат."""
