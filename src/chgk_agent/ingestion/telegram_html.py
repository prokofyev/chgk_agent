"""Разбор HTML-экспорта Telegram-канала с вопросами ЧГК.

Ожидаемая разметка — экспорт истории канала: сообщения `div.message.default`,
внутри которых находится блок `.text` с полями «Вопрос», «Ответ», «Зачёт»,
«Комментарий», «Источник» и «Автор», разделёнными переводами строк.
Сообщения с медиа пропускаются.
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from chgk_agent.models.domain import ParsedQuestion, ParseIssue, ParseResult

MEDIA_SELECTORS = (
    ".media_wrap",
    ".photo_wrap",
    ".video_wrap",
    ".voice_message",
    ".audio_file",
    ".video_file",
    ".sticker",
    ".location",
    ".poll",
)

_QUESTION_MARKER = re.compile(r"^Вопрос\s*:?\s*$")
_FIELD_MARKER = re.compile(
    r"^(Ответ|Зачёт|Зачет|Комментарий|Источник\(и\)|Источники|Источник|Автор)\s*:\s*(.*)$"
)
_FIELD_NAMES = {
    "Ответ": "answer",
    "Зачёт": "pass_criteria",
    "Зачет": "pass_criteria",
    "Комментарий": "comment",
    "Источник": "sources",
    "Источники": "sources",
    "Источник(и)": "sources",
    "Автор": "author",
}


def _clean_lines(text: str) -> list[str]:
    """Разбить текст сообщения на строки без хвостовых пробелов."""

    return [line.strip() for line in text.splitlines()]


def _join(lines: list[str]) -> str | None:
    """Склеить непустые строки значения поля."""

    value = "\n".join(line for line in lines if line).strip()
    return value or None


def _message_has_media(message: Tag) -> bool:
    """Содержит ли сообщение вложение."""

    return any(message.select_one(selector) is not None for selector in MEDIA_SELECTORS)


def _parse_message_text(text: str) -> ParsedQuestion | None:
    """Разобрать текст сообщения в вопрос или вернуть None."""

    lines = _clean_lines(text)

    question_index = next(
        (index for index, line in enumerate(lines) if _QUESTION_MARKER.match(line)),
        None,
    )
    if question_index is None:
        return None

    status_notes = tuple(line for line in lines[:question_index] if line)
    fields: dict[str, list[str]] = {"question": []}
    current = "question"

    for line in lines[question_index + 1 :]:
        marker = _FIELD_MARKER.match(line)
        if marker:
            current = _FIELD_NAMES[marker.group(1)]
            fields.setdefault(current, [])
            remainder = marker.group(2).strip()
            if remainder:
                fields[current].append(remainder)
            continue
        fields.setdefault(current, []).append(line)

    question_text = _join(fields.get("question", []))
    answer_text = _join(fields.get("answer", []))
    if not question_text or not answer_text:
        return None

    return ParsedQuestion(
        source_key="",
        question_text=question_text,
        answer_text=answer_text,
        pass_criteria=_join(fields.get("pass_criteria", [])),
        comment=_join(fields.get("comment", [])),
        sources=_join(fields.get("sources", [])),
        author=_join(fields.get("author", [])),
        status_notes=status_notes,
    )


class TelegramHtmlParser:
    """Разборщик HTML-экспорта Telegram-канала."""

    name = "telegram-html"

    def supports(self, path: Path) -> bool:
        """Проверить, похож ли файл на экспорт Telegram."""

        if path.suffix.lower() not in {".html", ".htm"}:
            return False
        try:
            head = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        except OSError:
            return False
        return "class=\"page_wrap\"" in head and "class=\"history\"" in head

    def parse(self, path: Path) -> ParseResult:
        """Разобрать файл экспорта."""

        result = ParseResult(location=str(path))
        document = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(document, "lxml")

        messages = soup.select("div.message.default")
        if not messages:
            result.issues.append(
                ParseIssue(
                    location=str(path),
                    message="не найдено ни одного сообщения с вопросами",
                )
            )
            return result

        skipped_keys: list[str] = []
        no_answer_keys: list[str] = []

        for message in messages:
            message_id = message.get("id") or ""
            if _message_has_media(message):
                result.skipped_with_media += 1
                skipped_keys.append(message_id)
                continue

            text_element = message.select_one(".text")
            if text_element is None:
                result.skipped_not_a_question += 1
                continue

            parsed = _parse_message_text(text_element.get_text("\n"))
            if parsed is None:
                if "Вопрос" in text_element.get_text():
                    result.skipped_no_answer += 1
                    no_answer_keys.append(message_id)
                else:
                    result.skipped_not_a_question += 1
                continue

            result.questions.append(
                ParsedQuestion(
                    source_key=message_id,
                    question_text=parsed.question_text,
                    answer_text=parsed.answer_text,
                    pass_criteria=parsed.pass_criteria,
                    comment=parsed.comment,
                    sources=parsed.sources,
                    author=parsed.author,
                    status_notes=parsed.status_notes,
                )
            )

        result.skipped = (
            result.skipped_with_media + result.skipped_no_answer + result.skipped_not_a_question
        )

        if not result.questions:
            result.issues.append(
                ParseIssue(
                    location=str(path),
                    message="не удалось извлечь ни одной пары «вопрос — ответ»",
                )
            )

        if skipped_keys:
            result.issues.append(
                ParseIssue(
                    location=str(path),
                    message=(
                        f"пропущено сообщений с медиа: {len(skipped_keys)} "
                        f"({', '.join(skipped_keys[:5])}{'…' if len(skipped_keys) > 5 else ''})"
                    ),
                )
            )
        if no_answer_keys:
            result.issues.append(
                ParseIssue(
                    location=str(path),
                    message=(
                        f"пропущено сообщений без ответа: {len(no_answer_keys)} "
                        f"({', '.join(no_answer_keys[:5])}{'…' if len(no_answer_keys) > 5 else ''})"
                    ),
                )
            )

        return result
