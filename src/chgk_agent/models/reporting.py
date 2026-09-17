"""Формирование отчёта об импорте по результатам разбора."""

from collections import Counter
from datetime import UTC, datetime

from chgk_agent.models.domain import ImportReport, ParseIssue, ParseResult


def build_parse_report(
    operation_id: str,
    results: list[ParseResult],
    *,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> ImportReport:
    """Собрать черновой отчёт об импорте по результатам разбора.

    Счётчики `added`/`updated`/`unchanged`/`unembedded` заполняются позже,
    когда результаты разбора попадут в базу.
    """

    report = ImportReport(operation_id=operation_id)
    report.locations = [result.location for result in results]
    report.started_at = started_at or datetime.now(UTC)
    report.finished_at = finished_at or datetime.now(UTC)
    report.processed = sum(len(result.questions) for result in results)
    report.skipped = sum(result.skipped for result in results)

    for result in results:
        report.issues.extend(result.issues)

    return report


def summarizing_issue(results: list[ParseResult]) -> ParseIssue | None:
    """Сформировать сводку по пропущенным сообщениям."""

    counters: Counter[str] = Counter()
    for result in results:
        if result.skipped_with_media:
            counters["with_media"] += result.skipped_with_media
        if result.skipped_no_answer:
            counters["no_answer"] += result.skipped_no_answer
        if result.skipped_not_a_question:
            counters["not_a_question"] += result.skipped_not_a_question

    if not counters:
        return None

    parts = []
    labels = {
        "with_media": "с медиа",
        "no_answer": "без ответа",
        "not_a_question": "без пары «вопрос — ответ»",
    }
    for key, value in counters.most_common():
        parts.append(f"{labels[key]}: {value}")

    return ParseIssue(location="итого", message="пропущено сообщений — " + ", ".join(parts))
