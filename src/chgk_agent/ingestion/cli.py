"""CLI разбора и импорта HTML-источников вопросов."""

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from chgk_agent.ingestion.discovery import discover_files, parse_all
from chgk_agent.models.reporting import build_parse_report, summarizing_issue


def build_parser() -> argparse.ArgumentParser:
    """Собрать разборщик аргументов командной строки."""

    parser = argparse.ArgumentParser(
        prog="chgk-agent-parse",
        description="Разобрать HTML-файлы с вопросами ЧГК и показать отчёт.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="HTML-файлы или каталоги с вопросами",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Вывести отчёт в формате JSON",
    )
    parser.add_argument(
        "--show-questions",
        type=int,
        default=0,
        metavar="N",
        help="Показать первые N разобранных вопросов",
    )
    return parser


def run(args: argparse.Namespace, *, out: object = None) -> int:
    """Выполнить разбор и напечатать отчёт."""

    out = out if out is not None else sys.stdout

    files = discover_files(args.paths)
    if not files:
        print("Не найдено ни одного HTML-файла по указанным путям.", file=sys.stderr)
        return 1

    results = [result for _, result in parse_all(files)]
    report = build_parse_report(uuid.uuid4().hex, results)
    summary = summarizing_issue(results)
    if summary is not None:
        report.issues.append(summary)

    if args.as_json:
        payload = {
            "operation_id": report.operation_id,
            "locations": report.locations,
            "processed": report.processed,
            "skipped": report.skipped,
            "issues": [asdict(issue) for issue in report.issues],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=out)
    else:
        print(f"Операция: {report.operation_id}", file=out)
        print(f"Файлов: {len(results)}", file=out)
        print(f"Разобрано вопросов: {report.processed}", file=out)
        print(f"Пропущено: {report.skipped}", file=out)
        for issue in report.issues:
            print(f"  - {issue.location}: {issue.message}", file=out)

    if args.show_questions:
        shown = 0
        for result in results:
            for question in result.questions:
                if shown >= args.show_questions:
                    break
                print(
                    f"\n[{question.source_key}] {question.question_text[:160]}\n"
                    f"Ответ: {question.answer_text[:120]}",
                    file=out,
                )
                shown += 1
            if shown >= args.show_questions:
                break

    return 0 if report.processed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI."""

    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
