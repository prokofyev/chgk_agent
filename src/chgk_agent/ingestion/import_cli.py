"""CLI импорта HTML-источников в базу с векторизацией."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from chgk_agent.config import get_settings
from chgk_agent.db.session import create_engine, create_session_factory
from chgk_agent.embeddings.gigachat import GigaChatEmbeddingProvider
from chgk_agent.embeddings.validation import verify_embedding_dimension
from chgk_agent.ingestion.service import IngestionService
from chgk_agent.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Собрать разборщик аргументов командной строки."""

    parser = argparse.ArgumentParser(
        prog="chgk-import",
        description="Импортировать HTML-файлы с вопросами ЧГК в базу.",
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
        "--skip-embeddings",
        action="store_true",
        help="Импортировать без обращения к GigaChat",
    )
    parser.add_argument(
        "--operation-id",
        default=None,
        help="Идентификатор операции (по умолчанию генерируется)",
    )
    return parser


async def _run_import(args: argparse.Namespace):
    """Выполнить импорт и вернуть отчёт."""

    settings = get_settings()
    configure_logging(
        json_logs=settings.observability.json_logs,
        level=settings.observability.log_level,
    )

    engine = create_engine(settings)
    provider = None
    if not args.skip_embeddings:
        provider = GigaChatEmbeddingProvider(settings.gigachat)
        verify_embedding_dimension(provider, settings)

    try:
        service = IngestionService(
            create_session_factory(engine),
            embedding_provider=provider,
        )
        return await service.run_import(args.paths, operation_id=args.operation_id)
    finally:
        await engine.dispose()


def run(args: argparse.Namespace, *, out: object = None) -> int:
    """Выполнить импорт и напечатать отчёт."""

    out = out if out is not None else sys.stdout

    report = asyncio.run(_run_import(args))

    if args.as_json:
        payload = {
            "operation_id": report.operation_id,
            "locations": report.locations,
            "processed": report.processed,
            "added": report.added,
            "updated": report.updated,
            "unchanged": report.unchanged,
            "skipped": report.skipped,
            "unembedded": report.unembedded,
            "issues": [asdict(issue) for issue in report.issues],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=out)
    else:
        print(f"Операция: {report.operation_id}", file=out)
        print(f"Обработано: {report.processed}", file=out)
        print(f"Добавлено: {report.added}", file=out)
        print(f"Обновлено: {report.updated}", file=out)
        print(f"Без изменений: {report.unchanged}", file=out)
        print(f"Пропущено: {report.skipped}", file=out)
        print(f"Без эмбеддинга: {report.unembedded}", file=out)
        for issue in report.issues:
            print(f"  - {issue.location}: {issue.message}", file=out)

    return 0 if not report.is_partial else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI импорта."""

    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
