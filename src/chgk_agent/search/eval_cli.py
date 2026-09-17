"""CLI измерения базовых метрик качества поиска.

Считает `recall@k` и `MRR` на наборе «описание → ожидаемый вопрос» и
сохраняет отчёт артефактом в JSON.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from chgk_agent.config import get_settings
from chgk_agent.db.session import create_engine, create_session_factory
from chgk_agent.embeddings.gigachat import GigaChatEmbeddingProvider
from chgk_agent.embeddings.validation import verify_embedding_dimension
from chgk_agent.logging_setup import configure_logging
from chgk_agent.search.evaluation import (
    DEFAULT_K,
    evaluate_search_quality,
    load_cases,
)
from chgk_agent.search.local import LocalSearch

DEFAULT_CASES = Path("resources/search_quality_cases.json")
DEFAULT_OUTPUT = Path("resources/search_quality_report.json")


def build_parser() -> argparse.ArgumentParser:
    """Собрать разборщик аргументов командной строки."""

    parser = argparse.ArgumentParser(
        prog="chgk-eval",
        description="Измерить recall@k и MRR локального поиска на наборе случаев.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help=f"JSON-набор «описание → ожидаемый вопрос» (по умолчанию {DEFAULT_CASES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"куда сохранить отчёт-артефакт (по умолчанию {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="глубина выдачи для recall@k",
    )
    parser.add_argument(
        "--no-lexical",
        action="store_true",
        help="считать только векторную ветку",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="напечатать отчёт в stdout",
    )
    return parser


async def _evaluate(args: argparse.Namespace):
    """Выполнить измерение качества на локальной базе."""

    settings = get_settings()
    configure_logging(
        json_logs=settings.observability.json_logs,
        level=settings.observability.log_level,
    )

    cases = load_cases(args.cases)
    provider = GigaChatEmbeddingProvider(settings.gigachat)
    verify_embedding_dimension(provider, settings)

    engine = create_engine(settings)
    try:
        async with create_session_factory(engine)() as session:
            search = LocalSearch(
                session,
                provider,
                use_lexical=not args.no_lexical,
                semantic_min_score=settings.search.semantic_min_score,
                lexical_min_score=settings.search.lexical_min_score,
            )
            return await evaluate_search_quality(search, cases, k=args.k)
    finally:
        await engine.dispose()


def run(args: argparse.Namespace, *, out: object = None) -> int:
    """Выполнить измерение и сохранить отчёт."""

    out = out if out is not None else sys.stdout
    report = asyncio.run(_evaluate(args))
    path = report.save(args.output)

    if args.as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), file=out)
    else:
        print(f"Случаев: {report.cases}", file=out)
        print(f"recall@{report.k}: {report.recall_at_k:.3f}", file=out)
        print(f"MRR: {report.mrr:.3f}", file=out)
        print(f"Отчёт сохранён: {path}", file=out)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI оценки качества."""

    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
