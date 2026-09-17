"""CLI переиндексации: векторизация очереди без повторного разбора HTML."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from chgk_agent.config import get_settings
from chgk_agent.db.session import create_engine, create_session_factory
from chgk_agent.embeddings.gigachat import GigaChatEmbeddingProvider
from chgk_agent.embeddings.reindexer import drain_queue
from chgk_agent.embeddings.validation import verify_embedding_dimension
from chgk_agent.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Собрать разборщик аргументов командной строки."""

    parser = argparse.ArgumentParser(
        prog="chgk-reindex",
        description="Векторизовать вопросы, оставшиеся без эмбеддинга.",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Ограничить переиндексацию одним источником",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Размер батча обращения к GigaChat",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "пересчитать векторы, посчитанные другой моделью (нужно после "
            "смены embedding-модели)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Вывести результат в формате JSON",
    )
    return parser


async def _run_reindex(args: argparse.Namespace):
    """Выполнить переиндексацию и вернуть результат."""

    settings = get_settings()
    configure_logging(
        json_logs=settings.observability.json_logs,
        level=settings.observability.log_level,
    )

    provider = GigaChatEmbeddingProvider(settings.gigachat)
    verify_embedding_dimension(provider, settings)
    if args.force:
        logger.info(
            "принудительная переиндексация",
            model=provider.model,
            dimension=provider.dimension,
        )

    engine = create_engine(settings)
    try:
        async with create_session_factory(engine)() as session:
            run = await drain_queue(
                session,
                provider,
                batch_size=args.batch_size,
                location=args.location,
                force=args.force,
            )
            await session.commit()
            return run
    finally:
        await engine.dispose()


def run(args: argparse.Namespace, *, out: object = None) -> int:
    """Выполнить переиндексацию и напечатать результат."""

    out = out if out is not None else sys.stdout

    result = asyncio.run(_run_reindex(args))

    if args.as_json:
        print(
            json.dumps(
                {
                    "embedded": result.embedded,
                    "failed": result.failed,
                    "remaining": result.remaining,
                    "errors": result.errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=out,
        )
    else:
        print(f"Векторизовано: {result.embedded}", file=out)
        print(f"Ошибок: {result.failed}", file=out)
        print(f"Осталось в очереди: {result.remaining}", file=out)
        for error in result.errors:
            print(f"  - {error}", file=out)

    return 0 if not result.failed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI переиндексации."""

    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
