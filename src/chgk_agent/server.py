"""Запуск сервиса: API и веб-интерфейс в одном ASGI-процессе."""

import os
from collections.abc import Sequence

import uvicorn

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
APP_FACTORY = "chgk_agent.api.app:build_default_app"


def build_parser():
    """Собрать разборщик аргументов командной строки."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="chgk-serve",
        description="Запустить API и веб-интерфейс поиска вопросов ЧГК.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true", help="перезапуск при изменении кода")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить сервер через uvicorn."""

    args = build_parser().parse_args(argv)
    port = args.port or int(os.environ.get("PORT", DEFAULT_PORT))
    uvicorn.run(
        APP_FACTORY,
        factory=True,
        host=args.host,
        port=port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
