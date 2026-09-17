"""Соблюдение правил `robots.txt` внешнего источника.

Страница поиска gotquestions.online закрыта правилами сайта
(`Disallow: /search`), но владелец сервиса явно принял решение работать с
ней, ограничившись rate limiting. Поэтому по умолчанию правила не
применяются, а флаг `respect_robots` включает строгий режим без релиза.

Правила кэшируются на время жизни политики, недоступный `robots.txt`
трактуется как разрешение: сбой вспомогательного запроса не должен
блокировать основной поиск.
"""

import time
import urllib.robotparser
from collections.abc import Callable

import httpx

from chgk_agent.logging_setup import get_logger

logger = get_logger(__name__)

ROBOTS_PATH = "/robots.txt"
DEFAULT_TTL_SECONDS = 3600.0


class RobotsPolicy:
    """Кэширующая проверка правил `robots.txt` для одного сайта."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        user_agent: str,
        enabled: bool = True,
        path: str = ROBOTS_PATH,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._enabled = enabled
        self._path = path
        self._ttl = ttl_seconds
        self._clock = clock
        self._parser: urllib.robotparser.RobotFileParser | None = None
        self._fetched_at: float | None = None

    @property
    def enabled(self) -> bool:
        """Включён ли строгий режим соблюдения правил."""

        return self._enabled

    async def allows(self, url: str) -> bool:
        """Разрешён ли запрос к `url` правилами сайта."""

        if not self._enabled:
            return True

        parser = await self._load()
        if parser is None:
            return True
        return parser.can_fetch(self._user_agent, url)

    async def _load(self) -> urllib.robotparser.RobotFileParser | None:
        """Загрузить `robots.txt` с учётом времени жизни кэша."""

        now = self._clock()
        if (
            self._parser is not None
            and self._fetched_at is not None
            and now - self._fetched_at < self._ttl
        ):
            return self._parser

        try:
            response = await self._client.get(self._path)
        except httpx.HTTPError as error:
            logger.warning("robots.txt недоступен", error=str(error))
            self._parser = None
            self._fetched_at = now
            return None

        if response.status_code >= 400:
            logger.warning("robots.txt недоступен", status=response.status_code)
            self._parser = None
            self._fetched_at = now
            return None

        parser = urllib.robotparser.RobotFileParser()
        parser.parse(response.text.splitlines())
        self._parser = parser
        self._fetched_at = now
        return parser


__all__ = ["ROBOTS_PATH", "RobotsPolicy"]
