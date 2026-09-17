"""Ограничение частоты запросов и предохранитель внешнего источника."""

import asyncio
import time


class RateLimiter:
    """Пропускает не более одного запроса за заданный интервал.

    Интервал отсчитывается от начала предыдущего запроса, поэтому
    гарантируется пауза не меньше `min_interval_seconds` между запросами
    даже при одновременных вызовах.
    """

    def __init__(
        self,
        min_interval_seconds: float = 1.0,
        *,
        clock: object = time.monotonic,
        sleep: object = asyncio.sleep,
    ) -> None:
        self._interval = max(min_interval_seconds, 0.0)
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last_started: float | None = None
        self.last_wait_seconds = 0.0

    async def acquire(self) -> float:
        """Дождаться разрешения на запрос и вернуть время ожидания."""

        async with self._lock:
            now = self._clock()
            waited = 0.0
            if self._last_started is not None:
                elapsed = now - self._last_started
                remaining = self._interval - elapsed
                if remaining > 0:
                    await self._sleep(remaining)
                    waited = remaining
                    now = self._clock()
            self._last_started = now
            self.last_wait_seconds = waited
            return waited


class CircuitBreaker:
    """Предохранитель: после серии ошибок ветка сразу возвращает отказ."""

    def __init__(
        self,
        *,
        threshold: int = 5,
        reset_seconds: float = 60.0,
        clock: object = time.monotonic,
    ) -> None:
        self._threshold = max(threshold, 1)
        self._reset_seconds = reset_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def failures(self) -> int:
        """Текущее число подряд идущих ошибок."""

        return self._failures

    @property
    def is_open(self) -> bool:
        """Разомкнут ли предохранитель прямо сейчас."""

        if self._opened_at is None:
            return False
        if self._clock() - self._opened_at >= self._reset_seconds:
            self._failures = 0
            self._opened_at = None
            return False
        return True

    def allow(self) -> bool:
        """Можно ли выполнять запрос."""

        return not self.is_open

    def record_success(self) -> None:
        """Зафиксировать успешный запрос."""

        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Зафиксировать ошибку и при необходимости разомкнуть цепь."""

        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._clock()
