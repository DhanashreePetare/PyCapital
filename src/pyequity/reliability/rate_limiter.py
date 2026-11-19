"""Minimum-interval rate limiter, one independent clock per key (host)."""

import threading
import time
from typing import Callable, Dict


class RateLimiter:
    """Guarantee at least ``min_interval`` seconds between calls per key.

    Thread-safe. Each caller reserves the next free slot under a lock and
    sleeps outside it, so concurrent callers are spaced out instead of
    firing together. Used to avoid hammering exchange servers.
    """

    def __init__(
        self,
        min_interval: float = 1.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a limiter.

        Args:
            min_interval: Minimum seconds between two calls for the same key
                (0 disables waiting).
            clock: Monotonic clock (injectable for tests).
            sleep: Sleep function (injectable for tests).
        """
        self._min_interval = max(0.0, float(min_interval))
        self._clock = clock
        self._sleep = sleep
        self._next_free: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, key: str = "default") -> float:
        """Block until a call for ``key`` is allowed.

        Args:
            key: Independent bucket, normally the host name.

        Returns:
            Seconds slept (0.0 if no wait was needed).
        """
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_free.get(key, now))
            self._next_free[key] = slot + self._min_interval
        delay = slot - now
        if delay > 0:
            self._sleep(delay)
            return delay
        return 0.0