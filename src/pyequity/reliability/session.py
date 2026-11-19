"""Reusable, self-refreshing HTTP sessions.

Creating an NSE session costs several seconds of warm-up, and
``create_nse_session`` raises if the connection drops, which breaks the
rule that connectors never raise. ``SessionManager`` reuses one session per
key, replaces it when it is old or has been invalidated (for example after
a 401/403), and turns a failed creation into a ``FAILED`` outcome.

``_session_utils.create_nse_session`` is wrapped, not modified.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from pyequity.reliability.outcome import FetchOutcome, Status
from pyequity.reliability.retry import retry_call

logger = logging.getLogger(__name__)

# Assumption, not verified against NSE: NSE session cookies stay valid for
# several minutes. Ten minutes is a conservative refresh interval.
DEFAULT_MAX_AGE_SECONDS = 600.0


def _default_factory(key: Optional[str]) -> Any:
    """Create an NSE session (imported lazily to avoid an import cycle)."""
    from pyequity.connectors._session_utils import create_nse_session
    return create_nse_session(key)


class SessionManager:
    """Hand out one live session per key and refresh it when needed."""

    def __init__(
        self,
        factory: Callable[[Optional[str]], Any] = _default_factory,
        *,
        max_age: float = DEFAULT_MAX_AGE_SECONDS,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a manager.

        Args:
            factory: Called with the key; returns a new session. Defaults to
                ``create_nse_session`` (key = NSE symbol or None).
            max_age: Seconds after which a session is replaced.
            max_attempts: Tries when creating a session.
            base_delay: First backoff delay between creation attempts.
            clock: Monotonic clock (injectable for tests).
            sleep: Sleep function (injectable for tests).
        """
        self._factory = factory
        self._max_age = max_age
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._clock = clock
        self._sleep = sleep
        self._sessions: Dict[Optional[str], Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: Optional[str] = None) -> FetchOutcome:
        """Return a usable session for ``key``.

        Args:
            key: Session identity, e.g. an NSE symbol, or None for the
                generic session.

        Returns:
            ``FetchOutcome`` with status OK and the session as ``data``, or
            FAILED (with the reason) if a session could not be created.
            Never raises.
        """
        with self._lock:
            cached = self._sessions.get(key)
            if cached is not None:
                session, created = cached
                if self._clock() - created < self._max_age:
                    return FetchOutcome(Status.OK, session)
                logger.info("[session] session for %r expired; creating a new one", key)
                self._discard(key)

            outcome = retry_call(
                lambda: self._factory(key),
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
                is_empty=lambda _value: False,
                label=f"create session {key!r}",
                sleep=self._sleep,
            )
            if outcome.ok:
                self._sessions[key] = (outcome.data, self._clock())
            return outcome

    def invalidate(self, key: Optional[str] = None) -> None:
        """Drop the session for ``key`` so the next ``get`` builds a fresh one.

        Call this after a 401/403 or an unparseable response.

        Args:
            key: Session identity used in ``get``.
        """
        with self._lock:
            self._discard(key)

    def close(self) -> None:
        """Close and forget every session."""
        with self._lock:
            for key in list(self._sessions):
                self._discard(key)

    def _discard(self, key: Optional[str]) -> None:
        cached = self._sessions.pop(key, None)
        if cached is not None:
            close = getattr(cached[0], "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - closing is best effort
                    logger.debug("[session] error while closing session for %r", key, exc_info=True)