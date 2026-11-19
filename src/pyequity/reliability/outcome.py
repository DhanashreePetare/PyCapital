"""Result type that keeps "the request failed" apart from "the answer is empty".

Connectors historically returned an empty DataFrame for both cases, which
hid real problems (for example an undecodable response looked exactly like
an unknown symbol). ``FetchOutcome`` carries the difference so callers and
the public API can react correctly: a ``FAILED`` fetch may be retried or
sent to a fallback source, an ``EMPTY`` one is a real answer.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Status(str, Enum):
    """Final state of a fetch."""

    OK = "ok"          # data was obtained
    EMPTY = "empty"    # the source answered, and the answer has no rows
    FAILED = "failed"  # no usable answer could be obtained


@dataclass(frozen=True)
class FetchOutcome:
    """Outcome of one resilient fetch.

    Attributes:
        status: ``Status.OK``, ``Status.EMPTY`` or ``Status.FAILED``.
        data: The fetched value for OK and EMPTY, otherwise ``None``.
        reason: Short human-readable explanation (empty for OK).
        attempts: How many times the operation was tried.
    """

    status: Status
    data: Any = None
    reason: str = ""
    attempts: int = 1

    @property
    def ok(self) -> bool:
        """True if data was obtained."""
        return self.status is Status.OK

    @property
    def empty(self) -> bool:
        """True if the source answered with no rows."""
        return self.status is Status.EMPTY

    @property
    def failed(self) -> bool:
        """True if no usable answer could be obtained."""
        return self.status is Status.FAILED