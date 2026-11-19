"""Retry with exponential backoff that never raises.

``retry_call`` runs an operation and returns a ``FetchOutcome``. It retries
only failures that can plausibly be transient (connection problems,
timeouts, HTTP 429/403/5xx, responses that cannot be parsed). It does NOT
retry a genuine empty result, so an invalid symbol is answered at once
instead of after several slow attempts. Programming errors are not retried
and are logged with their traceback.
"""

import logging
import random
import time
from typing import Any, Callable, Optional

import requests

from pyequity.reliability.outcome import FetchOutcome, Status

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying. 403 is included because NSE and BSE answer
# 403 to sessions they have temporarily stopped trusting.
RETRYABLE_STATUSES = frozenset({403, 408, 429, 500, 502, 503, 504})


class RetryableError(Exception):
    """Raise inside an operation to say "this failure may be transient"."""


class PermanentError(Exception):
    """Raise inside an operation to say "retrying cannot help"."""


def describe_response(response: Any) -> str:
    """Summarise an HTTP response for log messages.

    Logging only "Expecting value" hid an undecodable brotli body for a
    long time; the status, content type and encoding make that diagnosable.

    Args:
        response: A ``requests.Response`` (or any object with the same
            ``status_code`` and ``headers`` attributes), or ``None``.

    Returns:
        A one-line description such as
        ``"status=200 content-type=application/json content-encoding=br"``.
    """
    if response is None:
        return "no response"
    headers = getattr(response, "headers", None) or {}
    return (
        f"status={getattr(response, 'status_code', '?')} "
        f"content-type={headers.get('Content-Type', '?')} "
        f"content-encoding={headers.get('Content-Encoding', 'none')}"
    )


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether an exception is worth another attempt."""
    if isinstance(exc, RetryableError):
        return True
    if isinstance(exc, PermanentError):
        return False
    if isinstance(exc, requests.HTTPError):
        status = getattr(exc.response, "status_code", None)
        return status in RETRYABLE_STATUSES
    if isinstance(exc, (requests.ConnectionError, requests.Timeout,
                        requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.ContentDecodingError)):
        return True
    # json.JSONDecodeError and UnicodeDecodeError are ValueError subclasses:
    # an unparseable body may be a transient block page.
    return isinstance(exc, ValueError)


def _default_is_empty(value: Any) -> bool:
    """None, or anything with no length / an ``empty`` flag, counts as empty."""
    if value is None:
        return True
    empty_flag = getattr(value, "empty", None)
    if isinstance(empty_flag, bool):
        return empty_flag
    try:
        return len(value) == 0
    except TypeError:
        return False


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Read a numeric ``Retry-After`` header from an HTTP error, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        return max(0.0, float(headers.get("Retry-After")))
    except (TypeError, ValueError):
        return None


def retry_call(
    operation: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    is_empty: Optional[Callable[[Any], bool]] = None,
    label: str = "operation",
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> FetchOutcome:
    """Run ``operation`` with retries and return a ``FetchOutcome``.

    Never raises ``Exception`` subclasses. ``KeyboardInterrupt`` and
    ``SystemExit`` are not caught.

    Args:
        operation: Zero-argument callable that performs one attempt and
            returns the fetched value. Raise ``RetryableError`` or
            ``PermanentError`` to classify a failure explicitly.
        max_attempts: Total tries including the first (at least 1).
        base_delay: Seconds before the second attempt; doubles each time.
        max_delay: Upper bound for any single wait, in seconds.
        jitter: Random spread of each wait as a fraction (0.1 = +/-10%).
        is_empty: Predicate saying a returned value is a genuine empty
            answer. Defaults to None / empty DataFrame / zero length.
        label: Name used in log messages.
        sleep: Sleep function (injectable for tests).
        rng: Function returning a float in [0, 1) (injectable for tests).

    Returns:
        ``FetchOutcome``: OK with the value, EMPTY with the empty value (not
        retried), or FAILED with a reason and the number of attempts used.
    """
    max_attempts = max(1, int(max_attempts))
    empty_check = is_empty or _default_is_empty
    last_reason = "unknown error"

    for attempt in range(1, max_attempts + 1):
        try:
            value = operation()
        except Exception as exc:  # noqa: BLE001 - classified below, never re-raised
            if isinstance(exc, requests.HTTPError):
                detail = describe_response(exc.response)
            else:
                detail = ""
            last_reason = f"{type(exc).__name__}: {exc}" + (f" ({detail})" if detail else "")

            if not _is_retryable(exc):
                logger.error("[retry] %s failed permanently on attempt %d: %s",
                             label, attempt, last_reason,
                             exc_info=not isinstance(exc, (PermanentError, requests.RequestException)))
                return FetchOutcome(Status.FAILED, None, last_reason, attempt)

            if attempt == max_attempts:
                break

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            retry_after = _retry_after_seconds(exc)
            if retry_after is not None:
                delay = min(max_delay, max(delay, retry_after))
            delay *= 1 + jitter * (rng() * 2 - 1)
            delay = max(0.0, delay)
            logger.warning("[retry] %s attempt %d/%d failed (%s); retrying in %.1fs",
                           label, attempt, max_attempts, last_reason, delay)
            sleep(delay)
            continue

        if empty_check(value):
            return FetchOutcome(Status.EMPTY, value, "source returned no rows", attempt)
        return FetchOutcome(Status.OK, value, "", attempt)

    logger.error("[retry] %s failed after %d attempts: %s", label, max_attempts, last_reason)
    return FetchOutcome(Status.FAILED, None, last_reason, max_attempts)