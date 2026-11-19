"""Argument checking shared by the public API modules."""

import datetime as _dt
from typing import Tuple, Union

from pyequity.api._errors import InvalidArgumentError

DateLike = Union[str, _dt.date, _dt.datetime]


def check_date(value: DateLike, name: str) -> str:
    """Validate a date and return it as ``YYYY-MM-DD``.

    Args:
        value: An ISO string (``"2026-09-23"``), a ``date`` or a ``datetime``
            (pandas ``Timestamp`` works too).
        name:  Argument name used in the error message.

    Returns:
        The date as a ``YYYY-MM-DD`` string.

    Raises:
        InvalidArgumentError: If the value is not a valid date.
    """
    if isinstance(value, _dt.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, _dt.date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        try:
            return _dt.datetime.strptime(value.strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    raise InvalidArgumentError(
        f"{name} must be a date in YYYY-MM-DD format, got {value!r}")


def check_range(start: DateLike, end: DateLike) -> Tuple[str, str]:
    """Validate a start/end pair.

    Args:
        start: Start date (see ``check_date``).
        end:   End date (see ``check_date``).

    Returns:
        ``(start, end)`` as ``YYYY-MM-DD`` strings.

    Raises:
        InvalidArgumentError: If either date is invalid or start is after end.
    """
    start_s = check_date(start, "start")
    end_s = check_date(end, "end")
    if start_s > end_s:
        raise InvalidArgumentError(f"start ({start_s}) cannot be after end ({end_s})")
    return start_s, end_s