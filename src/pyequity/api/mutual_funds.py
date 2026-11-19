"""Public API for mutual fund data.

Example:
    >>> from pyequity.api import mutual_funds
    >>> nav = mutual_funds.get_nav(119551, "2026-09-15", "2026-09-23")
"""

import logging
import warnings
from typing import Union

import pandas as pd

from pyequity.api._args import DateLike, check_range
from pyequity.api._errors import (
    DataQualityWarning, DataUnavailableError, EmptyResultWarning,
    FallbackWarning, IncompleteDataWarning, InvalidArgumentError)
from pyequity.connectors import amfi
from pyequity.processing.normalize import normalize, validate
from pyequity.schema import empty_frame

logger = logging.getLogger(__name__)

_SOURCES = ("auto", "mfapi", "official")


def _check_scheme_code(scheme_code: Union[int, str]) -> int:
    """Return the scheme code as a positive int or raise InvalidArgumentError."""
    if isinstance(scheme_code, bool):
        raise InvalidArgumentError(f"scheme_code must be an integer code, got {scheme_code!r}")
    try:
        if isinstance(scheme_code, str):
            text = scheme_code.strip()
            if not text.isdigit():
                raise ValueError
            code = int(text)
        elif isinstance(scheme_code, int) or float(scheme_code) == int(scheme_code):
            code = int(scheme_code)
        else:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise InvalidArgumentError(
            f"scheme_code must be an AMFI scheme code such as 119551, got {scheme_code!r}"
        ) from None
    if code <= 0:
        raise InvalidArgumentError(f"scheme_code must be positive, got {scheme_code!r}")
    return code


def _finish(df: pd.DataFrame, start: str, end: str, requested: str,
            fallback_used: bool, failed_chunks: list) -> pd.DataFrame:
    """Normalise, integrity-check and label the result."""
    out = normalize(df, "nav")
    if not out.empty:
        report = validate(out, "nav", start=start, end=end)
        if report.errors:
            warnings.warn(f"NAV data failed integrity checks: {report.summary()}",
                          DataQualityWarning, stacklevel=3)
    out.attrs["source"] = (str(out["source"].iloc[0]) if not out.empty else "")
    out.attrs["requested_source"] = requested
    out.attrs["fallback_used"] = fallback_used
    out.attrs["failed_chunks"] = list(failed_chunks)
    return out


def get_nav(scheme_code: Union[int, str], start: DateLike, end: DateLike, *,
            source: str = "auto") -> pd.DataFrame:
    """Get NAV history for one mutual fund scheme.

    With ``source="auto"`` (default) mfapi.in is used. Only if mfapi.in could
    not answer (a failed request, not an empty answer) are AMFI's own
    published files used instead; this is announced with a
    ``FallbackWarning`` and the ``source`` column then reads
    ``"AMFI_OFFICIAL"``. An unknown scheme code is not retried elsewhere.

    The returned frame has the standard ``nav`` schema (date, scheme_code,
    scheme_name, nav, source), sorted oldest to newest. ``df.attrs`` holds
    ``source``, ``requested_source``, ``fallback_used`` and
    ``failed_chunks`` (attrs may be dropped by later pandas operations).

    Args:
        scheme_code: AMFI scheme code, e.g. ``119551`` (int or digit string).
        start: First date, ``"YYYY-MM-DD"`` or a date object.
        end:   Last date, ``"YYYY-MM-DD"`` or a date object.
        source: ``"auto"``, ``"mfapi"`` (never fall back) or ``"official"``
            (AMFI files only; slow for long ranges, about 1 MB per day).

    Returns:
        pandas DataFrame of NAVs. May be empty, in which case an
        ``EmptyResultWarning`` explains why.

    Raises:
        InvalidArgumentError: Bad scheme code, date, range or ``source``.
        DataUnavailableError: No source could answer, so it is unknown
            whether data exists.
    """
    code = _check_scheme_code(scheme_code)
    start_s, end_s = check_range(start, end)
    if source not in _SOURCES:
        raise InvalidArgumentError(f"source must be one of {_SOURCES}, got {source!r}")

    if source in ("auto", "mfapi"):
        outcome = amfi.fetch_nav_outcome(code, start_s, end_s)
        if outcome.ok:
            return _finish(outcome.data, start_s, end_s, source, False, [])
        if outcome.empty:
            warnings.warn(
                f"mfapi.in has no NAV rows for scheme {code} between {start_s} and {end_s} "
                f"(unknown scheme code, or no NAV published in that range).",
                EmptyResultWarning, stacklevel=2)
            return _finish(empty_frame("nav"), start_s, end_s, source, False, [])
        if source == "mfapi":
            raise DataUnavailableError(
                f"mfapi.in could not be reached for scheme {code}: {outcome.reason}")

        warnings.warn(
            f"mfapi.in failed for scheme {code} ({outcome.reason}); using AMFI's official "
            f"files instead. The source column will read AMFI_OFFICIAL.",
            FallbackWarning, stacklevel=2)
        official = amfi.fetch_nav_official(code, start_s, end_s)
        failed = list(official.attrs.get("failed_chunks", []))
        if official.empty:
            raise DataUnavailableError(
                f"No source could answer for scheme {code}: mfapi.in failed "
                f"({outcome.reason}) and AMFI's official files returned no rows"
                + (f" (unreadable: {', '.join(failed)})." if failed else "."))
        if failed:
            warnings.warn(
                f"AMFI official data for scheme {code} is incomplete; could not read: "
                f"{', '.join(failed)}.", IncompleteDataWarning, stacklevel=2)
        return _finish(official, start_s, end_s, source, True, failed)

    official = amfi.fetch_nav_official(code, start_s, end_s)
    failed = list(official.attrs.get("failed_chunks", []))
    if failed:
        warnings.warn(
            f"AMFI official data for scheme {code} is incomplete; could not read: "
            f"{', '.join(failed)}.", IncompleteDataWarning, stacklevel=2)
    if official.empty:
        warnings.warn(
            f"AMFI's official files returned no NAV rows for scheme {code} between "
            f"{start_s} and {end_s} (none published, or the files could not be read).",
            EmptyResultWarning, stacklevel=2)
    return _finish(official, start_s, end_s, source, False, failed)