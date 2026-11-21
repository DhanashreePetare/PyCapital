"""Public API for NSE corporate actions (dividends, bonuses, splits, rights, ...).

Example:
    >>> from pyequity.api import actions
    >>> everything = actions.get_actions("RELIANCE")                    # full history
    >>> window = actions.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
    >>> since = actions.get_actions("RELIANCE", start="2024-01-01")     # up to a year ahead

Caching (shared disk cache, ``pyequity.reliability.cache``):

- With BOTH ``start`` and ``end`` the request is a closed ex-date range and
  goes through the range-aware ``get_series``: repeating it, or asking for an
  overlapping range, only fetches the missing part. Dates that are not yet
  settled (today, and ex-dates in the future) are returned but never stored.
- With a missing ``start`` or ``end`` the request is open-ended (it reaches
  back to the connector's default start, or forward to announced upcoming
  actions). No date boundary is invented for the cache: the whole answer is
  stored with ``get_snapshot`` under an exact key (symbol, start, end) and
  expires after the cache's snapshot TTL, so new announcements appear.

Empty and failed fetches are never cached.
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from pyequity.api._args import DateLike, check_date, check_range
from pyequity.api._errors import (
    DataQualityWarning, DataUnavailableError, EmptyResultWarning,
    IncompleteDataWarning, InvalidArgumentError)
from pyequity.connectors import corporate as _connector
from pyequity.processing.normalize import normalize, validate
from pyequity.reliability.cache import get_default_cache
from pyequity.schema import empty_frame

logger = logging.getLogger(__name__)

_DATASET = "corporate_actions"
_SOURCE = "NSE"


def _check_symbol(symbol: str) -> str:
    """Return the symbol stripped and upper-cased, or raise InvalidArgumentError."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise InvalidArgumentError(
            f"symbol must be a non-empty string such as 'RELIANCE', got {symbol!r}")
    return symbol.strip().upper()


def _check_bounds(start: Optional[DateLike], end: Optional[DateLike]):
    """Validate optional start/end; return them as ``YYYY-MM-DD`` or None.

    The connector fills a missing start with 1990-01-01 and a missing end with
    one year ahead. A one-sided request whose given date lies beyond that
    default is contradictory, so it is rejected here rather than sent.
    """
    if start is not None and end is not None:
        return check_range(start, end)
    start_s = check_date(start, "start") if start is not None else None
    end_s = check_date(end, "end") if end is not None else None
    if start_s is not None:
        latest = (datetime.now() + timedelta(days=_connector._DEFAULT_DAYS_AHEAD)).date()
        if start_s > latest.isoformat():
            raise InvalidArgumentError(
                f"start ({start_s}) is later than the default end ({latest}, one year "
                f"ahead); give an explicit end")
    if end_s is not None and end_s < _connector._DEFAULT_FROM:
        raise InvalidArgumentError(
            f"end ({end_s}) is earlier than the default start ({_connector._DEFAULT_FROM}); "
            f"give an explicit start")
    return start_s, end_s


def get_actions(symbol: str, start: Optional[DateLike] = None,
                end: Optional[DateLike] = None) -> pd.DataFrame:
    """Get corporate actions for one NSE symbol, filtered by ex-date.

    ``start`` and ``end`` are each optional and bound the ex-date:

    - neither: the full available history, plus announced upcoming actions
      (up to a year ahead);
    - ``start`` only: from ``start`` up to a year ahead;
    - ``end`` only: from the earliest available history up to ``end``;
    - both: exactly that range (``start`` may not be after ``end``).

    The returned frame has the standard ``corporate_actions`` schema, sorted
    by ex-date, oldest to newest (rows with the same ex-date keep NSE's
    order). Prices are never adjusted here. ``parse_status`` is
    ``"needs_review"`` for demergers, mergers, schemes and text that was not
    parsed cleanly: do not adjust prices from such rows. ``df.attrs`` holds
    ``source`` and ``failed_ranges`` (attrs may be dropped by later pandas
    operations).

    NSE answers an unknown or renamed symbol exactly like a symbol with no
    actions, so an empty result does not prove the symbol is invalid (for
    example TATAMOTORS returned nothing while TMPV had the history).

    Args:
        symbol: NSE symbol, e.g. ``"RELIANCE"``.
        start: Optional first ex-date, ``"YYYY-MM-DD"`` or a date object.
        end:   Optional last ex-date.

    Returns:
        pandas DataFrame. May be empty, in which case an
        ``EmptyResultWarning`` is issued.

    Raises:
        InvalidArgumentError: Bad symbol or dates, a reversed range, or a
            one-sided range that contradicts the default bound.
        DataUnavailableError: NSE could not answer, so it is unknown whether
            actions exist.

    Warns:
        EmptyResultWarning: NSE answered and no rows match.
        IncompleteDataWarning: Some parts of a closed date range failed but
            others returned data; the result is missing rows.
        DataQualityWarning: The data failed integrity checks.
    """
    sym = _check_symbol(symbol)
    start_s, end_s = _check_bounds(start, end)

    failed: List[dict] = []

    def fetch_range(span_start: Optional[str], span_end: Optional[str]) -> pd.DataFrame:
        outcome = _connector.fetch_corporate_actions_outcome(sym, span_start, span_end)
        if outcome.failed:
            logger.error(f"[Actions API] {sym} {span_start}..{span_end} failed: "
                         f"{outcome.reason}")
            failed.append({"start": span_start, "end": span_end, "reason": outcome.reason})
            return empty_frame(_DATASET)
        return outcome.data  # OK frame, or the standard empty frame for EMPTY

    cache = get_default_cache()
    if start_s is not None and end_s is not None:
        def fetch_closed(span_start: str, span_end: str) -> pd.DataFrame:
            frame = fetch_range(span_start, span_end)
            # A closed ex-date range cannot match a row that has no ex-date; the
            # cache would drop such rows on a miss but not on a hit, so drop them
            # here for a result that is the same either way.
            undated = frame["ex_date"].isna()
            if undated.any():
                logger.warning(f"[Actions API] {sym}: left out {int(undated.sum())} row(s) "
                               f"without an ex-date from a dated range request")
                frame = frame[~undated].reset_index(drop=True)
            return frame

        identity = {"source": _SOURCE, "symbol": sym}
        df = cache.get_series(_DATASET, identity, start_s, end_s, fetch_closed)
    else:
        # Open-ended: no invented boundary, so cache the exact request as a snapshot.
        identity = {"source": _SOURCE, "symbol": sym, "start": start_s, "end": end_s}
        df = cache.get_snapshot(_DATASET, identity, lambda: fetch_range(start_s, end_s))

    window = f"between {start_s or 'the start of history'} and {end_s or 'the latest announced'}"
    if failed:
        if df.empty:
            raise DataUnavailableError(
                f"Could not fetch corporate actions for {sym} {window}: "
                f"{failed[0]['reason']}")
        ranges = ", ".join(f"{f['start']}..{f['end']}" for f in failed)
        warnings.warn(
            f"Corporate actions for {sym} are incomplete; could not fetch: {ranges}.",
            IncompleteDataWarning, stacklevel=2)
    elif df.empty:
        warnings.warn(
            f"NSE returned no corporate actions for {sym} {window}. NSE answers an "
            f"unknown or renamed symbol the same way, so this does not prove the "
            f"symbol is invalid.", EmptyResultWarning, stacklevel=2)

    out = normalize(df, _DATASET)
    if not out.empty:
        report = validate(out, _DATASET, start=start_s, end=end_s)
        if report.errors:
            warnings.warn(f"Corporate actions data failed integrity checks: "
                          f"{report.summary()}", DataQualityWarning, stacklevel=2)
    out.attrs["source"] = _SOURCE if not out.empty else ""
    out.attrs["failed_ranges"] = [f"{f['start']}..{f['end']}" for f in failed]
    return out