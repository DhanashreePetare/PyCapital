"""Public API for NSE futures and options (F&O) data.

Example:
    >>> from pyequity.api import fno
    >>> fut = fno.get_history("RELIANCE", "2026-01-01", "2026-03-31")
    >>> opt = fno.get_history("RELIANCE", "2026-01-01", "2026-01-31",
    ...                       instrument="OPTSTK", expiry="2026-01-27",
    ...                       strike=1250, option_type="CE")
    >>> expiries = fno.list_expiries("RELIANCE")

Caching: ``get_history`` goes through the shared disk cache
(``pyequity.reliability.cache``). Only complete, non-empty, settled data is
stored, so an empty or failed fetch is asked again next time. ``list_expiries``
is deliberately not cached: it is a live list of strings, and the cache
stores schema DataFrames only.
"""

import datetime as _dt
import logging
import warnings
from typing import List, Optional, Union

import pandas as pd

from pyequity.api._args import DateLike, check_range
from pyequity.api._errors import (
    DataQualityWarning, DataUnavailableError, EmptyResultWarning,
    IncompleteDataWarning, InvalidArgumentError)
from pyequity.connectors import fno as _connector
from pyequity.processing.normalize import normalize, validate
from pyequity.reliability.cache import get_default_cache
from pyequity.schema import empty_frame

logger = logging.getLogger(__name__)

_DATASET = "fno"
_SOURCE = "NSE_FO"
_OPTION_TYPES = ("CE", "PE")


def _check_symbol(symbol: str) -> str:
    """Return the symbol stripped and upper-cased, or raise InvalidArgumentError."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise InvalidArgumentError(
            f"symbol must be a non-empty string such as 'RELIANCE', got {symbol!r}")
    return symbol.strip().upper()


def _check_instrument(instrument: str) -> str:
    """Return a valid instrument type (case-insensitive) or raise."""
    text = instrument.strip().upper() if isinstance(instrument, str) else ""
    if text not in _connector.VALID_INSTRUMENTS:
        raise InvalidArgumentError(
            f"instrument must be one of {_connector.VALID_INSTRUMENTS}, got {instrument!r}")
    return text


def _check_expiry(expiry: Union[str, _dt.date, None]) -> Optional[str]:
    """Return an expiry as ``YYYY-MM-DD`` (None stays None) or raise."""
    if expiry is None:
        return None
    if isinstance(expiry, _dt.date):  # datetime and pandas Timestamp are dates too
        return expiry.strftime("%Y-%m-%d")
    parsed = _connector._parse_expiry(expiry) if isinstance(expiry, str) else None
    if parsed is None:
        raise InvalidArgumentError(
            f"expiry must be a date as YYYY-MM-DD or e.g. 27-Oct-2026, got {expiry!r}")
    return parsed.strftime("%Y-%m-%d")


def _check_strike(strike: object) -> Optional[float]:
    """Return a positive finite strike as float (None stays None) or raise."""
    if strike is None:
        return None
    text = _connector._format_strike(strike)
    if text is None:
        raise InvalidArgumentError(f"strike must be a positive number, got {strike!r}")
    return float(text)


def _check_option_type(option_type: Optional[str]) -> Optional[str]:
    """Return ``CE``/``PE`` (case-insensitive) or None, or raise."""
    if option_type is None:
        return None
    text = option_type.strip().upper() if isinstance(option_type, str) else ""
    if text not in _OPTION_TYPES:
        raise InvalidArgumentError(
            f"option_type must be 'CE' or 'PE', got {option_type!r}")
    return text


def get_history(symbol: str, start: DateLike, end: DateLike, *,
                instrument: str = "FUTSTK",
                expiry: Union[str, _dt.date, None] = None,
                strike: Optional[float] = None,
                option_type: Optional[str] = None) -> pd.DataFrame:
    """Get daily F&O history for one underlying.

    Data comes from NSE and is never silently cut off: the connector splits
    requests that reach NSE's row cap, and reports a failure rather than
    returning partial data. Results are cached on disk (complete, settled,
    non-empty data only), so repeating a request, or asking for an
    overlapping range, only fetches what is missing.

    Options (``OPTSTK``/``OPTIDX``) need both ``strike`` and ``option_type``;
    futures must not have them.

    The returned frame has the standard ``fno`` schema, sorted oldest to
    newest, with ``source`` equal to ``"NSE_FO"``. ``df.attrs`` holds
    ``source`` and ``failed_ranges`` (date ranges that could not be fetched;
    attrs may be dropped by later pandas operations).

    Args:
        symbol: Underlying, e.g. ``"RELIANCE"`` or ``"BANKNIFTY"``.
        start: First date, ``"YYYY-MM-DD"`` or a date object.
        end:   Last date.
        instrument: ``"FUTSTK"`` (default), ``"OPTSTK"``, ``"FUTIDX"`` or
            ``"OPTIDX"``.
        expiry: Optional expiry, ``"YYYY-MM-DD"``, NSE style
            ``"27-Oct-2026"`` or a date. Past expiries are allowed. None
            returns every contract listed on each day.
        strike: Strike price (positive number). Required for options.
        option_type: ``"CE"`` or ``"PE"``. Required for options.

    Returns:
        pandas DataFrame. May be empty, in which case an
        ``EmptyResultWarning`` explains why.

    Raises:
        InvalidArgumentError: Bad symbol, instrument, dates, range, expiry,
            strike or option type, or a strike/option type combination that
            does not fit the instrument.
        DataUnavailableError: NSE could not answer, so it is unknown whether
            data exists.

    Warns:
        EmptyResultWarning: NSE answered and has no rows.
        IncompleteDataWarning: Some date ranges failed but others returned
            data; the result is missing rows.
        DataQualityWarning: The data failed integrity checks.
    """
    sym = _check_symbol(symbol)
    start_s, end_s = check_range(start, end)
    instrument_s = _check_instrument(instrument)
    expiry_s = _check_expiry(expiry)
    strike_f = _check_strike(strike)
    opt = _check_option_type(option_type)

    if instrument_s.startswith("OPT"):
        if strike_f is None or opt is None:
            raise InvalidArgumentError(
                f"{instrument_s} needs both strike and option_type "
                f"(without them NSE's row cap would lose data)")
    elif strike_f is not None or opt is not None:
        raise InvalidArgumentError(
            f"strike and option_type only apply to options, not {instrument_s}")

    failed: List[dict] = []

    def fetch(span_start: str, span_end: str) -> pd.DataFrame:
        outcome = _connector.fetch_fno_outcome(
            sym, span_start, span_end, expiry_s, instrument_s,
            strike_price=strike_f, option_type=opt)
        if outcome.failed:
            logger.error(f"[FNO API] {sym} {span_start}..{span_end} failed: {outcome.reason}")
            failed.append({"start": span_start, "end": span_end, "reason": outcome.reason})
            return empty_frame(_DATASET)
        return outcome.data  # OK frame, or the standard empty frame for EMPTY

    # Every parameter that changes the result is part of the cache identity.
    # ``None`` means "no filter" (cache.make_key keeps it distinct from a value).
    identity = {
        "source": _SOURCE, "symbol": sym, "instrument_type": instrument_s,
        "expiry_date": expiry_s, "strike_price": strike_f, "option_type": opt,
    }
    df = get_default_cache().get_series(_DATASET, identity, start_s, end_s, fetch)

    if failed:
        ranges = ", ".join(f"{f['start']}..{f['end']}" for f in failed)
        if df.empty:
            raise DataUnavailableError(
                f"Could not fetch {instrument_s} data for {sym} between "
                f"{start_s} and {end_s}: {failed[0]['reason']}")
        warnings.warn(
            f"F&O data for {sym} is incomplete; could not fetch: {ranges}.",
            IncompleteDataWarning, stacklevel=2)
    elif df.empty:
        warnings.warn(
            f"NSE has no {instrument_s} rows for {sym} between {start_s} and "
            f"{end_s} with these filters (nothing traded, or no such contract).",
            EmptyResultWarning, stacklevel=2)

    out = normalize(df, _DATASET)
    if not out.empty:
        report = validate(out, _DATASET, start=start_s, end=end_s)
        if report.errors:
            warnings.warn(f"F&O data failed integrity checks: {report.summary()}",
                          DataQualityWarning, stacklevel=2)
    out.attrs["source"] = _SOURCE if not out.empty else ""
    out.attrs["failed_ranges"] = [f"{f['start']}..{f['end']}" for f in failed]
    return out


def list_expiries(symbol: str) -> List[str]:
    """List the currently listed F&O expiry dates for an underlying.

    Only live contracts are listed; historical data does not need this (call
    ``get_history`` without ``expiry``). The answer is not cached: it is a
    short live list that changes as contracts are listed and expire.

    Args:
        symbol: Underlying, e.g. ``"RELIANCE"`` or ``"BANKNIFTY"``.

    Returns:
        Sorted list of ``"YYYY-MM-DD"`` strings. An empty list means NSE
        lists nothing (an ``EmptyResultWarning`` is issued); a failed
        request raises instead.

    Raises:
        InvalidArgumentError: Empty or non-string symbol.
        DataUnavailableError: NSE could not answer.

    Warns:
        EmptyResultWarning: NSE answered that no expiries are listed (this
            is also what it says for an unknown symbol).
    """
    sym = _check_symbol(symbol)
    outcome = _connector.list_expiries_outcome(sym)
    if outcome.failed:
        raise DataUnavailableError(
            f"Could not list expiries for {sym}: {outcome.reason}")
    if outcome.empty:
        warnings.warn(
            f"NSE lists no F&O expiries for {sym} (unknown symbol, or it has no "
            f"F&O contracts).", EmptyResultWarning, stacklevel=2)
        return []
    return list(outcome.data)