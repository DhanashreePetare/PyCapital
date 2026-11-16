"""
NSE Derivatives (F&O) historical data connector for PyEquity.

Endpoints and field names were confirmed through DevTools and live probes
(2026-11). Facts this module relies on:

* The historical endpoint returns at most 70 rows per request, newest rows
  first. Anything beyond that is dropped silently by NSE, so every request
  is checked against the cap and split until each piece fits.
* ``expiryDate`` may be blank. NSE then returns every contract listed on
  each day, including expired contracts in past years.
* ``optionType`` (CE/PE) and ``strikePrice`` filter server-side.
* Futures rows come back with option_type "XX" and strike_price 0.
* ``option-chain-contract-info`` lists currently listed expiries only.

Policy: if the result cannot be guaranteed complete (a request failed, or
a single day still hits the row cap) the function returns an empty frame
and logs an error. It never returns partial data and never raises.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from pyequity.connectors._session_utils import create_nse_session
from pyequity.schema import SCHEMAS, columns_of, empty_frame

logger = logging.getLogger(__name__)

_HIST_URL = "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
_INFO_URL = "https://www.nseindia.com/api/option-chain-contract-info"

NSE_ROW_CAP = 70
VALID_INSTRUMENTS = ("FUTSTK", "OPTSTK", "FUTIDX", "OPTIDX")
_REQUEST_DELAY = 1.0  # seconds between requests; tests set this to 0

# Fixed English month names: strftime/strptime %b depend on the OS locale.
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_RENAME = {
    "FH_TIMESTAMP": "date",
    "FH_SYMBOL": "symbol",
    "FH_INSTRUMENT": "instrument_type",
    "FH_EXPIRY_DT": "expiry_date",
    "FH_OPTION_TYPE": "option_type",
    "FH_STRIKE_PRICE": "strike_price",
    "FH_OPENING_PRICE": "open",
    "FH_TRADE_HIGH_PRICE": "high",
    "FH_TRADE_LOW_PRICE": "low",
    "FH_CLOSING_PRICE": "close",
    "FH_SETTLE_PRICE": "settle_price",
    "FH_TOT_TRADED_QTY": "volume",
    "FH_TOT_TRADED_VAL": "value",
    "FH_OPEN_INT": "open_interest",
    "FH_CHANGE_IN_OI": "change_in_oi",
    "FH_UNDERLYING_VALUE": "underlying_value",
}


def _parse_expiry(value) -> Optional[datetime]:
    """
    Parse an expiry date given as "YYYY-MM-DD" or NSE style "27-Oct-2026".

    Args:
        value: Date string.

    Returns:
        datetime, or None if the value is not a valid date.
    """
    text = str(value).strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        pass
    try:
        day, mon, year = text.split("-")
        return datetime(int(year), _MONTHS.index(mon.title()) + 1, int(day))
    except (ValueError, TypeError):
        return None


def _to_nse_expiry(value) -> Optional[str]:
    """
    Convert an expiry date to NSE's "27-Oct-2026" format.

    Args:
        value: "YYYY-MM-DD" or an NSE-style "DD-Mon-YYYY" string.

    Returns:
        NSE-format string, or None if the value is not a valid date.
    """
    parsed = _parse_expiry(value)
    if parsed is None:
        return None
    return f"{parsed.day:02d}-{_MONTHS[parsed.month - 1]}-{parsed.year}"


def _format_strike(value) -> Optional[str]:
    """
    Format a strike price for the NSE request.

    Args:
        value: Number (int, float or numeric string), must be > 0.

    Returns:
        "1250" for whole strikes, "1250.5" for fractional ones,
        or None if the value is not a positive finite number.
    """
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (number > 0) or number == float("inf"):
        return None
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _fetch_fno_chunk(session, symbol: str, from_date: str, to_date: str,
                     expiry_date: str, instrument_type: str,
                     strike_price: str, option_type: str
                     ) -> Tuple[pd.DataFrame, str]:
    """
    Internal: fetch one request from NSE's F&O historical API.

    Args:
        session: Warmed-up NSE requests session.
        symbol: Underlying symbol.
        from_date: DD-MM-YYYY.
        to_date: DD-MM-YYYY.
        expiry_date: NSE format, or "" for all expiries.
        instrument_type: FUTSTK, OPTSTK, FUTIDX or OPTIDX.
        strike_price: Formatted strike, or "" for all strikes.
        option_type: "CE", "PE" or "".

    Returns:
        (DataFrame, status). status is "ok", "empty" or "error", so the
        caller can tell "no data" apart from "request failed". The frame is
        not filtered, so its length is the raw row count used for the cap
        check.
    """
    params = {
        "functionName": "getDerivativesHistoricalData",
        "symbol": symbol,
        "instrumentType": instrument_type,
        "year": "",
        "expiryDate": expiry_date,
        "strikePrice": strike_price,
        "optionType": option_type,
        "fromDate": from_date,
        "toDate": to_date,
    }
    try:
        response = session.get(_HIST_URL, params=params, timeout=30)
        if response.status_code != 200:
            logger.error(f"[FNO] HTTP {response.status_code} for "
                         f"{symbol} {from_date}..{to_date}")
            return empty_frame("fno"), "error"

        payload = response.json()
        if isinstance(payload, dict):
            if "data" not in payload:
                logger.error(f"[FNO] Unexpected response shape for {symbol}")
                return empty_frame("fno"), "error"
            data = payload["data"]
        else:
            data = payload
        if not isinstance(data, list):
            logger.error(f"[FNO] Unexpected response shape for {symbol}")
            return empty_frame("fno"), "error"
        if not data:
            return empty_frame("fno"), "empty"

        df = pd.DataFrame(data).rename(columns=_RENAME)
        df["source"] = "NSE_FO"
        df = df[columns_of("fno")].copy()

        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y",
                                    errors="coerce").astype("datetime64[ns]")
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], format="%d-%b-%Y",
                                           errors="coerce").astype("datetime64[ns]")
        for col in ["strike_price", "open", "high", "low", "close",
                    "settle_price", "underlying_value"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        for col in ["volume", "value", "open_interest", "change_in_oi"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        # pandas 3 uses a "str" dtype for text; the schema says "object".
        df = df.astype(SCHEMAS["fno"])
        return df, "ok"

    except Exception as e:
        logger.error(f"[FNO] Chunk fetch error: {e}")
        return empty_frame("fno"), "error"


def _fetch_range(session, symbol, start, end, expiry, instrument,
                 strike, opt_type, state) -> Optional[List[pd.DataFrame]]:
    """
    Internal: fetch [start, end], splitting the range whenever a response
    reaches the row cap, so no data is silently dropped.

    Args:
        session: NSE session.
        symbol: Underlying symbol.
        start, end: datetime range, inclusive.
        expiry, instrument, strike, opt_type: request filters (NSE format).
        state: dict with key "first" used to throttle requests.

    Returns:
        List of DataFrames (possibly empty), or None if completeness
        cannot be guaranteed (request error, or one day hits the cap).
    """
    if not state["first"]:
        time.sleep(_REQUEST_DELAY)
    state["first"] = False

    df, status = _fetch_fno_chunk(
        session, symbol, start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y"),
        expiry, instrument, strike, opt_type)

    if status == "error":
        logger.error(f"[FNO] Request failed for {start.date()} to {end.date()}")
        return None
    if status == "empty":
        return []
    if len(df) < NSE_ROW_CAP:
        return [df]

    if start >= end:
        logger.error(
            f"[FNO] {symbol} on {start.date()} returned {len(df)} rows, at "
            f"NSE's {NSE_ROW_CAP}-row cap, so the data may be incomplete. "
            f"Narrow the request with expiry_date, strike_price or option_type.")
        return None

    mid = start + timedelta(days=(end - start).days // 2)
    left = _fetch_range(session, symbol, start, mid, expiry, instrument,
                        strike, opt_type, state)
    if left is None:
        return None
    right = _fetch_range(session, symbol, mid + timedelta(days=1), end, expiry,
                         instrument, strike, opt_type, state)
    if right is None:
        return None
    return left + right


def fetch_fno_data(symbol: str, from_date: str, to_date: str,
                   expiry_date: Optional[str] = None,
                   instrument_type: str = "FUTSTK", *,
                   strike_price=None,
                   option_type: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch historical F&O data from NSE without silent truncation.

    Requests that reach NSE's 70-row cap are split until every piece fits.
    If completeness cannot be guaranteed (a request failed, or a single day
    still hits the cap) an empty frame is returned and an error is logged.

    Args:
        symbol: Underlying, e.g. "RELIANCE" or "BANKNIFTY".
        from_date: YYYY-MM-DD.
        to_date: YYYY-MM-DD.
        expiry_date: Optional. "YYYY-MM-DD" or NSE style "27-Oct-2026".
            Past expiries are allowed (NSE keeps their history). If None,
            all contracts listed on each day are returned.
        instrument_type: "FUTSTK" (default), "OPTSTK", "FUTIDX" or "OPTIDX".
        strike_price: Optional strike (number). Required for options.
        option_type: Optional "CE" or "PE" (case-insensitive). Required
            for options, not allowed for futures.

    Returns:
        pandas DataFrame with columns: date, symbol, instrument_type,
        expiry_date, option_type, strike_price, open, high, low, close,
        settle_price, volume, value, open_interest, change_in_oi,
        underlying_value, source. Futures rows carry option_type "XX" and
        strike_price 0 as NSE reports them. Empty frame on any failure.
    """
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        logger.error("[FNO] Invalid date format. Use YYYY-MM-DD")
        return empty_frame("fno")
    if start > end:
        logger.error("[FNO] from_date cannot be after to_date")
        return empty_frame("fno")

    symbol = str(symbol).strip().upper()
    if not symbol:
        logger.error("[FNO] symbol is required")
        return empty_frame("fno")

    instrument = str(instrument_type).strip().upper()
    if instrument not in VALID_INSTRUMENTS:
        logger.error(f"[FNO] Invalid instrument_type {instrument_type!r}. "
                     f"Use one of {VALID_INSTRUMENTS}")
        return empty_frame("fno")
    is_option = instrument.startswith("OPT")

    opt = ""
    if option_type is not None:
        opt = str(option_type).strip().upper()
        if opt not in ("CE", "PE"):
            logger.error(f"[FNO] Invalid option_type {option_type!r}. Use CE or PE")
            return empty_frame("fno")

    strike = ""
    if strike_price is not None:
        strike = _format_strike(strike_price)
        if strike is None:
            logger.error(f"[FNO] Invalid strike_price {strike_price!r}")
            return empty_frame("fno")

    if is_option and (not opt or not strike):
        logger.error(
            "[FNO] Options need both strike_price and option_type: without "
            "them one day exceeds NSE's row cap and data would be lost.")
        return empty_frame("fno")
    if not is_option and (opt or strike):
        logger.warning("[FNO] strike_price/option_type only apply to options; "
                       "request rejected.")
        return empty_frame("fno")

    expiry = ""
    if expiry_date is not None:
        expiry = _to_nse_expiry(expiry_date)
        if expiry is None:
            logger.error(f"[FNO] Invalid expiry_date {expiry_date!r}. "
                         f"Use YYYY-MM-DD or e.g. 27-Oct-2026")
            return empty_frame("fno")

    logger.info(f"[FNO] Fetching {symbol} ({instrument}) "
                f"from {from_date} to {to_date}...")
    try:
        session = create_nse_session()
    except Exception as e:
        logger.error(f"[FNO] Could not create NSE session: {e}")
        return empty_frame("fno")

    # A single contract fits ~43 rows per 60 days. With every expiry
    # returned, use smaller starting chunks; the cap check splits further.
    chunk_days = 60 if expiry else 20
    state = {"first": True}
    frames: List[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        part = _fetch_range(session, symbol, chunk_start, chunk_end, expiry,
                            instrument, strike, opt, state)
        if part is None:
            logger.error(f"[FNO] {symbol}: result could not be guaranteed "
                         f"complete, returning no data.")
            return empty_frame("fno")
        frames.extend(part)
        chunk_start = chunk_end + timedelta(days=1)

    if not frames:
        logger.warning(f"[FNO] No data returned for {symbol}")
        return empty_frame("fno")

    df = pd.concat(frames, ignore_index=True)

    # Safety net: keep only rows that match what was asked for.
    keep = df["instrument_type"] == instrument
    if opt:
        keep &= df["option_type"] == opt
    if strike:
        keep &= (df["strike_price"] - float(strike)).abs() < 1e-6
    if expiry:
        keep &= df["expiry_date"] == pd.to_datetime(expiry, format="%d-%b-%Y")
    dropped = int((~keep).sum())
    if dropped:
        logger.warning(f"[FNO] Dropped {dropped} rows that did not match the "
                       f"requested filters.")
        df = df[keep]

    df = df.drop_duplicates(subset=["date", "expiry_date", "option_type", "strike_price"])
    df = df.sort_values(["date", "expiry_date", "option_type", "strike_price"])
    df = df.reset_index(drop=True)
    if df.empty:
        return empty_frame("fno")

    logger.info(f"[FNO] {symbol}: {len(df)} rows fetched")
    return df


def list_expiries(symbol: str) -> List[str]:
    """
    List the currently listed F&O expiry dates for a symbol.

    Only live contracts are listed. Historical data does not need this:
    call fetch_fno_data with expiry_date=None to get past contracts.

    Args:
        symbol: Underlying, e.g. "RELIANCE" or "BANKNIFTY".

    Returns:
        Sorted list of expiry dates as "YYYY-MM-DD". Empty list on failure.
    """
    symbol = str(symbol).strip().upper()
    if not symbol:
        logger.error("[FNO] symbol is required")
        return []
    try:
        session = create_nse_session()
        response = session.get(_INFO_URL, params={"symbol": symbol}, timeout=30)
        if response.status_code != 200:
            logger.error(f"[FNO] HTTP {response.status_code} listing expiries "
                         f"for {symbol}")
            return []
        payload = response.json()
        raw = payload.get("expiryDates") if isinstance(payload, dict) else None
        if not isinstance(raw, list) or not raw:
            logger.warning(f"[FNO] No expiries listed for {symbol}")
            return []
        out = set()
        for item in raw:
            parsed = _parse_expiry(item)
            if parsed is None:
                logger.warning(f"[FNO] Skipping unparseable expiry {item!r}")
                continue
            out.add(parsed.strftime("%Y-%m-%d"))
        return sorted(out)
    except Exception as e:
        logger.error(f"[FNO] Error listing expiries for {symbol}: {e}")
        return []