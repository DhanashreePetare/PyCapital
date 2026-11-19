"""
BSE equity historical (OHLCV) connector.

BSE returns at most 31 rows per request, so long ranges are fetched in 30-day
chunks (about 22 trading days each) and stitched together.

Policy: if any chunk still fails after retries, or a chunk comes back at the
row cap, the result cannot be guaranteed complete. The function then returns
an empty frame and logs an error. It never returns partial data and never
raises. An empty answer from BSE (for example an unknown scripcode) is not
retried, because retrying cannot change it.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import requests

from pyequity.connectors.symbol_map import get_bse_code
from pyequity.schema import SCHEMAS, empty_frame

logger = logging.getLogger(__name__)

BSE_ROW_CAP = 31
_CHUNK_DAYS = 30
_ATTEMPTS = 3
_RETRY_DELAY = 2.0   # seconds, multiplied by the attempt number
_CHUNK_DELAY = 1.0   # seconds between chunks, to be gentle on BSE

_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Mobile Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          "https://www.bseindia.com",
    "Referer":         "https://www.bseindia.com/",
    "sec-fetch-dest":  "empty",
    "sec-fetch-mode":  "cors",
    "sec-fetch-site":  "same-site",
}


def _clean_number(value):
    """BSE sends numbers as strings with thousands separators."""
    if isinstance(value, str):
        return value.replace(",", "")
    return value


def _request_chunk(scripcode: str, from_date: str, to_date: str
                   ) -> Tuple[pd.DataFrame, str]:
    """
    Internal: one request to BSE.

    Args:
        scripcode: BSE stock code, e.g. "500325".
        from_date: YYYY-MM-DD.
        to_date: YYYY-MM-DD.

    Returns:
        (DataFrame, status). status is "ok", "empty" (BSE answered with no
        rows) or "error" (the request or its parsing failed).
    """
    try:
        fmt_from = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d/%m/%Y")
        fmt_to = datetime.strptime(to_date, "%Y-%m-%d").strftime("%d/%m/%Y")

        url = (
            f"https://api.bseindia.com/BseIndiaAPI/api/StockpricesearchData/w"
            f"?MonthDate={fmt_from}"
            f"&YearDate={fmt_to}"
            f"&pageType=0"
            f"&Scode={scripcode}"
            f"&Seg=C"
            f"&rbType=D"
            f"&SortOrder=true"
        )

        response = requests.get(url, headers=_HEADERS, timeout=30)

        if response.status_code != 200:
            logger.error(f"[BSE] Error: status code {response.status_code} for {scripcode}")
            return empty_frame("equity"), "error"

        data = response.json()
        records = data.get("StockData")

        if not records:
            logger.warning(f"[BSE] No records found for scripcode {scripcode}")
            return empty_frame("equity"), "empty"

        df = pd.DataFrame(records)
        df = df[["RealDate", "qe_open", "qe_high", "qe_low", "qe_close",
                 "no_of_shrs", "no_trades"]].copy()
        df = df.rename(columns={
            "RealDate":   "date",
            "qe_open":    "open",
            "qe_high":    "high",
            "qe_low":     "low",
            "qe_close":   "close",
            "no_of_shrs": "volume",
            "no_trades":  "trades",
        })

        df["date"] = pd.to_datetime(df["date"])
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].apply(_clean_number).astype("float64")
        for col in ("volume", "trades"):
            df[col] = df[col].apply(_clean_number).astype("float64").astype("int64")

        df["symbol"] = data.get("scrip_id") or scripcode
        df["source"] = "BSE"
        df = df.sort_values("date").reset_index(drop=True)

        return df, "ok"

    except Exception as e:
        logger.error(f"[BSE] Exception for {scripcode}: {e}")
        return empty_frame("equity"), "error"


def _chunk_with_retry(scripcode: str, from_date: str, to_date: str
                      ) -> Tuple[pd.DataFrame, str]:
    """Internal: retry real failures a few times; never retry an empty answer."""
    df, status = _request_chunk(scripcode, from_date, to_date)
    for attempt in range(1, _ATTEMPTS):
        if status != "error":
            break
        time.sleep(_RETRY_DELAY * attempt)
        df, status = _request_chunk(scripcode, from_date, to_date)
    return df, status


def _fetch_bse_chunk(scripcode: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Internal: fetch one chunk and return just the DataFrame.

    Kept for tools and tests that call it directly. Returns an empty frame on
    any failure.
    """
    return _request_chunk(scripcode, from_date, to_date)[0]


def fetch_bse_historical(scripcode: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a stock from BSE.

    Long ranges are split into 30-day chunks that stay under BSE's 31-row cap.
    Failed requests are retried. If a chunk still fails, or reaches the cap,
    an empty frame is returned and an error is logged, so the result is never
    silently incomplete.

    Args:
        scripcode : BSE stock code e.g. "500325" for Reliance
        from_date : Start date in YYYY-MM-DD format
        to_date   : End date   in YYYY-MM-DD format

    Returns:
        pandas DataFrame in the standard equity schema:
        date, open, high, low, close, volume, trades, symbol, source.
        Sorted oldest to newest. Empty frame on any failure.
    """
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        logger.error("[BSE] Invalid date format. Use YYYY-MM-DD")
        return empty_frame("equity")

    if start > end:
        logger.error("[BSE] from_date cannot be after to_date")
        return empty_frame("equity")

    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end)
        label = f"{chunk_start.date()} to {chunk_end.date()}"

        chunk_df, status = _chunk_with_retry(
            scripcode,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )

        if status == "error":
            logger.error(f"[BSE] {scripcode}: chunk {label} failed after {_ATTEMPTS} "
                         f"attempts; returning no data instead of a partial result.")
            return empty_frame("equity")

        if len(chunk_df) >= BSE_ROW_CAP:
            logger.error(f"[BSE] {scripcode}: chunk {label} returned {len(chunk_df)} rows, "
                         f"at BSE's {BSE_ROW_CAP}-row cap, so it may be incomplete; "
                         f"returning no data.")
            return empty_frame("equity")

        if status == "ok":
            chunks.append(chunk_df)

        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= end:
            time.sleep(_CHUNK_DELAY)

    if not chunks:
        return empty_frame("equity")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[list(SCHEMAS["equity"])].astype(SCHEMAS["equity"])


def fetch_bse_historical_by_symbol(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch BSE historical data using an NSE symbol instead of a BSE scripcode.

    Args:
        symbol    : NSE stock symbol, e.g. "RELIANCE"
        from_date : Start date in YYYY-MM-DD format
        to_date   : End date   in YYYY-MM-DD format

    Returns:
        Same DataFrame as fetch_bse_historical(). If the symbol has no BSE
        mapping, an error is logged and an empty frame is returned; nothing
        is guessed and nothing raises.
    """
    try:
        scripcode = get_bse_code(symbol)
    except ValueError as e:
        logger.error(f"[BSE] {e}")
        return empty_frame("equity")
    return fetch_bse_historical(scripcode, from_date, to_date)