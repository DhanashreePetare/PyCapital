import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pycapital.schema import empty_frame
from pycapital.connectors._session_utils import create_nse_session


def _fetch_nse_chunk(session: requests.Session, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Internal function — fetches one chunk from NSE API.
    from_date and to_date must be in DD-MM-YYYY format.
    Returns raw DataFrame or empty DataFrame on failure.
    """
    url = (
        f"https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
        f"?functionName=getHistoricalTradeData"
        f"&symbol={symbol}&series=EQ"
        f"&fromDate={from_date}&toDate={to_date}"
    )

    try:
        response = session.get(url, timeout=15)

        if response.status_code != 200:
            return empty_frame("equity")

        text = response.text.strip()
        if not text or text == "null" or text == "[]":
            return empty_frame("equity")

        data = json.loads(text)
        if not data:
            return empty_frame("equity")

        df = pd.DataFrame(data)

        df = df.rename(columns={
            "mtimestamp":       "date",
            "chOpeningPrice":   "open",
            "chTradeHighPrice": "high",
            "chTradeLowPrice":  "low",
            "chClosingPrice":   "close",
            "chTotTradedQty":   "volume",
            "chTotalTrades":    "trades"
        })

        df = df[["date", "open", "high", "low", "close", "volume", "trades"]].copy()

        df["date"]   = pd.to_datetime(df["date"], format="%d-%b-%Y")
        df["open"]   = df["open"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(int)
        df["trades"] = df["trades"].astype(int)

        return df

    except Exception as e:
        print(f"[NSE] Chunk fetch error: {e}")
        return empty_frame("equity")


def fetch_nse_historical(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a stock from NSE.
    Handles NSE's 70-row limit automatically by fetching in chunks.

    Args:
        symbol    : NSE stock symbol e.g. "RELIANCE", "TCS"
        from_date : Start date in YYYY-MM-DD format e.g. "2025-09-23"
        to_date   : End date   in YYYY-MM-DD format e.g. "2026-09-23"

    Returns:
        pandas DataFrame with columns:
        date, symbol, open, high, low, close, volume, trades, source
    """

    # Parse input dates
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end   = datetime.strptime(to_date,   "%Y-%m-%d")
    except ValueError:
        print(f"[NSE] Invalid date format. Use YYYY-MM-DD")
        return empty_frame("equity")

    if start > end:
        print(f"[NSE] from_date cannot be after to_date")
        return empty_frame("equity")

    # Create session once — reuse for all chunks
    print(f"[NSE] Fetching {symbol} from {from_date} to {to_date}...")
    session = create_nse_session(symbol)

    # NSE limit = 70 rows ≈ 60 trading days safely
    # We use 60-day chunks to stay well within limit
    CHUNK_DAYS = 60

    all_chunks = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)

        fmt_from = chunk_start.strftime("%d-%m-%Y")
        fmt_to   = chunk_end.strftime("%d-%m-%Y")

        print(f"[NSE] Fetching chunk: {fmt_from} to {fmt_to}...")

        chunk_df = _fetch_nse_chunk(session, symbol, fmt_from, fmt_to)

        if not chunk_df.empty:
            all_chunks.append(chunk_df)

        # Move to next chunk
        chunk_start = chunk_end + timedelta(days=1)

        # Small delay between chunks — be respectful to NSE servers
        if chunk_start <= end:
            time.sleep(1)

    if not all_chunks:
        print(f"[NSE] No data returned for {symbol}")
        return empty_frame("equity")

    # Combine all chunks
    df = pd.concat(all_chunks, ignore_index=True)

    # Remove any duplicate dates (overlap between chunks)
    df = df.drop_duplicates(subset=["date"])

    # Add identifier columns
    df["symbol"] = symbol.upper()
    df["source"] = "NSE"

    # Sort oldest to newest
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[NSE] ✅ {symbol} — {len(df)} rows fetched")

    return df