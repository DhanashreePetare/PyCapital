"""
NSE Derivatives (F&O) historical data connector for PyCapital.

Endpoint and real field names confirmed via DevTools + discovery run
on 2026-09-25. Handles NSE's ~70-row-per-request cap via chunking,
same pattern as nse.py / indices.py.
"""

import pandas as pd
import time
from datetime import datetime, timedelta
from connectors._session_utils import create_nse_session


def _fetch_fno_chunk(session, symbol, from_date, to_date, expiry_date, instrument_type):
    """
    Internal — fetches one chunk from NSE's F&O historical API.
    from_date/to_date must be DD-MM-YYYY.
    """
    columns = [
        "date", "symbol", "instrument_type", "expiry_date",
        "option_type", "strike_price", "open", "high", "low", "close",
        "settle_price", "volume", "value", "open_interest",
        "change_in_oi", "underlying_value", "source"
    ]

    url = "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
    params = {
        "functionName": "getDerivativesHistoricalData",
        "symbol": symbol,
        "instrumentType": instrument_type,
        "year": "",
        "expiryDate": expiry_date,
        "strikePrice": "",
        "optionType": "",
        "fromDate": from_date,
        "toDate": to_date
    }

    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()
        data = payload.get("data", []) if isinstance(payload, dict) else payload

        if not data:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(data)

        df = df.rename(columns={
            "FH_TIMESTAMP":        "date",
            "FH_SYMBOL":           "symbol",
            "FH_INSTRUMENT":       "instrument_type",
            "FH_EXPIRY_DT":        "expiry_date",
            "FH_OPTION_TYPE":      "option_type",
            "FH_STRIKE_PRICE":     "strike_price",
            "FH_OPENING_PRICE":    "open",
            "FH_TRADE_HIGH_PRICE": "high",
            "FH_TRADE_LOW_PRICE":  "low",
            "FH_CLOSING_PRICE":    "close",
            "FH_SETTLE_PRICE":     "settle_price",
            "FH_TOT_TRADED_QTY":   "volume",
            "FH_TOT_TRADED_VAL":   "value",
            "FH_OPEN_INT":         "open_interest",
            "FH_CHANGE_IN_OI":     "change_in_oi",
            "FH_UNDERLYING_VALUE": "underlying_value"
        })

        df["source"] = "NSE_FO"
        df = df[columns[:-1] + ["source"]].copy() if all(
            c in df.columns for c in columns if c != "source"
        ) else df

        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], format="%d-%b-%Y", errors="coerce")

        for col in ["strike_price", "open", "high", "low", "close", "settle_price", "underlying_value"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        for col in ["volume", "value", "open_interest", "change_in_oi"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

        return df

    except Exception as e:
        print(f"[FNO] Chunk fetch error: {e}")
        return pd.DataFrame(columns=columns)


def fetch_fno_data(symbol, from_date, to_date, expiry_date, instrument_type="FUTSTK"):
    """
    Fetch historical F&O data from NSE, handling NSE's ~70-row cap
    automatically via chunking.

    Args:
        symbol          : e.g. "RELIANCE"
        from_date       : YYYY-MM-DD
        to_date         : YYYY-MM-DD
        expiry_date     : NSE format, e.g. "27-Oct-2026" — must match
                          a currently-listed expiry (check the site's
                          dropdown, don't guess an expired date)
        instrument_type : "FUTSTK" (default, stock futures) or
                          "OPTSTK" (stock options). Always pass this
                          explicitly — leaving it blank lets NSE
                          silently return a different instrument type
                          than you asked for.

    Returns:
        pandas DataFrame with columns:
        date, symbol, instrument_type, expiry_date, option_type,
        strike_price, open, high, low, close, settle_price, volume,
        value, open_interest, change_in_oi, underlying_value, source
    """
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end   = datetime.strptime(to_date,   "%Y-%m-%d")
    except ValueError:
        print(f"[FNO] Invalid date format. Use YYYY-MM-DD")
        return pd.DataFrame()

    if start > end:
        print(f"[FNO] from_date cannot be after to_date")
        return pd.DataFrame()

    print(f"[FNO] Fetching {symbol} ({instrument_type}) from {from_date} to {to_date}...")
    session = create_nse_session()

    CHUNK_DAYS = 60
    all_chunks = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)

        fmt_from = chunk_start.strftime("%d-%m-%Y")
        fmt_to   = chunk_end.strftime("%d-%m-%Y")

        print(f"[FNO] Fetching chunk: {fmt_from} to {fmt_to}...")
        chunk_df = _fetch_fno_chunk(session, symbol, fmt_from, fmt_to, expiry_date, instrument_type)

        if not chunk_df.empty:
            all_chunks.append(chunk_df)

        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= end:
            time.sleep(1)

    if not all_chunks:
        print(f"[FNO] No data returned for {symbol}")
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "expiry_date", "option_type", "strike_price"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[FNO] ✅ {symbol} — {len(df)} rows fetched")
    return df