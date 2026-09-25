import requests
import pandas as pd
import json
import time


def fetch_nse_historical(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a stock from NSE.

    Args:
        symbol    : NSE stock symbol e.g. "RELIANCE", "TCS"
        from_date : Start date in YYYY-MM-DD format e.g. "2026-08-23"
        to_date   : End date   in YYYY-MM-DD format e.g. "2026-09-23"

    Returns:
        pandas DataFrame with columns:
        date, symbol, open, high, low, close, volume, trades, source
    """

    # Convert YYYY-MM-DD to DD-MM-YYYY for NSE API
    from datetime import datetime
    fmt_from = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d-%m-%Y")
    fmt_to   = datetime.strptime(to_date,   "%Y-%m-%d").strftime("%d-%m-%Y")

    session = requests.Session()
    session.headers.update({
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         f"https://www.nseindia.com/get-quote/equity/{symbol}",
        "sec-fetch-dest":  "empty",
        "sec-fetch-mode":  "cors",
        "sec-fetch-site":  "same-origin",
        "Connection":      "keep-alive"
    })

    # Step 1 — Warm up session (get cookies)
    session.get("https://www.nseindia.com", timeout=15)
    time.sleep(3)

    # Step 2 — Visit stock page (strengthens session)
    session.get(f"https://www.nseindia.com/get-quote/equity/{symbol}", timeout=15)
    time.sleep(2)

    # Step 3 — Fetch historical data
    url = (
        f"https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
        f"?functionName=getHistoricalTradeData"
        f"&symbol={symbol}&series=EQ"
        f"&fromDate={fmt_from}&toDate={fmt_to}"
    )

    try:
        response = session.get(url, timeout=15)

        if response.status_code != 200:
            print(f"[NSE] Error: status code {response.status_code} for {symbol}")
            return pd.DataFrame()

        text = response.text.strip()

        if not text or text == "null" or text == "[]":
            print(f"[NSE] Empty response for {symbol}")
            return pd.DataFrame()

        data = json.loads(text)

        if not data:
            print(f"[NSE] No data returned for {symbol}")
            return pd.DataFrame()

        # Build DataFrame
        df = pd.DataFrame(data)

        # Rename NSE internal column names to clean names
        df = df.rename(columns={
            "mtimestamp":       "date",
            "chOpeningPrice":   "open",
            "chTradeHighPrice": "high",
            "chTradeLowPrice":  "low",
            "chClosingPrice":   "close",
            "chTotTradedQty":   "volume",
            "chTotalTrades":    "trades"
        })

        # Keep only needed columns
        df = df[["date", "open", "high", "low", "close", "volume", "trades"]]

        # Fix data types
        df["date"]   = pd.to_datetime(df["date"], format="%d-%b-%Y")
        df["open"]   = df["open"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(int)
        df["trades"] = df["trades"].astype(int)

        # Add identifier columns
        df["symbol"] = symbol.upper()
        df["source"] = "NSE"

        # Sort oldest to newest
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except Exception as e:
        print(f"[NSE] Exception for {symbol}: {e}")
        return pd.DataFrame()