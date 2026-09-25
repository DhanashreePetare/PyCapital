import requests
import pandas as pd
import json


def fetch_bse_historical(scripcode: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a stock from BSE.

    Args:
        scripcode : BSE stock code e.g. "500325" for Reliance
        from_date : Start date in YYYY-MM-DD format e.g. "2026-08-24"
        to_date   : End date   in YYYY-MM-DD format e.g. "2026-09-24"

    Returns:
        pandas DataFrame with columns:
        date, symbol, open, high, low, close, volume, trades, source
    """

    from datetime import datetime

    # Convert YYYY-MM-DD to DD/MM/YYYY for BSE API
    fmt_from = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    fmt_to   = datetime.strptime(to_date,   "%Y-%m-%d").strftime("%d/%m/%Y")

    headers = {
        "User-Agent":      "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Mobile Safari/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin":          "https://www.bseindia.com",
        "Referer":         "https://www.bseindia.com/",
        "sec-fetch-dest":  "empty",
        "sec-fetch-mode":  "cors",
        "sec-fetch-site":  "same-site"
    }

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

    try:
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"[BSE] Error: status code {response.status_code} for {scripcode}")
            return pd.DataFrame()

        data = response.json()

        # Extract StockData
        records = data.get("StockData")

        if not records:
            print(f"[BSE] No records found for scripcode {scripcode}")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Keep only needed columns
        df = df[[
            "RealDate",
            "qe_open",
            "qe_high",
            "qe_low",
            "qe_close",
            "no_of_shrs",
            "no_trades",
            "scrip_id" if "scrip_id" in df.columns else "qe_close"
        ]].copy() if "scrip_id" in df.columns else df[[
            "RealDate",
            "qe_open",
            "qe_high",
            "qe_low",
            "qe_close",
            "no_of_shrs",
            "no_trades"
        ]].copy()

        # Rename to standard column names
        df = df.rename(columns={
            "RealDate":   "date",
            "qe_open":    "open",
            "qe_high":    "high",
            "qe_low":     "low",
            "qe_close":   "close",
            "no_of_shrs": "volume",
            "no_trades":  "trades"
        })

        # Fix date
        df["date"] = pd.to_datetime(df["date"])

        # Remove commas from number strings and convert to proper types
        def clean_number(val):
            if isinstance(val, str):
                return val.replace(",", "")
            return val

        df["open"]   = df["open"].apply(clean_number).astype(float)
        df["high"]   = df["high"].apply(clean_number).astype(float)
        df["low"]    = df["low"].apply(clean_number).astype(float)
        df["close"]  = df["close"].apply(clean_number).astype(float)
        df["volume"] = df["volume"].apply(clean_number).astype(float).astype(int)
        df["trades"] = df["trades"].apply(clean_number).astype(float).astype(int)

        # Add identifier columns
        symbol = data.get("scrip_id", scripcode)
        df["symbol"] = symbol
        df["source"] = "BSE"

        # Sort oldest to newest
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except Exception as e:
        print(f"[BSE] Exception for {scripcode}: {e}")
        return pd.DataFrame()