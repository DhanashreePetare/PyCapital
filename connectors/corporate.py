import pandas as pd
from datetime import datetime
from connectors._session_utils import create_nse_session


def fetch_corporate_actions(symbol):
    """
    Fetch corporate actions for a given symbol from NSE.

    Args:
        symbol: NSE stock symbol, e.g. "RELIANCE"

    Returns:
        pandas DataFrame with columns:
        ex_date, symbol, action_type, action_detail, record_date, source
    """
    columns = [
        "ex_date", "symbol", "action_type",
        "action_detail", "record_date", "source"
    ]

    try:
        session = create_nse_session()

        url = "https://www.nseindia.com/api/corporates-corporateActions"
        params = {"index": "equities", "symbol": symbol}

        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        if not data:
            print(f"[Corporate] No actions found for {symbol}")
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(data)

        df = df.rename(columns={
            "exDate": "ex_date",
            "subject": "action_detail",
            "recDate": "record_date"
        })

        df["action_type"] = df["action_detail"].apply(extract_action_type)

        # Force the symbol column to what was requested, not the raw
        # API field, so it's guaranteed consistent for cross-source joins
        df["symbol"] = symbol.upper()
        df["source"] = "NSE"

        df = df[columns]

        df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce")
        df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")

        return df

    except Exception as e:
        print(f"[Corporate] Error fetching actions for {symbol}: {e}")
        return pd.DataFrame(columns=columns)


def extract_action_type(action_detail):
    """Extract a simple action type from NSE's subject field."""
    if pd.isna(action_detail):
        return ""
    action_detail = str(action_detail).upper()
    if "DIVIDEND" in action_detail:
        return "DIVIDEND"
    if "BONUS" in action_detail:
        return "BONUS"
    if "SPLIT" in action_detail:
        return "SPLIT"
    if "RIGHT" in action_detail:
        return "RIGHTS"
    if "BUYBACK" in action_detail:
        return "BUYBACK"
    if "MERGER" in action_detail:
        return "MERGER"
    return "OTHER"