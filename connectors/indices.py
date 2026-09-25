import pandas as pd
import time
from datetime import datetime, timedelta
from connectors._session_utils import create_nse_session


def _fetch_indices_chunk(session, index_name, from_date, to_date):
    """
    Internal — fetches one chunk from NSE indices API.
    from_date/to_date must be DD-MM-YYYY.
    """
    columns = ["date", "index_name", "open", "high", "low", "close", "volume", "source"]

    url = "https://www.nseindia.com/api/historicalOR/indicesHistory"
    params = {"indexType": index_name, "from": from_date, "to": to_date}

    try:
        response = session.get(url, params=params, timeout=20)
        response.raise_for_status()

        data = response.json().get("data", [])
        if not data:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(data)

        df = df.rename(columns={
            "EOD_TIMESTAMP": "date",
            "EOD_OPEN_INDEX_VAL": "open",
            "EOD_HIGH_INDEX_VAL": "high",
            "EOD_LOW_INDEX_VAL": "low",
            "EOD_CLOSE_INDEX_VAL": "close",
            "HIT_TRADED_QTY": "volume"
        })

        df["index_name"] = index_name
        df["source"] = "NSE"
        df = df[columns].copy()

        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

        return df

    except Exception as e:
        print(f"[Indices] Chunk fetch error: {e}")
        return pd.DataFrame(columns=columns)


def fetch_index_history(index_name, from_date, to_date):
    """
    Fetch historical index data from NSE.
    Handles NSE's ~70-row limit automatically by fetching in chunks,
    same pattern as fetch_nse_historical() in nse.py.

    Args:
        index_name : e.g. "NIFTY 50"
        from_date  : YYYY-MM-DD
        to_date    : YYYY-MM-DD

    Returns:
        pandas DataFrame with columns:
        date, index_name, open, high, low, close, volume, source
    """
    columns = ["date", "index_name", "open", "high", "low", "close", "volume", "source"]

    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end   = datetime.strptime(to_date,   "%Y-%m-%d")
    except ValueError:
        print(f"[Indices] Invalid date format. Use YYYY-MM-DD")
        return pd.DataFrame(columns=columns)

    if start > end:
        print(f"[Indices] from_date cannot be after to_date")
        return pd.DataFrame(columns=columns)

    print(f"[Indices] Fetching {index_name} from {from_date} to {to_date}...")
    session = create_nse_session()

    CHUNK_DAYS = 60
    all_chunks = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)

        fmt_from = chunk_start.strftime("%d-%m-%Y")
        fmt_to   = chunk_end.strftime("%d-%m-%Y")

        print(f"[Indices] Fetching chunk: {fmt_from} to {fmt_to}...")
        chunk_df = _fetch_indices_chunk(session, index_name, fmt_from, fmt_to)

        if not chunk_df.empty:
            all_chunks.append(chunk_df)

        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= end:
            time.sleep(1)

    if not all_chunks:
        print(f"[Indices] No data returned for {index_name}")
        return pd.DataFrame(columns=columns)

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[Indices] ✅ {index_name} — {len(df)} rows fetched")
    return df