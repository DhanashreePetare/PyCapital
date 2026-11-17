"""Capture real NSE prices around corporate-action ex-dates.

Validation data for processing/adjust.py. Run manually from the repo root:
    python tools/capture_adjust_cases.py

Saves CSVs to tests/unit/golden/adjust/ and prints, for each case, the price
move across the ex-date before and after applying the expected factor. A
correct case shows a large raw move and a small adjusted move.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pyequity.connectors.nse import fetch_nse_historical

OUT = Path(__file__).resolve().parents[1] / "tests" / "unit" / "golden" / "adjust"
DAYS = 12
# (symbol, ex-date, expected price factor)
CASES = [
    ("RELIANCE", "2024-10-28", 0.5),      # bonus 1:1
    ("HDFCBANK", "2019-09-19", 0.5),      # split Rs 2 -> Re 1
    ("INFY", "2018-09-04", 0.5),          # bonus 1:1
    ("BAJFINANCE", "2025-06-16", 0.1),    # bonus 4:1 and split 2 -> 1 on one day
]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    for symbol, ex, factor in CASES:
        ex_ts = pd.Timestamp(ex)
        start = (datetime.strptime(ex, "%Y-%m-%d") - timedelta(days=DAYS)).strftime("%Y-%m-%d")
        end = (datetime.strptime(ex, "%Y-%m-%d") + timedelta(days=DAYS)).strftime("%Y-%m-%d")
        df = fetch_nse_historical(symbol, start, end)
        if df.empty or not (df["date"] == ex_ts).any():
            print(f"FAIL  {symbol} {ex}: no data, or the ex-date is not a trading day in it")
            continue
        path = OUT / f"case_{symbol}_{ex}.csv"
        df.to_csv(path, index=False)
        prev = df.loc[df["date"] < ex_ts, "close"].iloc[-1]
        on = df.loc[df["date"] >= ex_ts, "close"].iloc[0]
        print(f"OK    {symbol} {ex}: {len(df)} rows | raw move {on / prev - 1:+.1%} | "
              f"adjusted move {on / (prev * factor) - 1:+.1%} -> {path.name}")


if __name__ == "__main__":
    main()