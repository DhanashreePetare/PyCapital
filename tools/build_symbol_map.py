"""Builds the NSE symbol -> BSE scripcode map by joining the official
NSE equity list and BSE Bhav Copy (Equity with ISIN) files on ISIN.
Run manually; not part of the library.

Usage: python tools/build_symbol_map.py <EQUITY_L.csv> "<bse_files_glob>"
"""
import glob
import sys
import pandas as pd

NSE_SYMBOL, NSE_SERIES, NSE_ISIN = "SYMBOL", "SERIES", "ISIN NUMBER"
BSE_CODE, BSE_ISIN = "FinInstrmId", "ISIN"

nse = pd.read_csv(sys.argv[1])
nse.columns = nse.columns.str.strip()

files = glob.glob(sys.argv[2])
print(f"BSE files found: {len(files)}")
bse = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
bse.columns = bse.columns.str.strip()
bse = bse.dropna(subset=[BSE_CODE, BSE_ISIN]).drop_duplicates(subset=[BSE_CODE])

nse = nse[nse[NSE_SERIES].astype(str).str.strip() == "EQ"].copy()
nse["isin"] = nse[NSE_ISIN].astype(str).str.strip().str.upper()
bse["isin"] = bse[BSE_ISIN].astype(str).str.strip().str.upper()

m = nse.merge(bse, on="isin", how="inner")
m["symbol"] = m[NSE_SYMBOL].astype(str).str.strip().str.upper()
m["code"] = m[BSE_CODE].astype(int).astype(str)

# Never guess: drop anything ambiguous and report it.
amb = pd.concat([m[m.duplicated("symbol", keep=False)],
                 m[m.duplicated("code", keep=False)]])
amb = amb[~amb.index.duplicated()]
m = m.drop(amb.index).sort_values("symbol")
print(f"NSE EQ: {len(nse)} | BSE rows: {len(bse)} | matched: {len(m)} | ambiguous dropped: {len(amb)}")
if len(amb):
    print(amb[["symbol", "code", "isin"]].to_string(index=False))

# Safety check: the 15 mappings we already use must be unchanged.
from pyequity.connectors.symbol_map import SYMBOL_TO_BSE as OLD
new = dict(zip(m["symbol"], m["code"]))
for s, c in OLD.items():
    if new.get(s) != c:
        print(f"MISMATCH for {s}: old={c} new={new.get(s)}")

lines = ["SYMBOL_TO_BSE = {"] + [f'    "{s}": "{c}",' for s, c in new.items()] + ["}"]
open("generated_map.txt", "w").write("\n".join(lines))
print("written generated_map.txt")