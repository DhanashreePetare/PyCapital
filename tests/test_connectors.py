import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from connectors.nse import fetch_nse_historical
from connectors.bse import fetch_bse_historical

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def validate_df(df, source, symbol):
    """Validates a DataFrame has correct structure and types."""
    if df.empty:
        print(f"  ❌ {source} {symbol} — Empty DataFrame")
        return False

    required_cols = ["date", "open", "high", "low", "close", "volume", "trades", "symbol", "source"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ❌ {source} {symbol} — Missing columns: {missing}")
        return False

    # Check data types
    errors = []
    if str(df["date"].dtype) != "datetime64[ns]":
        errors.append(f"date should be datetime64[ns], got {df['date'].dtype}")
    if str(df["open"].dtype) != "float64":
        errors.append(f"open should be float64, got {df['open'].dtype}")
    if str(df["volume"].dtype) != "int64":
        errors.append(f"volume should be int64, got {df['volume'].dtype}")

    if errors:
        for e in errors:
            print(f"  ❌ Type error: {e}")
        return False

    # Check no null values in critical columns
    nulls = df[["date","open","high","low","close","volume"]].isnull().sum()
    if nulls.any():
        print(f"  ⚠️  Null values found: {nulls[nulls > 0].to_dict()}")

    # Check high >= low always
    invalid = df[df["high"] < df["low"]]
    if not invalid.empty:
        print(f"  ❌ {len(invalid)} rows where high < low — data corruption")
        return False

    # Check close is between low and high
    invalid2 = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
    if not invalid2.empty:
        print(f"  ⚠️  {len(invalid2)} rows where close is outside high-low range")

    # Check dates are sorted
    if not df["date"].is_monotonic_increasing:
        print(f"  ⚠️  Dates are not sorted in ascending order")

    # Check no duplicate dates
    dupes = df["date"].duplicated().sum()
    if dupes > 0:
        print(f"  ❌ {dupes} duplicate dates found")
        return False

    print(f"  ✅ {source} {symbol} — {len(df)} rows — {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"     close range: {df['close'].min()} to {df['close'].max()}")
    print(f"     columns: {list(df.columns)}")
    return True


# ============================================================
# TEST 1 — Basic fetch — Reliance 1 month
# ============================================================
print_header("TEST 1 — Basic Fetch — RELIANCE 1 month")

df_nse = fetch_nse_historical("RELIANCE", "2026-08-23", "2026-09-23")
validate_df(df_nse, "NSE", "RELIANCE")

df_bse = fetch_bse_historical("500325", "2026-08-23", "2026-09-23")
validate_df(df_bse, "BSE", "RELIANCE")


# ============================================================
# TEST 2 — Long date range — 6 months
# ============================================================
print_header("TEST 2 — Long Range — RELIANCE 6 months")

df_nse_6m = fetch_nse_historical("RELIANCE", "2026-03-01", "2026-09-23")
nse_rows = len(df_nse_6m) if not df_nse_6m.empty else 0
print(f"  NSE 6 months — got {nse_rows} rows")
if nse_rows > 0:
    print(f"  Date range: {df_nse_6m['date'].min().date()} to {df_nse_6m['date'].max().date()}")
    expected_roughly = 120  # ~6 months trading days
    if nse_rows < expected_roughly:
        print(f"  ⚠️  WARNING — Expected ~{expected_roughly} rows for 6 months but got {nse_rows}")
        print(f"  ⚠️  NSE may have a limit on rows per request — PAGINATION NEEDED")
    else:
        print(f"  ✅ 6 month range works fine")


# ============================================================
# TEST 3 — Very long date range — 1 year
# ============================================================
print_header("TEST 3 — Long Range — RELIANCE 1 year")

df_nse_1y = fetch_nse_historical("RELIANCE", "2025-09-23", "2026-09-23")
nse_rows_1y = len(df_nse_1y) if not df_nse_1y.empty else 0
print(f"  NSE 1 year — got {nse_rows_1y} rows")
if nse_rows_1y > 0:
    print(f"  Date range returned: {df_nse_1y['date'].min().date()} to {df_nse_1y['date'].max().date()}")
    expected_1y = 240  # ~1 year trading days
    if nse_rows_1y < expected_1y:
        print(f"  ⚠️  WARNING — Expected ~{expected_1y} rows for 1 year but got {nse_rows_1y}")
        print(f"  ⚠️  NSE IS LIMITING DATA — PAGINATION NEEDS TO BE BUILT")
    else:
        print(f"  ✅ 1 year range works fine — no pagination needed")


# ============================================================
# TEST 4 — Multiple stocks
# ============================================================
print_header("TEST 4 — Multiple Stocks")

stocks = [
    ("TCS",        "532540"),
    ("HDFCBANK",   "500180"),
    ("INFY",       "500209"),
    ("SBIN",       "500112"),
    ("BAJFINANCE", "500034"),
]

for nse_sym, bse_code in stocks:
    df_n = fetch_nse_historical(nse_sym, "2026-08-01", "2026-09-23")
    df_b = fetch_bse_historical(bse_code, "2026-08-01", "2026-09-23")
    nse_ok = validate_df(df_n, "NSE", nse_sym)
    bse_ok = validate_df(df_b, "BSE", nse_sym)

    # Row count consistency check
    if nse_ok and bse_ok:
        diff = abs(len(df_n) - len(df_b))
        if diff > 2:
            print(f"  ⚠️  Row count mismatch — NSE:{len(df_n)} BSE:{len(df_b)} diff:{diff}")


# ============================================================
# TEST 5 — Error handling — wrong symbol
# ============================================================
print_header("TEST 5 — Error Handling — Wrong Symbol")

df_bad = fetch_nse_historical("WRONGSYMBOL123", "2026-08-01", "2026-09-23")
if df_bad.empty:
    print("  ✅ NSE correctly returned empty DataFrame for wrong symbol")
else:
    print(f"  ❌ NSE returned data for invalid symbol — something is wrong")

df_bad_bse = fetch_bse_historical("999999", "2026-08-01", "2026-09-23")
if df_bad_bse.empty:
    print("  ✅ BSE correctly returned empty DataFrame for wrong scripcode")
else:
    print(f"  ❌ BSE returned data for invalid scripcode — something is wrong")


# ============================================================
# TEST 6 — Edge case — very short range (1 week)
# ============================================================
print_header("TEST 6 — Edge Case — 1 Week Range")

df_week = fetch_nse_historical("RELIANCE", "2026-09-15", "2026-09-23")
if not df_week.empty:
    print(f"  ✅ NSE 1 week — {len(df_week)} rows")
    print(df_week[["date","open","high","low","close","volume"]].to_string(index=False))
else:
    print("  ❌ NSE 1 week failed")


# ============================================================
# TEST 7 — Data sanity check — NSE vs BSE prices match
# ============================================================
print_header("TEST 7 — Sanity Check — NSE vs BSE prices match")

df_nse_r = fetch_nse_historical("RELIANCE", "2026-09-01", "2026-09-23")
df_bse_r = fetch_bse_historical("500325",   "2026-09-01", "2026-09-23")

if not df_nse_r.empty and not df_bse_r.empty:
    # Compare close prices for dates that exist in both
    df_nse_r["date_only"] = df_nse_r["date"].dt.date
    df_bse_r["date_only"] = df_bse_r["date"].dt.date

    merged = df_nse_r.merge(df_bse_r, on="date_only", suffixes=("_nse","_bse"))

    if not merged.empty:
        merged["price_diff"] = abs(merged["close_nse"] - merged["close_bse"])
        merged["diff_pct"]   = (merged["price_diff"] / merged["close_nse"] * 100).round(3)

        print(f"  Comparing {len(merged)} common trading days")
        print(f"  Max price difference: ₹{merged['price_diff'].max():.2f}")
        print(f"  Avg price difference: ₹{merged['price_diff'].mean():.2f}")

        big_diff = merged[merged["diff_pct"] > 1.0]
        if big_diff.empty:
            print(f"  ✅ NSE and BSE prices are consistent (all within 1%)")
        else:
            print(f"  ⚠️  {len(big_diff)} days with >1% price difference between NSE and BSE")
            print(big_diff[["date_only","close_nse","close_bse","diff_pct"]].to_string(index=False))


# ============================================================
# SUMMARY
# ============================================================
print_header("SUMMARY")
print("  If all tests above show ✅ — NSE and BSE connectors are production ready")
print("  If any show ⚠️  PAGINATION WARNING — fetch_nse needs chunking logic added")
print("  If any show ❌ — that issue must be fixed before merging")