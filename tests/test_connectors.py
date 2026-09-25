import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime

from connectors.nse import fetch_nse_historical
from connectors.bse import fetch_bse_historical
from connectors.corporate import fetch_corporate_actions
from connectors.fno import fetch_fno_data
from connectors.indices import fetch_index_history
from connectors.sebi import fetch_bulk_deals
from connectors.amfi import fetch_nav


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def validate_df(df, source, symbol):
    """Validates an OHLCV DataFrame (equity-style: nse.py / bse.py) has correct structure and types."""
    if df.empty:
        print(f"  ❌ {source} {symbol} — Empty DataFrame")
        return False

    required_cols = ["date", "open", "high", "low", "close", "volume", "trades", "symbol", "source"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ❌ {source} {symbol} — Missing columns: {missing}")
        return False

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

    nulls = df[["date", "open", "high", "low", "close", "volume"]].isnull().sum()
    if nulls.any():
        print(f"  ⚠️  Null values found: {nulls[nulls > 0].to_dict()}")

    invalid = df[df["high"] < df["low"]]
    if not invalid.empty:
        print(f"  ❌ {len(invalid)} rows where high < low — data corruption")
        return False

    invalid2 = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
    if not invalid2.empty:
        print(f"  ⚠️  {len(invalid2)} rows where close is outside high-low range")

    if not df["date"].is_monotonic_increasing:
        print(f"  ⚠️  Dates are not sorted in ascending order")

    dupes = df["date"].duplicated().sum()
    if dupes > 0:
        print(f"  ❌ {dupes} duplicate dates found")
        return False

    print(f"  ✅ {source} {symbol} — {len(df)} rows — {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"     close range: {df['close'].min()} to {df['close'].max()}")
    print(f"     columns: {list(df.columns)}")
    return True


def validate_corporate_df(df, symbol):
    """Validates a corporate actions DataFrame structure and types."""
    if df.empty:
        print(f"  ⚠️  Corporate {symbol} — Empty DataFrame (may genuinely have no recent actions)")
        return False

    required_cols = ["ex_date", "symbol", "action_type", "action_detail", "record_date", "source"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ❌ Corporate {symbol} — Missing columns: {missing}")
        return False

    if str(df["ex_date"].dtype) != "datetime64[ns]":
        print(f"  ❌ Corporate {symbol} — ex_date should be datetime64[ns], got {df['ex_date'].dtype}")
        return False

    if (df["symbol"] != symbol.upper()).any():
        print(f"  ❌ Corporate {symbol} — symbol column has values other than the requested symbol")
        return False

    unknown_types = df[~df["action_type"].isin(
        ["DIVIDEND", "BONUS", "SPLIT", "RIGHTS", "BUYBACK", "MERGER", "OTHER", ""]
    )]
    if not unknown_types.empty:
        print(f"  ⚠️  {len(unknown_types)} rows with unrecognized action_type")

    print(f"  ✅ Corporate {symbol} — {len(df)} actions found")
    print(f"     action types: {df['action_type'].value_counts().to_dict()}")
    return True


def validate_fno_df(df, symbol):
    """Validates an F&O DataFrame structure and types."""
    if df.empty:
        print(f"  ❌ FNO {symbol} — Empty DataFrame")
        return False

    required_cols = [
        "date", "symbol", "expiry_date", "option_type", "strike_price",
        "open", "high", "low", "close", "volume", "open_interest", "source"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ❌ FNO {symbol} — Missing columns: {missing}")
        return False

    errors = []
    if str(df["date"].dtype) != "datetime64[ns]":
        errors.append(f"date should be datetime64[ns], got {df['date'].dtype}")
    if str(df["open"].dtype) != "float64":
        errors.append(f"open should be float64, got {df['open'].dtype}")
    if str(df["volume"].dtype) != "int64":
        errors.append(f"volume should be int64, got {df['volume'].dtype}")
    if str(df["open_interest"].dtype) != "int64":
        errors.append(f"open_interest should be int64, got {df['open_interest'].dtype}")

    if errors:
        for e in errors:
            print(f"  ❌ Type error: {e}")
        return False

    invalid = df[df["high"] < df["low"]]
    if not invalid.empty:
        print(f"  ❌ {len(invalid)} rows where high < low — data corruption")
        return False

    print(f"  ✅ FNO {symbol} — {len(df)} rows — {df['date'].min().date()} to {df['date'].max().date()}")
    return True


def validate_indices_df(df, index_name):
    """Validates an indices DataFrame structure and types."""
    if df.empty:
        print(f"  ❌ Indices {index_name} — Empty DataFrame")
        return False

    required_cols = ["date", "index_name", "open", "high", "low", "close", "volume", "source"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ❌ Indices {index_name} — Missing columns: {missing}")
        return False

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

    invalid = df[df["high"] < df["low"]]
    if not invalid.empty:
        print(f"  ❌ {len(invalid)} rows where high < low — data corruption")
        return False

    dupes = df["date"].duplicated().sum()
    if dupes > 0:
        print(f"  ❌ {dupes} duplicate dates found")
        return False

    print(f"  ✅ Indices {index_name} — {len(df)} rows — {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"     close range: {df['close'].min()} to {df['close'].max()}")
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
    expected_roughly = 120
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
    expected_1y = 240
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
    print(df_week[["date", "open", "high", "low", "close", "volume"]].to_string(index=False))
else:
    print("  ❌ NSE 1 week failed")


# ============================================================
# TEST 7 — Data sanity check — NSE vs BSE prices match
# ============================================================
print_header("TEST 7 — Sanity Check — NSE vs BSE prices match")

df_nse_r = fetch_nse_historical("RELIANCE", "2026-09-01", "2026-09-23")
df_bse_r = fetch_bse_historical("500325",   "2026-09-01", "2026-09-23")

if not df_nse_r.empty and not df_bse_r.empty:
    df_nse_r["date_only"] = df_nse_r["date"].dt.date
    df_bse_r["date_only"] = df_bse_r["date"].dt.date

    merged = df_nse_r.merge(df_bse_r, on="date_only", suffixes=("_nse", "_bse"))

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
            print(big_diff[["date_only", "close_nse", "close_bse", "diff_pct"]].to_string(index=False))


# ============================================================
# TEST 8 — Pagination Test — 1 year data (equity)
# ============================================================
print_header("TEST 8 — Pagination Test — RELIANCE 1 year")

df_1y = fetch_nse_historical("RELIANCE", "2025-09-23", "2026-09-23")
if not df_1y.empty:
    print(f"  Rows fetched: {len(df_1y)}")
    print(f"  Date range:   {df_1y['date'].min().date()} to {df_1y['date'].max().date()}")
    if len(df_1y) > 200:
        print(f"  ✅ Pagination working — got full year of data")
    else:
        print(f"  ⚠️  Still limited — check chunk logic")
else:
    print("  ❌ Failed")


# ============================================================
# TEST 9 — Corporate Actions — Basic fetch
# ============================================================
print_header("TEST 9 — Corporate Actions — RELIANCE")

df_corp = fetch_corporate_actions("RELIANCE")
validate_corporate_df(df_corp, "RELIANCE")

print("\n  Checking a second stock (TCS) for comparison...")
df_corp_tcs = fetch_corporate_actions("TCS")
validate_corporate_df(df_corp_tcs, "TCS")


# ============================================================
# TEST 10 — F&O — Correct instrument type, 1 month
# ============================================================
print_header("TEST 10 — F&O — RELIANCE FUTSTK 1 month")

df_fno = fetch_fno_data(
    symbol="RELIANCE",
    from_date="2026-08-26",
    to_date="2026-09-26",
    expiry_date="27-Oct-2026",
    instrument_type="FUTSTK"
)
if not df_fno.empty:
    wrong_instrument = df_fno[df_fno["instrument_type"] != "FUTSTK"]
    if not wrong_instrument.empty:
        print(f"  ❌ {len(wrong_instrument)} rows have instrument_type != FUTSTK — data mismatch")
    else:
        print(f"  ✅ F&O — {len(df_fno)} rows, all confirmed FUTSTK")
        print(df_fno[["date","expiry_date","open","high","low","close","volume","open_interest"]].head(5).to_string(index=False))
else:
    print("  ❌ F&O fetch failed or returned no data")


# ============================================================
# TEST 11 — F&O — Contract lifecycle sanity check (not pagination)
# ============================================================
print_header("TEST 11 — F&O — RELIANCE FUTSTK Full Contract Lifecycle")

# NOTE: A single expiry contract only trades for ~3 months before it
# expires, so this is NOT a "did we get 6 months of rows" test like
# nse.py's — it's a check that the returned data starts exactly when
# this contract began trading and ends at (or near) today, with no
# gaps in between.

df_fno_life = fetch_fno_data(
    symbol="RELIANCE",
    from_date="2026-03-26",
    to_date="2026-09-26",
    expiry_date="27-Oct-2026",
    instrument_type="FUTSTK"
)
if not df_fno_life.empty:
    print(f"  Rows fetched: {len(df_fno_life)}")
    print(f"  Data starts: {df_fno_life['date'].min().date()} (contract's real listing date)")
    print(f"  Data ends:   {df_fno_life['date'].max().date()}")
    gaps = df_fno_life["date"].diff().dt.days
    big_gaps = gaps[gaps > 5]
    if not big_gaps.empty:
        print(f"  ⚠️  {len(big_gaps)} gaps of >5 days found — possible missing data, investigate")
    else:
        print(f"  ✅ No suspicious gaps — data looks continuous and complete")
else:
    print("  ❌ F&O fetch failed")


# ============================================================
# TEST 12 — Indices — 6 month range (pagination check)
# ============================================================
print_header("TEST 12 — Indices Pagination Check — NIFTY 50 6 months")

df_idx_6m = fetch_index_history("NIFTY 50", "2026-03-23", "2026-09-23")
idx_rows = len(df_idx_6m) if not df_idx_6m.empty else 0
print(f"  NIFTY 50 6 months — got {idx_rows} rows")

if idx_rows > 0:
    validate_indices_df(df_idx_6m, "NIFTY 50")
    expected_roughly = 120  # indices trade every trading day, same as equities
    if idx_rows < expected_roughly:
        print(f"  ⚠️  WARNING — Expected ~{expected_roughly} rows for 6 months but got {idx_rows}")
        print(f"  ⚠️  Indices endpoint appears to have the same NSE pagination cap")
        print(f"  ⚠️  as the equity endpoint — indices.py NEEDS chunking logic added,")
        print(f"  ⚠️  same pattern as _fetch_nse_chunk() in nse.py")
    else:
        print(f"  ✅ 6 month range works fine — no pagination needed")
else:
    print("  ❌ Indices fetch failed")


# ============================================================
# TEST 13 — Indices — 1 year range (confirm Test 12 finding)
# ============================================================
print_header("TEST 13 — Indices Pagination Check — NIFTY 50 1 year")

df_idx_1y = fetch_index_history("NIFTY 50", "2025-09-23", "2026-09-23")
idx_rows_1y = len(df_idx_1y) if not df_idx_1y.empty else 0
print(f"  NIFTY 50 1 year — got {idx_rows_1y} rows")

if idx_rows_1y > 0:
    print(f"  Date range returned: {df_idx_1y['date'].min().date()} to {df_idx_1y['date'].max().date()}")
    expected_1y = 240
    if idx_rows_1y < expected_1y:
        print(f"  ⚠️  WARNING — Expected ~{expected_1y} rows for 1 year but got {idx_rows_1y}")
        print(f"  ⚠️  CONFIRMS pagination is needed in indices.py")
    else:
        print(f"  ✅ 1 year range works fine — no pagination needed")
else:
    print("  ❌ Indices 1 year fetch failed")


# ============================================================
# TEST 14 — Indices — short range sanity check (1 week)
# ============================================================
print_header("TEST 14 — Indices Edge Case — 1 Week Range")

df_idx_week = fetch_index_history("NIFTY 50", "2026-09-15", "2026-09-23")
if not df_idx_week.empty:
    print(f"  ✅ NIFTY 50 1 week — {len(df_idx_week)} rows")
    print(df_idx_week[["date", "open", "high", "low", "close", "volume"]].to_string(index=False))
else:
    print("  ❌ NIFTY 50 1 week failed")


# ============================================================
# TEST 15 — Deep Historical Test — does NSE serve OLD data?
# ============================================================
print_header("TEST 15 — Deep Historical Check — RELIANCE Jan 2018")

df_old = fetch_nse_historical("RELIANCE", "2018-01-01", "2018-01-31")
if not df_old.empty:
    actual_years = df_old["date"].dt.year.unique()
    print(f"  Rows fetched: {len(df_old)}")
    print(f"  Date range returned: {df_old['date'].min().date()} to {df_old['date'].max().date()}")
    print(f"  Years present in data: {sorted(actual_years)}")
    if 2018 in actual_years:
        print(f"  ✅ Genuine old historical data confirmed — NOT limited to recent-only")
        print(df_old[["date","open","high","low","close","volume"]].head(5).to_string(index=False))
    else:
        print(f"  ❌ Returned dates don't actually match 2018 — investigate silent fallback")
else:
    print("  ❌ NSE returned nothing for 2018 — endpoint may only serve recent data")
    print("  ⚠️  If this fails, your friend's concern is confirmed valid for this endpoint")



# ============================================================
# TEST 16 — AMFI — Basic NAV fetch
# ============================================================
print_header("TEST 16 — AMFI — Scheme NAV Fetch")

df_amfi = fetch_nav(119551)
if not df_amfi.empty:
    required_cols = ["date", "scheme_code", "scheme_name", "nav", "source"]
    missing = [c for c in required_cols if c not in df_amfi.columns]
    if missing:
        print(f"  ❌ AMFI — Missing columns: {missing}")
    else:
        print(f"  ✅ AMFI — {len(df_amfi)} rows — scheme: {df_amfi.iloc[0]['scheme_name']}")
        print(f"     date range: {df_amfi['date'].min().date()} to {df_amfi['date'].max().date()}")
else:
    print("  ❌ AMFI fetch failed or returned no data")

print("\n  Checking error handling — invalid scheme code...")
df_amfi_bad = fetch_nav(999999999)
if df_amfi_bad.empty:
    print("  ✅ AMFI correctly returned empty DataFrame for invalid scheme code (no crash)")
else:
    print("  ❌ AMFI returned data for an invalid scheme code — something is wrong")


# ============================================================
# TEST 17 — SEBI Bulk Deals — Adaptive chunking check, 1 week
# ============================================================
print_header("TEST 17 — SEBI Bulk Deals — 1 Week (Adaptive Chunking)")

df_sebi = fetch_bulk_deals("2026-09-19", "2026-09-26")
if not df_sebi.empty:
    unique_dates = df_sebi["date"].dt.date.nunique()
    print(f"  ✅ SEBI — {len(df_sebi)} bulk deals across {unique_dates} distinct dates")
    print(f"     date range: {df_sebi['date'].min().date()} to {df_sebi['date'].max().date()}")
    if unique_dates <= 1:
        print(f"  ⚠️  Only 1 distinct date returned for a 7-day request — check if this")
        print(f"  ⚠️  week genuinely only had trading/deals on one day, or if capping persists")
    print(df_sebi.head(5).to_string(index=False))
else:
    print("  ❌ SEBI fetch failed or returned no data")


# ============================================================
# TEST 18 — SEBI Bulk Deals — Adaptive chunking check, 1 month
# ============================================================
print_header("TEST 18 — SEBI Bulk Deals — 1 Month (Adaptive Chunking)")

df_sebi_1m = fetch_bulk_deals("2026-08-26", "2026-09-26")
sebi_rows = len(df_sebi_1m) if not df_sebi_1m.empty else 0
if sebi_rows > 0:
    unique_dates_1m = df_sebi_1m["date"].dt.date.nunique()
    print(f"  ✅ SEBI 1 month — {sebi_rows} rows across {unique_dates_1m} distinct trading days")
    print(f"     date range: {df_sebi_1m['date'].min().date()} to {df_sebi_1m['date'].max().date()}")
    if df_sebi_1m["date"].max().date() < datetime.strptime("2026-09-24", "%Y-%m-%d").date():
        print(f"  ⚠️  Data doesn't reach close to the end date — may still be losing recent days")
    else:
        print(f"  ✅ Data reaches close to the requested end date — looks complete")
else:
    print("  ❌ SEBI 1 month fetch failed")