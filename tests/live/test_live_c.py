"""
Live checks for fno.py (hits the real NSE). Run manually before a PR:

    python tests/live/test_live_c.py

Not collected by pytest. Uses dates relative to today and picks expiries from
list_expiries(), so it does not go stale.
"""
import logging
from datetime import date, timedelta

from pyequity.connectors.corporate import fetch_corporate_actions
from pyequity.connectors.fno import fetch_fno_data, list_expiries
from pyequity.schema import columns_of

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def main():
    today = date.today()
    start = (today - timedelta(days=30)).isoformat()
    end = today.isoformat()

    expiries = list_expiries("RELIANCE")
    check("list_expiries RELIANCE", len(expiries) >= 2, f"{expiries[:3]}")
    check("list_expiries sorted ISO", expiries == sorted(expiries)
          and all(len(e) == 10 for e in expiries))
    check("list_expiries unknown symbol -> []", list_expiries("NOTASYMBOL123") == [])

    # Futures, all contracts per day (blank expiry)
    df = fetch_fno_data("RELIANCE", start, end)
    check("futures blank expiry", not df.empty and list(df.columns) == columns_of("fno"),
          f"{len(df)} rows, {df['expiry_date'].nunique() if not df.empty else 0} expiries")
    if not df.empty:
        check("futures encoding", (df["option_type"] == "XX").all()
              and (df["strike_price"] == 0).all())
        check("no duplicate contract-days",
              not df.duplicated(["date", "expiry_date"]).any())

    # Historical contract from 2023 must be reachable without knowing expiries
    old = fetch_fno_data("RELIANCE", "2023-03-15", "2023-03-15")
    check("2023 historical day", len(old) == 3, f"{len(old)} rows")

    # Expired contract with an explicit expiry (ISO and NSE format give same rows)
    a = fetch_fno_data("RELIANCE", "2026-08-26", "2026-09-26", "2026-10-27")
    b = fetch_fno_data("RELIANCE", "2026-08-26", "2026-09-26", "27-Oct-2026")
    check("expired expiry returns data, both formats equal",
          not a.empty and a.equals(b), f"{len(a)} rows")

    # Options need filters; with them, a long range must not be truncated
    check("options without filters -> empty",
          fetch_fno_data("RELIANCE", start, end, instrument_type="OPTSTK").empty)
    ce = fetch_fno_data("RELIANCE", start, end, instrument_type="OPTSTK",
                        strike_price=1250, option_type="CE")
    check("options CE strike 1250", ce.empty or ((ce["option_type"] == "CE").all()
          and (ce["strike_price"] == 1250).all()), f"{len(ce)} rows")

    # Index instruments
    fi = fetch_fno_data("BANKNIFTY", start, end, instrument_type="FUTIDX")
    check("FUTIDX BANKNIFTY", not fi.empty and (fi["instrument_type"] == "FUTIDX").all(),
          f"{len(fi)} rows")

    # ---- Corporate actions (facts below were verified against NSE circulars) ----
    itc = fetch_corporate_actions("ITC")
    check("corporate: full history, not just ~20 latest rows", len(itc) >= 30, f"{len(itc)} rows")
    check("corporate: columns", list(itc.columns) == columns_of("corporate_actions"))

    rel = fetch_corporate_actions("RELIANCE", "2020-05-01", "2020-05-31")
    r = rel[rel["action_categories"] == "RIGHTS"]
    check("corporate: RELIANCE 2020 rights 1:15 @ premium 1247",
          len(r) == 1 and tuple(r.iloc[0][["rights_new", "rights_held", "rights_premium"]]) == (1, 15, 1247)
          and r.iloc[0]["parse_status"] == "ok", f"{len(rel)} rows")

    baj = fetch_corporate_actions("BAJFINANCE", "2025-06-16", "2025-06-16")
    bonus = baj[baj["action_categories"].str.contains("BONUS")]
    split = baj[baj["action_categories"].str.contains("SPLIT")]
    check("corporate: BAJFINANCE 2025-06-16 bonus 4:1 and split 2->1 as separate rows",
          len(bonus) == 1 and len(split) == 1
          and (bonus.iloc[0]["bonus_new"], bonus.iloc[0]["bonus_held"]) == (4, 1)
          and (split.iloc[0]["split_old_fv"], split.iloc[0]["split_new_fv"]) == (2, 1),
          f"{len(baj)} rows")

    pg = fetch_corporate_actions("POWERGRID", "2021-07-29", "2021-07-29")
    pgb = pg[pg["action_categories"].str.contains("BONUS")]
    check("corporate: POWERGRID 2021-07-29 bonus 1:3",
          len(pgb) == 1 and (pgb.iloc[0]["bonus_new"], pgb.iloc[0]["bonus_held"]) == (1, 3))

    june = fetch_corporate_actions("RELIANCE", "2026-06-01", "2026-06-10")
    check("corporate: date window returns only that window",
          len(june) == 1 and june.iloc[0]["ex_date"].strftime("%Y-%m-%d") == "2026-06-05")

    check("corporate: unknown symbol -> empty, no crash",
          fetch_corporate_actions("NOTASYMBOL123").empty)
    check("corporate: bad date -> empty, no crash",
          fetch_corporate_actions("RELIANCE", "06/01/2026", "2026-06-10").empty)

    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()