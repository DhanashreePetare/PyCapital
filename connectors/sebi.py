"""
SEBI-mandated Bulk Deals connector for PyCapital.

SEBI mandates same-day public disclosure of Bulk Deals; NSE hosts the
actual data. This connector uses NSE's official historical Bulk Deals
API, confirmed via DevTools.

KNOWN LIMITATION (documented, not a bug):
NSE's own Bulk & Block Deals page states outright: "This page displays
data upto 70 records only. Kindly download the csv file for more
records." We investigated the CSV download path (DevTools, Network
tab, multiple filter attempts) and found no corresponding server
request — the export appears to be generated client-side in the
browser from data not exposed via a discoverable API call. As a
result, this connector inherits NSE's stated 70-record cap on any
single query. For a date range with more than 70 total bulk deals
across the market, only the most recent ~70 will be returned, and
older records in that range will be silently missing from the result.

Practical implication: this connector is reliable for near-term /
recent bulk deal monitoring (e.g. "what large trades happened this
week"), but is NOT reliable for exhaustive historical bulk deals
research over long date ranges. Users needing complete historical
bulk deals data should be directed to NSE's official CSV export on
their website until a documented, stable programmatic export endpoint
is found.
"""

import pandas as pd
from datetime import datetime
from connectors._session_utils import create_nse_session


def fetch_bulk_deals(from_date, to_date):
    """
    Fetch SEBI-mandated Bulk Deals disclosures from NSE.

    LIMITATION: NSE caps this endpoint at 70 records per query (see
    module docstring). For date ranges likely to contain more than 70
    total deals across the market, results may be incomplete — this
    function does NOT attempt to work around that cap, since NSE's own
    UI does not expose a way to do so.

    Args:
        from_date : YYYY-MM-DD
        to_date   : YYYY-MM-DD

    Returns:
        pandas DataFrame with columns:
        date, symbol, scrip_name, client_name, deal_type,
        quantity, price, source
        Sorted oldest to newest within the returned (possibly
        capped) result set. Empty DataFrame on failure.
    """
    columns = ["date", "symbol", "scrip_name", "client_name",
               "deal_type", "quantity", "price", "source"]

    try:
        fmt_from = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d-%m-%Y")
        fmt_to   = datetime.strptime(to_date,   "%Y-%m-%d").strftime("%d-%m-%Y")

        session = create_nse_session()

        url = "https://www.nseindia.com/api/historicalOR/bulk-block-short-deals"
        params = {"optionType": "bulk_deals", "from": fmt_from, "to": fmt_to}

        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()
        data = payload.get("data", []) if isinstance(payload, dict) else payload

        if not data:
            print(f"[SEBI] No bulk deals found for {from_date} to {to_date}")
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(data)

        df = df.rename(columns={
            "BD_DT_DATE":     "date",
            "BD_SYMBOL":      "symbol",
            "BD_SCRIP_NAME":  "scrip_name",
            "BD_CLIENT_NAME": "client_name",
            "BD_BUY_SELL":    "deal_type",
            "BD_QTY_TRD":     "quantity",
            "BD_TP_WATP":     "price"
        })

        df["source"] = "SEBI_NSE"
        df = df[columns]

        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype("int64")
        df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")

        df = df.sort_values("date").reset_index(drop=True)

        if len(df) >= 70:
            print(f"[SEBI] ⚠️  Warning: {len(df)} rows returned — at NSE's known 70-row "
                  f"cap. Older deals within {from_date} to {to_date} may be missing. "
                  f"See module docstring for details.")

        return df

    except Exception as e:
        print(f"[SEBI] Exception fetching bulk deals: {e}")
        return pd.DataFrame(columns=columns)



# This version keeps a runtime warning so anyone using the library gets told in real time when they're likely hitting the cap — that's the honest, transparent behavior your project doc's "Data Integrity Safety" section calls for, rather than either pretending it's complete or over-engineering a fix for something NSE itself doesn't expose a real path around.
# My recommendation: ship sebi.py exactly as it currently is — using the confirmed, real, working bulk-block-short-deals endpoint — but document the 70-row cap explicitly and honestly in the docstring, rather than sinking more hours into an unclear path. Here's that final version: