"""
SEBI-mandated Bulk Deals connector for PyEquity.

SEBI mandates same-day public disclosure of Bulk Deals; NSE hosts the
actual data. Three functions are provided:

- ``fetch_bulk_deals``: NSE's historical Bulk Deals API for a date range.
  Capped by NSE (see below). Output ``source`` is ``"SEBI_NSE"``.
- ``fetch_bulk_deals_by_day``: the same API queried one day at a time, so
  every day in a range is represented, and every day that hit the cap is
  reported (in the log and in ``df.attrs``). Output ``source`` is
  ``"SEBI_NSE"``.
- ``fetch_latest_bulk_deals``: NSE's ``bulk.csv`` archive file, which holds
  the COMPLETE list for the latest trading day only. Output ``source`` is
  ``"SEBI_NSE_CSV"`` so complete data can be told apart from capped data.

KNOWN LIMITATION (documented, not a bug):
The historical API returns at most 70 records per query. Measured on real
responses (November 2026):
- A single-day query returned exactly 70 rows on every day tried, while the
  complete file for the latest day held 134 to 136 rows. So the cap applies
  to a single day, and busy days are cut roughly in half.
- Every row the API returned was also present in the complete file, so the
  API truncates but does not corrupt.
- A two-day query returned 70 rows from the first day only. NSE does not
  document which rows it keeps, and within a single day the 70 returned
  were not simply the first 70 of the file. Rows are missing; which ones
  is not predictable.
- The CSV download behind NSE's own page does not use a discoverable API
  call, and ``bulk.csv`` holds the latest day only, so complete history
  could not be found.

Practical implication: only the latest trading day can be obtained
completely. For older days this module returns what NSE serves and says
loudly when a query was capped. Do not use older results for exhaustive
research. Users needing complete history should use NSE's own CSV export
on their website.
"""

import csv
import io
import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import requests

from pyequity.connectors._session_utils import create_nse_session
from pyequity.reliability.rate_limiter import RateLimiter
from pyequity.reliability.retry import retry_call
from pyequity.reliability.session import SessionManager
from pyequity.schema import SCHEMAS, empty_frame

logger = logging.getLogger(__name__)

_API_URL = "https://www.nseindia.com/api/historicalOR/bulk-block-short-deals"

# Tried in order. Both served the same file when probed in November 2026.
_BULK_CSV_URLS = (
    "https://nsearchives.nseindia.com/content/equities/bulk.csv",
    "https://archives.nseindia.com/content/equities/bulk.csv",
)
_CSV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}

_NSE_ROW_CAP = 70
_CSV_SOURCE = "SEBI_NSE_CSV"
_COLUMNS = ["date", "symbol", "scrip_name", "client_name",
            "deal_type", "quantity", "price", "source"]
_CSV_REQUIRED = ("Date", "Symbol", "Security Name", "Client Name",
                 "Buy/Sell", "Quantity Traded")


def _get_bulk_records(session, fmt_from: str, fmt_to: str) -> list:
    """
    Request bulk deals for one date range from the NSE API.

    Args:
        session  : warmed-up NSE session
        fmt_from : start date, DD-MM-YYYY
        fmt_to   : end date, DD-MM-YYYY

    Returns:
        List of raw record dicts (possibly empty). Raises on network or
        HTTP errors; callers decide how to handle that.
    """
    params = {"optionType": "bulk_deals", "from": fmt_from, "to": fmt_to}
    response = session.get(_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", []) if isinstance(payload, dict) else payload


def _records_to_frame(data: list) -> pd.DataFrame:
    """Convert raw NSE API records to the standard bulk_deals frame (source SEBI_NSE)."""
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
    df = df[_COLUMNS].copy()

    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    df["date"] = df["date"].astype("datetime64[ns]")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype("int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")

    df = df.sort_values("date").reset_index(drop=True)
    return df.astype(SCHEMAS["bulk_deals"])


def fetch_bulk_deals(from_date, to_date):
    """
    Fetch SEBI-mandated Bulk Deals disclosures from NSE.

    LIMITATION: NSE caps this endpoint at 70 records per query (see
    module docstring). When 70 rows come back the result is incomplete and
    a warning is logged. This function does NOT work around the cap; use
    fetch_bulk_deals_by_day to cover every day of a range, and
    fetch_latest_bulk_deals for the complete latest day.

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
    try:
        fmt_from = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d-%m-%Y")
        fmt_to   = datetime.strptime(to_date,   "%Y-%m-%d").strftime("%d-%m-%Y")

        session = create_nse_session()

        data = _get_bulk_records(session, fmt_from, fmt_to)

        if not data:
            logger.warning(f"[SEBI] No bulk deals found for {from_date} to {to_date}")
            return empty_frame("bulk_deals")

        df = _records_to_frame(data)

        if len(df) >= _NSE_ROW_CAP:
            logger.warning(f"[SEBI] ⚠️  Warning: {len(df)} rows returned, at NSE's known "
                           f"{_NSE_ROW_CAP}-row cap. The result for {from_date} to {to_date} "
                           f"is INCOMPLETE: some deals are missing (which ones is not "
                           f"documented by NSE). See module docstring for details.")

        return df

    except Exception as e:
        logger.error(f"[SEBI] Exception fetching bulk deals: {e}")
        return empty_frame("bulk_deals")


def _weekdays(start: datetime, end: datetime):
    """Yield every Monday-to-Friday date from ``start`` to ``end`` inclusive."""
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def fetch_bulk_deals_by_day(from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch Bulk Deals one calendar day at a time, so no day of the range is
    silently skipped, and report every day whose result hit NSE's cap.

    Saturdays and Sundays are skipped. Exchange holidays simply return no
    rows. Each day is a separate request, so a long range is slow.

    A day that returns 70 rows is INCOMPLETE (NSE's cap, see module
    docstring). This function cannot complete such a day; it flags it:
    a warning names the days, and the returned frame carries
    ``df.attrs["capped_days"]`` (list of YYYY-MM-DD strings) and
    ``df.attrs["failed_days"]`` (days that could not be fetched).
    attrs are kept on the frame returned here but may be dropped by later
    pandas operations. For the latest trading day, fetch_latest_bulk_deals
    returns the complete list.

    Args:
        from_date : Start date in YYYY-MM-DD format
        to_date   : End date in YYYY-MM-DD format

    Returns:
        pandas DataFrame with columns:
        date, symbol, scrip_name, client_name, deal_type,
        quantity, price, source ("SEBI_NSE")
        Sorted oldest to newest. Empty DataFrame on failure; never raises.
    """
    def _empty(capped=None, failed=None) -> pd.DataFrame:
        frame = empty_frame("bulk_deals")
        frame.attrs["capped_days"] = capped or []
        frame.attrs["failed_days"] = failed or []
        return frame

    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        logger.error("[SEBI] Invalid date format. Use YYYY-MM-DD")
        return _empty()

    if start > end:
        logger.error("[SEBI] from_date cannot be after to_date")
        return _empty()

    # One session is reused for the whole range and replaced only if a day
    # fails (a stale or blocked session is the usual cause). Each day's
    # request is retried on transient errors, and requests are spaced out.
    sessions = SessionManager(lambda _key: create_nse_session(), sleep=time.sleep)
    limiter = RateLimiter(1.0, sleep=time.sleep)

    frames: List[pd.DataFrame] = []
    capped: List[str] = []
    failed: List[str] = []

    day = start
    while day <= end:
        if day.weekday() < 5:
            label = day.strftime("%Y-%m-%d")
            fmt = day.strftime("%d-%m-%Y")

            session_outcome = sessions.get()
            if session_outcome.failed:
                # No point retrying a dead network for every remaining day:
                # report them all as failed and stop.
                logger.error(f"[SEBI] Could not open an NSE session: {session_outcome.reason}")
                failed.extend(d.strftime("%Y-%m-%d") for d in _weekdays(day, end))
                break

            limiter.wait("nseindia.com")
            outcome = retry_call(
                lambda: _get_bulk_records(session_outcome.data, fmt, fmt),
                base_delay=2.0,
                label=f"SEBI bulk deals {label}",
                sleep=time.sleep,
            )
            if outcome.ok:
                try:
                    frame = _records_to_frame(outcome.data)
                except Exception as e:
                    logger.error(f"[SEBI] Could not read bulk deals records for {label}: {e}")
                    failed.append(label)
                else:
                    frames.append(frame)
                    if len(frame) >= _NSE_ROW_CAP:
                        capped.append(label)
            elif outcome.failed:
                logger.error(f"[SEBI] Bulk deals request failed for {label}: {outcome.reason}")
                failed.append(label)
                sessions.invalidate()
        day += timedelta(days=1)

    if capped:
        logger.warning(f"[SEBI] ⚠️  {len(capped)} day(s) hit NSE's {_NSE_ROW_CAP}-row cap and are "
                       f"INCOMPLETE: {', '.join(capped)}")
    if failed:
        logger.warning(f"[SEBI] Could not fetch {len(failed)} day(s), result is incomplete: "
                       f"{', '.join(failed)}")

    if not frames:
        logger.warning(f"[SEBI] No bulk deals found for {from_date} to {to_date}")
        return _empty(capped, failed)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df.attrs["capped_days"] = capped
    df.attrs["failed_days"] = failed
    return df


def _parse_bulk_csv(text: str) -> Optional[pd.DataFrame]:
    """
    Parse NSE's bulk.csv text into the standard bulk_deals frame.

    Columns are found by header name. Returns None if a required column
    is missing, so a changed layout is rejected instead of guessed at.
    Rows with the wrong number of fields, or an unreadable date, quantity
    or price, are dropped and counted in a warning. Identical rows are
    kept: the same client can genuinely trade the same size twice.
    """
    reader = csv.reader(io.StringIO(text))
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration:
        return None

    if any(col not in header for col in _CSV_REQUIRED):
        return None
    price_idx = next((i for i, h in enumerate(header) if h.startswith("Trade Price")), None)
    if price_idx is None:
        return None
    idx = {name: header.index(name) for name in _CSV_REQUIRED}

    rows = []
    wrong_width = 0
    for fields in reader:
        if not fields or all(not f.strip() for f in fields):
            continue
        if len(fields) != len(header):
            wrong_width += 1
            continue
        rows.append([f.strip() for f in fields])

    if not rows:
        if wrong_width:
            logger.warning(f"[SEBI] Dropped {wrong_width} unreadable row(s) from bulk.csv "
                           f"({wrong_width} with the wrong number of fields)")
        return empty_frame("bulk_deals")

    raw = pd.DataFrame({
        "date": [r[idx["Date"]] for r in rows],
        "symbol": [r[idx["Symbol"]] for r in rows],
        "scrip_name": [r[idx["Security Name"]] for r in rows],
        "client_name": [r[idx["Client Name"]] for r in rows],
        "deal_type": [r[idx["Buy/Sell"]].upper() for r in rows],
        "quantity": [r[idx["Quantity Traded"]].replace(",", "") for r in rows],
        "price": [r[price_idx].replace(",", "") for r in rows],
    })

    raw["date"] = pd.to_datetime(raw["date"], format="%d-%b-%Y", errors="coerce")
    raw["date"] = raw["date"].astype("datetime64[ns]")
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")

    unreadable = raw["date"].isna() | raw["quantity"].isna() | raw["price"].isna()
    dropped = int(unreadable.sum()) + wrong_width
    if dropped:
        logger.warning(f"[SEBI] Dropped {dropped} unreadable row(s) from bulk.csv "
                       f"({wrong_width} with the wrong number of fields)")
    raw = raw[~unreadable].copy()

    raw["quantity"] = raw["quantity"].astype("int64")
    raw["price"] = raw["price"].astype("float64")
    raw["source"] = _CSV_SOURCE
    raw = raw[_COLUMNS]
    raw = raw.sort_values("date", kind="stable").reset_index(drop=True)
    return raw.astype(SCHEMAS["bulk_deals"])


def fetch_latest_bulk_deals() -> pd.DataFrame:
    """
    Fetch the COMPLETE bulk deals list for the latest trading day from
    NSE's bulk.csv archive file.

    Unlike the historical API this file is not capped (134 to 136 rows
    were returned on days where the API returned 70). It holds only the
    latest trading day, with no date parameter, so the date in the result
    is whatever NSE has published; check the ``date`` column. No history
    is available from this source.

    Returns:
        pandas DataFrame with columns:
        date, symbol, scrip_name, client_name, deal_type,
        quantity, price, source ("SEBI_NSE_CSV")
        The CSV's Remarks column is not kept. Empty DataFrame on any
        failure, including an unrecognised file layout; never raises.
    """
    try:
        text = None
        for url in _BULK_CSV_URLS:
            try:
                response = requests.get(url, headers=_CSV_HEADERS, timeout=30)
            except Exception as e:
                logger.warning(f"[SEBI] bulk.csv request failed for {url}: {e}")
                continue
            if response.status_code != 200:
                logger.warning(f"[SEBI] bulk.csv returned status {response.status_code} for {url}")
                continue
            text = response.content.decode("utf-8-sig", errors="replace")
            break

        if text is None:
            logger.error("[SEBI] Could not download bulk.csv from any known address")
            return empty_frame("bulk_deals")

        df = _parse_bulk_csv(text)
        if df is None:
            logger.error("[SEBI] Unrecognised bulk.csv layout; NSE may have changed its format")
            return empty_frame("bulk_deals")
        if df.empty:
            logger.warning("[SEBI] bulk.csv contained no readable deals")
            return empty_frame("bulk_deals")

        dates = sorted({d.strftime("%Y-%m-%d") for d in df["date"]})
        if len(dates) > 1:
            logger.warning(f"[SEBI] bulk.csv holds more than one date: {', '.join(dates)}")
        logger.info(f"[SEBI] bulk.csv: {len(df)} deals for {', '.join(dates)}")
        return df

    except Exception as e:
        logger.error(f"[SEBI] Exception reading bulk.csv: {e}")
        return empty_frame("bulk_deals")