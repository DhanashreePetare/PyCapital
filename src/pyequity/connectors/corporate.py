"""
NSE corporate actions connector.

Fetches dividends, bonuses, splits, rights issues and similar actions for one
symbol, and parses NSE's free-text description into structured columns
(see ``_action_parser``). Facts used here were confirmed with live probes
(2026-11):

* ``from_date`` / ``to_date`` (DD-MM-YYYY) filter on the ex-date.
* A request without dates returns only about 20 rows, which silently drops
  older history, so this connector always sends an explicit date window.
* Date-filtered requests are not capped at 20 (one whole-year, all-equities
  request returned 1,927 rows), so no splitting is needed per symbol.
* An unknown symbol and a valid symbol with no actions both return an empty
  list with HTTP 200. An empty result therefore does NOT prove the symbol is
  invalid. A symbol that changed name keeps its history under the old or the
  new symbol depending on NSE (TATAMOTORS returned nothing; TMPV did).
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from pyequity.connectors._action_parser import parse_action_details
from pyequity.connectors._session_utils import create_nse_session
from pyequity.schema import SCHEMAS, columns_of, empty_frame

logger = logging.getLogger(__name__)

_URL = "https://www.nseindia.com/api/corporates-corporateActions"
# Start of the default window. Not a claim that data exists this far back.
_DEFAULT_FROM = "1990-01-01"
# Announced actions can have an ex-date in the future; include up to a year ahead.
_DEFAULT_DAYS_AHEAD = 365
_DATASET = "corporate_actions"


def _to_dates(series: pd.Series) -> pd.Series:
    """Parse NSE's DD-Mon-YYYY dates; "-" and junk become NaT."""
    out = pd.to_datetime(series, format="%d-%b-%Y", errors="coerce")
    return out.astype("datetime64[ns]")


def _clean_isin(value: object) -> Optional[str]:
    """Return a stripped ISIN string, or None when missing."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and value != "-" else None


def fetch_corporate_actions(symbol: str, from_date: Optional[str] = None,
                            to_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch corporate actions for one NSE symbol.

    Without dates the full available history is requested (from 1990-01-01,
    up to one year ahead so announced upcoming actions are included). Earlier
    versions sent no dates and got only about 20 rows.

    Failures (network error, bad response, bad arguments) are logged and give
    an empty frame; nothing raises. An empty frame can also mean the symbol
    has no actions, or the symbol is unknown or was renamed: NSE does not
    distinguish these.

    Args:
        symbol: NSE symbol, e.g. "RELIANCE".
        from_date: Optional start of the ex-date range, YYYY-MM-DD.
        to_date: Optional end of the ex-date range, YYYY-MM-DD.

    Returns:
        pandas DataFrame in the standard ``corporate_actions`` schema:
        ex_date, symbol, action_type, action_detail, record_date, source,
        isin, action_categories, bonus_new, bonus_held, split_old_fv,
        split_new_fv, rights_new, rights_held, rights_premium, parse_status.
        ``action_detail`` is NSE's text unchanged. ``action_type`` is the
        legacy single label. ``action_categories`` lists every detected
        category (e.g. "BONUS|SPLIT"). Ratios are recorded as written, new
        shares first (``Bonus 1:3`` is 1 new for 3 held). ``parse_status`` is
        "needs_review" for demergers, mergers, schemes and any text that was
        not parsed cleanly; do not adjust prices from such rows.
        Rows are in NSE's order (newest first).
    """
    symbol = str(symbol).strip().upper() if symbol is not None else ""
    if not symbol:
        logger.error("[Corporate] symbol is required")
        return empty_frame(_DATASET)

    try:
        start = datetime.strptime(from_date or _DEFAULT_FROM, "%Y-%m-%d")
        if to_date is None:
            end = datetime.now() + timedelta(days=_DEFAULT_DAYS_AHEAD)
        else:
            end = datetime.strptime(to_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        logger.error("[Corporate] Invalid date format. Use YYYY-MM-DD")
        return empty_frame(_DATASET)
    if start > end:
        logger.error("[Corporate] from_date cannot be after to_date")
        return empty_frame(_DATASET)

    try:
        session = create_nse_session()
        params = {
            "index": "equities",
            "symbol": symbol,
            "from_date": start.strftime("%d-%m-%Y"),
            "to_date": end.strftime("%d-%m-%Y"),
        }
        response = session.get(_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.error(f"[Corporate] Error fetching actions for {symbol}: {e}")
        return empty_frame(_DATASET)

    try:
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            payload = payload["data"]
        if not isinstance(payload, list) or not all(isinstance(r, dict) for r in payload):
            logger.error(f"[Corporate] Unexpected response shape for {symbol}")
            return empty_frame(_DATASET)
        if not payload:
            logger.warning(
                f"[Corporate] No actions returned for {symbol} between "
                f"{start.date()} and {end.date()} (NSE returns the same empty "
                f"answer for an unknown or renamed symbol)")
            return empty_frame(_DATASET)

        raw = pd.DataFrame(payload)
        missing = [c for c in ("exDate", "subject") if c not in raw.columns]
        if missing:
            logger.error(f"[Corporate] Response for {symbol} lacks field(s) {missing}")
            return empty_frame(_DATASET)
        for col in ("recDate", "isin", "symbol"):
            if col not in raw.columns:
                raw[col] = None

        # Keep only rows for the requested symbol (the API symbol is kept as is).
        api_symbol = raw["symbol"].where(raw["symbol"].notna(), symbol)
        api_symbol = api_symbol.astype(str).str.strip().str.upper()
        wrong = ~(api_symbol == symbol)
        if wrong.any():
            logger.warning(f"[Corporate] Dropped {int(wrong.sum())} row(s) that "
                           f"belong to a different symbol than {symbol}")
            raw, api_symbol = raw[~wrong], api_symbol[~wrong]
        raw = raw.reset_index(drop=True)
        api_symbol = api_symbol.reset_index(drop=True)

        ex_date = _to_dates(raw["exDate"])
        lost = ex_date.isna() & ~raw["exDate"].astype(str).str.strip().isin(["", "-", "None", "nan"])
        if lost.any():
            logger.warning(f"[Corporate] {int(lost.sum())} ex-date value(s) could "
                           f"not be parsed and became NaT")

        out = pd.DataFrame({
            "ex_date": ex_date,
            "symbol": api_symbol,
            "action_type": raw["subject"].map(extract_action_type),
            "action_detail": raw["subject"],
            "record_date": _to_dates(raw["recDate"]),
            "source": "NSE",
            "isin": raw["isin"].map(_clean_isin),
        })
        parsed = pd.DataFrame(parse_action_details(raw["subject"]))
        out = pd.concat([out, parsed], axis=1)

        # Safety net: the API filters by ex-date; drop anything outside the window.
        outside = out["ex_date"].notna() & ((out["ex_date"] < pd.Timestamp(start)) |
                                            (out["ex_date"] > pd.Timestamp(end)))
        if outside.any():
            logger.warning(f"[Corporate] Dropped {int(outside.sum())} row(s) outside "
                           f"{start.date()}..{end.date()}")
            out = out[~outside].reset_index(drop=True)
        if out.empty:
            return empty_frame(_DATASET)

        out = out[columns_of(_DATASET)].astype(SCHEMAS[_DATASET])

        review = int((out["parse_status"] == "needs_review").sum())
        logger.info(f"[Corporate] {symbol}: {len(out)} rows, {review} need review")
        return out

    except Exception as e:
        logger.error(f"[Corporate] Error processing actions for {symbol}: {e}")
        return empty_frame(_DATASET)


def extract_action_type(action_detail):
    """
    Extract a simple action type from NSE's subject field.

    Legacy label, kept unchanged so existing callers see the same values. It
    keeps only the FIRST match, so combined actions lose parts (e.g. "Bonus +
    Split" is BONUS) and it mislabels some texts (e.g. "Demerger" is MERGER).
    Use the ``action_categories`` and ``parse_status`` columns for anything
    that depends on the exact action.

    Args:
        action_detail: NSE subject text.

    Returns:
        One of DIVIDEND, BONUS, SPLIT, RIGHTS, BUYBACK, MERGER, OTHER, or an
        empty string for missing text.
    """
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