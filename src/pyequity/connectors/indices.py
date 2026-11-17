"""NSE index history connector.

Integrity rules (data integrity beats availability):

* A chunk is ``ok``, ``empty`` or ``error``. HTTP errors (NSE answers bad
  requests with HTTP 422), non-JSON bodies, error payloads, unexpected shapes,
  missing fields and unparseable dates/prices are all ``error``.
* If any chunk fails, ``fetch_index_history`` logs the failed range and
  returns an empty standard frame. It never returns partial history.
* NSE caps each response at about 70 rows and, observed on NIFTY 50, the cap
  cuts the NEWEST data. A response with 70 or more rows is therefore never
  trusted: its date range is split in half and both halves are fetched again.
  A single day that still reaches the cap makes the whole call fail.
* NSE accepts long/display index names and ignores case. For many indices the
  short trading name returns nothing. Aliases are resolved lazily from NSE's
  live ``/api/index-names`` list (nothing is hardcoded), only when the first
  chunk comes back empty, so the normal path makes the same requests as before.
* ``index_name`` in the output is NSE's ``EOD_INDEX_NAME``. If that field is
  missing or blank the requested name is used and a warning is logged.
* Volume (``HIT_TRADED_QTY``) is validated as an exact integer straight from
  the raw JSON value, never through float64. Missing or non-numeric values
  become 0 and are counted in a warning. Negative values, fractional
  quantities, values above the int64 range, and floats above 2**53 (which may
  already have been rounded by the JSON decoder and cannot be verified) are
  errors: the whole call returns empty. Nothing is truncated or rounded.

Known limitations:

* The alias behaviour was verified for one date window (03-16 Nov 2026).
* ``HIT_TURN_OVER`` is deliberately not part of the shared schema.
* For indices without traded quantity (for example INDIA VIX) a missing volume
  and a zero volume are indistinguishable once the schema forces ``int64``.
  The number of volumes replaced with 0 is logged.
"""
import difflib
import logging
import time
from decimal import Decimal, InvalidOperation
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from pyequity.connectors._session_utils import create_nse_session
from pyequity.schema import SCHEMAS, empty_frame

logger = logging.getLogger(__name__)

_HISTORY_URL = "https://www.nseindia.com/api/historicalOR/indicesHistory"
_NAMES_URL = "https://www.nseindia.com/api/index-names"

_CHUNK_DAYS = 60
_ROW_CAP = 70            # observed NSE cap per response, not a documented limit
_REQUEST_DELAY_SECONDS = 1
_INT64_MAX = 2 ** 63 - 1
_MAX_EXACT_FLOAT = 2 ** 53     # above this a float may already be rounded
_TIMEOUT_SECONDS = 20

_COLUMNS = list(SCHEMAS["index"])
_PRICE_FIELDS = {
    "EOD_OPEN_INDEX_VAL": "open",
    "EOD_HIGH_INDEX_VAL": "high",
    "EOD_LOW_INDEX_VAL": "low",
    "EOD_CLOSE_INDEX_VAL": "close",
}
_REQUIRED_FIELDS = ("EOD_TIMESTAMP",) + tuple(_PRICE_FIELDS)

# Entries in /api/index-names that are stock-eligibility lists, not indices.
# They returned no history under any spelling in the probe.
_NON_INDEX_ENTRIES = frozenset({"securities in f&o", "permitted to trade"})

# Process-wide cache of the (short, long) pairs from /api/index-names.
_INDEX_NAME_PAIRS: Optional[List[Tuple[str, str]]] = None


class _ChunkResult(NamedTuple):
    status: str                      # "ok" | "empty" | "capped" | "error"
    frame: pd.DataFrame
    detail: str = ""
    volume_defaulted: int = 0
    name_fallbacks: int = 0


def _error(detail: str) -> _ChunkResult:
    return _ChunkResult("error", empty_frame("index"), detail)


def _conform(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with exactly the shared schema columns, order and dtypes."""
    df = df[_COLUMNS].copy()
    for column, dtype in SCHEMAS["index"].items():
        df[column] = df[column].astype(dtype)
    return df.reset_index(drop=True)


def _get_json(session: Any, url: str, params: Optional[Dict[str, str]]) -> Tuple[Any, str]:
    """GET url and decode JSON. Returns (payload, "") or (None, error detail)."""
    try:
        response = session.get(url, params=params, timeout=_TIMEOUT_SECONDS)
        try:
            response.raise_for_status()
        except Exception as exc:
            body = str(getattr(response, "text", ""))[:200]
            return None, f"{exc} body={body!r}"
        return response.json(), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _exact_volume(value: Any) -> Tuple[str, int]:
    """Classify one raw HIT_TRADED_QTY value without ever rounding it.

    Returns ("ok", n) for a non-negative integer that fits int64 exactly,
    ("missing", 0) for None or a non-numeric value (the caller defaults it to
    0 and counts it), or ("invalid", 0) for a numeric value that is negative,
    fractional, too large, or a float too large to be verified exact.
    """
    if value is None or isinstance(value, bool):
        return "missing", 0
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "missing", 0
        if value != int(value) or abs(value) > _MAX_EXACT_FLOAT:
            return "invalid", 0
        number = int(value)
    elif isinstance(value, str):
        try:
            decimal = Decimal(value.strip())
        except InvalidOperation:
            return "missing", 0
        if not decimal.is_finite():
            return "missing", 0
        if decimal.adjusted() > 19:                 # far above int64: never expand it
            return "invalid", 0
        if decimal != decimal.to_integral_value():
            return "invalid", 0
        number = int(decimal)
    else:
        return "missing", 0
    if number < 0 or number > _INT64_MAX:
        return "invalid", 0
    return "ok", number


def _parse_records(records: List[Any], requested_name: str) -> _ChunkResult:
    """Validate raw NSE records and map them to the shared index schema."""
    if not all(isinstance(r, dict) for r in records):
        return _error("records are not JSON objects")
    raw = pd.DataFrame(records)

    missing = [f for f in _REQUIRED_FIELDS if f not in raw.columns]
    if missing:
        return _error(f"missing fields {missing}")

    dates = pd.to_datetime(raw["EOD_TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
    if dates.isna().any():
        return _error(f"{int(dates.isna().sum())} unparseable EOD_TIMESTAMP value(s)")

    out = pd.DataFrame({"date": dates})
    if "EOD_INDEX_NAME" in raw.columns:
        names = raw["EOD_INDEX_NAME"].astype(object).where(raw["EOD_INDEX_NAME"].notna(), "")
        names = names.map(lambda v: " ".join(str(v).split()))
        fallbacks = int((names == "").sum())
        out["index_name"] = names.where(names != "", requested_name)
    else:
        fallbacks = len(raw)
        out["index_name"] = requested_name

    for field, column in _PRICE_FIELDS.items():
        values = pd.to_numeric(raw[field], errors="coerce").astype("float64")
        if (~np.isfinite(values)).any():
            return _error(f"{int((~np.isfinite(values)).sum())} missing or malformed {field} value(s)")
        out[column] = values

    # Volumes are read from the raw records, not from the DataFrame: pandas
    # turns a column of big ints with a None into float64 and rounds them.
    classified = [_exact_volume(record.get("HIT_TRADED_QTY")) for record in records]
    invalid = sum(1 for kind, _ in classified if kind == "invalid")
    if invalid:
        return _error(f"{invalid} invalid HIT_TRADED_QTY value(s) (negative, fractional, "
                      "above int64, or a float too large to be verified exact)")
    defaulted = sum(1 for kind, _ in classified if kind == "missing")
    out["volume"] = pd.Series([number for _, number in classified], index=raw.index, dtype="int64")
    out["source"] = "NSE"
    return _ChunkResult("ok", _conform(out), "", defaulted, fallbacks)


def _request_chunk(session: Any, index_name: str, from_date: str, to_date: str) -> _ChunkResult:
    """One request, validated. Dates are DD-MM-YYYY. Never raises, never logs."""
    payload, detail = _get_json(
        session, _HISTORY_URL,
        {"indexType": index_name, "from": from_date, "to": to_date},
    )
    if payload is None:
        return _error(detail)
    if not isinstance(payload, dict):
        return _error("response is not a JSON object")
    if payload.get("error"):
        return _error(f"NSE error payload: {str(payload)[:200]}")
    records = payload.get("data")
    if not isinstance(records, list):
        return _error("response has no 'data' list")
    if not records:
        return _ChunkResult("empty", empty_frame("index"))
    if len(records) >= _ROW_CAP:
        return _ChunkResult("capped", empty_frame("index"),
                            f"{len(records)} rows (cap {_ROW_CAP})")
    try:
        return _parse_records(records, index_name)
    except Exception as exc:  # defensive: malformed payload must not raise
        return _error(f"could not parse records: {type(exc).__name__}: {exc}")


def _fetch_indices_chunk(session: Any, index_name: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Fetch one chunk (kept for tools/capture_fixtures.py).

    Args:
        session: An NSE session (anything with ``get``).
        index_name: NSE index name.
        from_date: Start date, DD-MM-YYYY.
        to_date: End date, DD-MM-YYYY.

    Returns:
        DataFrame in the shared "index" schema. Empty on failure, on no data,
        or if the response reached the row cap (it cannot be trusted).
    """
    result = _request_chunk(session, index_name, from_date, to_date)
    if result.status == "error":
        logger.error("[Indices] Chunk fetch error: %s", result.detail)
    elif result.status == "capped":
        logger.error("[Indices] Chunk reached the row cap: %s", result.detail)
    elif result.name_fallbacks:
        logger.warning("[Indices] %d record(s) had no EOD_INDEX_NAME; used the requested name %r",
                       result.name_fallbacks, index_name)
    return result.frame


def _clean_index_name(index_name: Any) -> Optional[str]:
    """Strip and collapse whitespace. Case is kept (NSE ignores it)."""
    if not isinstance(index_name, str):
        return None
    cleaned = " ".join(index_name.split())
    return cleaned or None


def _load_index_names(session: Any) -> Optional[List[Tuple[str, str]]]:
    """Return (short, long) pairs from NSE's live list, cached per process."""
    global _INDEX_NAME_PAIRS
    if _INDEX_NAME_PAIRS is not None:
        return _INDEX_NAME_PAIRS
    payload, detail = _get_json(session, _NAMES_URL, None)
    if payload is None:
        logger.warning("[Indices] Could not fetch the index-name list: %s", detail)
        return None
    stn = payload.get("stn") if isinstance(payload, dict) else None
    pairs: List[Tuple[str, str]] = []
    if isinstance(stn, list):
        for entry in stn:
            if (isinstance(entry, (list, tuple)) and len(entry) == 2
                    and all(isinstance(v, str) for v in entry)):
                pairs.append((" ".join(entry[0].split()), " ".join(entry[1].split())))
            else:
                pairs = []
                break
    if not pairs:
        logger.warning("[Indices] Index-name list has an unexpected shape; ignoring it")
        return None
    _INDEX_NAME_PAIRS = pairs
    return pairs


def _reset_index_name_cache() -> None:
    """Forget the cached name list (used by tests)."""
    global _INDEX_NAME_PAIRS
    _INDEX_NAME_PAIRS = None


def _resolve_alias(session: Any, name: str) -> Optional[str]:
    """Return the long name if name is a known short alias, else None.

    Logs an error with close suggestions if name is not in NSE's list at all.
    Case-insensitive. An ambiguous name is never aliased.
    """
    pairs = _load_index_names(session)
    if pairs is None:
        return None
    key = name.lower()
    longs = {long.lower() for _, long in pairs}
    targets: Dict[str, set] = {}
    for short, long in pairs:
        if short.lower() != long.lower():
            targets.setdefault(short.lower(), set()).add(long)
    if key in longs:
        return None
    if key in targets:
        if len(targets[key]) == 1:
            return next(iter(targets[key]))
        logger.warning("[Indices] %r maps to several long names; not resolving it", name)
        return None
    known = sorted({n for pair in pairs for n in pair})
    by_lower = {k.lower(): k for k in known}
    suggestions = difflib.get_close_matches(key, list(by_lower), n=3, cutoff=0.6)
    logger.error("[Indices] %r is not in NSE's index list. Close matches: %s. "
                 "Continuing with the name as given in case the list is stale.",
                 name, [by_lower[s] for s in suggestions] or "none")
    return None


class _Walker:
    """Fetches date ranges one request at a time with pacing and a budget."""

    def __init__(self, session: Any, max_requests: int) -> None:
        self.session = session
        self.max_requests = max_requests
        self.requests = 0
        self.volume_defaulted = 0
        self.name_fallbacks = 0

    def pace(self) -> None:
        if self.requests:
            time.sleep(_REQUEST_DELAY_SECONDS)
        self.requests += 1

    def fetch_range(self, name: str, start: date, end: date
                    ) -> Tuple[List[pd.DataFrame], str, bool]:
        """Fetch [start, end] inclusive, splitting capped responses.

        Returns (frames, error, saw_data). error is "" on success. Terminates
        because every split makes both halves strictly shorter, a single day is
        never split, and a request budget backs this up.
        """
        frames: List[pd.DataFrame] = []
        stack = [(start, end)]
        while stack:
            lo, hi = stack.pop()
            if self.requests >= self.max_requests:
                return [], "request budget exhausted", False
            self.pace()
            result = _request_chunk(self.session, name, lo.strftime("%d-%m-%Y"), hi.strftime("%d-%m-%Y"))
            if result.status == "error":
                return [], f"{name} {lo} to {hi}: {result.detail}", False
            if result.status == "capped":
                if lo == hi:
                    return [], f"{name} {lo}: single day reached the row cap ({result.detail})", False
                mid = lo + timedelta(days=(hi - lo).days // 2)
                logger.info("[Indices] %s to %s reached the row cap; splitting at %s", lo, hi, mid)
                stack.append((mid + timedelta(days=1), hi))
                stack.append((lo, mid))
                continue
            if result.status == "ok":
                self.volume_defaulted += result.volume_defaulted
                self.name_fallbacks += result.name_fallbacks
                frames.append(result.frame)
        return frames, "", bool(frames)


def _max_requests(total_days: int, first_chunk_days: int) -> int:
    """Upper bound on requests for one call, with a safety margin.

    Splitting a span of L days into single days takes at most 2L - 1 requests
    (a binary tree with L leaves), so all chunks together need at most
    2 * total_days. An alias retry re-walks the first chunk once (at most
    2 * first_chunk_days - 1 more) and fetches the name list once (1 more).
    """
    return 2 * total_days + 2 * first_chunk_days + 1 + 10


def _fail(message: str) -> pd.DataFrame:
    logger.error("[Indices] %s", message)
    return empty_frame("index")


def _fetch_history(name: str, start: date, end: date) -> pd.DataFrame:
    session = create_nse_session()
    spans: List[Tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=_CHUNK_DAYS), end)
        spans.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)

    first_days = (spans[0][1] - spans[0][0]).days + 1
    walker = _Walker(session, _max_requests((end - start).days + 1, first_days))
    active_name = name
    frames: List[pd.DataFrame] = []
    for position, (lo, hi) in enumerate(spans):
        got, error, saw_data = walker.fetch_range(active_name, lo, hi)
        if error:
            return _fail(f"History incomplete, returning no data. {error}")
        if position == 0 and not saw_data:
            walker.pace()
            long_name = _resolve_alias(session, name)
            if long_name and long_name != active_name:
                logger.info("[Indices] %r returned nothing; retrying as %r", name, long_name)
                active_name = long_name
                got, error, _ = walker.fetch_range(active_name, lo, hi)
                if error:
                    return _fail(f"History incomplete, returning no data. {error}")
        frames.extend(got)

    if not frames:
        logger.warning("[Indices] No data returned for %s", name)
        return empty_frame("index")

    df = pd.concat(frames, ignore_index=True)
    inside = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    if not inside.all():
        logger.warning("[Indices] Dropped %d row(s) outside %s to %s", int((~inside).sum()), start, end)
        df = df[inside]
    if df.empty:
        logger.warning("[Indices] No data returned for %s", name)
        return empty_frame("index")

    repeated = df.duplicated(subset=["date"], keep=False)
    if repeated.any() and df[repeated].drop_duplicates().duplicated(subset=["date"], keep=False).any():
        return _fail("Conflicting records for the same date, returning no data")
    df = df.drop_duplicates().sort_values("date", kind="stable")

    if walker.volume_defaulted:
        logger.warning("[Indices] %d volume value(s) were missing and set to 0", walker.volume_defaulted)
    if walker.name_fallbacks:
        logger.warning("[Indices] %d record(s) had no EOD_INDEX_NAME; used the requested name %r",
                       walker.name_fallbacks, active_name)
    logger.info("[Indices] %s: %d rows fetched", name, len(df))
    return _conform(df)


def fetch_index_history(index_name: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Fetch historical index data from NSE.

    Long ranges are fetched in 60-day chunks; any response that reaches NSE's
    ~70-row cap is split and refetched. If any chunk fails, or a single day
    reaches the cap, the result is an empty frame and the failure is logged:
    partial history is never returned. The name may be a long/display name or
    a short alias (resolved from NSE's live list when needed); case is ignored.

    Args:
        index_name: Index name, e.g. "NIFTY 50" or "NIFTY FINANCIAL SERVICES".
        from_date: Start date, YYYY-MM-DD.
        to_date: End date, YYYY-MM-DD.

    Returns:
        DataFrame with columns date, index_name, open, high, low, close,
        volume, source (shared "index" schema), oldest first. ``index_name`` is
        NSE's own label. Empty standard frame on invalid input or any failure.
    """
    name = _clean_index_name(index_name)
    if name is None:
        return _fail("index_name must be a non-empty string")
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = datetime.strptime(to_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return _fail("Invalid date format. Use YYYY-MM-DD")
    if start > end:
        return _fail("from_date cannot be after to_date")

    logger.info("[Indices] Fetching %s from %s to %s", name, from_date, to_date)
    try:
        return _fetch_history(name, start, end)
    except Exception as exc:
        return _fail(f"Unexpected failure ({type(exc).__name__}: {exc}), returning no data")


def list_indices() -> List[str]:
    """List the index names that ``fetch_index_history`` accepts.

    Built from NSE's live ``/api/index-names`` list (long/display names, no
    duplicates, NSE's order), minus entries that are not indices
    ("Securities in F&O", "Permitted to Trade").

    Limitations: appearing in this list means NSE currently publishes the name,
    NOT that history exists. Some listed indices (for example a matured bond
    index) may have no recent data, and recent availability does not guarantee
    historical availability for an older window. Short aliases are also
    accepted by ``fetch_index_history`` but are not listed here.

    Returns:
        List of index names, or an empty list if NSE could not be reached or
        returned an unexpected response (the failure is logged).
    """
    try:
        pairs = _INDEX_NAME_PAIRS
        if pairs is None:
            pairs = _load_index_names(create_nse_session())
    except Exception as exc:
        logger.error("[Indices] Could not list indices: %s: %s", type(exc).__name__, exc)
        return []
    if pairs is None:
        return []
    names: List[str] = []
    seen = set()
    for short, long in pairs:
        if short.lower() in _NON_INDEX_ENTRIES or long.lower() in _NON_INDEX_ENTRIES:
            continue
        if long.lower() not in seen:
            seen.add(long.lower())
            names.append(long)
    return names
