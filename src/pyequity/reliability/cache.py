"""
Local disk cache for PyEquity (storage layer).

This module holds the building blocks of the cache: location and on/off
switch, key handling, interval arithmetic, TTL rules and a SQLite-backed
store. It never talks to an exchange and never imports a connector.

Rules the storage layer enforces
--------------------------------
* A range is recorded as *covered* only through ``CacheStore.record_series``,
  and only when the caller says the fetch was complete. An empty result is
  never recorded as covered, so a failed fetch is simply asked again.
* Rows are stored exactly: floats keep every digit, NaN and NaT come back as
  NaN and NaT, and a cache hit has the same columns, dtypes and order as
  ``schema.empty_frame(dataset)``.
* Raw data only. Adjusted prices are never cached.
* The store never breaks retrieval: any SQLite problem (corrupt file, locked
  file, read-only disk) is logged and reported through return values
  (``None`` / ``False``), never raised. Callers fall back to the fetcher.
* Unknown dataset name -> ``ValueError`` (developer mistake, as in
  ``normalize.py``).

``DiskCache`` (bottom of this file) sits on top of the store and does the
range-aware work: it fetches only the missing date spans, records coverage only
for settled, non-empty, identity-checked data, and always falls back to the
caller's fetcher if the cache cannot be used.

Storage format: one SQLite file (standard library only). Each row is stored
as a JSON array in schema column order plus its ISO date, so range queries
stay cheap and floats round-trip exactly.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import sys
import threading
import time
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from pyequity.schema import SCHEMAS, empty_frame
from pyequity.processing.normalize import DATE_COLUMN

logger = logging.getLogger(__name__)

ENV_DIR = "PYEQUITY_CACHE_DIR"
ENV_SWITCH = "PYEQUITY_CACHE"
DB_NAME = "pyequity_cache.sqlite3"
IST = timezone(timedelta(hours=5, minutes=30))

Interval = Tuple[date, date]  # inclusive on both ends


# ----------------------------------------------------------------- settings
def cache_enabled() -> bool:
    """Return False when ``PYEQUITY_CACHE`` is off/0/false/no, else True."""
    return os.environ.get(ENV_SWITCH, "on").strip().lower() not in (
        "off", "0", "false", "no")


def default_cache_dir() -> Path:
    """Cache folder: ``PYEQUITY_CACHE_DIR``, else a per-user folder.

    Windows: ``%LOCALAPPDATA%\\pyequity\\cache``; macOS:
    ``~/Library/Caches/pyequity``; elsewhere ``$XDG_CACHE_HOME/pyequity``
    or ``~/.cache/pyequity``.

    Returns:
        Path of the folder (not created here).
    """
    override = os.environ.get(ENV_DIR)
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "pyequity" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "pyequity"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "pyequity"


def today_ist(now: Optional[float] = None) -> date:
    """Current calendar date in IST (exchange time), from epoch seconds."""
    ts = time.time() if now is None else now
    return datetime.fromtimestamp(ts, tz=IST).date()


# --------------------------------------------------------------------- keys
def make_key(identity: Mapping[str, Any]) -> str:
    """Canonical, order-independent string key for a dataset identity.

    Args:
        identity: e.g. ``{"source": "NSE", "symbol": "RELIANCE", "series": "EQ"}``.
            Values are upper-cased and stripped when they are strings; None
            values are kept (they distinguish "no filter" from a value).

    Returns:
        A JSON string with sorted keys.
    """
    clean: Dict[str, Any] = {}
    for k, v in identity.items():
        if isinstance(v, str):
            v = v.strip().upper()
        elif isinstance(v, float) and math.isnan(v):
            v = None
        clean[str(k)] = v
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)


def schema_signature(dataset: str) -> str:
    """Fingerprint of a dataset's columns and dtypes (changes invalidate rows)."""
    _check_dataset(dataset)
    return json.dumps(list(SCHEMAS[dataset].items()), separators=(",", ":"))


def _check_dataset(dataset: str) -> None:
    if dataset not in SCHEMAS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(SCHEMAS)}")


# ---------------------------------------------------------------- intervals
def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    """Sort and merge overlapping or *adjacent* (consecutive-day) intervals."""
    out: List[Interval] = []
    for a, b in sorted(i for i in intervals if i[0] <= i[1]):
        if out and a <= out[-1][1] + timedelta(days=1):
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def subtract_intervals(request: Interval, covered: List[Interval]) -> List[Interval]:
    """Parts of ``request`` not inside any covered interval (inclusive ends)."""
    start, end = request
    if start > end:
        return []
    missing: List[Interval] = []
    cursor = start
    for a, b in merge_intervals(covered):
        if b < cursor:
            continue
        if a > end:
            break
        if a > cursor:
            missing.append((cursor, a - timedelta(days=1)))
        cursor = max(cursor, b + timedelta(days=1))
        if cursor > end:
            break
    if cursor <= end:
        missing.append((cursor, end))
    return missing


def settled_limit(now: Optional[float] = None, settle_days: int = 1) -> date:
    """Latest date that may be recorded as covered.

    Dates from ``today - settle_days + 1`` onward (IST) are still moving
    (partial sessions, late corrections) and are never recorded.
    """
    return today_ist(now) - timedelta(days=settle_days)


def ttl_for(interval_end: date, *, now: Optional[float], ttl_settled: float,
            ttl_recent: float, recent_days: int) -> float:
    """TTL in seconds: short for data in the last ``recent_days``, long before."""
    age_days = (today_ist(now) - interval_end).days
    return ttl_recent if age_days <= recent_days else ttl_settled


# -------------------------------------------------------------------- store
_DDL = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS coverage (
    dataset TEXT NOT NULL, key TEXT NOT NULL,
    start TEXT NOT NULL, end TEXT NOT NULL, fetched_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_cov ON coverage (dataset, key);
CREATE TABLE IF NOT EXISTS rows (
    dataset TEXT NOT NULL, key TEXT NOT NULL, d TEXT NOT NULL,
    seq INTEGER NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (dataset, key, d, seq)
);
CREATE TABLE IF NOT EXISTS snapshots (
    dataset TEXT NOT NULL, key TEXT NOT NULL, fetched_at REAL NOT NULL,
    expires_at REAL NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (dataset, key)
);
"""


def _enc(v: Any) -> Any:
    """Frame value -> JSON-safe value (NaN/NaT/None -> None, times -> ISO)."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalar
        v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    return v


def frame_to_rows(df: pd.DataFrame, dataset: str, *,
                  require_date: bool = True) -> List[Tuple[str, str]]:
    """Encode a standard frame as ``(iso_date, json_payload)`` tuples.

    Args:
        df: Frame with every schema column.
        dataset: Dataset name.
        require_date: True (series): rows without a date are skipped, since
            they can never be looked up. False (snapshots): every row is kept,
            with an empty date string when the date is missing.

    Raises:
        ValueError: unknown dataset, or the frame lacks a schema column.
    """
    _check_dataset(dataset)
    cols = list(SCHEMAS[dataset])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing columns {missing}")
    dcol = DATE_COLUMN[dataset]
    out: List[Tuple[str, str]] = []
    for rec in df[cols].itertuples(index=False, name=None):
        vals = [_enc(v) for v in rec]
        d = vals[cols.index(dcol)]
        if d is None and require_date:
            continue  # a row without a date can never be looked up
        out.append(("" if d is None else str(d)[:10],
                    json.dumps(vals, separators=(",", ":"))))
    return out


def rows_to_frame(payloads: List[str], dataset: str) -> pd.DataFrame:
    """Decode stored payloads into a frame with the exact schema dtypes."""
    _check_dataset(dataset)
    schema = SCHEMAS[dataset]
    if not payloads:
        return empty_frame(dataset)
    data = [json.loads(p) for p in payloads]
    out = {}
    for i, (col, dtype) in enumerate(schema.items()):
        values = [r[i] for r in data]
        if dtype.startswith("datetime64"):
            out[col] = pd.to_datetime(pd.Series(values), errors="coerce").astype(dtype)
        elif dtype == "object":
            out[col] = pd.Series(values, dtype="object")
        else:
            out[col] = pd.Series(values).astype(dtype)
    return pd.DataFrame(out)


class CacheStore:
    """SQLite persistence. Thread-safe; every method swallows DB errors.

    Args:
        directory: Folder for the database file (created if missing).
            Defaults to :func:`default_cache_dir`.
    """

    def __init__(self, directory: Optional[os.PathLike] = None) -> None:
        self.path = Path(directory) if directory else default_cache_dir()
        self.db_file = self.path / DB_NAME
        self._lock = threading.RLock()
        self._ok = self._init()

    @property
    def usable(self) -> bool:
        """False when the database could not be opened or created."""
        return self._ok

    BUSY_TIMEOUT = 10.0  # seconds to wait for another writer before giving up

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_file), timeout=self.BUSY_TIMEOUT)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init(self) -> bool:
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            with self._lock, closing(self._connect()) as con, con:
                con.executescript(_DDL)
                row = con.execute("SELECT v FROM meta WHERE k='schemas'").fetchone()
                sig = json.dumps({d: schema_signature(d) for d in sorted(SCHEMAS)})
                if row is not None and row[0] != sig:
                    logger.warning("cache: schema changed, clearing old cache")
                    for t in ("coverage", "rows", "snapshots"):
                        con.execute(f"DELETE FROM {t}")
                con.execute("INSERT OR REPLACE INTO meta VALUES ('schemas', ?)", (sig,))
            return True
        except (sqlite3.Error, OSError) as exc:
            logger.warning("cache disabled (cannot open %s): %s", self.db_file, exc)
            return False

    def _run(self, fn, default):
        if not self._ok:
            return default
        try:
            with self._lock, closing(self._connect()) as con, con:
                return fn(con)
        except (sqlite3.Error, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("cache operation failed, falling back to fetcher: %s", exc)
            return default

    # ---- series -----------------------------------------------------
    def covered(self, dataset: str, key: str, now: float) -> List[Interval]:
        """Unexpired covered intervals for a dataset/key, merged."""
        _check_dataset(dataset)

        def fn(con):
            cur = con.execute(
                "SELECT start, end FROM coverage WHERE dataset=? AND key=? "
                "AND expires_at>?", (dataset, key, now))
            return merge_intervals([(date.fromisoformat(a), date.fromisoformat(b))
                                    for a, b in cur.fetchall()])
        return self._run(fn, [])

    def record_series(self, dataset: str, key: str, interval: Interval,
                      df: pd.DataFrame, *, now: float, ttl: float) -> bool:
        """Replace rows in ``interval`` with ``df`` and record it as covered.

        Refuses (returns False) for an empty frame, so a failed or empty
        fetch is never remembered as "we have this range".
        """
        _check_dataset(dataset)
        if df is None or df.empty or interval[0] > interval[1]:
            return False
        try:
            rows = frame_to_rows(df, dataset)
        except ValueError as exc:
            logger.warning("cache: not storing, %s", exc)
            return False
        a, b = interval[0].isoformat(), interval[1].isoformat()
        rows = [r for r in rows if a <= r[0] <= b]
        if not rows:
            return False

        def fn(con):
            con.execute("DELETE FROM rows WHERE dataset=? AND key=? AND d BETWEEN ? AND ?",
                        (dataset, key, a, b))
            seq: Dict[str, int] = {}
            batch = []
            for d, p in rows:
                seq[d] = seq.get(d, -1) + 1
                batch.append((dataset, key, d, seq[d], p))
            con.executemany("INSERT INTO rows VALUES (?,?,?,?,?)", batch)
            con.execute("DELETE FROM coverage WHERE dataset=? AND key=? "
                        "AND start>=? AND end<=?", (dataset, key, a, b))
            con.execute("DELETE FROM coverage WHERE dataset=? AND key=? "
                        "AND expires_at<=?", (dataset, key, now))
            con.execute("INSERT INTO coverage VALUES (?,?,?,?,?,?)",
                        (dataset, key, a, b, now, now + ttl))
            return True
        return bool(self._run(fn, False))

    def read_series(self, dataset: str, key: str, start: date, end: date, *,
                    now: Optional[float] = None) -> Optional[pd.DataFrame]:
        """Stored rows with ``start <= date <= end`` in insertion order.

        Args:
            dataset: Dataset name.
            key: Key from :func:`make_key`.
            start, end: Inclusive date range.
            now: Epoch seconds. When given, the read first checks, in the same
                transaction, that unexpired coverage still includes the whole
                range. If another process cleared or replaced it in the
                meantime the result is None, never a silently short frame.

        Returns:
            A schema-exact frame, or None if the database failed or (with
            ``now``) the range is no longer covered.
        """
        _check_dataset(dataset)

        def fn(con):
            if now is not None:
                con.execute("BEGIN")  # one snapshot for the check and the read
                cur = con.execute(
                    "SELECT start, end FROM coverage WHERE dataset=? AND key=? "
                    "AND expires_at>?", (dataset, key, now))
                cov = [(date.fromisoformat(a), date.fromisoformat(b))
                       for a, b in cur.fetchall()]
                if subtract_intervals((start, end), cov):
                    logger.warning("cache: coverage for %s %s..%s disappeared before "
                                   "it could be read", dataset, start, end)
                    return None
            cur = con.execute(
                "SELECT payload FROM rows WHERE dataset=? AND key=? AND d BETWEEN ? AND ? "
                "ORDER BY d, seq", (dataset, key, start.isoformat(), end.isoformat()))
            return rows_to_frame([r[0] for r in cur.fetchall()], dataset)
        return self._run(fn, None)

    # ---- snapshots --------------------------------------------------
    def put_snapshot(self, dataset: str, key: str, df: pd.DataFrame, *,
                     now: float, ttl: float) -> bool:
        """Store a whole result under an exact key. Empty frames are refused."""
        _check_dataset(dataset)
        if df is None or df.empty:
            return False
        try:
            payloads = [p for _, p in frame_to_rows(df, dataset, require_date=False)]
        except ValueError as exc:
            logger.warning("cache: not storing snapshot, %s", exc)
            return False
        return bool(self._run(lambda con: con.execute(
            "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)",
            (dataset, key, now, now + ttl, json.dumps(payloads))) is not None, False))

    def get_snapshot(self, dataset: str, key: str, now: float
                     ) -> Optional[pd.DataFrame]:
        """Unexpired snapshot, or None on miss, expiry or DB failure."""
        _check_dataset(dataset)

        def fn(con):
            row = con.execute("SELECT payload FROM snapshots WHERE dataset=? AND key=? "
                              "AND expires_at>?", (dataset, key, now)).fetchone()
            return None if row is None else rows_to_frame(json.loads(row[0]), dataset)
        return self._run(fn, None)

    # ---- maintenance ------------------------------------------------
    def clear(self, dataset: Optional[str] = None, key: Optional[str] = None) -> bool:
        """Delete everything, one dataset, or one dataset/key."""
        def fn(con):
            for t in ("coverage", "rows", "snapshots"):
                sql, args = f"DELETE FROM {t}", []
                if dataset is not None:
                    sql += " WHERE dataset=?"
                    args.append(dataset)
                    if key is not None:
                        sql += " AND key=?"
                        args.append(key)
                con.execute(sql, args)
            return True
        return bool(self._run(fn, False))

    def stats(self) -> Dict[str, int]:
        """Row, coverage-interval and snapshot counts."""
        def fn(con):
            return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("rows", "coverage", "snapshots")}
        return self._run(fn, {"rows": 0, "coverage": 0, "snapshots": 0})


# ---------------------------------------------------------------- DiskCache
MAX_SPAN_FETCHES = 3  # more missing spans than this -> one fetch over their hull


def _parse_day(value: Any) -> Optional[date]:
    """YYYY-MM-DD string or date -> date; anything else -> None."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _conform(df: Any, dataset: str) -> Optional[pd.DataFrame]:
    """Copy of ``df`` with exactly the schema columns, order and dtypes.

    Returns None (and logs) when that is impossible, e.g. a column is missing
    or an integer column holds nulls. The input is never modified.
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning("cache: fetcher returned %s, not a DataFrame", type(df).__name__)
        return None
    schema = SCHEMAS[dataset]
    try:
        return df[list(schema)].copy().astype(schema)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("cache: fetched %s frame does not match the schema (%s); "
                       "it will not be cached", dataset, exc)
        return None


def _identity_matches(df: pd.DataFrame, dataset: str,
                      identity: Mapping[str, Any]) -> bool:
    """True when every row agrees with the identity the caller asked for.

    Checked for ``source`` and for every identity key that is a schema column
    (symbol, index_name, ...). A None value means "no filter". Identity keys
    that are not columns (e.g. ``series``) only separate cache keys.
    """
    schema = SCHEMAS[dataset]
    for k, v in identity.items():
        if v is None or not (k == "source" or k in schema):
            continue
        col = df[k]
        try:
            dtype = schema[k]
            if dtype == "object":
                ok = col.astype(str).str.strip().str.upper() == str(v).strip().upper()
            elif dtype.startswith("datetime64"):
                ok = col == pd.Timestamp(v)
            else:
                ok = pd.to_numeric(col, errors="coerce") == float(v)
        except (ValueError, TypeError):
            return False
        if not bool(ok.all()):
            return False
    return True


class DiskCache:
    """Range-aware disk cache for time-series datasets.

    The cache never calls a connector itself: callers pass a ``fetch``
    function. It stores only raw data, and only when the fetch looked complete
    (non-empty, right identity, right source, settled dates).

    Args:
        directory: Cache folder; default :func:`default_cache_dir`.
        ttl_settled: Seconds that data older than ``recent_days`` stays valid.
        ttl_recent: Seconds that data from the last ``recent_days`` stays valid.
        recent_days: Size of the "recent" window in days.
        settle_days: Dates within the last ``settle_days`` days (IST) are never
            recorded as covered. With 1, today is never stored.
        snapshot_ttl: Default snapshot TTL in seconds (used in stage 2).
        clock: Function returning epoch seconds; inject a fake in tests.
    """

    def __init__(self, directory: Optional[os.PathLike] = None, *,
                 ttl_settled: float = 30 * 86400.0, ttl_recent: float = 3600.0,
                 recent_days: int = 7, settle_days: int = 1,
                 snapshot_ttl: float = 86400.0,
                 clock: Optional[Callable[[], float]] = None) -> None:
        self.ttl_settled = float(ttl_settled)
        self.ttl_recent = float(ttl_recent)
        self.recent_days = int(recent_days)
        self.settle_days = int(settle_days)
        self.snapshot_ttl = float(snapshot_ttl)
        self._clock = clock or time.time
        self._directory = directory
        self._store_obj: Optional[CacheStore] = None
        self._store_lock = threading.Lock()

    @property
    def _store(self) -> CacheStore:
        """The SQLite store, created on first use (never when the cache is off)."""
        if self._store_obj is None:
            with self._store_lock:
                if self._store_obj is None:
                    self._store_obj = CacheStore(self._directory)
        return self._store_obj

    # ---- helpers ----------------------------------------------------
    def _active(self) -> bool:
        return cache_enabled() and self._store.usable

    def _record(self, dataset: str, key: str, span: Interval,
                df: pd.DataFrame, now: float) -> None:
        """Record settled parts of ``span``, split so each part gets its TTL."""
        end = min(span[1], settled_limit(now, self.settle_days))
        if end < span[0]:
            return
        boundary = today_ist(now) - timedelta(days=self.recent_days)
        pieces: List[Interval] = []
        if span[0] < boundary:
            pieces.append((span[0], min(end, boundary - timedelta(days=1))))
        if end >= boundary:
            pieces.append((max(span[0], boundary), end))
        for piece in pieces:
            if piece[0] > piece[1]:
                continue
            ttl = ttl_for(piece[1], now=now, ttl_settled=self.ttl_settled,
                          ttl_recent=self.ttl_recent, recent_days=self.recent_days)
            self._store.record_series(dataset, key, piece, df, now=now, ttl=ttl)

    # ---- public -----------------------------------------------------
    def get_series(self, dataset: str, identity: Mapping[str, Any], start: str,
                   end: str, fetch: Callable[[str, str], pd.DataFrame]
                   ) -> pd.DataFrame:
        """Return a dataset for ``start..end``, fetching only what is missing.

        Args:
            dataset: Name in ``schema.SCHEMAS`` (e.g. "index", "fno").
            identity: What the data is, e.g. ``{"source": "NSE",
                "index_name": "NIFTY 50"}``. Include ``source`` so a fallback
                to another exchange can never be stored under this key.
            start: First date, YYYY-MM-DD.
            end: Last date, YYYY-MM-DD (inclusive).
            fetch: ``fetch(start_iso, end_iso) -> DataFrame``. It must return
                complete data or an empty frame. Exceptions it raises are not
                caught, and nothing is stored for that call.

        Returns:
            Frame with the standard columns and dtypes, sorted by date, index
            0..n-1. Bad arguments give ``empty_frame(dataset)``. If the cache
            is off or unusable, whatever ``fetch(start, end)`` returns.

        Raises:
            ValueError: unknown dataset name.
        """
        _check_dataset(dataset)
        a, b = _parse_day(start), _parse_day(end)
        if (a is None or b is None or a > b or not isinstance(identity, Mapping)
                or not callable(fetch)):
            logger.warning("cache.get_series: invalid arguments (start=%r, end=%r)",
                           start, end)
            return empty_frame(dataset)
        if not self._active():
            return fetch(a.isoformat(), b.isoformat())

        now = self._clock()
        key = make_key(identity)
        covered = self._store.covered(dataset, key, now)
        missing = subtract_intervals((a, b), covered)
        if len(missing) > MAX_SPAN_FETCHES:
            missing = [(missing[0][0], missing[-1][1])]

        parts: List[pd.DataFrame] = []
        replaced: List[Interval] = []  # spans whose rows came from a fresh fetch
        for span in missing:
            raw = fetch(span[0].isoformat(), span[1].isoformat())
            frame = _conform(raw, dataset)
            if frame is None:  # unusable shape: hand it back, never store it
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    parts.append(raw)
                continue
            dcol = DATE_COLUMN[dataset]
            undated = int(frame[dcol].isna().sum())
            if undated:
                logger.warning("cache: %d %s row(s) have no date and are left out "
                               "of the cached result", undated, dataset)
            frame = frame[(frame[dcol] >= pd.Timestamp(span[0]))
                          & (frame[dcol] < pd.Timestamp(span[1]) + pd.Timedelta(days=1))]
            if frame.empty:
                logger.info("cache: empty result for %s %s..%s, not recorded",
                            dataset, span[0], span[1])
                continue
            replaced.append(span)
            if _identity_matches(frame, dataset, identity):
                self._record(dataset, key, span, frame, now)
            else:
                logger.warning("cache: fetched rows do not match identity %s; "
                               "returning them but not caching", key)
            parts.append(frame)

        # Cached rows are used only where coverage is valid and no fresh fetch
        # replaced them (an empty fetch leaves the old valid rows in place).
        valid = [(max(c0, a), min(c1, b)) for c0, c1 in covered if c1 >= a and c0 <= b]
        to_read: List[Interval] = []
        for iv in valid:
            to_read.extend(subtract_intervals(iv, replaced))
        for span in to_read:
            stored = self._store.read_series(dataset, key, span[0], span[1], now=now)
            if stored is None:  # database failed mid-way: ask the fetcher directly
                stored = _conform(fetch(span[0].isoformat(), span[1].isoformat()),
                                  dataset)
            if stored is not None and not stored.empty:
                parts.append(stored)

        parts = [p for p in parts if not p.empty]
        if not parts:
            return empty_frame(dataset)
        out = pd.concat(parts, ignore_index=True)
        if all(list(p.columns) == list(SCHEMAS[dataset]) for p in parts):
            out = out.sort_values(DATE_COLUMN[dataset], kind="mergesort")
        return out.reset_index(drop=True)

    def get_snapshot(self, dataset: str, identity: Mapping[str, Any],
                     fetch: Callable[[], pd.DataFrame],
                     ttl: Optional[float] = None) -> pd.DataFrame:
        """Return a whole result cached under an exact key, valid for a TTL.

        For data that is not a date series, such as corporate actions: new
        announcements show up once the entry expires. Empty results, frames
        that do not match the schema or the requested identity/source, and
        exceptions from ``fetch`` are never stored.

        Args:
            dataset: Name in ``schema.SCHEMAS``.
            identity: What the data is, e.g. ``{"source": "NSE", "symbol":
                "RELIANCE"}``. Include ``source`` so it is checked.
            fetch: ``fetch() -> DataFrame``. Exceptions it raises are not caught.
            ttl: Seconds the stored result stays valid. None uses the cache's
                ``snapshot_ttl`` (24 hours). A TTL of 0 or less, or one that is
                not a finite number, means "do not use the cache for this
                call". The TTL is fixed when the entry is written; a later
                call with a different ``ttl`` does not shorten it.

        Returns:
            Frame with the standard columns and dtypes and index 0..n-1, in
            the order ``fetch`` returned. Bad arguments give
            ``empty_frame(dataset)``. If the cache is off or unusable,
            whatever ``fetch()`` returns.

        Raises:
            ValueError: unknown dataset name.
        """
        _check_dataset(dataset)
        if not isinstance(identity, Mapping) or not callable(fetch):
            logger.warning("cache.get_snapshot: invalid arguments")
            return empty_frame(dataset)
        if not self._active():
            return fetch()

        life = self.snapshot_ttl if ttl is None else ttl
        usable_ttl = (isinstance(life, (int, float)) and not isinstance(life, bool)
                      and math.isfinite(life) and life > 0)
        now = self._clock()
        key = make_key(identity)
        if usable_ttl:
            hit = self._store.get_snapshot(dataset, key, now)
            if hit is not None:
                return hit

        raw = fetch()
        frame = _conform(raw, dataset)
        if frame is None:  # unusable shape: hand it back, never store it
            return raw if isinstance(raw, pd.DataFrame) else empty_frame(dataset)
        frame = frame.reset_index(drop=True)
        if frame.empty or not usable_ttl:
            return frame
        if _identity_matches(frame, dataset, identity):
            self._store.put_snapshot(dataset, key, frame, now=now, ttl=float(life))
        else:
            logger.warning("cache: fetched rows do not match identity %s; "
                           "returning them but not caching", key)
        return frame

    def invalidate(self, dataset: str, identity: Optional[Mapping[str, Any]] = None
                   ) -> bool:
        """Delete cached data for a dataset, or for one identity within it.

        Args:
            dataset: Name in ``schema.SCHEMAS``.
            identity: Same dict that was used to cache it; None removes the
                whole dataset.

        Returns:
            True if the deletion ran; False if the cache is off or unusable,
            or ``identity`` is not a mapping.

        Raises:
            ValueError: unknown dataset name.
        """
        _check_dataset(dataset)
        if identity is not None and not isinstance(identity, Mapping):
            logger.warning("cache.invalidate: identity must be a mapping or None")
            return False
        if not self._active():
            return False
        key = None if identity is None else make_key(identity)
        return self._store.clear(dataset, key)

    def clear(self) -> bool:
        """Delete everything in the cache. False if it is off or unusable."""
        if not self._active():
            return False
        return self._store.clear()

    def stats(self) -> Dict[str, int]:
        """Counts of stored ``rows``, ``coverage`` intervals and ``snapshots``.

        All zero if the cache is off or unusable (nothing is created on disk).
        """
        if not self._active():
            return {"rows": 0, "coverage": 0, "snapshots": 0}
        return self._store.stats()


# ------------------------------------------------------------ shared instance
_default_cache: Optional[DiskCache] = None
_default_lock = threading.Lock()


def get_default_cache() -> DiskCache:
    """The process-wide shared cache, created on first use.

    It uses :func:`default_cache_dir` and the standard policy. With
    ``PYEQUITY_CACHE=off`` it still returns an object, but that object never
    reads or writes the disk.
    """
    global _default_cache
    with _default_lock:
        if _default_cache is None:
            _default_cache = DiskCache()
        return _default_cache


def reset_default_cache() -> None:
    """Forget the shared instance (for tests, or after changing the cache dir)."""
    global _default_cache
    with _default_lock:
        _default_cache = None