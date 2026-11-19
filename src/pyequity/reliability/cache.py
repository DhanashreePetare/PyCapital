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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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


def frame_to_rows(df: pd.DataFrame, dataset: str) -> List[Tuple[str, str]]:
    """Encode a standard frame as ``(iso_date, json_payload)`` tuples.

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
        if d is None:
            continue  # a row without a date can never be looked up
        out.append((str(d)[:10], json.dumps(vals, separators=(",", ":"))))
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

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_file), timeout=10)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init(self) -> bool:
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            with self._lock, self._connect() as con:
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
            with self._lock, self._connect() as con:
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
            con.execute("INSERT INTO coverage VALUES (?,?,?,?,?,?)",
                        (dataset, key, a, b, now, now + ttl))
            return True
        return bool(self._run(fn, False))

    def read_series(self, dataset: str, key: str, start: date, end: date
                    ) -> Optional[pd.DataFrame]:
        """Stored rows with ``start <= date <= end`` in insertion order.

        Returns:
            A schema-exact frame, or None if the database failed.
        """
        _check_dataset(dataset)

        def fn(con):
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
            payloads = [p for _, p in frame_to_rows(df, dataset)]
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