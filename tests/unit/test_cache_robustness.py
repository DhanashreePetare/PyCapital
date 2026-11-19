"""Offline robustness tests for the cache (reliability/cache.py, stage 3):
concurrency, cross-process access, locked/read-only storage, races between
the coverage check and the read, stale data, and connection hygiene.
"""
import os
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyequity.reliability import cache as c
from pyequity.schema import SCHEMAS, empty_frame

NOW0 = pd.Timestamp("2025-06-10 12:00", tz="Asia/Kolkata").timestamp()
ID = {"source": "NSE", "symbol": "RELIANCE", "series": "EQ"}
EQUITY_SCHEMA = dict(SCHEMAS["equity"])  # frozen: tests may patch SCHEMAS
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
POSIX_PERMS = os.name != "nt" and not IS_ROOT


class Clock:
    def __init__(self, t=NOW0):
        self.t = t

    def __call__(self):
        return self.t


class FakeEquity:
    """Thread-safe business-day fetcher that records its calls."""

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, a, b):
        with self._lock:
            self.calls.append((a, b))
        days = pd.date_range(a, b, freq="B")
        if len(days) == 0:
            return empty_frame("equity")
        n = len(days)
        df = pd.DataFrame({"date": days, "open": 100.0 + days.dayofyear,
                           "high": 101.5 + days.dayofyear, "low": 99.25 + days.dayofyear,
                           "close": 100.1 + days.dayofyear, "volume": np.arange(n) + 1000,
                           "trades": np.arange(n) + 10, "symbol": "RELIANCE", "source": "NSE"})
        return df.astype(EQUITY_SCHEMA)


def get(dc, f, a, b, ident=ID):
    return dc.get_series("equity", ident, a, b, f)


def dates(df):
    return df["date"].dt.strftime("%Y-%m-%d").tolist()


def expected_dates(a, b):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(a, b)]


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def dc(tmp_path, clock):
    return c.DiskCache(tmp_path, clock=clock)


def run_threads(targets):
    errors, threads = [], []
    barrier = threading.Barrier(len(targets))

    def wrap(fn):
        def inner():
            try:
                barrier.wait()
                fn()
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)
        return inner
    threads = [threading.Thread(target=wrap(t)) for t in targets]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert not any(t.is_alive() for t in threads), "a thread hung"
    assert errors == []


# ---- concurrency ------------------------------------------------------------
def test_threads_on_disjoint_ranges_share_one_cache(dc):
    f = FakeEquity()
    months = [(f"2025-{m:02d}-01", f"2025-{m:02d}-28") for m in range(1, 6)]  # Jan-May: settled
    results = {}

    def job(a, b):
        return lambda: results.__setitem__(a, get(dc, f, a, b))
    run_threads([job(a, b) for a, b in months])
    for a, b in months:
        assert dates(results[a]) == expected_dates(a, b)
    total = sum(len(expected_dates(a, b)) for a, b in months)
    assert dc.stats()["rows"] == total
    f.calls.clear()
    full = get(dc, f, "2025-01-01", "2025-05-28")
    assert sorted(f.calls) == [("2025-01-29", "2025-01-31"), ("2025-03-29", "2025-03-31"),
                               ("2025-04-29", "2025-04-30")]      # only the gaps
    assert dates(full) == expected_dates("2025-01-01", "2025-05-28")


def test_threads_on_the_same_range_never_duplicate_rows(dc):
    f = FakeEquity()
    outs = []
    run_threads([lambda: outs.append(get(dc, f, "2025-01-01", "2025-03-31")) for _ in range(12)])
    want = expected_dates("2025-01-01", "2025-03-31")
    assert all(dates(o) == want for o in outs)
    assert dc.stats()["rows"] == len(want)
    assert dc.stats()["coverage"] == 1


def test_separate_instances_like_separate_processes_on_one_folder(tmp_path, clock):
    f = FakeEquity()
    caches = [c.DiskCache(tmp_path, clock=clock) for _ in range(6)]
    outs = []

    def job(i):
        def run():
            d = caches[i]
            outs.append(get(d, f, "2025-01-01", "2025-02-28"))
            d.get_snapshot("corporate_actions", {"source": "NSE", "symbol": "X"},
                           lambda: empty_frame("corporate_actions"))
            if i % 3 == 0:
                d.invalidate("corporate_actions")
            outs.append(get(d, f, "2025-02-01", "2025-03-15"))
        return run
    run_threads([job(i) for i in range(6)])
    for o in outs:
        assert not o["date"].duplicated().any() and o["date"].is_monotonic_increasing
    final = c.DiskCache(tmp_path, clock=clock)
    again = FakeEquity()
    out = get(final, again, "2025-01-01", "2025-03-15")
    assert dates(out) == expected_dates("2025-01-01", "2025-03-15")
    assert again.calls == []


def test_first_use_from_many_threads_creates_one_store(tmp_path, clock, monkeypatch):
    made = []

    class Counting(c.CacheStore):
        def __init__(self, *a, **k):
            made.append(1)
            super().__init__(*a, **k)
    monkeypatch.setattr(c, "CacheStore", Counting)
    d = c.DiskCache(tmp_path, clock=clock)
    f = FakeEquity()
    run_threads([lambda: get(d, f, "2025-01-01", "2025-01-31") for _ in range(16)])
    assert len(made) == 1


SCRIPT = '''
import sys
import numpy as np, pandas as pd
from pyequity.reliability.cache import DiskCache
from pyequity.schema import SCHEMAS

def fetch(a, b):
    days = pd.date_range(a, b, freq="B")
    if len(days) == 0:
        return pd.DataFrame({k: pd.Series(dtype=t) for k, t in SCHEMAS["equity"].items()})
    n = len(days)
    return pd.DataFrame({"date": days, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                         "volume": np.arange(n), "trades": np.arange(n),
                         "symbol": "RELIANCE", "source": "NSE"}).astype(SCHEMAS["equity"])

d = DiskCache(sys.argv[1])
for _ in range(4):
    d.get_series("equity", {"source": "NSE", "symbol": "RELIANCE", "series": "EQ"},
                 sys.argv[2], sys.argv[3], fetch)
'''


def test_real_processes_writing_the_same_cache_at_once(tmp_path):
    src = str(Path(c.__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    ranges = [("2025-01-01", "2025-02-28"), ("2025-03-01", "2025-04-30"),
              ("2025-05-01", "2025-06-30"), ("2025-01-01", "2025-06-30")]
    procs = [subprocess.Popen([sys.executable, "-c", SCRIPT, str(tmp_path), a, b], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) for a, b in ranges]
    for p in procs:
        out, err = p.communicate(timeout=180)
        assert p.returncode == 0, err.decode()
    d = c.DiskCache(tmp_path)  # real clock; the dates are long settled
    f = FakeEquity()
    out = get(d, f, "2025-01-01", "2025-06-30")
    assert f.calls == []
    assert dates(out) == expected_dates("2025-01-01", "2025-06-30")
    assert d.stats()["rows"] == len(out)


# ---- races between the coverage check and the read ---------------------------
def test_cache_cleared_between_check_and_read_refetches_instead_of_dropping_rows(dc, monkeypatch):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-31")
    real = dc._store.read_series

    def racing(*a, **k):
        dc._store.clear()          # someone else wipes the cache right here
        return real(*a, **k)
    monkeypatch.setattr(dc._store, "read_series", racing)
    f.calls.clear()
    out = get(dc, f, "2025-01-01", "2025-01-31")
    assert dates(out) == expected_dates("2025-01-01", "2025-01-31")
    assert f.calls == [("2025-01-01", "2025-01-31")]


def test_read_series_with_now_requires_live_coverage(tmp_path):
    s = c.CacheStore(tmp_path)
    df = FakeEquity()("2025-01-02", "2025-01-10")
    iv = (date(2025, 1, 2), date(2025, 1, 10))
    s.record_series("equity", "k", iv, df, now=NOW0, ttl=100)
    assert len(s.read_series("equity", "k", *iv, now=NOW0 + 50)) == len(df)
    assert s.read_series("equity", "k", *iv, now=NOW0 + 500) is None       # expired
    assert s.read_series("equity", "k", date(2025, 1, 1), iv[1], now=NOW0) is None  # not covered
    assert len(s.read_series("equity", "k", *iv)) == len(df)               # legacy call unchanged


def test_coverage_check_and_row_read_are_one_snapshot(tmp_path, monkeypatch):
    """A clear that lands between the two SELECTs must not produce a short frame."""
    s = c.CacheStore(tmp_path)
    df = FakeEquity()("2025-01-02", "2025-01-10")
    iv = (date(2025, 1, 2), date(2025, 1, 10))
    s.record_series("equity", "k", iv, df, now=NOW0, ttl=100)
    real = c.subtract_intervals

    def racing(request, covered):          # runs between the coverage and row queries
        monkeypatch.setattr(c, "subtract_intervals", real)
        s.clear()
        return real(request, covered)
    monkeypatch.setattr(c, "subtract_intervals", racing)
    out = s.read_series("equity", "k", *iv, now=NOW0)
    assert out is not None and len(out) == len(df)    # complete (old) data, never partial


def test_expired_rows_are_never_served_as_if_valid(dc, clock):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-31")
    clock.t = NOW0 + 31 * 86400
    out = get(dc, lambda a, b: empty_frame("equity"), "2025-01-01", "2025-01-31")
    assert out.empty            # stale rows stay on disk but are not returned


def test_schema_change_between_sessions_forces_a_refetch(tmp_path, clock, monkeypatch):
    f = FakeEquity()
    get(c.DiskCache(tmp_path, clock=clock), f, "2025-01-01", "2025-01-31")
    monkeypatch.setitem(c.SCHEMAS, "equity", {**SCHEMAS["equity"], "extra": "float64"})

    def new_shape(a, b):
        return f(a, b).assign(extra=1.5)
    d2 = c.DiskCache(tmp_path, clock=clock)
    assert d2.stats()["rows"] == 0
    f.calls.clear()
    out = get(d2, new_shape, "2025-01-01", "2025-01-31")
    assert f.calls and "extra" in out.columns


# ---- locked / read-only storage -----------------------------------------------
def test_locked_database_degrades_quickly_then_recovers(tmp_path, clock, monkeypatch):
    monkeypatch.setattr(c.CacheStore, "BUSY_TIMEOUT", 0.2)
    d = c.DiskCache(tmp_path, clock=clock)
    d.stats()                                   # create the database
    blocker = sqlite3.connect(str(tmp_path / c.DB_NAME))
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        f = FakeEquity()
        started = time.time()
        out = get(d, f, "2025-01-01", "2025-01-31")
        assert time.time() - started < 10
        assert dates(out) == expected_dates("2025-01-01", "2025-01-31")
    finally:
        blocker.rollback()
        blocker.close()
    f.calls.clear()
    get(d, f, "2025-01-01", "2025-01-31")       # the failed write was not remembered
    assert len(f.calls) == 1
    get(d, f, "2025-01-01", "2025-01-31")       # and now it works normally
    assert len(f.calls) == 1


@pytest.mark.skipif(not POSIX_PERMS, reason="needs POSIX permissions and a non-root user")
def test_read_only_database_file_degrades(tmp_path, clock):
    d = c.DiskCache(tmp_path, clock=clock)
    get(d, FakeEquity(), "2025-01-01", "2025-01-31")
    del d
    files = [p for p in tmp_path.iterdir()]
    for p in files:
        os.chmod(p, 0o444)
    try:
        d2 = c.DiskCache(tmp_path, clock=clock)
        f = FakeEquity()
        out = get(d2, f, "2025-01-01", "2025-01-31")
        assert dates(out) == expected_dates("2025-01-01", "2025-01-31")
        assert f.calls                          # went to the fetcher, did not raise
        assert d2.invalidate("equity") in (True, False)
    finally:
        for p in files:
            os.chmod(p, 0o644)


@pytest.mark.skipif(not POSIX_PERMS, reason="needs POSIX permissions and a non-root user")
def test_read_only_directory_degrades(tmp_path, clock):
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o555)
    try:
        d = c.DiskCache(ro / "cache", clock=clock)
        f = FakeEquity()
        out = get(d, f, "2025-01-01", "2025-01-31")
        assert len(out) and f.calls == [("2025-01-01", "2025-01-31")]
        assert d.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}
    finally:
        os.chmod(ro, 0o755)


# ---- connection hygiene ----------------------------------------------------------
def test_every_sqlite_connection_is_closed(tmp_path, clock, monkeypatch):
    opened = []
    real = sqlite3.connect

    def tracking(*a, **k):
        con = real(*a, **k)
        opened.append(con)
        return con
    monkeypatch.setattr(c.sqlite3, "connect", tracking)
    d = c.DiskCache(tmp_path, clock=clock)
    f = FakeEquity()
    get(d, f, "2025-01-01", "2025-01-31")
    get(d, f, "2025-01-01", "2025-02-15")
    d.get_snapshot("corporate_actions", {"source": "NSE", "symbol": "X"},
                   lambda: empty_frame("corporate_actions"))
    d.invalidate("equity")
    d.stats()
    d.clear()
    assert len(opened) > 5
    for con in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            con.execute("select 1")


def test_cache_folder_can_be_deleted_after_use(tmp_path, clock):
    import shutil
    folder = tmp_path / "cache"
    d = c.DiskCache(folder, clock=clock)
    get(d, FakeEquity(), "2025-01-01", "2025-01-31")
    shutil.rmtree(folder)                       # fails on Windows if a connection leaked
    assert not folder.exists()