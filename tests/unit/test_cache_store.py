"""Offline tests for the cache storage layer (reliability/cache.py, step 1)."""
import logging
import threading
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pyequity.reliability import cache as c
from pyequity.schema import SCHEMAS, empty_frame

NOW = 1_700_000_000.0  # fixed clock; tests never read the real time


def D(s):
    return date.fromisoformat(s)


def equity_frame(days, symbol="RELIANCE", source="NSE"):
    n = len(days)
    return pd.DataFrame({
        "date": pd.to_datetime(days).astype("datetime64[ns]"),
        "open": [100.1 + i for i in range(n)], "high": [101.3 + i for i in range(n)],
        "low": [99.7 + i for i in range(n)], "close": [0.1 + 0.2 + i for i in range(n)],
        "volume": np.arange(n, dtype="int64") + 10, "trades": np.arange(n, dtype="int64") + 1,
        "symbol": symbol, "source": source,
    })[list(SCHEMAS["equity"])].astype(SCHEMAS["equity"])


@pytest.fixture
def store(tmp_path):
    return c.CacheStore(tmp_path)


# ---- intervals --------------------------------------------------------
def test_merge_joins_overlap_and_adjacent():
    got = c.merge_intervals([(D("2025-01-05"), D("2025-01-09")), (D("2025-01-01"), D("2025-01-04")),
                             (D("2025-03-01"), D("2025-03-02"))])
    assert got == [(D("2025-01-01"), D("2025-01-09")), (D("2025-03-01"), D("2025-03-02"))]


@pytest.mark.parametrize("req,cov,want", [
    (("2025-01-01", "2025-01-31"), [], [("2025-01-01", "2025-01-31")]),
    (("2025-01-01", "2025-01-31"), [("2025-01-01", "2025-01-31")], []),
    (("2025-01-05", "2025-01-10"), [("2025-01-01", "2025-01-31")], []),
    (("2025-01-01", "2025-01-31"), [("2025-01-10", "2025-01-20")],
     [("2025-01-01", "2025-01-09"), ("2025-01-21", "2025-01-31")]),
    (("2025-01-01", "2025-01-31"), [("2024-12-01", "2025-01-15")], [("2025-01-16", "2025-01-31")]),
    (("2025-01-01", "2025-01-31"), [("2025-01-20", "2025-02-28")], [("2025-01-01", "2025-01-19")]),
    (("2025-01-01", "2025-01-10"), [("2025-01-01", "2025-01-05"), ("2025-01-06", "2025-01-10")], []),
    (("2025-02-01", "2025-01-01"), [], []),
])
def test_subtract(req, cov, want):
    got = c.subtract_intervals((D(req[0]), D(req[1])), [(D(a), D(b)) for a, b in cov])
    assert got == [(D(a), D(b)) for a, b in want]


# ---- keys / ttl / settle ---------------------------------------------
def test_key_order_and_case_independent():
    assert c.make_key({"a": "nse", "b": " eq "}) == c.make_key({"b": "EQ", "a": "NSE"})
    assert c.make_key({"s": "A"}) != c.make_key({"s": "B"})
    assert c.make_key({"x": None}) != c.make_key({"x": "NONE"})


def test_settle_and_ttl():
    now = pd.Timestamp("2025-06-10 12:00", tz="Asia/Kolkata").timestamp()
    assert c.settled_limit(now) == D("2025-06-09")
    kw = dict(now=now, ttl_settled=30.0, ttl_recent=1.0, recent_days=7)
    assert c.ttl_for(D("2025-06-05"), **kw) == 1.0
    assert c.ttl_for(D("2025-05-01"), **kw) == 30.0


def test_ist_date_rolls_over_at_ist_midnight():
    utc_1830 = pd.Timestamp("2025-06-10 18:30", tz="UTC").timestamp()
    assert c.today_ist(utc_1830) == D("2025-06-11")


def test_env_switch_and_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(c.ENV_SWITCH, "OFF")
    assert not c.cache_enabled()
    monkeypatch.setenv(c.ENV_SWITCH, "on")
    assert c.cache_enabled()
    monkeypatch.setenv(c.ENV_DIR, str(tmp_path))
    assert c.default_cache_dir() == tmp_path


# ---- round trip --------------------------------------------------------
def test_series_roundtrip_exact(store):
    df = equity_frame(["2025-01-01", "2025-01-02", "2025-01-03"])
    assert store.record_series("equity", "k", (D("2025-01-01"), D("2025-01-03")), df, now=NOW, ttl=100)
    got = store.read_series("equity", "k", D("2025-01-01"), D("2025-01-03"))
    pd.testing.assert_frame_equal(got, df, check_exact=True)
    assert list(got.dtypes.astype(str)) == list(SCHEMAS["equity"].values())


def test_nan_and_nat_roundtrip_fno(store):
    sch = SCHEMAS["fno"]
    row = {k: (np.nan if v == "float64" else (pd.NaT if v.startswith("datetime") else
           (0 if v == "int64" else "X"))) for k, v in sch.items()}
    row["date"] = pd.Timestamp("2025-01-02")
    df = pd.DataFrame([row]).astype(sch)
    store.record_series("fno", "k", (D("2025-01-01"), D("2025-01-31")), df, now=NOW, ttl=100)
    got = store.read_series("fno", "k", D("2025-01-01"), D("2025-01-31"))
    pd.testing.assert_frame_equal(got, df, check_exact=True)


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_empty_read_has_schema(store, name):
    got = store.read_series(name, "nothing", D("2025-01-01"), D("2025-01-02"))
    pd.testing.assert_frame_equal(got, empty_frame(name))


def test_same_day_rows_keep_order(store):
    df = equity_frame(["2025-01-02", "2025-01-02"], symbol="A")
    df["close"] = [5.5, 1.5]
    store.record_series("equity", "k", (D("2025-01-01"), D("2025-01-05")), df, now=NOW, ttl=100)
    got = store.read_series("equity", "k", D("2025-01-01"), D("2025-01-05"))
    assert got["close"].tolist() == [5.5, 1.5]


# ---- coverage rules ----------------------------------------------------
def test_empty_never_recorded(store):
    assert not store.record_series("equity", "k", (D("2025-01-01"), D("2025-01-31")),
                                   empty_frame("equity"), now=NOW, ttl=100)
    assert store.covered("equity", "k", NOW) == []
    assert not store.put_snapshot("equity", "k", empty_frame("equity"), now=NOW, ttl=10)


def test_rerecord_replaces_rows_no_duplicates(store):
    iv = (D("2025-01-01"), D("2025-01-03"))
    store.record_series("equity", "k", iv, equity_frame(["2025-01-01", "2025-01-02"]), now=NOW, ttl=100)
    new = equity_frame(["2025-01-02", "2025-01-03"])
    store.record_series("equity", "k", iv, new, now=NOW, ttl=100)
    got = store.read_series("equity", "k", *iv)
    assert got["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-01-03"]
    assert store.covered("equity", "k", NOW) == [iv]


def test_coverage_expires(store):
    iv = (D("2025-01-01"), D("2025-01-03"))
    store.record_series("equity", "k", iv, equity_frame(["2025-01-02"]), now=NOW, ttl=100)
    assert store.covered("equity", "k", NOW + 99) == [iv]
    assert store.covered("equity", "k", NOW + 101) == []


def test_rows_outside_interval_are_not_stored(store):
    iv = (D("2025-01-01"), D("2025-01-02"))
    store.record_series("equity", "k", iv, equity_frame(["2025-01-02", "2025-02-01"]), now=NOW, ttl=100)
    assert store.stats()["rows"] == 1


def test_key_isolation(store):
    iv = (D("2025-01-01"), D("2025-01-03"))
    store.record_series("equity", "a", iv, equity_frame(["2025-01-02"]), now=NOW, ttl=100)
    assert store.covered("equity", "b", NOW) == []
    assert store.read_series("equity", "b", *iv).empty
    assert store.covered("index", "a", NOW) == []


def test_snapshot_ttl(store):
    df = equity_frame(["2025-01-02"])
    assert store.put_snapshot("equity", "k", df, now=NOW, ttl=50)
    pd.testing.assert_frame_equal(store.get_snapshot("equity", "k", NOW + 10), df, check_exact=True)
    assert store.get_snapshot("equity", "k", NOW + 60) is None


def test_missing_column_not_stored(store):
    bad = equity_frame(["2025-01-02"]).drop(columns=["trades"])
    assert not store.record_series("equity", "k", (D("2025-01-01"), D("2025-01-03")), bad, now=NOW, ttl=9)


def test_input_frame_not_mutated(store):
    df = equity_frame(["2025-01-02"])
    before = df.copy(deep=True)
    store.record_series("equity", "k", (D("2025-01-01"), D("2025-01-03")), df, now=NOW, ttl=9)
    pd.testing.assert_frame_equal(df, before)


def test_unknown_dataset_raises(store):
    with pytest.raises(ValueError):
        store.covered("nope", "k", NOW)
    with pytest.raises(ValueError):
        c.schema_signature("nope")


# ---- maintenance / robustness ------------------------------------------
def test_clear_and_stats(store):
    iv = (D("2025-01-01"), D("2025-01-03"))
    store.record_series("equity", "a", iv, equity_frame(["2025-01-02"]), now=NOW, ttl=100)
    store.record_series("equity", "b", iv, equity_frame(["2025-01-02"]), now=NOW, ttl=100)
    assert store.stats() == {"rows": 2, "coverage": 2, "snapshots": 0}
    store.clear("equity", "a")
    assert store.stats()["rows"] == 1
    store.clear()
    assert store.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}


def test_schema_change_clears_cache(tmp_path, monkeypatch):
    s1 = c.CacheStore(tmp_path)
    s1.record_series("equity", "k", (D("2025-01-01"), D("2025-01-03")),
                     equity_frame(["2025-01-02"]), now=NOW, ttl=100)
    monkeypatch.setitem(SCHEMAS, "nav", {**SCHEMAS["nav"], "extra": "float64"})
    s2 = c.CacheStore(tmp_path)
    assert s2.stats()["rows"] == 0


def test_corrupt_file_degrades_without_raising(tmp_path, caplog):
    (tmp_path / c.DB_NAME).write_bytes(b"this is not a sqlite database" * 50)
    with caplog.at_level(logging.WARNING):
        s = c.CacheStore(tmp_path)
    assert not s.usable
    iv = (D("2025-01-01"), D("2025-01-03"))
    assert s.covered("equity", "k", NOW) == []
    assert s.read_series("equity", "k", *iv) is None
    assert not s.record_series("equity", "k", iv, equity_frame(["2025-01-02"]), now=NOW, ttl=9)
    assert "cache" in caplog.text


def test_unwritable_directory_degrades(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    s = c.CacheStore(blocker / "sub")  # parent is a file -> cannot mkdir
    assert not s.usable
    assert s.get_snapshot("equity", "k", NOW) is None


def test_concurrent_writers(store):
    errors = []

    def work(i):
        try:
            d = f"2025-01-{i + 1:02d}"
            store.record_series("equity", "k", (D(d), D(d)), equity_frame([d]), now=NOW, ttl=100)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    assert store.stats()["rows"] == 20
    assert store.covered("equity", "k", NOW) == [(D("2025-01-01"), D("2025-01-20"))]