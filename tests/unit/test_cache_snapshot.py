"""Offline tests for DiskCache snapshots, maintenance and the shared instance
(reliability/cache.py, stage 2). Real fixtures go through the real parsers;
only the network session is faked.
"""
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyequity.connectors import corporate, fno
from pyequity.connectors.indices import _fetch_indices_chunk
from pyequity.reliability import cache as c
from pyequity.schema import SCHEMAS, empty_frame

FIX = Path(__file__).resolve().parents[1] / "fixtures"
NOW0 = pd.Timestamp("2026-11-19 12:00", tz="Asia/Kolkata").timestamp()
CA_ID = {"source": "NSE", "symbol": "RELIANCE"}


class Clock:
    def __init__(self, t=NOW0):
        self.t = t

    def __call__(self):
        return self.t


class Resp:
    def __init__(self, payload):
        self._p = payload
        self.content = json.dumps(payload).encode()
        self.text = self.content.decode()
        self.status_code = 200

    def json(self):
        return self._p

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, payload):
        self.payload = payload

    def get(self, *a, **k):
        return Resp(self.payload)


def ca_frame(monkeypatch):
    payload = json.loads((FIX / "corporate_actions_reliance.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(corporate, "create_nse_session", lambda *a, **k: Session(payload))
    return corporate.fetch_corporate_actions("RELIANCE")


class Counter:
    """Wraps a frame in a zero-argument fetcher that counts calls."""

    def __init__(self, df):
        self.df, self.calls = df, 0

    def __call__(self):
        self.calls += 1
        return self.df.copy(deep=True)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def dc(tmp_path, clock):
    return c.DiskCache(tmp_path, clock=clock)


@pytest.fixture(autouse=True)
def _fresh_default():
    c.reset_default_cache()
    yield
    c.reset_default_cache()


@pytest.fixture
def ca(monkeypatch):
    df = ca_frame(monkeypatch)
    assert len(df) == 20
    return df


# ---- snapshot hits / misses (real corporate-actions fixture) ---------------
def test_snapshot_miss_then_hit_is_identical_to_a_fresh_fetch(dc, ca):
    f = Counter(ca)
    first = dc.get_snapshot("corporate_actions", CA_ID, f)
    second = dc.get_snapshot("corporate_actions", CA_ID, f)
    assert f.calls == 1
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    pd.testing.assert_frame_equal(second, ca.reset_index(drop=True), check_exact=True)
    assert {k: str(v) for k, v in second.dtypes.items()} == SCHEMAS["corporate_actions"]


def test_snapshot_keeps_fetcher_row_order(dc, ca):
    shuffled = ca.sample(frac=1, random_state=3).reset_index(drop=True)
    dc.get_snapshot("corporate_actions", CA_ID, lambda: shuffled)
    hit = dc.get_snapshot("corporate_actions", CA_ID, lambda: pytest.fail("fetched"))
    pd.testing.assert_frame_equal(hit, shuffled, check_exact=True)


def test_snapshot_keeps_rows_with_missing_ex_date(dc, ca):
    odd = ca.copy()
    odd.loc[odd.index[2], "ex_date"] = pd.NaT
    dc.get_snapshot("corporate_actions", CA_ID, lambda: odd)
    hit = dc.get_snapshot("corporate_actions", CA_ID, lambda: pytest.fail("fetched"))
    assert len(hit) == len(odd) and hit["ex_date"].isna().sum() == 1
    pd.testing.assert_frame_equal(hit, odd.reset_index(drop=True), check_exact=True)


# ---- snapshot TTL ------------------------------------------------------------
def test_snapshot_default_ttl_is_24_hours(dc, clock, ca):
    f = Counter(ca)
    dc.get_snapshot("corporate_actions", CA_ID, f)
    clock.t = NOW0 + 86399
    dc.get_snapshot("corporate_actions", CA_ID, f)
    assert f.calls == 1
    clock.t = NOW0 + 86401
    dc.get_snapshot("corporate_actions", CA_ID, f)
    assert f.calls == 2


def test_snapshot_ttl_override(dc, clock, ca):
    f = Counter(ca)
    dc.get_snapshot("corporate_actions", CA_ID, f, ttl=60)
    clock.t = NOW0 + 59
    dc.get_snapshot("corporate_actions", CA_ID, f, ttl=60)
    assert f.calls == 1
    clock.t = NOW0 + 61
    dc.get_snapshot("corporate_actions", CA_ID, f, ttl=60)
    assert f.calls == 2


def test_constructor_snapshot_ttl_is_used(tmp_path, clock, ca):
    d = c.DiskCache(tmp_path, clock=clock, snapshot_ttl=10)
    f = Counter(ca)
    d.get_snapshot("corporate_actions", CA_ID, f)
    clock.t = NOW0 + 11
    d.get_snapshot("corporate_actions", CA_ID, f)
    assert f.calls == 2


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf"), True, "60"])
def test_unusable_ttl_means_no_caching_for_that_call(dc, ca, bad):
    f = Counter(ca)
    a = dc.get_snapshot("corporate_actions", CA_ID, f, ttl=bad)
    dc.get_snapshot("corporate_actions", CA_ID, f, ttl=bad)
    assert f.calls == 2
    pd.testing.assert_frame_equal(a, ca.reset_index(drop=True), check_exact=True)
    assert dc.stats()["snapshots"] == 0


def test_refresh_after_expiry_replaces_the_entry(dc, clock, ca):
    dc.get_snapshot("corporate_actions", CA_ID, Counter(ca))
    clock.t = NOW0 + 90000
    fewer = ca.head(5)
    dc.get_snapshot("corporate_actions", CA_ID, Counter(fewer))
    hit = dc.get_snapshot("corporate_actions", CA_ID, lambda: pytest.fail("fetched"))
    assert len(hit) == 5 and dc.stats()["snapshots"] == 1


# ---- snapshot integrity ------------------------------------------------------
def test_empty_snapshot_is_never_stored(dc):
    f = Counter(empty_frame("corporate_actions"))
    out = dc.get_snapshot("corporate_actions", CA_ID, f)
    dc.get_snapshot("corporate_actions", CA_ID, f)
    assert out.empty and list(out.columns) == list(SCHEMAS["corporate_actions"])
    assert f.calls == 2 and dc.stats()["snapshots"] == 0


def test_identity_isolation(dc, ca):
    f = Counter(ca)
    dc.get_snapshot("corporate_actions", CA_ID, f)
    other = ca.assign(symbol="TCS")
    g = Counter(other)
    dc.get_snapshot("corporate_actions", {"source": "NSE", "symbol": "TCS"}, g)
    assert f.calls == g.calls == 1 and dc.stats()["snapshots"] == 2


def test_wrong_source_or_symbol_is_returned_but_not_stored(dc, ca, caplog):
    with caplog.at_level(logging.WARNING):
        out = dc.get_snapshot("corporate_actions", CA_ID, lambda: ca.assign(source="BSE"))
    assert len(out) == 20 and "identity" in caplog.text
    out2 = dc.get_snapshot("corporate_actions", CA_ID, lambda: ca.assign(symbol="INFY"))
    assert len(out2) == 20
    assert dc.stats()["snapshots"] == 0


def test_non_conforming_frame_is_returned_not_stored(dc, ca):
    out = dc.get_snapshot("corporate_actions", CA_ID, lambda: ca.drop(columns=["isin"]))
    assert len(out) == 20 and "isin" not in out.columns
    assert dc.stats()["snapshots"] == 0
    assert dc.get_snapshot("corporate_actions", CA_ID, lambda: None).empty


def test_fetcher_exception_propagates_and_stores_nothing(dc):
    def boom():
        raise RuntimeError("down")
    with pytest.raises(RuntimeError):
        dc.get_snapshot("corporate_actions", CA_ID, boom)
    assert dc.stats()["snapshots"] == 0


def test_fetcher_frame_not_mutated_and_hit_is_independent(dc, ca):
    original, before = ca.copy(deep=True), ca.copy(deep=True)
    dc.get_snapshot("corporate_actions", CA_ID, lambda: original)
    pd.testing.assert_frame_equal(original, before)
    first = dc.get_snapshot("corporate_actions", CA_ID, lambda: pytest.fail("fetched"))
    first["symbol"] = "CHANGED"
    again = dc.get_snapshot("corporate_actions", CA_ID, lambda: pytest.fail("fetched"))
    assert (again["symbol"] == "RELIANCE").all()


# ---- degraded modes ------------------------------------------------------------
def test_cache_off_snapshot_calls_fetcher_and_never_touches_disk(tmp_path, ca, monkeypatch):
    monkeypatch.setenv(c.ENV_SWITCH, "off")
    target = tmp_path / "never_created"
    d = c.DiskCache(target, clock=Clock())
    f = Counter(ca)
    d.get_snapshot("corporate_actions", CA_ID, f)
    d.get_snapshot("corporate_actions", CA_ID, f)
    assert f.calls == 2
    assert d.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}
    assert d.invalidate("corporate_actions") is False and d.clear() is False
    assert not target.exists()


def test_unusable_store_snapshot_falls_back(tmp_path, ca):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    d = c.DiskCache(blocker / "sub", clock=Clock())
    f = Counter(ca)
    out = d.get_snapshot("corporate_actions", CA_ID, f)
    assert len(out) == 20 and f.calls == 1
    assert d.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}


def test_corrupt_db_snapshot_still_returns_data(tmp_path, ca):
    (tmp_path / c.DB_NAME).write_bytes(b"junk" * 200)
    d = c.DiskCache(tmp_path, clock=Clock())
    assert len(d.get_snapshot("corporate_actions", CA_ID, Counter(ca))) == 20


def test_snapshot_bad_arguments_and_unknown_dataset(dc, ca):
    assert dc.get_snapshot("corporate_actions", "RELIANCE", Counter(ca)).empty
    assert dc.get_snapshot("corporate_actions", CA_ID, None).empty
    with pytest.raises(ValueError):
        dc.get_snapshot("nope", CA_ID, Counter(ca))


# ---- real index and F&O fixtures through get_series --------------------------
def test_real_index_fixture_round_trips_through_get_series(dc):
    body = (FIX / "indices_nifty50.json").read_bytes()
    fresh = _fetch_indices_chunk(Session(json.loads(body)), "NIFTY 50", "15-09-2026", "23-09-2026")
    assert len(fresh) == 7
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        return fresh.copy(deep=True)
    ident = {"source": "NSE", "index_name": "NIFTY 50"}
    first = dc.get_series("index", ident, "2026-09-15", "2026-09-23", fetch)
    second = dc.get_series("index", ident, "2026-09-15", "2026-09-23", fetch)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    pd.testing.assert_frame_equal(second, fresh.sort_values("date", kind="mergesort")
                                  .reset_index(drop=True), check_exact=True)


def test_real_fno_fixture_round_trips_through_get_series(dc):
    rows = json.loads((FIX / "fno_reliance_futstk.json").read_text(encoding="utf-8"))
    fresh, status = fno._fetch_fno_chunk(Session(rows), "RELIANCE", "15-09-2026",
                                         "23-09-2026", "", "FUTSTK", "", "")
    assert len(fresh) == 7, status
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        return fresh.copy(deep=True)
    ident = {"source": "NSE_FO", "symbol": "RELIANCE", "instrument_type": "FUTSTK"}  # connector labels F&O "NSE_FO"
    first = dc.get_series("fno", ident, "2026-09-15", "2026-09-23", fetch)
    second = dc.get_series("fno", ident, "2026-09-15", "2026-09-23", fetch)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert {k: str(v) for k, v in second.dtypes.items()} == SCHEMAS["fno"]
    assert len(second) == 7


def test_fno_wrong_instrument_type_is_not_stored(dc):
    rows = json.loads((FIX / "fno_reliance_futstk.json").read_text(encoding="utf-8"))
    fresh, _ = fno._fetch_fno_chunk(Session(rows), "RELIANCE", "15-09-2026",
                                    "23-09-2026", "", "FUTSTK", "", "")
    ident = {"source": "NSE_FO", "symbol": "RELIANCE", "instrument_type": "OPTSTK"}
    out = dc.get_series("fno", ident, "2026-09-15", "2026-09-23", lambda a, b: fresh)
    assert len(out) == 7 and dc.stats()["rows"] == 0


def test_fno_source_label_must_match_the_connector_label(dc):
    rows = json.loads((FIX / "fno_reliance_futstk.json").read_text(encoding="utf-8"))
    fresh, _ = fno._fetch_fno_chunk(Session(rows), "RELIANCE", "15-09-2026",
                                    "23-09-2026", "", "FUTSTK", "", "")
    assert set(fresh["source"]) == {"NSE_FO"}
    out = dc.get_series("fno", {"source": "NSE", "symbol": "RELIANCE"}, "2026-09-15",
                        "2026-09-23", lambda a, b: fresh)
    assert len(out) == 7 and dc.stats()["rows"] == 0  # "NSE" is not "NSE_FO"


def test_get_series_warns_about_rows_without_a_date(dc, caplog):
    body = json.loads((FIX / "indices_nifty50.json").read_bytes())
    fresh = _fetch_indices_chunk(Session(body), "NIFTY 50", "15-09-2026", "23-09-2026")
    fresh.loc[fresh.index[0], "date"] = pd.NaT
    ident = {"source": "NSE", "index_name": "NIFTY 50"}
    with caplog.at_level(logging.WARNING):
        out = dc.get_series("index", ident, "2026-09-15", "2026-09-23", lambda a, b: fresh)
    assert len(out) == 6 and "no date" in caplog.text


# ---- invalidate / clear / stats -------------------------------------------------
def _series_fetch(a, b):
    days = pd.date_range(a, b, freq="B")
    if len(days) == 0:
        return empty_frame("equity")
    return pd.DataFrame({"date": days, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                         "volume": 5, "trades": 2, "symbol": "RELIANCE", "source": "NSE"}
                        ).astype(SCHEMAS["equity"])


EQ = {"source": "NSE", "symbol": "RELIANCE"}


def test_stats_counts_series_and_snapshots(dc, ca):
    assert dc.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}
    dc.get_series("equity", EQ, "2026-01-05", "2026-01-09", _series_fetch)
    dc.get_snapshot("corporate_actions", CA_ID, Counter(ca))
    assert dc.stats() == {"rows": 5, "coverage": 1, "snapshots": 1}


def test_invalidate_one_identity_only(dc):
    other = {"source": "NSE", "symbol": "TCS"}
    dc.get_series("equity", EQ, "2026-01-05", "2026-01-09", _series_fetch)
    dc.get_series("equity", other, "2026-01-05", "2026-01-09",
                  lambda a, b: _series_fetch(a, b).assign(symbol="TCS"))
    assert dc.invalidate("equity", EQ) is True
    calls = []
    dc.get_series("equity", other, "2026-01-05", "2026-01-09",
                  lambda a, b: calls.append(1) or empty_frame("equity"))
    assert calls == []                       # TCS still cached
    dc.get_series("equity", EQ, "2026-01-05", "2026-01-09",
                  lambda a, b: calls.append(1) or _series_fetch(a, b))
    assert calls == [1]                      # RELIANCE refetched


def test_invalidate_whole_dataset_leaves_other_datasets(dc, ca):
    dc.get_series("equity", EQ, "2026-01-05", "2026-01-09", _series_fetch)
    dc.get_snapshot("corporate_actions", CA_ID, Counter(ca))
    assert dc.invalidate("equity") is True
    assert dc.stats() == {"rows": 0, "coverage": 0, "snapshots": 1}


def test_invalidate_clears_snapshots_too(dc, ca):
    dc.get_snapshot("corporate_actions", CA_ID, Counter(ca))
    dc.invalidate("corporate_actions", CA_ID)
    f = Counter(ca)
    dc.get_snapshot("corporate_actions", CA_ID, f)
    assert f.calls == 1


def test_invalidate_argument_checks(dc):
    assert dc.invalidate("equity", "RELIANCE") is False
    with pytest.raises(ValueError):
        dc.invalidate("nope")


def test_clear_removes_everything(dc, ca):
    dc.get_series("equity", EQ, "2026-01-05", "2026-01-09", _series_fetch)
    dc.get_snapshot("corporate_actions", CA_ID, Counter(ca))
    assert dc.clear() is True
    assert dc.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}


def test_persistence_across_instances(tmp_path, clock, ca):
    a = c.DiskCache(tmp_path, clock=clock)
    a.get_series("equity", EQ, "2026-01-05", "2026-01-09", _series_fetch)
    a.get_snapshot("corporate_actions", CA_ID, Counter(ca))
    b = c.DiskCache(tmp_path, clock=clock)
    calls = []
    b.get_series("equity", EQ, "2026-01-05", "2026-01-09", lambda x, y: calls.append(1))
    b.get_snapshot("corporate_actions", CA_ID, lambda: calls.append(1))
    assert calls == []


# ---- shared default instance ----------------------------------------------------
def test_default_cache_is_a_singleton_until_reset():
    a = c.get_default_cache()
    assert a is c.get_default_cache() and isinstance(a, c.DiskCache)
    c.reset_default_cache()
    assert c.get_default_cache() is not a


def test_default_cache_uses_env_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(c.ENV_DIR, str(tmp_path / "pyeq"))
    d = c.get_default_cache()
    d.get_series("equity", EQ, "2026-01-05", "2026-01-09", _series_fetch)
    assert (tmp_path / "pyeq" / c.DB_NAME).exists()


def test_default_cache_creates_nothing_when_off(monkeypatch, tmp_path):
    monkeypatch.setenv(c.ENV_DIR, str(tmp_path / "pyeq"))
    monkeypatch.setenv(c.ENV_SWITCH, "off")
    d = c.get_default_cache()
    d.get_series("equity", EQ, "2026-01-05", "2026-01-09", _series_fetch)
    d.stats()
    d.clear()
    assert not (tmp_path / "pyeq").exists()


# ---- storage-layer fix: snapshot keeps undated rows ---------------------------
def test_store_snapshot_keeps_undated_rows_but_series_skips_them(tmp_path):
    s = c.CacheStore(tmp_path)
    df = pd.DataFrame({"date": [pd.Timestamp("2026-01-05"), pd.NaT], "open": 1.0, "high": 2.0,
                       "low": 0.5, "close": 1.5, "volume": 1, "trades": 1,
                       "symbol": "A", "source": "NSE"}).astype(SCHEMAS["equity"])
    s.put_snapshot("equity", "k", df, now=NOW0, ttl=100)
    assert len(s.get_snapshot("equity", "k", NOW0)) == 2
    s.record_series("equity", "k", (date(2026, 1, 1), date(2026, 1, 31)), df, now=NOW0, ttl=100)
    assert len(s.read_series("equity", "k", date(2026, 1, 1), date(2026, 1, 31))) == 1