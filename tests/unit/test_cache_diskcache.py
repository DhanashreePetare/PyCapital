"""Offline tests for DiskCache.get_series (reliability/cache.py, stage 1).

All fetchers are fakes and the clock is injected; nothing touches the network.
Fixed "now": Tuesday 2025-06-10 12:00 IST -> settled limit 2025-06-09,
recent window starts 2025-06-03.
"""
import logging
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pyequity.reliability import cache as c
from pyequity.schema import SCHEMAS, empty_frame

NOW0 = pd.Timestamp("2025-06-10 12:00", tz="Asia/Kolkata").timestamp()


class Clock:
    def __init__(self, t=NOW0):
        self.t = t

    def __call__(self):
        return self.t


class FakeEquity:
    """Business-day equity rows; records every call; value depends on version."""

    def __init__(self, symbol="RELIANCE", source="NSE", version=1):
        self.symbol, self.source, self.version = symbol, source, version
        self.calls = []

    def __call__(self, a, b):
        self.calls.append((a, b))
        days = pd.date_range(a, b, freq="B")
        if len(days) == 0:
            return empty_frame("equity")
        n = len(days)
        df = pd.DataFrame({
            "date": days, "open": 100.0 + days.dayofyear, "high": 101.5 + days.dayofyear,
            "low": 99.25 + days.dayofyear, "close": 100.1 + days.dayofyear + self.version / 10,
            "volume": np.arange(n) + 1000, "trades": np.arange(n) + 10,
            "symbol": self.symbol, "source": self.source})
        return df.astype(SCHEMAS["equity"])


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def dc(tmp_path, clock):
    return c.DiskCache(tmp_path, clock=clock)


ID = {"source": "NSE", "symbol": "RELIANCE", "series": "EQ"}


def get(dc, fetch, a, b, ident=ID):
    return dc.get_series("equity", ident, a, b, fetch)


def dates(df):
    return df["date"].dt.strftime("%Y-%m-%d").tolist()


# ---- hits / misses ------------------------------------------------------
def test_miss_then_hit_makes_no_second_fetch(dc):
    f = FakeEquity()
    first = get(dc, f, "2025-01-01", "2025-01-31")
    second = get(dc, f, "2025-01-01", "2025-01-31")
    assert f.calls == [("2025-01-01", "2025-01-31")]
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_result_equals_uncached_fetch(dc):
    f = FakeEquity()
    got = get(dc, f, "2025-01-01", "2025-01-31")
    pd.testing.assert_frame_equal(got, FakeEquity()("2025-01-01", "2025-01-31"), check_exact=True)
    assert list(got.columns) == list(SCHEMAS["equity"])
    assert got.index.tolist() == list(range(len(got)))


def test_subset_of_cached_range_needs_no_fetch(dc):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-31")
    sub = get(dc, f, "2025-01-10", "2025-01-15")
    assert len(f.calls) == 1
    assert dates(sub)[0] == "2025-01-10" and dates(sub)[-1] == "2025-01-15"


# ---- partial ranges -----------------------------------------------------
def test_extend_right_fetches_only_new_span(dc):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-10")
    out = get(dc, f, "2025-01-01", "2025-01-20")
    assert f.calls[-1] == ("2025-01-11", "2025-01-20")
    assert dates(out) == dates(FakeEquity()("2025-01-01", "2025-01-20"))


def test_extend_left(dc):
    f = FakeEquity()
    get(dc, f, "2025-01-11", "2025-01-20")
    out = get(dc, f, "2025-01-01", "2025-01-20")
    assert f.calls[-1] == ("2025-01-01", "2025-01-10")
    assert dates(out) == dates(FakeEquity()("2025-01-01", "2025-01-20"))


def test_hole_in_the_middle(dc):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-10")
    get(dc, f, "2025-01-21", "2025-01-31")
    out = get(dc, f, "2025-01-01", "2025-01-31")
    assert f.calls[-1] == ("2025-01-11", "2025-01-20")
    assert dates(out) == dates(FakeEquity()("2025-01-01", "2025-01-31"))


def test_adjacent_ranges_merge(dc):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-10")
    get(dc, f, "2025-01-11", "2025-01-20")
    n = len(f.calls)
    get(dc, f, "2025-01-01", "2025-01-20")
    assert len(f.calls) == n


def test_up_to_three_missing_spans_fetched_separately(dc):
    f = FakeEquity()
    for d in ("2025-01-03", "2025-01-08"):  # two islands -> 3 missing spans
        get(dc, f, d, d)
    f.calls.clear()
    out = get(dc, f, "2025-01-01", "2025-01-15")
    assert f.calls == [("2025-01-01", "2025-01-02"), ("2025-01-04", "2025-01-07"),
                       ("2025-01-09", "2025-01-15")]
    assert dates(out) == dates(FakeEquity()("2025-01-01", "2025-01-15"))


def test_more_than_three_spans_use_one_hull_fetch_without_duplicates(dc):
    f = FakeEquity(version=1)
    for d in ("2025-01-03", "2025-01-08", "2025-01-13"):  # 3 islands -> 4 spans
        get(dc, f, d, d)
    f.calls.clear()
    f.version = 2
    out = get(dc, f, "2025-01-01", "2025-01-15")
    assert f.calls == [("2025-01-01", "2025-01-15")]
    assert not out["date"].duplicated().any()
    assert dates(out) == dates(FakeEquity()("2025-01-01", "2025-01-15"))
    # the refreshed hull replaced the old island rows
    assert (out["close"].round(3) == (100.1 + out["date"].dt.dayofyear + 0.2).round(3)).all()
    f.calls.clear()
    get(dc, f, "2025-01-01", "2025-01-15")
    assert f.calls == []


# ---- TTL / settle -------------------------------------------------------
def test_old_data_ttl_30_days(dc, clock):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-31")
    clock.t = NOW0 + 29 * 86400
    get(dc, f, "2025-01-01", "2025-01-31")
    assert len(f.calls) == 1
    clock.t = NOW0 + 31 * 86400
    get(dc, f, "2025-01-01", "2025-01-31")
    assert len(f.calls) == 2


def test_recent_data_ttl_one_hour(dc, clock):
    f = FakeEquity()
    get(dc, f, "2025-06-03", "2025-06-09")
    clock.t = NOW0 + 3599
    get(dc, f, "2025-06-03", "2025-06-09")
    assert len(f.calls) == 1
    clock.t = NOW0 + 3601
    get(dc, f, "2025-06-03", "2025-06-09")
    assert len(f.calls) == 2


def test_expired_refresh_replaces_rows_without_duplicates(dc, clock):
    f = FakeEquity(version=1)
    first = get(dc, f, "2025-01-01", "2025-01-31")
    f.version = 2
    clock.t = NOW0 + 31 * 86400
    second = get(dc, f, "2025-01-01", "2025-01-31")
    assert len(second) == len(first) and not second["date"].duplicated().any()
    assert (second["close"] - first["close"]).round(6).eq(0.1).all()
    assert dc._store.stats()["rows"] == len(first)


def test_interval_is_split_so_old_part_keeps_long_ttl(dc, clock):
    f = FakeEquity()
    get(dc, f, "2025-05-01", "2025-06-09")      # old part + recent part
    clock.t = NOW0 + 2 * 3600                   # recent part expired, old part not
    f.calls.clear()
    get(dc, f, "2025-05-01", "2025-06-09")
    assert f.calls == [("2025-06-03", "2025-06-09")]


def test_today_is_returned_but_never_recorded(dc):
    f = FakeEquity()
    out = get(dc, f, "2025-06-09", "2025-06-10")
    assert dates(out) == ["2025-06-09", "2025-06-10"]
    f.calls.clear()
    get(dc, f, "2025-06-09", "2025-06-10")
    assert f.calls == [("2025-06-10", "2025-06-10")]
    assert dc._store.stats()["rows"] == 1  # only 06-09 was stored


def test_settle_boundary_is_yesterday(dc):
    f = FakeEquity()
    get(dc, f, "2025-06-06", "2025-06-09")
    f.calls.clear()
    get(dc, f, "2025-06-06", "2025-06-09")
    assert f.calls == []


# ---- empty / failing fetches --------------------------------------------
def test_empty_fetch_not_recorded_and_retried(dc):
    f = FakeEquity()
    out = get(dc, f, "2025-01-04", "2025-01-05")   # weekend -> empty
    assert out.empty and list(out.columns) == list(SCHEMAS["equity"])
    get(dc, f, "2025-01-04", "2025-01-05")
    assert len(f.calls) == 2
    assert dc._store.stats()["coverage"] == 0


def test_empty_new_span_keeps_cached_rows_and_coverage(dc):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-10")
    g = lambda a, b: empty_frame("equity")
    out = get(dc, g, "2025-01-01", "2025-01-20")
    assert dates(out) == dates(FakeEquity()("2025-01-01", "2025-01-10"))
    assert dc._store.covered("equity", c.make_key(ID), NOW0) == [
        (date(2025, 1, 1), date(2025, 1, 10))]


def test_empty_hull_keeps_cached_middle_rows(dc):
    f = FakeEquity()
    for d in ("2025-01-03", "2025-01-08", "2025-01-13"):
        get(dc, f, d, d)
    out = get(dc, lambda a, b: empty_frame("equity"), "2025-01-01", "2025-01-15")
    assert dates(out) == ["2025-01-03", "2025-01-08", "2025-01-13"]


def test_fetcher_exception_propagates_and_stores_nothing(dc):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-10")

    def boom(a, b):
        raise RuntimeError("fetch failed")
    with pytest.raises(RuntimeError):
        get(dc, boom, "2025-01-01", "2025-01-20")
    assert dc._store.covered("equity", c.make_key(ID), NOW0) == [
        (date(2025, 1, 1), date(2025, 1, 10))]


# ---- isolation / integrity gates ------------------------------------------
def test_symbol_and_source_keys_are_isolated(dc):
    a, b = FakeEquity("RELIANCE", "NSE"), FakeEquity("TCS", "NSE")
    get(dc, a, "2025-01-01", "2025-01-10")
    get(dc, b, "2025-01-01", "2025-01-10", {"source": "NSE", "symbol": "TCS", "series": "EQ"})
    assert len(a.calls) == len(b.calls) == 1
    bse = FakeEquity("RELIANCE", "BSE")
    get(dc, bse, "2025-01-01", "2025-01-10", {"source": "BSE", "symbol": "RELIANCE", "series": "EQ"})
    assert len(bse.calls) == 1


def test_identity_key_is_case_insensitive(dc):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-10")
    get(dc, f, "2025-01-01", "2025-01-10", {"source": "nse", "symbol": "reliance", "series": "eq"})
    assert len(f.calls) == 1


def test_wrong_source_is_returned_but_not_stored(dc, caplog):
    bse_rows = FakeEquity("RELIANCE", "BSE")
    with caplog.at_level(logging.WARNING):
        out = get(dc, bse_rows, "2025-01-01", "2025-01-10")  # asked for NSE
    assert not out.empty and set(out["source"]) == {"BSE"}
    assert dc._store.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}
    assert "identity" in caplog.text


def test_wrong_symbol_is_returned_but_not_stored(dc):
    out = get(dc, FakeEquity("INFY", "NSE"), "2025-01-01", "2025-01-10")
    assert not out.empty
    assert dc._store.stats()["rows"] == 0


def test_mixed_symbols_in_one_frame_not_stored(dc):
    def mixed(a, b):
        df = FakeEquity()(a, b)
        df.loc[df.index[0], "symbol"] = "OTHER"
        return df
    get(dc, mixed, "2025-01-01", "2025-01-10")
    assert dc._store.stats()["rows"] == 0


def test_frame_missing_a_column_is_not_stored_and_does_not_raise(dc):
    f = FakeEquity()
    out = get(dc, lambda a, b: f(a, b).drop(columns=["trades"]), "2025-01-01", "2025-01-10")
    assert not out.empty
    assert dc._store.stats()["rows"] == 0


def test_index_dataset_with_index_name_gate(dc):
    def idx(name):
        def fn(a, b):
            days = pd.date_range(a, b, freq="B")
            if len(days) == 0:
                return empty_frame("index")
            return pd.DataFrame({"date": days, "index_name": name, "open": 1.0, "high": 2.0,
                                 "low": 0.5, "close": 1.5, "volume": 7, "source": "NSE"}
                                ).astype(SCHEMAS["index"])
        return fn
    ident = {"source": "NSE", "index_name": "NIFTY 50"}
    ok = dc.get_series("index", ident, "2025-01-01", "2025-01-10", idx("NIFTY 50"))
    assert dc._store.stats()["rows"] == len(ok) > 0
    dc2 = c.DiskCache(dc._store.path / "other", clock=Clock())
    dc2.get_series("index", ident, "2025-01-01", "2025-01-10", idx("NIFTY BANK"))
    assert dc2._store.stats()["rows"] == 0


# ---- schema / purity -------------------------------------------------------
def test_dtypes_are_schema_exact_even_if_fetcher_is_sloppy(dc):
    def sloppy(a, b):
        df = FakeEquity()(a, b)
        df["volume"] = df["volume"].astype("int32")
        df["date"] = df["date"].astype("datetime64[us]")
        return df[list(reversed(df.columns))]
    out = get(dc, sloppy, "2025-01-01", "2025-01-10")
    assert list(out.columns) == list(SCHEMAS["equity"])
    assert {k: str(v) for k, v in out.dtypes.items()} == SCHEMAS["equity"]
    again = get(dc, sloppy, "2025-01-01", "2025-01-10")
    pd.testing.assert_frame_equal(out, again, check_exact=True)


def test_fetcher_frame_is_not_mutated(dc):
    kept = {}

    def fn(a, b):
        df = FakeEquity()(a, b)
        kept["before"] = df.copy(deep=True)
        kept["obj"] = df
        return df
    get(dc, fn, "2025-01-01", "2025-01-31")
    pd.testing.assert_frame_equal(kept["obj"], kept["before"])


def test_returned_frame_is_independent_of_the_cache(dc):
    f = FakeEquity()
    first = get(dc, f, "2025-01-01", "2025-01-10")
    first["close"] = -1.0
    again = get(dc, f, "2025-01-01", "2025-01-10")
    assert (again["close"] > 0).all()


# ---- degraded modes ---------------------------------------------------------
def test_cache_off_calls_fetcher_every_time(dc, monkeypatch):
    monkeypatch.setenv(c.ENV_SWITCH, "off")
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-10")
    get(dc, f, "2025-01-01", "2025-01-10")
    assert f.calls == [("2025-01-01", "2025-01-10")] * 2
    assert dc._store.stats()["rows"] == 0


def test_unusable_store_falls_back_to_fetcher(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    dc = c.DiskCache(blocker / "sub", clock=Clock())
    f = FakeEquity()
    out = get(dc, f, "2025-01-01", "2025-01-10")
    assert dates(out) == dates(FakeEquity()("2025-01-01", "2025-01-10"))
    assert f.calls == [("2025-01-01", "2025-01-10")]


def test_read_failure_after_planning_falls_back_to_fetcher(dc, monkeypatch):
    f = FakeEquity()
    get(dc, f, "2025-01-01", "2025-01-10")
    monkeypatch.setattr(dc._store, "read_series", lambda *a, **k: None)
    f.calls.clear()
    out = get(dc, f, "2025-01-01", "2025-01-10")
    assert dates(out) == dates(FakeEquity()("2025-01-01", "2025-01-10"))
    assert f.calls == [("2025-01-01", "2025-01-10")]


def test_corrupt_db_file_still_returns_data(tmp_path):
    (tmp_path / c.DB_NAME).write_bytes(b"garbage" * 100)
    dc = c.DiskCache(tmp_path, clock=Clock())
    out = get(dc, FakeEquity(), "2025-01-01", "2025-01-10")
    assert not out.empty


# ---- argument handling ---------------------------------------------------
@pytest.mark.parametrize("start,end", [("2025-13-01", "2025-01-10"), ("2025-01-10", "2025-01-01"),
                                       ("01-01-2025", "10-01-2025"), (None, "2025-01-10")])
def test_bad_dates_give_empty_frame_and_no_fetch(dc, start, end):
    f = FakeEquity()
    out = get(dc, f, start, end)
    pd.testing.assert_frame_equal(out, empty_frame("equity"))
    assert f.calls == []


def test_bad_identity_or_fetch_give_empty_frame(dc):
    assert dc.get_series("equity", "RELIANCE", "2025-01-01", "2025-01-02", FakeEquity()).empty
    assert dc.get_series("equity", ID, "2025-01-01", "2025-01-02", None).empty


def test_unknown_dataset_raises(dc):
    with pytest.raises(ValueError):
        dc.get_series("nope", ID, "2025-01-01", "2025-01-02", FakeEquity())


def test_accepts_date_objects(dc):
    f = FakeEquity()
    out = dc.get_series("equity", ID, date(2025, 1, 1), date(2025, 1, 10), f)
    assert not out.empty and f.calls == [("2025-01-01", "2025-01-10")]


# ---- storage purge --------------------------------------------------------
def test_recording_purges_expired_coverage(tmp_path):
    s = c.CacheStore(tmp_path)
    df = FakeEquity()("2025-01-02", "2025-01-03")
    s.record_series("equity", "k", (date(2025, 1, 2), date(2025, 1, 3)), df, now=NOW0, ttl=10)
    s.record_series("equity", "k", (date(2025, 1, 6), date(2025, 1, 7)),
                    FakeEquity()("2025-01-06", "2025-01-07"), now=NOW0 + 100, ttl=10)
    assert s.stats()["coverage"] == 1