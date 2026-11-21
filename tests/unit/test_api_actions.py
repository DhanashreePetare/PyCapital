"""Offline tests for pyequity.api.actions (public corporate actions API + cache).

The connector function is replaced by a fake (or, for the end-to-end tests,
only the NSE session is faked, with the real recorded RELIANCE response);
nothing touches the network.
"""
import datetime as dt
import json
import logging
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pyequity.api import actions as api
from pyequity.api._errors import (
    DataUnavailableError, EmptyResultWarning, IncompleteDataWarning,
    InvalidArgumentError)
from pyequity.connectors import corporate as connector
from pyequity.processing.normalize import normalize
from pyequity.reliability import cache as cache_mod
from pyequity.reliability.outcome import FetchOutcome, Status
from pyequity.schema import SCHEMAS, columns_of, empty_frame

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "corporate_actions_reliance.json"
NOW0 = pd.Timestamp("2026-11-19 12:00", tz="Asia/Kolkata").timestamp()
DAY = 86400.0

# (ex-date, subject); NSE lists newest first.
RELIANCE = [("2026-06-05", "Dividend - Rs 6 Per Share"), ("2025-08-14", "Bonus 1:1"),
            ("2024-10-28", "Dividend - Rs 10 Per Share"), ("2023-08-21", "Dividend - Rs 9 Per Share"),
            ("2022-08-18", "Dividend - Rs 8 Per Share"), ("2020-05-13", "Rights 1:15 @ Premium Rs 1247")]


def actions_frame(rows, symbol="RELIANCE"):
    """A valid standard-schema frame from (ex_date, subject) pairs, in the given order."""
    n = len(rows)
    ex = pd.to_datetime([r[0] for r in rows])
    df = pd.DataFrame({
        "ex_date": ex, "symbol": symbol,
        "action_type": [("BONUS" if "Bonus" in r[1] else "RIGHTS" if "Rights" in r[1]
                         else "DIVIDEND") for r in rows],
        "action_detail": [r[1] for r in rows], "record_date": ex,
        "source": "NSE", "isin": "INE002A01018", "action_categories": "DIVIDEND",
        "bonus_new": [float("nan")] * n, "bonus_held": [float("nan")] * n,
        "split_old_fv": [float("nan")] * n, "split_new_fv": [float("nan")] * n,
        "rights_new": [float("nan")] * n, "rights_held": [float("nan")] * n,
        "rights_premium": [float("nan")] * n, "parse_status": "ok",
    })
    return df.astype(SCHEMAS[_DATASET])


_DATASET = "corporate_actions"


def check_schema(df):
    assert list(df.columns) == list(SCHEMAS[_DATASET])
    for col, dtype in SCHEMAS[_DATASET].items():
        assert str(df[col].dtype) == dtype, col


class Recorder:
    """Fake fetch_corporate_actions_outcome. ``handler(call) -> FetchOutcome``."""

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def __call__(self, symbol, from_date=None, to_date=None):
        call = {"symbol": symbol, "start": from_date, "end": to_date}
        self.calls.append(call)
        return self.handler(call)


def serving(universe):
    """Handler that answers like NSE: rows inside the ex-date window, else EMPTY."""
    def handler(call):
        d = universe["ex_date"]
        keep = pd.Series(True, index=universe.index)
        if call["start"]:
            keep &= d >= pd.Timestamp(call["start"])
        if call["end"]:
            keep &= d <= pd.Timestamp(call["end"])
        rows = universe[keep]
        if rows.empty:
            return FetchOutcome(Status.EMPTY, empty_frame(_DATASET), "no rows")
        return FetchOutcome(Status.OK, rows.reset_index(drop=True))
    return handler


def failing(reason="request failed: HTTPError: status 500"):
    return lambda call: FetchOutcome(Status.FAILED, None, reason)


def empty_outcome(call):
    return FetchOutcome(Status.EMPTY, empty_frame(_DATASET), "NSE returned no corporate actions")


class Clock:
    def __init__(self, t=NOW0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path, clock):
    """Every test gets its own cache folder and a fixed, movable clock (Nov 2026)."""
    monkeypatch.delenv(cache_mod.ENV_SWITCH, raising=False)
    cache = cache_mod.DiskCache(tmp_path / "cache", clock=clock)
    monkeypatch.setattr(api, "get_default_cache", lambda: cache)
    return cache


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("unexpected connector call")
    monkeypatch.setattr(connector, "fetch_corporate_actions_outcome", boom)


def install(monkeypatch, handler):
    rec = Recorder(handler)
    monkeypatch.setattr(connector, "fetch_corporate_actions_outcome", rec)
    return rec


def universe():
    return actions_frame(RELIANCE)


# ------------------------------- validation ---------------------------------

@pytest.mark.parametrize("symbol", ["", "   ", None, 5, ["RELIANCE"]])
def test_invalid_symbol_raises(monkeypatch, symbol):
    rec = install(monkeypatch, serving(universe()))
    with pytest.raises(InvalidArgumentError, match="symbol"):
        api.get_actions(symbol)
    assert rec.calls == []


@pytest.mark.parametrize("kwargs", [
    {"start": "01/01/2020", "end": "2026-06-30"},
    {"start": "2020-01-01", "end": "30-06-2026"},
    {"start": "2020-13-01"},
    {"end": "June 2026"},
    {"start": 20200101},
    {"start": ""},
])
def test_invalid_dates_raise_without_a_request(monkeypatch, kwargs):
    rec = install(monkeypatch, serving(universe()))
    with pytest.raises(InvalidArgumentError):
        api.get_actions("RELIANCE", **kwargs)
    assert rec.calls == []


def test_reversed_range_raises(monkeypatch):
    rec = install(monkeypatch, serving(universe()))
    with pytest.raises(InvalidArgumentError, match="after"):
        api.get_actions("RELIANCE", "2026-06-30", "2020-01-01")
    assert rec.calls == []


def test_one_sided_dates_are_allowed(monkeypatch):
    rec = install(monkeypatch, serving(universe()))
    only_start = api.get_actions("RELIANCE", start="2024-01-01")
    only_end = api.get_actions("RELIANCE", end="2022-12-31")
    assert only_start["ex_date"].min() >= pd.Timestamp("2024-01-01") and len(only_start) == 3
    assert only_end["ex_date"].max() <= pd.Timestamp("2022-12-31") and len(only_end) == 2
    assert [(c["start"], c["end"]) for c in rec.calls] == [("2024-01-01", None),
                                                           (None, "2022-12-31")]


def test_one_sided_date_that_contradicts_the_default_bound_raises(monkeypatch):
    rec = install(monkeypatch, serving(universe()))
    far = (dt.date.today() + dt.timedelta(days=800)).isoformat()
    with pytest.raises(InvalidArgumentError, match="start"):
        api.get_actions("RELIANCE", start=far)
    with pytest.raises(InvalidArgumentError, match="end"):
        api.get_actions("RELIANCE", end="1985-01-01")
    assert rec.calls == []


# ------------------------------- OK / EMPTY / FAILED ------------------------

def test_success_schema_order_metadata_and_arguments(monkeypatch):
    rec = install(monkeypatch, serving(universe()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = api.get_actions(" reliance ", dt.date(2020, 1, 1), dt.datetime(2026, 6, 30))
    check_schema(df)
    assert len(df) == len(RELIANCE)
    assert df["ex_date"].is_monotonic_increasing  # oldest to newest
    assert (df["symbol"] == "RELIANCE").all() and (df["source"] == "NSE").all()
    assert df.attrs["source"] == "NSE" and df.attrs["failed_ranges"] == []
    assert df["action_detail"].iloc[0] == "Rights 1:15 @ Premium Rs 1247"
    assert rec.calls == [{"symbol": "RELIANCE", "start": "2020-01-01", "end": "2026-06-30"}]


def test_no_dates_are_passed_to_the_connector_unchanged(monkeypatch):
    rec = install(monkeypatch, serving(universe()))
    df = api.get_actions("RELIANCE")
    assert len(df) == len(RELIANCE) and df["ex_date"].is_monotonic_increasing
    # the connector's own defaults apply: nothing is invented here
    assert rec.calls == [{"symbol": "RELIANCE", "start": None, "end": None}]


def test_same_day_rows_keep_their_order(monkeypatch):
    rows = [("2025-06-16", "Bonus 4:1"), ("2025-06-16", "Face Value Split"),
            ("2024-01-10", "Dividend - Rs 1")]
    install(monkeypatch, serving(actions_frame(rows)))
    df = api.get_actions("BAJFINANCE", "2024-01-01", "2025-12-31")
    assert df["action_detail"].tolist() == ["Dividend - Rs 1", "Bonus 4:1", "Face Value Split"]


def test_empty_outcome_returns_empty_frame_with_warning(monkeypatch):
    install(monkeypatch, empty_outcome)
    with pytest.warns(EmptyResultWarning, match="RELIANCE"):
        df = api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
    assert df.empty and list(df.columns) == columns_of(_DATASET)
    assert df.attrs["source"] == ""


def test_filters_matching_nothing_is_an_empty_result_not_an_error(monkeypatch):
    install(monkeypatch, serving(universe()))
    with pytest.warns(EmptyResultWarning):
        df = api.get_actions("RELIANCE", "2012-01-01", "2012-12-31")
    assert df.empty and list(df.columns) == columns_of(_DATASET)


def test_failed_outcome_raises_data_unavailable(monkeypatch):
    install(monkeypatch, failing("request failed: HTTPError: status 500"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no EmptyResultWarning may stand in for it
        with pytest.raises(DataUnavailableError, match="status 500"):
            api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
        with pytest.raises(DataUnavailableError, match="status 500"):
            api.get_actions("RELIANCE")


def test_failure_is_logged(monkeypatch, caplog):
    install(monkeypatch, failing("request failed: boom"))
    with caplog.at_level(logging.ERROR, logger="pyequity.api.actions"):
        with pytest.raises(DataUnavailableError):
            api.get_actions("RELIANCE")
    assert any("boom" in r.message for r in caplog.records)


# ------------------------------- cache: closed ranges -----------------------

def test_second_identical_call_is_a_cache_hit(monkeypatch, isolated_cache):
    rec = install(monkeypatch, serving(universe()))
    first = api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
    assert len(rec.calls) == 1 and isolated_cache.stats()["rows"] == len(RELIANCE)
    second = api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
    assert len(rec.calls) == 1
    pd.testing.assert_frame_equal(first, second)
    check_schema(second)


def test_sub_range_is_served_from_the_cache(monkeypatch):
    rec = install(monkeypatch, serving(universe()))
    api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
    part = api.get_actions("RELIANCE", "2023-01-01", "2025-12-31")
    assert len(rec.calls) == 1
    assert part["ex_date"].tolist() == [pd.Timestamp("2023-08-21"), pd.Timestamp("2024-10-28"),
                                        pd.Timestamp("2025-08-14")]


def test_overlapping_range_fetches_only_the_missing_part(monkeypatch):
    rec = install(monkeypatch, serving(universe()))
    api.get_actions("RELIANCE", "2020-01-01", "2023-12-31")
    df = api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
    assert [(c["start"], c["end"]) for c in rec.calls] == [
        ("2020-01-01", "2023-12-31"), ("2024-01-01", "2026-06-30")]
    assert len(df) == len(RELIANCE)
    assert df["ex_date"].is_monotonic_increasing and not df.duplicated().any()


def test_different_symbols_have_distinct_cache_entries(monkeypatch, isolated_cache):
    both = pd.concat([actions_frame(RELIANCE), actions_frame(RELIANCE[:2], symbol="TCS")],
                     ignore_index=True)

    def handler(call):
        return serving(both[both["symbol"] == call["symbol"]].reset_index(drop=True))(call)
    rec = install(monkeypatch, handler)
    a = api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
    b = api.get_actions("TCS", "2020-01-01", "2026-06-30")
    assert [c["symbol"] for c in rec.calls] == ["RELIANCE", "TCS"]
    assert len(a) == len(RELIANCE) and len(b) == 2 and (b["symbol"] == "TCS").all()
    assert isolated_cache.stats()["coverage"] == 2
    api.get_actions("reliance", "2020-01-01", "2026-06-30")  # case does not matter
    api.get_actions("TCS", "2020-01-01", "2026-06-30")
    assert len(rec.calls) == 2


def test_empty_results_are_not_cached(monkeypatch, isolated_cache):
    rec = install(monkeypatch, empty_outcome)
    for _ in range(2):
        with pytest.warns(EmptyResultWarning):
            api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
        with pytest.warns(EmptyResultWarning):
            api.get_actions("RELIANCE")
    assert len(rec.calls) == 4  # asked again every time
    assert isolated_cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}


def test_failed_results_are_not_cached_and_recovery_works(monkeypatch, isolated_cache):
    rec = install(monkeypatch, failing())
    for _ in range(2):
        with pytest.raises(DataUnavailableError):
            api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
        with pytest.raises(DataUnavailableError):
            api.get_actions("RELIANCE")
    assert len(rec.calls) == 4
    assert isolated_cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}

    rec = install(monkeypatch, serving(universe()))
    assert len(api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")) == len(RELIANCE)
    assert len(api.get_actions("RELIANCE")) == len(RELIANCE)
    assert len(rec.calls) == 2
    stats = isolated_cache.stats()
    assert stats["rows"] == len(RELIANCE) and stats["snapshots"] == 1


def test_partial_failure_warns_and_failed_part_is_not_covered(monkeypatch):
    good = serving(universe())
    install(monkeypatch, good)
    api.get_actions("RELIANCE", "2020-01-01", "2023-12-31")

    def handler(call):
        return failing("request failed: HTTPError: status 503")(call) \
            if call["start"] == "2024-01-01" else good(call)
    rec = install(monkeypatch, handler)
    with pytest.warns(IncompleteDataWarning, match="2024-01-01..2026-06-30"):
        df = api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
    assert [c["start"] for c in rec.calls] == ["2024-01-01"]
    assert df["ex_date"].max() <= pd.Timestamp("2023-12-31")
    assert df.attrs["failed_ranges"] == ["2024-01-01..2026-06-30"]
    install(monkeypatch, good)  # recovers: the failed part was never marked covered
    assert len(api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")) == len(RELIANCE)


def test_unsettled_and_future_ex_dates_are_returned_but_not_stored(monkeypatch, isolated_cache):
    rows = [("2026-12-15", "Dividend - Rs 5"), ("2026-11-10", "Dividend - Rs 4")]
    rec = install(monkeypatch, serving(actions_frame(rows)))
    first = api.get_actions("RELIANCE", "2026-11-01", "2026-12-31")
    assert len(first) == 2 and isolated_cache.stats()["rows"] == 1  # only 2026-11-10
    second = api.get_actions("RELIANCE", "2026-11-01", "2026-12-31")
    assert len(second) == 2
    # The cache records a piece as covered only when it holds rows: 2026-11-01..11-11
    # is stored, the empty recent piece (11-12..11-18) and the future part are asked again.
    assert [(c["start"], c["end"]) for c in rec.calls] == [
        ("2026-11-01", "2026-12-31"), ("2026-11-12", "2026-12-31")]


def test_rows_without_an_ex_date_are_left_out_of_a_dated_range_either_way(monkeypatch):
    frame = actions_frame(RELIANCE[:2])
    frame.loc[1, "ex_date"] = pd.NaT
    rec = install(monkeypatch, lambda call: FetchOutcome(Status.OK, frame.copy()))
    miss = api.get_actions("RELIANCE", "2020-01-01", "2026-10-31")
    hit = api.get_actions("RELIANCE", "2020-01-01", "2026-10-31")
    assert len(miss) == len(hit) == 1 and len(rec.calls) == 1
    pd.testing.assert_frame_equal(miss, hit)


# ------------------------------- cache: open-ended ranges -------------------

def test_open_ended_request_is_cached_as_a_snapshot(monkeypatch, isolated_cache):
    rec = install(monkeypatch, serving(universe()))
    first = api.get_actions("RELIANCE")
    assert isolated_cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 1}
    second = api.get_actions("RELIANCE")
    assert len(rec.calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_open_ended_keys_distinguish_start_and_end(monkeypatch, isolated_cache):
    rec = install(monkeypatch, serving(universe()))
    requests = [dict(), dict(start="2024-01-01"), dict(end="2022-12-31"),
                dict(start="2024-01-01", end=None), dict(start="2023-01-01")]
    sizes = [len(api.get_actions("RELIANCE", **kw)) for kw in requests]
    assert sizes == [6, 3, 2, 3, 4]
    assert len(rec.calls) == 4  # the 4th repeats the 2nd: a hit; the others are distinct
    assert isolated_cache.stats()["snapshots"] == 4
    for kw in requests:
        api.get_actions("RELIANCE", **kw)
    assert len(rec.calls) == 4


def test_open_ended_symbols_are_distinct(monkeypatch):
    both = pd.concat([actions_frame(RELIANCE), actions_frame(RELIANCE[:2], symbol="TCS")],
                     ignore_index=True)
    rec = install(monkeypatch, lambda call: serving(
        both[both["symbol"] == call["symbol"]].reset_index(drop=True))(call))
    assert len(api.get_actions("RELIANCE")) == 6
    assert len(api.get_actions("TCS")) == 2
    assert len(rec.calls) == 2


def test_open_ended_snapshot_expires_so_new_announcements_appear(monkeypatch, clock):
    rec = install(monkeypatch, serving(universe()))
    api.get_actions("RELIANCE")
    clock.t += DAY - 60
    api.get_actions("RELIANCE")
    assert len(rec.calls) == 1  # still within the 24 h snapshot TTL
    clock.t += 120
    new = pd.concat([actions_frame([("2026-12-20", "Dividend - Rs 7")]), universe()],
                    ignore_index=True)
    install(monkeypatch, serving(new))
    df = api.get_actions("RELIANCE")
    assert len(df) == 7 and df["ex_date"].max() == pd.Timestamp("2026-12-20")


def test_open_and_closed_requests_do_not_share_entries(monkeypatch):
    rec = install(monkeypatch, serving(universe()))
    api.get_actions("RELIANCE")
    api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
    assert len(rec.calls) == 2


def test_cache_can_be_switched_off(monkeypatch, isolated_cache):
    monkeypatch.setenv(cache_mod.ENV_SWITCH, "off")
    rec = install(monkeypatch, serving(universe()))
    for kw in (dict(), dict(start="2020-01-01", end="2026-06-30")):
        a = api.get_actions("RELIANCE", **kw)
        b = api.get_actions("RELIANCE", **kw)
        pd.testing.assert_frame_equal(a, b)
    assert len(rec.calls) == 4
    assert isolated_cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}


def test_shared_default_cache_is_used(monkeypatch, tmp_path):
    monkeypatch.undo()  # drop the isolation fixtures and use the real default cache
    monkeypatch.setenv(cache_mod.ENV_DIR, str(tmp_path / "shared"))
    monkeypatch.delenv(cache_mod.ENV_SWITCH, raising=False)
    cache_mod.reset_default_cache()
    try:
        rec = Recorder(serving(universe()))
        monkeypatch.setattr(connector, "fetch_corporate_actions_outcome", rec)
        api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
        api.get_actions("RELIANCE", "2020-01-01", "2026-06-30")
        assert len(rec.calls) == 1
        assert cache_mod.get_default_cache().stats()["rows"] > 0
    finally:
        cache_mod.reset_default_cache()


# ------------------- end to end through the real connector -------------------

class Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class NseSession:
    def __init__(self, payload, status=200):
        self.payload, self.status, self.calls = payload, status, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return Resp(self.payload, self.status)


@pytest.fixture
def real_connector(monkeypatch):
    monkeypatch.undo()  # keep the real connector function

    def install_session(session):
        monkeypatch.setattr(connector, "create_nse_session", lambda *a, **k: session)
        return session
    return install_session


@pytest.fixture
def real_cache(monkeypatch, tmp_path, clock):
    cache = cache_mod.DiskCache(tmp_path / "e2e", clock=clock)
    monkeypatch.setattr(api, "get_default_cache", lambda: cache)
    return cache


def fixture_rows():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_end_to_end_closed_range_matches_connector_and_cache_hit_is_identical(
        real_connector, real_cache):
    session = real_connector(NseSession(fixture_rows()))
    miss = api.get_actions("RELIANCE", "2010-01-01", "2026-06-30")
    expected = normalize(connector.fetch_corporate_actions(
        "RELIANCE", "2010-01-01", "2026-06-30"), _DATASET)
    check_schema(miss)
    pd.testing.assert_frame_equal(miss, expected)
    assert len(miss) == 20 and miss["ex_date"].is_monotonic_increasing
    sent = len(session.calls)
    hit = api.get_actions("RELIANCE", "2010-01-01", "2026-06-30")
    assert len(session.calls) == sent  # served from the cache
    pd.testing.assert_frame_equal(hit, miss)  # same values, dtypes and order


def test_end_to_end_open_ended_snapshot_round_trips(real_connector, real_cache):
    session = real_connector(NseSession(fixture_rows()))
    miss = api.get_actions("RELIANCE")
    assert session.calls[0][1]["from_date"] == "01-01-1990"  # connector default, not ours
    hit = api.get_actions("RELIANCE")
    assert len(session.calls) == 1
    check_schema(hit)
    pd.testing.assert_frame_equal(hit, miss)
    assert (hit["parse_status"] == "needs_review").any()  # e.g. the demerger row


def test_end_to_end_http_error_raises_and_nothing_is_cached(real_connector, real_cache):
    real_connector(NseSession([], status=500))
    with pytest.raises(DataUnavailableError, match="request failed"):
        api.get_actions("RELIANCE", "2010-01-01", "2026-06-30")
    with pytest.raises(DataUnavailableError, match="request failed"):
        api.get_actions("RELIANCE")
    assert real_cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}


@pytest.mark.parametrize("payload", [{"message": "blocked"}, [{"symbol": "RELIANCE"}]])
def test_end_to_end_bad_structure_raises(real_connector, real_cache, payload):
    real_connector(NseSession(payload))
    with pytest.raises(DataUnavailableError):
        api.get_actions("RELIANCE", "2010-01-01", "2026-06-30")


def test_end_to_end_no_actions_warns_and_nothing_is_cached(real_connector, real_cache):
    real_connector(NseSession([]))
    with pytest.warns(EmptyResultWarning, match="unknown or renamed"):
        assert api.get_actions("NOTASYMBOL123").empty
    assert real_cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}


def test_connector_wrapper_is_unchanged_by_the_api_layer(real_connector):
    real_connector(NseSession(fixture_rows()))
    df = connector.fetch_corporate_actions("RELIANCE")
    assert isinstance(df, pd.DataFrame) and len(df) == 20
    real_connector(NseSession([], status=500))
    assert connector.fetch_corporate_actions("RELIANCE").empty
    assert connector.fetch_corporate_actions_outcome("RELIANCE").failed