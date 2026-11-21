"""Offline tests for pyequity.api.fno (public F&O API and cache integration).

The connector functions are replaced by fakes (or, for the end-to-end tests,
only the NSE session is faked); nothing touches the network.
"""
import datetime as dt
import logging
import warnings

import pandas as pd
import pytest

from pyequity.api import fno as api
from pyequity.api._errors import (
    DataUnavailableError, EmptyResultWarning, IncompleteDataWarning,
    InvalidArgumentError)
from pyequity.connectors import fno as connector
from pyequity.processing.normalize import normalize
from pyequity.reliability import cache as cache_mod
from pyequity.reliability.outcome import FetchOutcome, Status
from pyequity.schema import SCHEMAS, columns_of, empty_frame

NOW0 = pd.Timestamp("2026-11-19 12:00", tz="Asia/Kolkata").timestamp()
JAN = list(pd.bdate_range("2026-01-01", "2026-01-30"))


def fno_frame(dates, *, symbol="RELIANCE", instrument="FUTSTK",
              expiry="2026-01-27", opt="XX", strike=0.0):
    """A valid standard-schema F&O frame, one row per date."""
    n = len(dates)
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)), "symbol": symbol,
        "instrument_type": instrument, "expiry_date": pd.Timestamp(expiry),
        "option_type": opt, "strike_price": float(strike),
        "open": [10.0] * n, "high": [12.0] * n, "low": [9.0] * n,
        "close": [11.0] * n, "settle_price": [11.0] * n,
        "volume": [100] * n, "value": [1000] * n, "open_interest": [50] * n,
        "change_in_oi": [5] * n, "underlying_value": [1300.0] * n,
        "source": "NSE_FO",
    }).astype(SCHEMAS["fno"])


def check_schema(df):
    assert list(df.columns) == list(SCHEMAS["fno"])
    for col, dtype in SCHEMAS["fno"].items():
        assert str(df[col].dtype) == dtype, col


class Recorder:
    """Fake fetch_fno_outcome. ``handler(call) -> FetchOutcome``."""

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def __call__(self, symbol, from_date, to_date, expiry_date=None,
                 instrument_type="FUTSTK", *, strike_price=None, option_type=None):
        call = {"symbol": symbol, "start": from_date, "end": to_date,
                "expiry": expiry_date, "instrument": instrument_type,
                "strike": strike_price, "option_type": option_type}
        self.calls.append(call)
        return self.handler(call)


def serving(universe):
    """Handler that answers from a frame like NSE: OK with rows, else EMPTY."""
    def handler(call):
        d = universe["date"]
        rows = universe[(d >= pd.Timestamp(call["start"])) & (d <= pd.Timestamp(call["end"]))]
        if rows.empty:
            return FetchOutcome(Status.EMPTY, empty_frame("fno"), "no rows")
        return FetchOutcome(Status.OK, rows.reset_index(drop=True))
    return handler


def failing(reason="request failed: HTTP 500"):
    return lambda call: FetchOutcome(Status.FAILED, None, reason)


def empty_outcome(call):
    return FetchOutcome(Status.EMPTY, empty_frame("fno"), "NSE returned no F&O rows")


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Every test gets its own cache folder and a fixed clock (Nov 2026)."""
    monkeypatch.delenv(cache_mod.ENV_SWITCH, raising=False)
    cache = cache_mod.DiskCache(tmp_path / "cache", clock=lambda: NOW0)
    monkeypatch.setattr(api, "get_default_cache", lambda: cache)
    return cache


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("unexpected connector call")
    monkeypatch.setattr(connector, "fetch_fno_outcome", boom)
    monkeypatch.setattr(connector, "list_expiries_outcome", boom)


def install(monkeypatch, handler):
    rec = Recorder(handler)
    monkeypatch.setattr(connector, "fetch_fno_outcome", rec)
    return rec


# ------------------------------- validation ---------------------------------

OPT = {"instrument": "OPTSTK", "strike": 1250, "option_type": "CE"}


@pytest.mark.parametrize("symbol", ["", "   ", None, 5, ["RELIANCE"]])
def test_invalid_symbol_raises(monkeypatch, symbol):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with pytest.raises(InvalidArgumentError, match="symbol"):
        api.get_history(symbol, "2026-01-01", "2026-01-30")
    assert rec.calls == []


@pytest.mark.parametrize("instrument", ["XYZ", "", None, 3, "FUT"])
def test_invalid_instrument_raises(monkeypatch, instrument):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with pytest.raises(InvalidArgumentError, match="instrument"):
        api.get_history("RELIANCE", "2026-01-01", "2026-01-30", instrument=instrument)
    assert rec.calls == []


@pytest.mark.parametrize("start,end", [
    ("01/01/2026", "2026-01-30"), ("2026-01-01", "30-01-2026"),
    ("2026-13-01", "2026-01-30"), (None, "2026-01-30"), (20260101, "2026-01-30"),
])
def test_invalid_dates_raise(monkeypatch, start, end):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with pytest.raises(InvalidArgumentError):
        api.get_history("RELIANCE", start, end)
    assert rec.calls == []


def test_reversed_range_raises(monkeypatch):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with pytest.raises(InvalidArgumentError, match="after"):
        api.get_history("RELIANCE", "2026-02-01", "2026-01-01")
    assert rec.calls == []


@pytest.mark.parametrize("expiry", ["not-a-date", "2026-02-30", "", 20260127, "27/01/2026"])
def test_invalid_expiry_raises(monkeypatch, expiry):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with pytest.raises(InvalidArgumentError, match="expiry"):
        api.get_history("RELIANCE", "2026-01-01", "2026-01-30", expiry=expiry)
    assert rec.calls == []


@pytest.mark.parametrize("strike", ["abc", -5, 0, True, float("nan"), float("inf")])
def test_invalid_strike_raises(monkeypatch, strike):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with pytest.raises(InvalidArgumentError, match="strike"):
        api.get_history("RELIANCE", "2026-01-01", "2026-01-30",
                        instrument="OPTSTK", strike=strike, option_type="CE")
    assert rec.calls == []


@pytest.mark.parametrize("option_type", ["XX", "call", "", 1])
def test_invalid_option_type_raises(monkeypatch, option_type):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with pytest.raises(InvalidArgumentError, match="option_type"):
        api.get_history("RELIANCE", "2026-01-01", "2026-01-30",
                        instrument="OPTSTK", strike=1250, option_type=option_type)
    assert rec.calls == []


@pytest.mark.parametrize("kwargs", [
    {"instrument": "OPTSTK"},
    {"instrument": "OPTSTK", "strike": 1250},
    {"instrument": "OPTIDX", "option_type": "PE"},
])
def test_options_need_strike_and_option_type(monkeypatch, kwargs):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with pytest.raises(InvalidArgumentError, match="both strike and option_type"):
        api.get_history("NIFTY", "2026-01-01", "2026-01-30", **kwargs)
    assert rec.calls == []


@pytest.mark.parametrize("kwargs", [{"strike": 1250}, {"option_type": "CE"},
                                    {"instrument": "FUTIDX", "strike": 1250, "option_type": "CE"}])
def test_futures_reject_strike_and_option_type(monkeypatch, kwargs):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with pytest.raises(InvalidArgumentError, match="only apply to options"):
        api.get_history("RELIANCE", "2026-01-01", "2026-01-30", **kwargs)
    assert rec.calls == []


# ------------------------------- OK / EMPTY / FAILED ------------------------

def test_success_returns_standard_frame_and_normalized_arguments(monkeypatch):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = api.get_history(" reliance ", dt.date(2026, 1, 1), dt.datetime(2026, 1, 30))
    check_schema(df)
    assert len(df) == len(JAN) and (df["source"] == "NSE_FO").all()
    assert df["date"].is_monotonic_increasing
    assert df.attrs["source"] == "NSE_FO" and df.attrs["failed_ranges"] == []
    assert rec.calls == [{"symbol": "RELIANCE", "start": "2026-01-01", "end": "2026-01-30",
                          "expiry": None, "instrument": "FUTSTK",
                          "strike": None, "option_type": None}]


def test_option_arguments_reach_the_connector_normalized(monkeypatch):
    frame = fno_frame(JAN, instrument="OPTSTK", opt="CE", strike=1250)
    rec = install(monkeypatch, serving(frame))
    df = api.get_history("reliance", "2026-01-01", "2026-01-30", instrument="optstk",
                         expiry="27-Jan-2026", strike="1250", option_type="ce")
    assert len(df) == len(JAN)
    c = rec.calls[0]
    assert (c["instrument"], c["expiry"], c["strike"], c["option_type"]) == \
        ("OPTSTK", "2026-01-27", 1250.0, "CE")


@pytest.mark.parametrize("given", ["2026-01-27", "27-Jan-2026", "27-jan-2026",
                                   dt.date(2026, 1, 27), pd.Timestamp("2026-01-27")])
def test_expiry_forms_are_normalized(monkeypatch, given):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    api.get_history("RELIANCE", "2026-01-01", "2026-01-30", expiry=given)
    assert rec.calls[0]["expiry"] == "2026-01-27"


def test_empty_outcome_returns_empty_frame_with_warning(monkeypatch):
    install(monkeypatch, empty_outcome)
    with pytest.warns(EmptyResultWarning, match="RELIANCE"):
        df = api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert df.empty and list(df.columns) == columns_of("fno")
    assert df.attrs["source"] == ""


def test_failed_outcome_raises_data_unavailable(monkeypatch):
    install(monkeypatch, failing("request failed: HTTP 500"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no EmptyResultWarning may hide the failure
        with pytest.raises(DataUnavailableError, match="HTTP 500"):
            api.get_history("RELIANCE", "2026-01-01", "2026-01-30")


def test_failed_is_never_returned_as_an_empty_result(monkeypatch):
    install(monkeypatch, failing())
    with pytest.raises(DataUnavailableError):
        api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    install(monkeypatch, empty_outcome)
    with pytest.warns(EmptyResultWarning):
        assert api.get_history("RELIANCE", "2026-01-01", "2026-01-30").empty


# ------------------------------- cache --------------------------------------

def test_second_identical_call_is_a_cache_hit(monkeypatch, isolated_cache):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    first = api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert len(rec.calls) == 1
    assert isolated_cache.stats()["rows"] == len(JAN)
    second = api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert len(rec.calls) == 1  # served from the cache, no request
    pd.testing.assert_frame_equal(first, second)
    check_schema(second)


def test_cache_key_ignores_case_and_expiry_format(monkeypatch):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    api.get_history("reliance", "2026-01-01", "2026-01-30", expiry="27-Jan-2026")
    api.get_history("RELIANCE", "2026-01-01", "2026-01-30", expiry="2026-01-27")
    assert len(rec.calls) == 1


def test_overlapping_range_fetches_only_the_missing_part(monkeypatch):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    api.get_history("RELIANCE", "2026-01-01", "2026-01-15")
    df = api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert [(c["start"], c["end"]) for c in rec.calls] == [
        ("2026-01-01", "2026-01-15"), ("2026-01-16", "2026-01-30")]
    assert len(df) == len(JAN)
    assert df["date"].is_monotonic_increasing and not df["date"].duplicated().any()


def test_different_parameters_get_distinct_cache_entries(monkeypatch, isolated_cache):
    universe = pd.concat([
        fno_frame(JAN),
        fno_frame(JAN, symbol="TCS"),
        fno_frame(JAN, instrument="OPTSTK", opt="CE", strike=1250),
        fno_frame(JAN, instrument="OPTSTK", opt="PE", strike=1250),
        fno_frame(JAN, instrument="OPTSTK", opt="CE", strike=1300),
        fno_frame(JAN, expiry="2026-02-24"),
    ], ignore_index=True)

    def handler(call):
        u = universe[universe["symbol"] == call["symbol"]]
        u = u[u["instrument_type"] == call["instrument"]]
        if call["expiry"]:
            u = u[u["expiry_date"] == pd.Timestamp(call["expiry"])]
        if call["option_type"]:
            u = u[u["option_type"] == call["option_type"]]
        if call["strike"]:
            u = u[u["strike_price"] == call["strike"]]
        return serving(u.reset_index(drop=True))(call) if len(u) else empty_outcome(call)
    rec = install(monkeypatch, handler)

    requests = [
        dict(symbol="RELIANCE", expiry="2026-01-27"),
        dict(symbol="TCS", expiry="2026-01-27"),
        dict(symbol="RELIANCE", expiry="2026-02-24"),
        dict(symbol="RELIANCE", expiry="2026-01-27", **OPT),
        dict(symbol="RELIANCE", expiry="2026-01-27", instrument="OPTSTK", strike=1250,
             option_type="PE"),
        dict(symbol="RELIANCE", expiry="2026-01-27", instrument="OPTSTK", strike=1300,
             option_type="CE"),
    ]
    results = []
    for kw in requests:
        results.append(api.get_history(start="2026-01-01", end="2026-01-30", **kw))
    assert len(rec.calls) == len(requests)  # no request was answered from another's entry
    assert isolated_cache.stats()["coverage"] == len(requests)
    assert results[1]["symbol"].eq("TCS").all()
    assert results[2]["expiry_date"].eq(pd.Timestamp("2026-02-24")).all()
    assert results[4]["option_type"].eq("PE").all() and results[5]["strike_price"].eq(1300).all()

    for kw in requests:  # every one is now a hit
        api.get_history(start="2026-01-01", end="2026-01-30", **kw)
    assert len(rec.calls) == len(requests)


def test_unfiltered_request_is_not_served_by_a_filtered_entry(monkeypatch):
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    api.get_history("RELIANCE", "2026-01-01", "2026-01-30", expiry="2026-01-27")
    api.get_history("RELIANCE", "2026-01-01", "2026-01-30")  # no expiry filter
    assert len(rec.calls) == 2


def test_empty_results_are_not_cached(monkeypatch, isolated_cache):
    rec = install(monkeypatch, empty_outcome)
    for _ in range(2):
        with pytest.warns(EmptyResultWarning):
            api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert len(rec.calls) == 2  # asked again each time
    assert isolated_cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}


def test_failed_results_are_not_cached(monkeypatch, isolated_cache):
    rec = install(monkeypatch, failing())
    for _ in range(2):
        with pytest.raises(DataUnavailableError):
            api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert len(rec.calls) == 2
    assert isolated_cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}

    # Once NSE recovers the same request fetches and caches normally.
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    assert len(api.get_history("RELIANCE", "2026-01-01", "2026-01-30")) == len(JAN)
    assert len(rec.calls) == 1 and isolated_cache.stats()["rows"] == len(JAN)


def test_partial_failure_warns_and_keeps_only_good_data_cached(monkeypatch, isolated_cache):
    universe = fno_frame(JAN)
    good = serving(universe)
    install(monkeypatch, good)
    api.get_history("RELIANCE", "2026-01-01", "2026-01-15")

    def handler(call):  # the second part of the range fails
        return failing("request failed: HTTP 503")(call) if call["start"] == "2026-01-16" \
            else good(call)
    rec = install(monkeypatch, handler)
    with pytest.warns(IncompleteDataWarning, match="2026-01-16..2026-01-30"):
        df = api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert [c["start"] for c in rec.calls] == ["2026-01-16"]
    assert df["date"].max() <= pd.Timestamp("2026-01-15")  # not presented as complete
    assert df.attrs["failed_ranges"] == ["2026-01-16..2026-01-30"]
    # the failed part was not recorded as covered
    install(monkeypatch, good)
    assert len(api.get_history("RELIANCE", "2026-01-01", "2026-01-30")) == len(JAN)


def test_unsettled_days_are_returned_but_not_cached(monkeypatch, isolated_cache):
    today = cache_mod.today_ist(NOW0)  # 2026-11-19: the fixed clock's "today"
    days = [today - dt.timedelta(days=2), today - dt.timedelta(days=1), today]
    rec = install(monkeypatch, serving(fno_frame(days)))
    start, end = days[0].isoformat(), days[-1].isoformat()
    first = api.get_history("RELIANCE", start, end)
    assert len(first) == 3
    api.get_history("RELIANCE", start, end)
    assert len(rec.calls) == 2  # today is never treated as covered
    assert isolated_cache.stats()["rows"] == 2  # only the two settled days are stored


def test_cache_can_be_switched_off(monkeypatch, isolated_cache):
    monkeypatch.setenv(cache_mod.ENV_SWITCH, "off")
    rec = install(monkeypatch, serving(fno_frame(JAN)))
    api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert len(rec.calls) == 2
    assert isolated_cache.stats()["rows"] == 0


def test_shared_default_cache_is_used(monkeypatch, tmp_path):
    monkeypatch.undo()  # drop the isolation fixtures; use the real default cache
    monkeypatch.setenv(cache_mod.ENV_DIR, str(tmp_path / "shared"))
    monkeypatch.delenv(cache_mod.ENV_SWITCH, raising=False)
    cache_mod.reset_default_cache()
    try:
        rec = Recorder(serving(fno_frame(JAN)))
        monkeypatch.setattr(connector, "fetch_fno_outcome", rec)
        api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
        api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
        assert len(rec.calls) == 1
        assert cache_mod.get_default_cache().stats()["rows"] == len(JAN)
    finally:
        cache_mod.reset_default_cache()


def test_failure_is_logged(monkeypatch, caplog):
    install(monkeypatch, failing("request failed: HTTP 500"))
    with caplog.at_level(logging.ERROR, logger="pyequity.api.fno"):
        with pytest.raises(DataUnavailableError):
            api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert any("HTTP 500" in r.message for r in caplog.records)


# ------------------------------- list_expiries ------------------------------

def install_expiries(monkeypatch, outcome):
    calls = []
    monkeypatch.setattr(connector, "list_expiries_outcome",
                        lambda symbol: calls.append(symbol) or outcome)
    return calls


def test_list_expiries_ok(monkeypatch):
    calls = install_expiries(monkeypatch, FetchOutcome(
        Status.OK, ["2026-11-24", "2026-12-29"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert api.list_expiries(" reliance ") == ["2026-11-24", "2026-12-29"]
    assert calls == ["RELIANCE"]


def test_list_expiries_empty_warns_and_returns_empty_list(monkeypatch):
    install_expiries(monkeypatch, FetchOutcome(Status.EMPTY, [], "none listed"))
    with pytest.warns(EmptyResultWarning, match="NOTASYMBOL"):
        assert api.list_expiries("NOTASYMBOL") == []


def test_list_expiries_failed_raises_not_empty(monkeypatch):
    install_expiries(monkeypatch, FetchOutcome(Status.FAILED, None, "HTTP 500 listing expiries"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DataUnavailableError, match="HTTP 500"):
            api.list_expiries("RELIANCE")


@pytest.mark.parametrize("symbol", ["", "  ", None, 7])
def test_list_expiries_invalid_symbol_raises(monkeypatch, symbol):
    calls = install_expiries(monkeypatch, FetchOutcome(Status.OK, ["2026-11-24"]))
    with pytest.raises(InvalidArgumentError, match="symbol"):
        api.list_expiries(symbol)
    assert calls == []


def test_list_expiries_returns_a_copy_not_the_outcome_list(monkeypatch):
    data = ["2026-11-24"]
    install_expiries(monkeypatch, FetchOutcome(Status.OK, data))
    out = api.list_expiries("RELIANCE")
    out.append("junk")
    assert data == ["2026-11-24"]


# ------------------- end to end through the real connector -------------------

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def nse_date(d):
    return f"{d.day:02d}-{MONTHS[d.month - 1]}-{d.year}"


def nse_row(day, expiry=dt.date(2026, 1, 27)):
    return {
        "FH_TIMESTAMP": nse_date(day), "FH_SYMBOL": "RELIANCE", "FH_INSTRUMENT": "FUTSTK",
        "FH_EXPIRY_DT": nse_date(expiry), "FH_OPTION_TYPE": "XX", "FH_STRIKE_PRICE": 0,
        "FH_OPENING_PRICE": 10.0, "FH_TRADE_HIGH_PRICE": 12.0, "FH_TRADE_LOW_PRICE": 9.0,
        "FH_CLOSING_PRICE": 11.0, "FH_SETTLE_PRICE": 11.0, "FH_TOT_TRADED_QTY": 100,
        "FH_TOT_TRADED_VAL": 1000, "FH_OPEN_INT": 50, "FH_CHANGE_IN_OI": 5,
        "FH_UNDERLYING_VALUE": 1300.0,
    }


class Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload


class NseSession:
    def __init__(self, rows, status=200, info=None):
        self.rows, self.status, self.info, self.calls = rows, status, info, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if self.status != 200:
            return Resp({}, self.status)
        if url == connector._INFO_URL:
            return Resp(self.info)
        d1 = dt.datetime.strptime(params["fromDate"], "%d-%m-%Y")
        d2 = dt.datetime.strptime(params["toDate"], "%d-%m-%Y")
        return Resp([r for r in self.rows
                     if d1 <= dt.datetime.strptime(r["FH_TIMESTAMP"], "%d-%b-%Y") <= d2])


@pytest.fixture
def real_connector(monkeypatch):
    monkeypatch.undo()  # keep the real connector functions
    monkeypatch.setattr(connector, "_REQUEST_DELAY", 0)

    def install_session(session):
        monkeypatch.setattr(connector, "create_nse_session", lambda *a, **k: session)
        return session
    return install_session


@pytest.fixture
def real_cache(monkeypatch, tmp_path):
    cache = cache_mod.DiskCache(tmp_path / "e2e", clock=lambda: NOW0)
    monkeypatch.setattr(api, "get_default_cache", lambda: cache)
    return cache


def test_end_to_end_matches_the_connector_and_caches(real_connector, real_cache):
    session = real_connector(NseSession([nse_row(d) for d in JAN]))
    df = api.get_history("RELIANCE", "2026-01-01", "2026-01-30", expiry="27-Jan-2026")
    expected = normalize(connector.fetch_fno_data(
        "RELIANCE", "2026-01-01", "2026-01-30", "27-Jan-2026"), "fno")
    pd.testing.assert_frame_equal(df, expected)
    check_schema(df)
    sent = len(session.calls)
    api.get_history("RELIANCE", "2026-01-01", "2026-01-30", expiry="2026-01-27")
    assert len(session.calls) == sent  # cache hit: no further NSE requests


def test_end_to_end_http_error_raises_and_nothing_is_cached(real_connector, real_cache):
    real_connector(NseSession([nse_row(d) for d in JAN], status=500))
    with pytest.raises(DataUnavailableError, match="a request failed"):
        api.get_history("RELIANCE", "2026-01-01", "2026-01-30")
    assert real_cache.stats()["rows"] == 0


def test_end_to_end_no_rows_warns_and_nothing_is_cached(real_connector, real_cache):
    real_connector(NseSession([]))
    with pytest.warns(EmptyResultWarning):
        assert api.get_history("RELIANCE", "2026-01-01", "2026-01-30").empty
    assert real_cache.stats()["rows"] == 0


def test_end_to_end_list_expiries(real_connector):
    real_connector(NseSession([], info={"expiryDates": ["29-Dec-2026", "24-Nov-2026"]}))
    assert api.list_expiries("reliance") == ["2026-11-24", "2026-12-29"]
    real_connector(NseSession([], info={}))  # NSE's answer for an unknown symbol
    with pytest.warns(EmptyResultWarning):
        assert api.list_expiries("NOTASYMBOL") == []
    real_connector(NseSession([], status=500))
    with pytest.raises(DataUnavailableError):
        api.list_expiries("RELIANCE")


def test_connector_wrappers_are_unchanged_by_the_api_layer(real_connector):
    real_connector(NseSession([nse_row(d) for d in JAN]))
    df = connector.fetch_fno_data("RELIANCE", "2026-01-01", "2026-01-30")
    assert isinstance(df, pd.DataFrame) and len(df) == len(JAN)
    real_connector(NseSession([], status=500))
    assert connector.fetch_fno_data("RELIANCE", "2026-01-01", "2026-01-30").empty
    assert connector.fetch_fno_outcome("RELIANCE", "2026-01-01", "2026-01-30").failed