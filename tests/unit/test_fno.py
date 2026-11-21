"""Offline tests for pyequity.connectors.fno (no internet needed)."""
import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest

from pyequity.connectors import fno
from pyequity.reliability.outcome import FetchOutcome, Status
from pyequity.schema import SCHEMAS, columns_of, empty_frame

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def nse_date(d):
    return f"{d.day:02d}-{MONTHS[d.month - 1]}-{d.year}"


def make_row(day, expiry, instrument="FUTSTK", opt="XX", strike=0, symbol="RELIANCE"):
    return {
        "FH_TIMESTAMP": nse_date(day), "FH_SYMBOL": symbol,
        "FH_INSTRUMENT": instrument, "FH_EXPIRY_DT": nse_date(expiry),
        "FH_OPTION_TYPE": opt, "FH_STRIKE_PRICE": strike,
        "FH_OPENING_PRICE": 10.0, "FH_TRADE_HIGH_PRICE": 12.0,
        "FH_TRADE_LOW_PRICE": 9.0, "FH_CLOSING_PRICE": 11.0,
        "FH_SETTLE_PRICE": 11.0, "FH_TOT_TRADED_QTY": 100,
        "FH_TOT_TRADED_VAL": 1000, "FH_OPEN_INT": 50, "FH_CHANGE_IN_OI": 5,
        "FH_UNDERLYING_VALUE": 1300.0,
    }


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class FakeSession:
    """Mimics NSE: filters server-side, newest first, hard 70-row cap."""

    def __init__(self, universe=None, fail_on_call=None, info=None):
        self.universe = universe or []
        self.calls = []
        self.fail_on_call = fail_on_call
        self.info = info

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            return FakeResponse({}, status=500)
        if url == fno._INFO_URL:
            return FakeResponse(self.info)
        d1 = datetime.strptime(params["fromDate"], "%d-%m-%Y")
        d2 = datetime.strptime(params["toDate"], "%d-%m-%Y")
        rows = []
        for r in self.universe:
            d = datetime.strptime(r["FH_TIMESTAMP"], "%d-%b-%Y")
            if not d1 <= d <= d2:
                continue
            if params["expiryDate"] and r["FH_EXPIRY_DT"] != params["expiryDate"]:
                continue
            if params["optionType"] and r["FH_OPTION_TYPE"] != params["optionType"]:
                continue
            if params["strikePrice"] and float(r["FH_STRIKE_PRICE"]) != float(params["strikePrice"]):
                continue
            if r["FH_INSTRUMENT"] != params["instrumentType"]:
                continue
            rows.append(r)
        rows.sort(key=lambda r: datetime.strptime(r["FH_TIMESTAMP"], "%d-%b-%Y"),
                  reverse=True)
        return FakeResponse(rows[:70])


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(fno, "_REQUEST_DELAY", 0)


def install(monkeypatch, session):
    monkeypatch.setattr(fno, "create_nse_session", lambda *a, **k: session)
    return session


def weekdays(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


EXPIRIES = [datetime(2026, 11, 24), datetime(2026, 12, 29), datetime(2027, 1, 25)]


def futures_universe(n_days):
    return [make_row(d, e) for d in weekdays(datetime(2026, 1, 1), n_days)
            for e in EXPIRIES]


# ---------- input validation: nothing is fetched, nothing raises ----------

@pytest.mark.parametrize("args,kwargs", [
    (("RELIANCE", "01/01/2026", "2026-02-01"), {}),
    (("RELIANCE", "2026-03-01", "2026-02-01"), {}),
    (("RELIANCE", "2026-01-01", "2026-02-01"), {"instrument_type": "XYZ"}),
    (("RELIANCE", "2026-01-01", "2026-02-01"), {"instrument_type": "OPTSTK"}),
    (("RELIANCE", "2026-01-01", "2026-02-01"),
     {"instrument_type": "OPTSTK", "strike_price": 1250}),
    (("RELIANCE", "2026-01-01", "2026-02-01"),
     {"instrument_type": "OPTSTK", "option_type": "CE"}),
    (("RELIANCE", "2026-01-01", "2026-02-01"),
     {"instrument_type": "OPTSTK", "option_type": "XX", "strike_price": 1250}),
    (("RELIANCE", "2026-01-01", "2026-02-01"),
     {"instrument_type": "OPTSTK", "option_type": "CE", "strike_price": "abc"}),
    (("RELIANCE", "2026-01-01", "2026-02-01"),
     {"instrument_type": "OPTSTK", "option_type": "CE", "strike_price": -5}),
    (("RELIANCE", "2026-01-01", "2026-02-01"), {"option_type": "CE"}),
    (("RELIANCE", "2026-01-01", "2026-02-01"), {"strike_price": 1250}),
    (("RELIANCE", "2026-01-01", "2026-02-01"), {"expiry_date": "not-a-date"}),
    (("", "2026-01-01", "2026-02-01"), {}),
])
def test_invalid_requests_return_empty_without_fetching(monkeypatch, args, kwargs):
    session = install(monkeypatch, FakeSession(futures_universe(30)))
    df = fno.fetch_fno_data(*args, **kwargs)
    assert df.empty and list(df.columns) == columns_of("fno")
    assert session.calls == []


def test_session_creation_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("dropped")
    monkeypatch.setattr(fno, "create_nse_session", boom)
    df = fno.fetch_fno_data("RELIANCE", "2026-01-01", "2026-02-01")
    assert df.empty and list(df.columns) == columns_of("fno")


# ---------- expiry and strike formatting ----------

@pytest.mark.parametrize("given", ["2026-12-29", "29-Dec-2026", "29-dec-2026"])
def test_expiry_formats_are_sent_in_nse_format(monkeypatch, given):
    session = install(monkeypatch, FakeSession(futures_universe(10)))
    fno.fetch_fno_data("RELIANCE", "2026-01-01", "2026-01-20", given)
    assert session.calls[0][1]["expiryDate"] == "29-Dec-2026"


@pytest.mark.parametrize("value,expected", [
    (1250, "1250"), (1250.0, "1250"), ("1250", "1250"), (1250.5, "1250.5"),
    (97.25, "97.25"), (0.05, "0.05"),
])
def test_strike_formatting(value, expected):
    assert fno._format_strike(value) == expected


@pytest.mark.parametrize("bad", [None, "abc", 0, -1, True, float("nan"), float("inf")])
def test_strike_formatting_rejects_bad_values(bad):
    assert fno._format_strike(bad) is None


def test_filters_reach_request_params(monkeypatch):
    rows = [make_row(d, EXPIRIES[0], "OPTSTK", "CE", 1250)
            for d in weekdays(datetime(2026, 1, 1), 5)]
    session = install(monkeypatch, FakeSession(rows))
    df = fno.fetch_fno_data("reliance", "2026-01-01", "2026-01-10",
                            instrument_type="optstk", strike_price=1250.0,
                            option_type="ce")
    p = session.calls[0][1]
    assert p["symbol"] == "RELIANCE" and p["instrumentType"] == "OPTSTK"
    assert p["strikePrice"] == "1250" and p["optionType"] == "CE"
    assert p["expiryDate"] == "" and len(df) == 5


# ---------- backward compatibility ----------

def test_old_style_call_and_output_schema(monkeypatch):
    install(monkeypatch, FakeSession(futures_universe(10)))
    df = fno.fetch_fno_data("RELIANCE", "2026-01-01", "2026-01-20",
                            "24-Nov-2026", "FUTSTK")
    assert list(df.columns) == columns_of("fno")
    for col, dtype in SCHEMAS["fno"].items():
        assert str(df[col].dtype) == dtype, col
    assert len(df) == 10 and (df["expiry_date"] == pd.Timestamp("2026-11-24")).all()
    assert (df["option_type"] == "XX").all() and (df["strike_price"] == 0).all()
    assert df["date"].is_monotonic_increasing


# ---------- row cap handling ----------

def test_large_range_is_never_truncated(monkeypatch):
    universe = futures_universe(150)  # 450 rows, far above the cap
    session = install(monkeypatch, FakeSession(universe))
    df = fno.fetch_fno_data("RELIANCE", "2026-01-01", "2026-12-31")
    assert len(df) == len(universe) == 450
    assert len(session.calls) > 1
    assert not df.duplicated(["date", "expiry_date"]).any()


def test_capped_range_is_split_not_trusted(monkeypatch):
    # 6 expiries per day: the first 20-day chunk holds 90 rows, above the cap.
    expiries = [datetime(2026, 11, 24) + timedelta(days=30 * i) for i in range(6)]
    universe = [make_row(d, e) for d in weekdays(datetime(2026, 1, 1), 40)
                for e in expiries]
    session = install(monkeypatch, FakeSession(universe))
    df = fno.fetch_fno_data("RELIANCE", "2026-01-01", "2026-01-31")
    expected = [r for r in universe
                if datetime.strptime(r["FH_TIMESTAMP"], "%d-%b-%Y") <= datetime(2026, 1, 31)]
    assert len(df) == len(expected) == 132
    assert len(session.calls) > 2


def test_single_day_at_cap_returns_empty_and_logs_error(monkeypatch, caplog):
    day = datetime(2026, 11, 13)
    universe = [make_row(day, day + timedelta(days=i)) for i in range(80)]
    install(monkeypatch, FakeSession(universe))
    with caplog.at_level(logging.ERROR, logger="pyequity.connectors.fno"):
        df = fno.fetch_fno_data("RELIANCE", "2026-11-13", "2026-11-13")
    assert df.empty
    assert any("cap" in r.message for r in caplog.records)


def test_failed_request_returns_empty_not_partial(monkeypatch):
    install(monkeypatch, FakeSession(futures_universe(150), fail_on_call=3))
    df = fno.fetch_fno_data("RELIANCE", "2026-01-01", "2026-12-31")
    assert df.empty and list(df.columns) == columns_of("fno")


def test_no_data_returns_empty(monkeypatch):
    install(monkeypatch, FakeSession([]))
    df = fno.fetch_fno_data("RELIANCE", "2026-01-01", "2026-01-31")
    assert df.empty and list(df.columns) == columns_of("fno")


def test_unexpected_response_shape_is_an_error(monkeypatch):
    class Odd(FakeSession):
        def get(self, url, params=None, timeout=None):
            return FakeResponse({"message": "blocked"})
    install(monkeypatch, Odd())
    assert fno.fetch_fno_data("RELIANCE", "2026-01-01", "2026-01-31").empty


def test_option_range_with_filters(monkeypatch):
    rows = []
    for d in weekdays(datetime(2026, 1, 1), 100):
        for e in EXPIRIES[:2]:
            rows.append(make_row(d, e, "OPTSTK", "CE", 1250))
            rows.append(make_row(d, e, "OPTSTK", "PE", 1250))
            rows.append(make_row(d, e, "OPTSTK", "CE", 1300))
    install(monkeypatch, FakeSession(rows))
    df = fno.fetch_fno_data("RELIANCE", "2026-01-01", "2026-12-31",
                            instrument_type="OPTSTK", strike_price=1250,
                            option_type="CE")
    assert len(df) == 200
    assert (df["option_type"] == "CE").all() and (df["strike_price"] == 1250).all()


def test_rows_not_matching_filters_are_dropped(monkeypatch, caplog):
    class Ignores(FakeSession):
        def get(self, url, params=None, timeout=None):
            d = datetime(2026, 1, 5)
            return FakeResponse([make_row(d, EXPIRIES[0], "FUTSTK"),
                                 make_row(d, EXPIRIES[0], "OPTSTK", "CE", 1250)])
    install(monkeypatch, Ignores())
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.fno"):
        df = fno.fetch_fno_data("RELIANCE", "2026-01-05", "2026-01-05")
    assert len(df) == 1 and df.iloc[0]["instrument_type"] == "FUTSTK"
    assert any("Dropped" in r.message for r in caplog.records)


# ---------- list_expiries ----------

def test_list_expiries_sorted_iso(monkeypatch):
    info = {"expiryDates": ["29-Dec-2026", "23-Nov-2026", "25-Jan-2027"],
            "strikePrice": ["1250"]}
    session = install(monkeypatch, FakeSession(info=info))
    assert fno.list_expiries("reliance") == ["2026-11-23", "2026-12-29", "2027-01-25"]
    assert session.calls[0][1] == {"symbol": "RELIANCE"}


@pytest.mark.parametrize("info", [{}, {"expiryDates": []}, {"expiryDates": "x"}, [], None])
def test_list_expiries_bad_payload_returns_empty_list(monkeypatch, info):
    install(monkeypatch, FakeSession(info=info))
    assert fno.list_expiries("RELIANCE") == []


def test_list_expiries_skips_unparseable_entries(monkeypatch):
    install(monkeypatch, FakeSession(info={"expiryDates": ["23-Nov-2026", "junk"]}))
    assert fno.list_expiries("RELIANCE") == ["2026-11-23"]


def test_list_expiries_session_failure_returns_empty_list(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("dropped")
    monkeypatch.setattr(fno, "create_nse_session", boom)
    assert fno.list_expiries("RELIANCE") == []


def test_list_expiries_empty_symbol():
    assert fno.list_expiries("  ") == []


# ---------- fetch_fno_outcome: OK / EMPTY / FAILED ----------

RANGE = ("RELIANCE", "2026-01-01", "2026-01-20")


def test_outcome_ok_carries_the_frame(monkeypatch):
    install(monkeypatch, FakeSession(futures_universe(10)))
    outcome = fno.fetch_fno_outcome(*RANGE, "24-Nov-2026", "FUTSTK")
    assert isinstance(outcome, FetchOutcome)
    assert outcome.status is Status.OK and outcome.ok
    assert not outcome.empty and not outcome.failed
    assert outcome.reason == "" and outcome.attempts == 1
    df = outcome.data
    assert len(df) == 10 and list(df.columns) == columns_of("fno")
    for col, dtype in SCHEMAS["fno"].items():
        assert str(df[col].dtype) == dtype, col
    assert (df["source"] == "NSE_FO").all()


def test_outcome_empty_when_nse_has_no_rows(monkeypatch):
    session = install(monkeypatch, FakeSession([]))
    outcome = fno.fetch_fno_outcome(*RANGE)
    assert outcome.status is Status.EMPTY and outcome.empty
    assert not outcome.ok and not outcome.failed
    assert outcome.reason
    assert outcome.data.empty and list(outcome.data.columns) == columns_of("fno")
    assert session.calls  # NSE really was asked


def test_outcome_empty_when_every_returned_row_is_filtered_out(monkeypatch):
    class OnlyOptions(FakeSession):
        def get(self, url, params=None, timeout=None):
            return FakeResponse([make_row(datetime(2026, 1, 5), EXPIRIES[0],
                                          "OPTSTK", "CE", 1250)])
    install(monkeypatch, OnlyOptions())
    outcome = fno.fetch_fno_outcome("RELIANCE", "2026-01-05", "2026-01-05")
    assert outcome.status is Status.EMPTY
    assert outcome.data.empty and list(outcome.data.columns) == columns_of("fno")


@pytest.mark.parametrize("args,kwargs,word", [
    (("RELIANCE", "01/01/2026", "2026-02-01"), {}, "date"),
    (("RELIANCE", "2026-03-01", "2026-02-01"), {}, "from_date"),
    (("", "2026-01-01", "2026-02-01"), {}, "symbol"),
    (("RELIANCE", "2026-01-01", "2026-02-01"), {"instrument_type": "XYZ"}, "instrument_type"),
    (("RELIANCE", "2026-01-01", "2026-02-01"), {"instrument_type": "OPTSTK"}, "both"),
    (("RELIANCE", "2026-01-01", "2026-02-01"),
     {"instrument_type": "OPTSTK", "option_type": "XX", "strike_price": 1250}, "option_type"),
    (("RELIANCE", "2026-01-01", "2026-02-01"),
     {"instrument_type": "OPTSTK", "option_type": "CE", "strike_price": -5}, "strike_price"),
    (("RELIANCE", "2026-01-01", "2026-02-01"), {"strike_price": 1250}, "only apply to options"),
    (("RELIANCE", "2026-01-01", "2026-02-01"), {"expiry_date": "not-a-date"}, "expiry_date"),
])
def test_outcome_failed_for_invalid_arguments_without_fetching(monkeypatch, args, kwargs, word):
    session = install(monkeypatch, FakeSession(futures_universe(30)))
    outcome = fno.fetch_fno_outcome(*args, **kwargs)
    assert outcome.status is Status.FAILED and outcome.failed
    assert outcome.data is None
    assert word in outcome.reason
    assert session.calls == []


def test_outcome_failed_when_session_cannot_be_created(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("dropped")
    monkeypatch.setattr(fno, "create_nse_session", boom)
    outcome = fno.fetch_fno_outcome(*RANGE)
    assert outcome.failed and outcome.data is None
    assert "session" in outcome.reason and "dropped" in outcome.reason


def test_outcome_failed_request_is_never_partial_and_not_retried(monkeypatch, caplog):
    session = install(monkeypatch, FakeSession(futures_universe(150), fail_on_call=3))
    with caplog.at_level(logging.ERROR, logger="pyequity.connectors.fno"):
        outcome = fno.fetch_fno_outcome("RELIANCE", "2026-01-01", "2026-12-31")
    assert outcome.failed and outcome.data is None
    assert "request failed" in outcome.reason
    assert outcome.attempts == 1
    assert len(session.calls) == 3  # stopped at the failing call, no retry
    assert any("could not be guaranteed" in r.message for r in caplog.records)


def test_outcome_failed_on_the_very_first_request(monkeypatch):
    session = install(monkeypatch, FakeSession(futures_universe(10), fail_on_call=1))
    outcome = fno.fetch_fno_outcome(*RANGE)
    assert outcome.failed and len(session.calls) == 1


def test_outcome_failed_when_single_day_reaches_the_row_cap(monkeypatch, caplog):
    day = datetime(2026, 11, 13)
    universe = [make_row(day, day + timedelta(days=i)) for i in range(80)]
    install(monkeypatch, FakeSession(universe))
    with caplog.at_level(logging.ERROR, logger="pyequity.connectors.fno"):
        outcome = fno.fetch_fno_outcome("RELIANCE", "2026-11-13", "2026-11-13")
    assert outcome.failed and outcome.data is None
    assert "row cap" in outcome.reason and "narrow" in outcome.reason
    assert any("cap" in r.message for r in caplog.records)


def test_outcome_failed_for_unexpected_response_shape(monkeypatch):
    class Odd(FakeSession):
        def get(self, url, params=None, timeout=None):
            return FakeResponse({"message": "blocked"})
    install(monkeypatch, Odd())
    outcome = fno.fetch_fno_outcome(*RANGE)
    assert outcome.failed and outcome.data is None


def test_outcome_failed_when_the_request_raises(monkeypatch):
    class Raises(FakeSession):
        def get(self, url, params=None, timeout=None):
            raise RuntimeError("socket closed")
    install(monkeypatch, Raises())
    outcome = fno.fetch_fno_outcome(*RANGE)  # must not raise
    assert outcome.failed and outcome.data is None


def test_failed_is_not_empty_and_empty_is_not_failed(monkeypatch):
    install(monkeypatch, FakeSession([]))
    empty = fno.fetch_fno_outcome(*RANGE)
    install(monkeypatch, FakeSession([], fail_on_call=1))
    failed = fno.fetch_fno_outcome(*RANGE)
    assert empty.status is Status.EMPTY and failed.status is Status.FAILED
    assert empty.data is not None and failed.data is None


# ---------- fetch_fno_data stays a faithful wrapper ----------

def _scenario(monkeypatch, name):
    if name == "ok":
        return install(monkeypatch, FakeSession(futures_universe(10)))
    if name == "empty":
        return install(monkeypatch, FakeSession([]))
    return install(monkeypatch, FakeSession(futures_universe(150), fail_on_call=3))


@pytest.mark.parametrize("name", ["ok", "empty", "failed"])
def test_wrapper_returns_exactly_what_the_outcome_holds(monkeypatch, name):
    args = ("RELIANCE", "2026-01-01", "2026-12-31" if name == "failed" else "2026-01-20")
    session = _scenario(monkeypatch, name)
    outcome = fno.fetch_fno_outcome(*args)
    calls_for_outcome = len(session.calls)

    session = _scenario(monkeypatch, name)
    legacy = fno.fetch_fno_data(*args)

    assert len(session.calls) == calls_for_outcome  # same requests, no extras
    expected = outcome.data if outcome.ok else empty_frame("fno")
    pd.testing.assert_frame_equal(legacy, expected)
    assert list(legacy.columns) == columns_of("fno")
    assert legacy.empty == (name != "ok")


def test_wrapper_passes_every_argument_through(monkeypatch):
    rows = [make_row(d, EXPIRIES[0], "OPTSTK", "CE", 1250)
            for d in weekdays(datetime(2026, 1, 1), 5)]
    session = install(monkeypatch, FakeSession(rows))
    df = fno.fetch_fno_data("reliance", "2026-01-01", "2026-01-10", "24-Nov-2026",
                            "optstk", strike_price=1250.0, option_type="ce")
    p = session.calls[0][1]
    assert p["expiryDate"] == "24-Nov-2026" and p["instrumentType"] == "OPTSTK"
    assert p["strikePrice"] == "1250" and p["optionType"] == "CE" and len(df) == 5


# ---------- list_expiries_outcome ----------

def test_expiries_outcome_ok_sorted_iso(monkeypatch):
    info = {"expiryDates": ["29-Dec-2026", "23-Nov-2026", "25-Jan-2027"]}
    session = install(monkeypatch, FakeSession(info=info))
    outcome = fno.list_expiries_outcome("reliance")
    assert outcome.status is Status.OK and outcome.ok
    assert outcome.data == ["2026-11-23", "2026-12-29", "2027-01-25"]
    assert outcome.reason == "" and outcome.attempts == 1
    assert session.calls[0][1] == {"symbol": "RELIANCE"}


def test_expiries_outcome_empty_for_an_explicit_empty_list(monkeypatch, caplog):
    install(monkeypatch, FakeSession(info={"expiryDates": []}))
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.fno"):
        outcome = fno.list_expiries_outcome("NOTASYMBOL123")
    assert outcome.status is Status.EMPTY and outcome.empty
    assert outcome.data == [] and outcome.reason
    assert any("No expiries listed" in r.message for r in caplog.records)


@pytest.mark.parametrize("info", [
    {"expiryDates": "x"}, {"expiryDates": None}, [], None,
    {"message": "blocked"}, {"strikePrice": ["560"]}, "{}", 0,
])
def test_expiries_outcome_failed_when_there_is_no_list_of_expiries(monkeypatch, info):
    install(monkeypatch, FakeSession(info=info))
    outcome = fno.list_expiries_outcome("RELIANCE")
    assert outcome.status is Status.FAILED and outcome.data is None
    assert "shape" in outcome.reason


def test_expiries_outcome_empty_for_the_empty_object_nse_sends_for_unknown_symbols(
        monkeypatch, caplog):
    # Observed live (2026-11): an unknown symbol gives HTTP 200 and exactly {}.
    install(monkeypatch, FakeSession(info={}))
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.fno"):
        outcome = fno.list_expiries_outcome("NOTASYMBOL123")
    assert outcome.status is Status.EMPTY and outcome.empty
    assert outcome.data == [] and outcome.reason
    assert any("No expiries listed" in r.message for r in caplog.records)


def test_expiries_outcome_failed_when_the_body_is_not_json(monkeypatch):
    class NotJson(FakeSession):
        def get(self, url, params=None, timeout=None):
            class Resp:
                status_code = 200
                def json(self):
                    raise ValueError("Expecting value")
            return Resp()
    install(monkeypatch, NotJson())
    outcome = fno.list_expiries_outcome("RELIANCE")
    assert outcome.failed and outcome.data is None and "Expecting value" in outcome.reason


def test_expiries_outcome_failed_on_http_error(monkeypatch):
    session = install(monkeypatch, FakeSession(info={"expiryDates": ["23-Nov-2026"]},
                                               fail_on_call=1))
    outcome = fno.list_expiries_outcome("RELIANCE")
    assert outcome.failed and outcome.data is None and "500" in outcome.reason
    assert len(session.calls) == 1  # no retry


def test_expiries_outcome_failed_when_session_cannot_be_created(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("dropped")
    monkeypatch.setattr(fno, "create_nse_session", boom)
    outcome = fno.list_expiries_outcome("RELIANCE")
    assert outcome.failed and "dropped" in outcome.reason


def test_expiries_outcome_failed_for_empty_symbol_without_a_request(monkeypatch):
    session = install(monkeypatch, FakeSession(info={"expiryDates": ["23-Nov-2026"]}))
    outcome = fno.list_expiries_outcome("  ")
    assert outcome.failed and "symbol" in outcome.reason
    assert session.calls == []


def test_expiries_outcome_skips_junk_but_fails_when_nothing_is_readable(monkeypatch):
    install(monkeypatch, FakeSession(info={"expiryDates": ["23-Nov-2026", "junk"]}))
    assert fno.list_expiries_outcome("RELIANCE").data == ["2026-11-23"]
    install(monkeypatch, FakeSession(info={"expiryDates": ["junk", "also junk"]}))
    outcome = fno.list_expiries_outcome("RELIANCE")
    assert outcome.failed and outcome.data is None and "read" in outcome.reason


@pytest.mark.parametrize("info,fail_on_call", [
    ({"expiryDates": ["29-Dec-2026", "23-Nov-2026"]}, None),   # ok
    ({"expiryDates": []}, None),                               # empty
    ({}, None),                                                # empty: unknown symbol
    ({"message": "blocked"}, None),                            # failed: bad shape
    ({"expiryDates": ["23-Nov-2026"]}, 1),                     # failed: HTTP error
])
def test_list_expiries_wrapper_matches_the_outcome(monkeypatch, info, fail_on_call):
    install(monkeypatch, FakeSession(info=info, fail_on_call=fail_on_call))
    outcome = fno.list_expiries_outcome("RELIANCE")
    install(monkeypatch, FakeSession(info=info, fail_on_call=fail_on_call))
    legacy = fno.list_expiries("RELIANCE")
    assert isinstance(legacy, list)
    assert legacy == (outcome.data if outcome.ok else [])
def test_outcome_failed_when_processing_raises_after_the_fetch(monkeypatch, caplog):
    install(monkeypatch, FakeSession(futures_universe(5)))

    def boom(self, *a, **k):
        raise RuntimeError("post-processing exploded")

    monkeypatch.setattr(pd.DataFrame, "drop_duplicates", boom)

    with caplog.at_level(logging.ERROR, logger="pyequity.connectors.fno"):
        outcome = fno.fetch_fno_outcome(*RANGE, "24-Nov-2026", "FUTSTK")
        legacy = fno.fetch_fno_data(*RANGE, "24-Nov-2026", "FUTSTK")

    assert outcome.failed and outcome.data is None
    assert "RuntimeError" in outcome.reason
    assert "post-processing exploded" in outcome.reason
    assert legacy.empty and list(legacy.columns) == columns_of("fno")
    assert any("post-processing exploded" in r.message for r in caplog.records)