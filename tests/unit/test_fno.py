"""Offline tests for pyequity.connectors.fno (no internet needed)."""
import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest

from pyequity.connectors import fno
from pyequity.schema import SCHEMAS, columns_of

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