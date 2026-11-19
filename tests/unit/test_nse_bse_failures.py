"""Offline tests: NSE/BSE equity connectors never return partial data, retry
real failures, and always match the schema dtypes (on any pandas version)."""
import json
import re
from datetime import datetime

import pytest
import requests
from pandas.testing import assert_frame_equal

from pyequity.connectors import bse, nse
from pyequity.schema import SCHEMAS, empty_frame


class NetworkBlocked(BaseException):
    """BaseException on purpose: connectors catch Exception and would hide it."""


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise NetworkBlocked("a test tried to use the network")
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def dtypes_of(df):
    return {c: str(t) for c, t in df.dtypes.items()}


# ------------------------------------------------------------------ NSE
def nse_row(day="23-Sep-2026", close=1248):
    return {"mtimestamp": day, "chOpeningPrice": 1242, "chTradeHighPrice": 1252.8,
            "chTradeLowPrice": 1238.9, "chClosingPrice": close,
            "chTotTradedQty": 8352048, "chTotalTrades": 147001}


class NseResponse:
    def __init__(self, body, status=200):
        self.status_code = status
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


class NseSession:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        return self.handler(self.calls[-1], len(self.calls))


def use_nse(monkeypatch, handler):
    session = NseSession(handler)
    monkeypatch.setattr(nse, "create_nse_session", lambda *a, **k: session)
    return session


def test_nse_output_matches_schema_dtypes(monkeypatch):
    use_nse(monkeypatch, lambda p, n: NseResponse([nse_row()]))
    df = nse.fetch_nse_historical("reliance", "2026-09-15", "2026-09-23")
    assert list(df.columns) == list(SCHEMAS["equity"])
    assert dtypes_of(df) == SCHEMAS["equity"]
    assert df["symbol"].tolist() == ["RELIANCE"] and df["source"].tolist() == ["NSE"]


def test_nse_chunks_are_stitched_in_date_order(monkeypatch):
    def handler(params, n):
        day = datetime.strptime(params["fromDate"], "%d-%m-%Y").strftime("%d-%b-%Y")
        return NseResponse([nse_row(day)])
    session = use_nse(monkeypatch, handler)
    df = nse.fetch_nse_historical("RELIANCE", "2026-01-01", "2026-05-10")
    assert len(session.calls) == 3 and len(df) == 3
    assert df["date"].is_monotonic_increasing


def test_nse_failed_chunk_returns_empty_not_partial(monkeypatch, caplog):
    def handler(params, n):
        if params["fromDate"] == "01-01-2026":
            return NseResponse([nse_row("02-Jan-2026")])
        return NseResponse("server error", status=500)
    session = use_nse(monkeypatch, handler)
    df = nse.fetch_nse_historical("RELIANCE", "2026-01-01", "2026-05-10")
    assert_frame_equal(df, empty_frame("equity"))
    assert len(session.calls) == 1 + nse._ATTEMPTS      # first chunk, then retries
    assert "partial" in caplog.text


def test_nse_transient_error_is_retried(monkeypatch):
    def handler(params, n):
        return NseResponse("oops", status=500) if n == 1 else NseResponse([nse_row()])
    session = use_nse(monkeypatch, handler)
    df = nse.fetch_nse_historical("RELIANCE", "2026-09-15", "2026-09-23")
    assert len(df) == 1 and len(session.calls) == 2


def test_nse_empty_answer_is_not_retried_and_not_an_error(monkeypatch):
    session = use_nse(monkeypatch, lambda p, n: NseResponse("[]"))
    df = nse.fetch_nse_historical("WRONGSYMBOL123", "2026-09-15", "2026-09-23")
    assert_frame_equal(df, empty_frame("equity"))
    assert len(session.calls) == 1


def test_nse_chunk_at_the_row_cap_is_rejected(monkeypatch, caplog):
    use_nse(monkeypatch, lambda p, n: NseResponse([nse_row()] * nse.NSE_ROW_CAP))
    df = nse.fetch_nse_historical("RELIANCE", "2026-09-15", "2026-09-23")
    assert_frame_equal(df, empty_frame("equity"))
    assert "cap" in caplog.text


def test_nse_session_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("dropped")
    monkeypatch.setattr(nse, "create_nse_session", boom)
    assert_frame_equal(nse.fetch_nse_historical("RELIANCE", "2026-09-15", "2026-09-23"),
                       empty_frame("equity"))


@pytest.mark.parametrize("symbol,start,end", [
    ("RELIANCE", "15/09/2026", "2026-09-23"),
    ("RELIANCE", "2026-09-23", "2026-09-15"),
    ("", "2026-09-15", "2026-09-23"),
])
def test_nse_bad_arguments_return_empty_without_a_request(monkeypatch, symbol, start, end):
    session = use_nse(monkeypatch, lambda p, n: NseResponse([nse_row()]))
    assert_frame_equal(nse.fetch_nse_historical(symbol, start, end), empty_frame("equity"))
    assert session.calls == []


def test_nse_special_character_symbol_is_sent_as_a_parameter(monkeypatch):
    session = use_nse(monkeypatch, lambda p, n: NseResponse([nse_row()]))
    nse.fetch_nse_historical("M&M", "2026-09-15", "2026-09-23")
    assert session.calls[0]["symbol"] == "M&M"


def test_nse_internal_chunk_helper_still_returns_a_dataframe():
    session = NseSession(lambda p, n: NseResponse([nse_row()]))
    df = nse._fetch_nse_chunk(session, "RELIANCE", "15-09-2026", "23-09-2026")
    assert len(df) == 1


# ------------------------------------------------------------------ BSE
def bse_record(day="2026-09-15T00:00:00"):
    return {"RealDate": day, "qe_open": "1,254.45", "qe_high": "1,259.50",
            "qe_low": "1,236.00", "qe_close": "1,236.00",
            "no_of_shrs": "11,57,037", "no_trades": "40,761"}


class BseResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload


def use_bse(monkeypatch, handler):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return handler(url, len(calls))
    monkeypatch.setattr(bse.requests, "get", fake_get)
    return calls


def ok_payload(n=1, day="2026-09-15T00:00:00"):
    return {"scrip_id": "RELIANCE", "StockData": [bse_record(day)] * n}


def day_from_url(url):
    d, m, y = re.search(r"MonthDate=(\d\d)/(\d\d)/(\d{4})", url).groups()
    return f"{y}-{m}-{d}T00:00:00"


def test_bse_output_matches_schema_dtypes(monkeypatch):
    use_bse(monkeypatch, lambda url, n: BseResponse(ok_payload()))
    df = bse.fetch_bse_historical("500325", "2026-09-15", "2026-09-23")
    assert list(df.columns) == list(SCHEMAS["equity"])
    assert dtypes_of(df) == SCHEMAS["equity"]
    assert df["close"].tolist() == [1236.0] and df["volume"].tolist() == [1157037]


def test_bse_chunks_are_stitched_in_date_order(monkeypatch):
    calls = use_bse(monkeypatch,
                    lambda url, n: BseResponse(ok_payload(day=day_from_url(url))))
    df = bse.fetch_bse_historical("500325", "2026-01-01", "2026-03-15")
    assert len(calls) == 3 and len(df) == 3
    assert df["date"].is_monotonic_increasing


def test_bse_failed_chunk_returns_empty_not_partial(monkeypatch, caplog):
    def handler(url, n):
        if "MonthDate=01/01/2026" in url:
            return BseResponse(ok_payload(day="2026-01-02T00:00:00"))
        raise requests.ConnectionError("timed out")
    calls = use_bse(monkeypatch, handler)
    df = bse.fetch_bse_historical("500325", "2026-01-01", "2026-03-15")
    assert_frame_equal(df, empty_frame("equity"))
    assert len(calls) == 1 + bse._ATTEMPTS
    assert "partial" in caplog.text


def test_bse_transient_error_is_retried(monkeypatch):
    def handler(url, n):
        if n == 1:
            raise requests.ConnectionError("timed out")
        return BseResponse(ok_payload())
    calls = use_bse(monkeypatch, handler)
    df = bse.fetch_bse_historical("500325", "2026-09-15", "2026-09-23")
    assert len(df) == 1 and len(calls) == 2


def test_bse_invalid_scripcode_is_empty_and_never_retried(monkeypatch):
    payload = {"StockData": None, "spnError": "The scrip code 999999 does not exist."}
    calls = use_bse(monkeypatch, lambda url, n: BseResponse(payload))
    df = bse.fetch_bse_historical("999999", "2026-01-01", "2026-03-15")
    assert_frame_equal(df, empty_frame("equity"))
    assert len(calls) == 3                 # one per chunk, no retries


def test_bse_chunk_at_the_row_cap_is_rejected(monkeypatch, caplog):
    use_bse(monkeypatch, lambda url, n: BseResponse(ok_payload(n=bse.BSE_ROW_CAP)))
    df = bse.fetch_bse_historical("500325", "2026-09-15", "2026-09-23")
    assert_frame_equal(df, empty_frame("equity"))
    assert "cap" in caplog.text


def test_bse_unmapped_symbol_returns_empty_and_makes_no_request(monkeypatch):
    calls = use_bse(monkeypatch, lambda url, n: BseResponse(ok_payload()))
    df = bse.fetch_bse_historical_by_symbol("NOTASYMBOL", "2026-09-15", "2026-09-23")
    assert_frame_equal(df, empty_frame("equity"))
    assert calls == []


def test_bse_mapped_symbol_uses_its_scripcode(monkeypatch):
    calls = use_bse(monkeypatch, lambda url, n: BseResponse(ok_payload()))
    df = bse.fetch_bse_historical_by_symbol("RELIANCE", "2026-09-15", "2026-09-23")
    assert len(df) == 1 and "Scode=500325" in calls[0]


@pytest.mark.parametrize("start,end", [("15/09/2026", "2026-09-23"),
                                       ("2026-09-23", "2026-09-15")])
def test_bse_bad_dates_return_empty_without_a_request(monkeypatch, start, end):
    calls = use_bse(monkeypatch, lambda url, n: BseResponse(ok_payload()))
    assert_frame_equal(bse.fetch_bse_historical("500325", start, end), empty_frame("equity"))
    assert calls == []