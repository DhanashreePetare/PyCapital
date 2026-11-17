"""Offline tests for connectors/sebi.py (no network: session and requests are faked)."""
import logging

import pandas as pd
import pytest

from pyequity.connectors import sebi
from pyequity.schema import SCHEMAS

COLS = list(SCHEMAS["bulk_deals"])


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.content = text.encode("utf-8")

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Answers API calls from a dict {DD-MM-YYYY: list-of-records or Exception}."""
    def __init__(self, by_day):
        self.by_day = by_day
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        value = self.by_day.get(params["from"], [])
        if isinstance(value, Exception):
            raise value
        return FakeResp(200, {"data": value})


def rec(day, n, client="C"):
    return [{"BD_DT_DATE": day, "BD_SYMBOL": "ABC", "BD_SCRIP_NAME": "Abc Ltd",
             "BD_CLIENT_NAME": f"{client}{i}", "BD_BUY_SELL": "BUY" if i % 2 else "SELL",
             "BD_QTY_TRD": 100 + i, "BD_TP_WATP": 10.5} for i in range(n)]


@pytest.fixture(autouse=True)
def locked(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("unexpected real network call")
    monkeypatch.setattr(sebi.requests, "get", boom)
    monkeypatch.setattr(sebi, "create_nse_session", boom)
    monkeypatch.setattr(sebi.time, "sleep", lambda s: None)


def use_session(monkeypatch, session):
    monkeypatch.setattr(sebi, "create_nse_session", lambda *a, **k: session)


def check_schema(df):
    assert list(df.columns) == COLS
    for c, t in SCHEMAS["bulk_deals"].items():
        assert str(df[c].dtype) == t, c


# --------------------------- fetch_bulk_deals (unchanged behaviour) ---------

def test_fetch_bulk_deals_basic(monkeypatch):
    s = FakeSession({"21-09-2026": rec("21-Sep-2026", 3)})
    use_session(monkeypatch, s)
    df = sebi.fetch_bulk_deals("2026-09-21", "2026-09-21")
    check_schema(df)
    assert len(df) == 3 and set(df["source"]) == {"SEBI_NSE"}
    assert s.calls == [{"optionType": "bulk_deals", "from": "21-09-2026", "to": "21-09-2026"}]


def test_fetch_bulk_deals_cap_warning_says_incomplete_and_not_older(monkeypatch, caplog):
    use_session(monkeypatch, FakeSession({"21-09-2026": rec("21-Sep-2026", 70)}))
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.sebi"):
        df = sebi.fetch_bulk_deals("2026-09-21", "2026-09-21")
    assert len(df) == 70
    msgs = " ".join(r.message for r in caplog.records)
    assert "70-row cap" in msgs and "INCOMPLETE" in msgs
    assert "Older deals" not in msgs and "older deals" not in msgs


def test_fetch_bulk_deals_empty_and_errors(monkeypatch):
    use_session(monkeypatch, FakeSession({}))
    assert sebi.fetch_bulk_deals("2026-09-21", "2026-09-21").empty
    assert sebi.fetch_bulk_deals("bad", "2026-09-21").empty
    use_session(monkeypatch, FakeSession({"21-09-2026": ConnectionError("x")}))
    df = sebi.fetch_bulk_deals("2026-09-21", "2026-09-21")
    assert df.empty
    check_schema(df)


# --------------------------- fetch_bulk_deals_by_day -------------------------

def test_by_day_covers_each_weekday_and_skips_weekend(monkeypatch):
    s = FakeSession({"18-09-2026": rec("18-Sep-2026", 2), "21-09-2026": rec("21-Sep-2026", 3)})
    use_session(monkeypatch, s)
    df = sebi.fetch_bulk_deals_by_day("2026-09-18", "2026-09-22")  # Fri..Tue
    check_schema(df)
    assert [c["from"] for c in s.calls] == ["18-09-2026", "21-09-2026", "22-09-2026"]
    assert all(c["from"] == c["to"] for c in s.calls)
    assert len(df) == 5 and df["date"].is_monotonic_increasing
    assert df.attrs["capped_days"] == [] and df.attrs["failed_days"] == []


def test_by_day_flags_capped_days_in_log_and_attrs(monkeypatch, caplog):
    s = FakeSession({"21-09-2026": rec("21-Sep-2026", 70), "22-09-2026": rec("22-Sep-2026", 5)})
    use_session(monkeypatch, s)
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.sebi"):
        df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-22")
    assert len(df) == 75
    assert df.attrs["capped_days"] == ["2026-09-21"]
    assert any("INCOMPLETE" in r.message and "2026-09-21" in r.message for r in caplog.records)


def test_by_day_failed_day_reported_and_others_kept(monkeypatch, caplog):
    s = FakeSession({"21-09-2026": ConnectionError("boom"), "22-09-2026": rec("22-Sep-2026", 4)})
    use_session(monkeypatch, s)
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.sebi"):
        df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-22")
    assert len(df) == 4
    assert df.attrs["failed_days"] == ["2026-09-21"]
    assert any("Could not fetch" in r.message for r in caplog.records)


def test_by_day_no_rows_returns_empty_frame_with_attrs(monkeypatch):
    use_session(monkeypatch, FakeSession({}))
    df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-22")
    assert df.empty
    check_schema(df)
    assert df.attrs["capped_days"] == []


def test_by_day_session_failure_and_bad_input_never_raise(monkeypatch):
    def broken(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(sebi, "create_nse_session", broken)
    assert sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-22").empty
    for a, b in [("x", "2026-09-22"), ("2026-09-22", "2026-09-21"), (None, None)]:
        df = sebi.fetch_bulk_deals_by_day(a, b)
        assert df.empty
        check_schema(df)


def test_by_day_opens_one_session(monkeypatch):
    opened = []
    s = FakeSession({})
    monkeypatch.setattr(sebi, "create_nse_session", lambda *a, **k: opened.append(1) or s)
    sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-25")
    assert len(opened) == 1 and len(s.calls) == 5


# --------------------------- fetch_latest_bulk_deals (bulk.csv) --------------

CSV_HEADER = "Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,Trade Price / Wght. Avg. Price,Remarks"
CSV_OK = "\n".join([
    CSV_HEADER,
    "13-NOV-2026,AAATECH,AAA Technologies Limited,VAIBHAV RAJENDRA DOSHI,BUY,80083,90.63,-",
    '13-NOV-2026,ALICON,"Alicon, Castalloy Limited",JUNOMONETA FINSOL PRIVATE LIMITED,sell,"95,589",941.01,-',
    "13-NOV-2026,ALICON,\"Alicon, Castalloy Limited\",JUNOMONETA FINSOL PRIVATE LIMITED,sell,\"95,589\",941.01,-",
])


def patch_csv(monkeypatch, handler):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return handler(url)
    monkeypatch.setattr(sebi.requests, "get", fake_get)
    return calls


def test_parse_csv_maps_columns_and_types():
    df = sebi._parse_bulk_csv(CSV_OK)
    check_schema(df)
    assert len(df) == 3
    assert set(df["source"]) == {"SEBI_NSE_CSV"}
    assert df["date"].iloc[0] == pd.Timestamp("2026-11-13")
    assert df["quantity"].tolist() == [80083, 95589, 95589]
    assert df["deal_type"].tolist() == ["BUY", "SELL", "SELL"]
    assert df["scrip_name"].iloc[1] == "Alicon, Castalloy Limited"
    assert df["price"].iloc[0] == 90.63


def test_parse_csv_keeps_genuine_identical_rows():
    assert len(sebi._parse_bulk_csv(CSV_OK)) == 3


def test_parse_csv_drops_unreadable_and_malformed_rows_with_warning(caplog):
    text = "\n".join([CSV_HEADER,
        "13-NOV-2026,A,A Ltd,C1,BUY,100,10.0,-",
        "13-NOV-2026,B,B Ltd,C2,BUY,abc,10.0,-",
        "13-NOV-2026,C,C Ltd,C3,BUY,100,-,-",
        "13-NOV-2026,D,D Ltd,C4,BUY",
        "not-a-date,E,E Ltd,C5,BUY,100,10.0,-"])
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.sebi"):
        df = sebi._parse_bulk_csv(text)
    assert df["symbol"].tolist() == ["A"]
    assert any("Dropped 4" in r.message and "1 with the wrong" in r.message for r in caplog.records)


def test_parse_csv_layout_change_returns_none():
    assert sebi._parse_bulk_csv("Date,Symbol\n13-NOV-2026,A") is None
    assert sebi._parse_bulk_csv("") is None
    no_price = CSV_HEADER.replace("Trade Price / Wght. Avg. Price", "Something")
    assert sebi._parse_bulk_csv(no_price + "\n") is None


def test_parse_csv_header_only_is_empty_not_none():
    df = sebi._parse_bulk_csv(CSV_HEADER + "\n")
    assert df.empty
    check_schema(df)


def test_latest_basic_and_utf8_bom(monkeypatch):
    patch_csv(monkeypatch, lambda u: FakeResp(200, text="\ufeff" + CSV_OK))
    df = sebi.fetch_latest_bulk_deals()
    check_schema(df)
    assert len(df) == 3


def test_latest_falls_back_to_second_address(monkeypatch):
    calls = patch_csv(monkeypatch, lambda u: FakeResp(503) if "nsearchives" in u else FakeResp(200, text=CSV_OK))
    df = sebi.fetch_latest_bulk_deals()
    assert len(df) == 3 and len(calls) == 2


def test_latest_all_addresses_fail_returns_empty_frame(monkeypatch):
    patch_csv(monkeypatch, lambda u: FakeResp(503))
    df = sebi.fetch_latest_bulk_deals()
    assert df.empty
    check_schema(df)


def test_latest_network_exception_never_raises(monkeypatch):
    def boom(u):
        raise ConnectionError("down")
    patch_csv(monkeypatch, boom)
    assert sebi.fetch_latest_bulk_deals().empty


def test_latest_layout_change_logs_error(monkeypatch, caplog):
    patch_csv(monkeypatch, lambda u: FakeResp(200, text="<html>blocked</html>"))
    with caplog.at_level(logging.ERROR, logger="pyequity.connectors.sebi"):
        df = sebi.fetch_latest_bulk_deals()
    assert df.empty
    assert any("layout" in r.message for r in caplog.records)


def test_latest_warns_when_file_has_multiple_dates(monkeypatch, caplog):
    text = CSV_OK + "\n14-NOV-2026,Z,Z Ltd,C9,BUY,5,1.0,-"
    patch_csv(monkeypatch, lambda u: FakeResp(200, text=text))
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.sebi"):
        sebi.fetch_latest_bulk_deals()
    assert any("more than one date" in r.message for r in caplog.records)