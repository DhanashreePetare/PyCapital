"""Offline tests for connectors/amfi.py (no network: requests.get is replaced)."""
import logging

import pandas as pd
import pytest

from pyequity.connectors import amfi
from pyequity.schema import SCHEMAS, empty_frame

NAV_COLS = list(SCHEMAS["nav"])


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.content = (text or "").encode("utf-8")

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("unexpected real network call")
    monkeypatch.setattr(amfi.requests, "get", boom)


def patch_get(monkeypatch, handler):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, params))
        return handler(url, params)

    monkeypatch.setattr(amfi.requests, "get", fake_get)
    return calls


def assert_nav_schema(df):
    assert list(df.columns) == NAV_COLS
    for col, dtype in SCHEMAS["nav"].items():
        assert str(df[col].dtype) == dtype, col


# ----------------------------- fetch_nav (mfapi) -----------------------------

def mfapi_payload(rows):
    return {"meta": {"scheme_name": "Test Fund - Direct Plan - Growth"},
            "data": [{"date": d, "nav": n} for d, n in rows]}


def test_fetch_nav_basic(monkeypatch):
    payload = mfapi_payload([("23-09-2026", "10.5"), ("22-09-2026", "10.25")])
    patch_get(monkeypatch, lambda u, p: FakeResponse(200, payload))
    df = amfi.fetch_nav(119551)
    assert_nav_schema(df)
    assert len(df) == 2
    assert df["date"].is_monotonic_increasing
    assert set(df["source"]) == {"AMFI"}
    assert df["scheme_name"].iloc[0] == "Test Fund - Direct Plan - Growth"
    assert df["nav"].tolist() == [10.25, 10.5]


def test_fetch_nav_drops_only_unreadable_rows(monkeypatch, caplog):
    payload = mfapi_payload([("23-09-2026", "10.5"), ("22-09-2026", "N.A."),
                             ("21-09-2026", ""), ("20-09-2026", "10.0")])
    patch_get(monkeypatch, lambda u, p: FakeResponse(200, payload))
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.amfi"):
        df = amfi.fetch_nav(119551)
    assert len(df) == 2
    assert df["nav"].tolist() == [10.0, 10.5]
    assert any("Dropped 2 of 4" in r.message for r in caplog.records)


def test_fetch_nav_unreadable_date_dropped(monkeypatch):
    payload = mfapi_payload([("23-09-2026", "10.5"), ("not-a-date", "9.0")])
    patch_get(monkeypatch, lambda u, p: FakeResponse(200, payload))
    assert len(amfi.fetch_nav(119551)) == 1


def test_fetch_nav_all_rows_unreadable_returns_empty_frame(monkeypatch):
    payload = mfapi_payload([("23-09-2026", "N.A.")])
    patch_get(monkeypatch, lambda u, p: FakeResponse(200, payload))
    df = amfi.fetch_nav(119551)
    assert df.empty
    assert_nav_schema(df)


def test_fetch_nav_date_filter_inclusive(monkeypatch):
    payload = mfapi_payload([("25-09-2026", "3"), ("24-09-2026", "2"), ("23-09-2026", "1")])
    patch_get(monkeypatch, lambda u, p: FakeResponse(200, payload))
    df = amfi.fetch_nav(119551, "2026-09-24", "2026-09-24")
    assert df["nav"].tolist() == [2.0]


def test_fetch_nav_bad_status_returns_empty_frame(monkeypatch):
    patch_get(monkeypatch, lambda u, p: FakeResponse(404, {}))
    df = amfi.fetch_nav(999999999)
    assert df.empty
    assert_nav_schema(df)


def test_fetch_nav_no_data_returns_empty_frame(monkeypatch):
    patch_get(monkeypatch, lambda u, p: FakeResponse(200, {"meta": {}, "data": []}))
    assert_nav_schema(amfi.fetch_nav(1))


def test_fetch_nav_never_raises(monkeypatch):
    def boom(u, p):
        raise ConnectionError("down")
    patch_get(monkeypatch, boom)
    assert amfi.fetch_nav(119551).empty


# ------------------------- fetch_nav_official -------------------------------

HISTORY_HEADER = ("Scheme Code;NAV Name;Plan;Option;ISIN Div Payout/ISIN Growth;"
                  "ISIN Div Reinvestment;Net Asset Value;Date")


def history_text(rows, header=HISTORY_HEADER):
    lines = [header, "", "Open Ended Schemes ( Debt Scheme )", "", "Aditya Birla Sun Life Mutual Fund", ""]
    for code, nav, date in rows:
        lines.append(f"{code};Some Fund  - DIRECT - IDCW;Direct Plan;IDCW-Re-investment;"
                     f"INF1;INF2;{nav};{date}")
    return "\n".join(lines)


NAVALL_TEXT = "\n".join([
    "Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date",
    "",
    "Open Ended Schemes(Children’s Fund)",
    "Axis Mutual Fund",
    "135762;INF846K01WO1;-;Axis Children's Fund;Direct Plan;Growth Option;31.0814;13-Nov-2026",
    "119551;INF209KA12Z1;INF209KA13Z9;Aditya Birla Sun Life Banking & PSU Debt Fund;Direct Plan;IDCW-Re-investment;107.2765;13-Nov-2026",
])


def official_handler(history, navall=NAVALL_TEXT):
    def handler(url, params):
        if url == amfi.NAVALL_URL:
            return FakeResponse(200, text=navall)
        return FakeResponse(200, text=history)
    return handler


def test_parse_history_by_header_and_skips_headings():
    text = history_text([(119551, "107.2", "15-Sep-2026"), (999, "5", "15-Sep-2026"),
                         (119551, "107.3", "16-Sep-2026")])
    df, bad, name = amfi._parse_official_history(text, 119551)
    assert bad == 0
    assert df["nav"].tolist() == [107.2, 107.3]
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert "DIRECT" in name


def test_parse_history_column_order_does_not_matter():
    header = "Date;Net Asset Value;Scheme Code;NAV Name"
    text = header + "\n15-Sep-2026;10.5;119551;Some Fund"
    df, bad, name = amfi._parse_official_history(text, 119551)
    assert df["nav"].tolist() == [10.5] and name == "Some Fund"


def test_parse_history_missing_required_header_returns_none():
    assert amfi._parse_official_history("Scheme Code;Foo;Bar\n1;2;3", 1) is None


def test_parse_history_no_header_returns_none():
    assert amfi._parse_official_history("<html>blocked</html>", 1) is None


def test_parse_history_nonnumeric_nav_counted_and_dropped():
    text = history_text([(1, "N.A.", "15-Sep-2026"), (1, "7.5", "16-Sep-2026")])
    df, bad, _ = amfi._parse_official_history(text, 1)
    assert bad == 1 and df["nav"].tolist() == [7.5]


def test_navall_name_matches_mfapi_style():
    assert amfi._parse_navall_name(NAVALL_TEXT, 119551) == (
        "Aditya Birla Sun Life Banking & PSU Debt Fund - Direct Plan - IDCW-Re-investment")


def test_navall_name_unknown_scheme_or_changed_layout():
    assert amfi._parse_navall_name(NAVALL_TEXT, 1) is None
    assert amfi._parse_navall_name("Scheme Code;Foo\n1;2", 1) is None


def test_official_basic(monkeypatch):
    text = history_text([(119551, "107.2", "15-Sep-2026"), (119551, "107.3", "16-Sep-2026")])
    patch_get(monkeypatch, official_handler(text))
    df = amfi.fetch_nav_official(119551, "2026-09-15", "2026-09-23")
    assert_nav_schema(df)
    assert len(df) == 2
    assert set(df["source"]) == {"AMFI_OFFICIAL"}
    assert df["scheme_name"].iloc[0].endswith("Direct Plan - IDCW-Re-investment")
    assert df["scheme_code"].iloc[0] == 119551


def test_official_uses_dd_mon_yyyy_dates_and_one_chunk_for_short_range(monkeypatch):
    text = history_text([(1, "1", "15-Sep-2026")])
    calls = patch_get(monkeypatch, official_handler(text))
    amfi.fetch_nav_official(1, "2026-09-15", "2026-09-23")
    history_calls = [c for c in calls if c[0] == amfi.HISTORY_URL]
    assert len(history_calls) == 1
    assert history_calls[0][1] == {"frmdt": "15-Sep-2026", "todt": "23-Sep-2026"}


def test_official_long_range_is_chunked_without_gaps_or_overlap(monkeypatch):
    calls = patch_get(monkeypatch, official_handler(history_text([])))
    amfi.fetch_nav_official(1, "2026-01-01", "2026-03-31")  # 90 days
    ranges = [c[1] for c in calls if c[0] == amfi.HISTORY_URL]
    assert len(ranges) == 3
    assert ranges[0] == {"frmdt": "01-Jan-2026", "todt": "30-Jan-2026"}
    assert ranges[1] == {"frmdt": "31-Jan-2026", "todt": "01-Mar-2026"}
    assert ranges[2] == {"frmdt": "02-Mar-2026", "todt": "31-Mar-2026"}


def test_official_dedupes_dates_across_chunks(monkeypatch):
    text = history_text([(1, "5", "30-Jan-2026")])
    patch_get(monkeypatch, official_handler(text))
    df = amfi.fetch_nav_official(1, "2026-01-01", "2026-03-31")
    assert len(df) == 1


def test_official_falls_back_to_history_name_when_navall_lacks_scheme(monkeypatch):
    text = history_text([(5, "9", "15-Sep-2026")])
    patch_get(monkeypatch, official_handler(text))
    df = amfi.fetch_nav_official(5, "2026-09-15", "2026-09-15")
    assert df["scheme_name"].iloc[0] == "Some Fund  - DIRECT - IDCW"


def test_official_layout_change_returns_empty_and_logs_error(monkeypatch, caplog):
    patch_get(monkeypatch, official_handler("Scheme Code;Foo\n1;2"))
    with caplog.at_level(logging.ERROR, logger="pyequity.connectors.amfi"):
        df = amfi.fetch_nav_official(1, "2026-09-15", "2026-09-23")
    assert df.empty
    assert_nav_schema(df)
    assert any("layout" in r.message for r in caplog.records)


def test_official_failed_chunk_is_reported_and_partial_data_kept(monkeypatch, caplog):
    good = history_text([(1, "5", "05-Jan-2026")])

    def handler(url, params):
        if url == amfi.NAVALL_URL:
            return FakeResponse(200, text=NAVALL_TEXT)
        if params["frmdt"] == "31-Jan-2026":
            return FakeResponse(503, text="")
        return FakeResponse(200, text=good)

    patch_get(monkeypatch, handler)
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.amfi"):
        df = amfi.fetch_nav_official(1, "2026-01-01", "2026-03-31")
    assert len(df) == 1
    assert any("may be incomplete" in r.message and "2026-01-31" in r.message
               for r in caplog.records)


def test_official_empty_chunks_are_not_reported_as_failures(monkeypatch, caplog):
    patch_get(monkeypatch, official_handler(history_text([])))
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.amfi"):
        df = amfi.fetch_nav_official(1, "2026-09-15", "2026-09-23")
    assert df.empty
    assert not any("may be incomplete" in r.message for r in caplog.records)


def test_official_nonnumeric_nav_dropped_with_warning(monkeypatch, caplog):
    text = history_text([(1, "N.A.", "15-Sep-2026"), (1, "7.5", "16-Sep-2026")])
    patch_get(monkeypatch, official_handler(text))
    with caplog.at_level(logging.WARNING, logger="pyequity.connectors.amfi"):
        df = amfi.fetch_nav_official(1, "2026-09-15", "2026-09-23")
    assert df["nav"].tolist() == [7.5]
    assert any("Dropped 1 official rows" in r.message for r in caplog.records)


def test_official_decodes_utf8_itself():
    class R:  # requests would guess latin-1 here; _decode must not
        content = "Children’s Fund".encode("utf-8")
        encoding = "ISO-8859-1"
    assert amfi._decode(R()) == "Children’s Fund"


@pytest.mark.parametrize("code,start,end", [
    ("abc", "2026-09-15", "2026-09-23"),
    (None, "2026-09-15", "2026-09-23"),
    (1, "15-09-2026", "2026-09-23"),
    (1, "2026-09-15", None),
    (1, "2026-09-23", "2026-09-15"),
])
def test_official_invalid_input_returns_empty_frame_without_network(code, start, end):
    df = amfi.fetch_nav_official(code, start, end)  # no_network fixture would raise
    assert df.empty
    assert_nav_schema(df)


def test_official_never_raises_on_network_error(monkeypatch):
    def boom(u, p):
        raise ConnectionError("down")
    patch_get(monkeypatch, boom)
    df = amfi.fetch_nav_official(1, "2026-09-15", "2026-09-23")
    assert df.empty
    assert_nav_schema(df)