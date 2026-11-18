"""
Offline replay tests for the AMFI official-file connector and the SEBI bulk
deals additions, using real raw responses captured by
tools/capture_fixtures_step1b.py (tests/fixtures/step1b/) and the existing
bulk deals API fixture. No network: requests.get and the NSE session are faked.

Expected values are computed independently from the raw fixture text (plain
string splitting / csv module), not by the code under test.
"""
import csv
import glob
import hashlib
import io
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from pyequity.connectors import amfi, sebi
from pyequity.processing.normalize import normalize, validate
from pyequity.schema import SCHEMAS

FIX = Path(__file__).resolve().parent.parent / "fixtures"
STEP = FIX / "step1b"
SCHEME = 119551
SCHEME_NAME = "Aditya Birla Sun Life Banking & PSU Debt Fund - Direct Plan - IDCW-Re-investment"
EXPECTED_DATES = ["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
                  "2026-09-21", "2026-09-22", "2026-09-23"]
FIRST_THREE_NAVS = [106.9321, 106.9851, 107.0008]   # seen on real data in the live check


def read(name):
    return (STEP / name).read_bytes()


HISTORY = "amfi_history_119551_2026-09-15_2026-09-23.txt"
NAVALL = "amfi_navall_sample.txt"
BULK = "bulk_latest.csv"


class Resp:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("unexpected real network call")
    monkeypatch.setattr(amfi.requests, "get", boom)
    monkeypatch.setattr(sebi.requests, "get", boom)
    monkeypatch.setattr(sebi, "create_nse_session", boom)
    monkeypatch.setattr(sebi.time, "sleep", lambda s: None)


def serve_amfi(monkeypatch, history_bytes=None):
    calls = []

    def fake(url, params=None, headers=None, timeout=None):
        calls.append((url, params))
        if url == amfi.NAVALL_URL:
            return Resp(read(NAVALL))
        return Resp(history_bytes if history_bytes is not None else read(HISTORY))
    monkeypatch.setattr(amfi.requests, "get", fake)
    return calls


# ------------------------------ manifest -------------------------------------

def test_step1b_fixtures_match_manifest():
    manifest = json.loads((STEP / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {HISTORY, NAVALL, BULK}
    for name, meta in manifest["files"].items():
        data = read(name)
        assert len(data) == meta["bytes"], name
        assert hashlib.sha256(data).hexdigest() == meta["sha256"], f"{name} was modified"


# ------------------------- AMFI official history -----------------------------

def independent_history_rows():
    text = read(HISTORY).decode("utf-8")
    rows = []
    for line in text.splitlines():
        f = [x.strip() for x in line.split(";")]
        if f[0] == str(SCHEME):
            rows.append((f[-1], float(f[-2])))
    return rows


def test_official_matches_independent_parse_of_raw_fixture(monkeypatch):
    serve_amfi(monkeypatch)
    df = amfi.fetch_nav_official(SCHEME, "2026-09-15", "2026-09-23")
    expected = independent_history_rows()
    assert len(expected) == 7
    assert len(df) == len(expected)
    got = [(d.strftime("%d-%b-%Y"), n) for d, n in zip(df["date"], df["nav"])]
    assert got == expected
    assert [d.strftime("%Y-%m-%d") for d in df["date"]] == EXPECTED_DATES
    assert df["nav"].tolist()[:3] == FIRST_THREE_NAVS


def test_official_schema_labels_and_name(monkeypatch):
    serve_amfi(monkeypatch)
    df = amfi.fetch_nav_official(SCHEME, "2026-09-15", "2026-09-23")
    assert list(df.columns) == list(SCHEMAS["nav"])
    for c, t in SCHEMAS["nav"].items():
        assert str(df[c].dtype) == t, c
    assert set(df["source"]) == {"AMFI_OFFICIAL"}
    assert set(df["scheme_code"]) == {SCHEME}
    assert set(df["scheme_name"]) == {SCHEME_NAME}


def test_official_passes_validate_and_normalize_is_a_no_op(monkeypatch):
    serve_amfi(monkeypatch)
    df = amfi.fetch_nav_official(SCHEME, "2026-09-15", "2026-09-23")
    assert validate(df, "nav").ok
    pd.testing.assert_frame_equal(normalize(df, "nav"), df)


def test_official_other_schemes_in_file_are_not_mixed_in(monkeypatch):
    serve_amfi(monkeypatch)
    df = amfi.fetch_nav_official(SCHEME, "2026-09-15", "2026-09-23")
    text = read(HISTORY).decode("utf-8")
    other_codes = {l.split(";")[0] for l in text.splitlines()
                   if l.split(";")[0].isdigit() and l.split(";")[0] != str(SCHEME)}
    assert other_codes, "fixture should contain other schemes"
    assert len(df) == 7


def test_official_two_chunks_same_data_is_deduplicated(monkeypatch):
    calls = serve_amfi(monkeypatch)
    df = amfi.fetch_nav_official(SCHEME, "2026-09-15", "2026-10-20")  # 36 days = 2 chunks
    history_calls = [c for c in calls if c[0] == amfi.HISTORY_URL]
    assert len(history_calls) == 2
    assert len(df) == 7


def test_official_changed_header_is_rejected(monkeypatch, caplog):
    broken = read(HISTORY).replace(b"Net Asset Value", b"NAV")
    serve_amfi(monkeypatch, broken)
    with caplog.at_level(logging.ERROR, logger="pyequity.connectors.amfi"):
        df = amfi.fetch_nav_official(SCHEME, "2026-09-15", "2026-09-23")
    assert df.empty
    assert any("layout" in r.message for r in caplog.records)


def test_official_scheme_missing_from_file_returns_empty_frame(monkeypatch):
    serve_amfi(monkeypatch)
    assert amfi.fetch_nav_official(999999999, "2026-09-15", "2026-09-23").empty


# ------------------------------ bulk.csv --------------------------------------

def independent_csv_rows():
    text = read(BULK).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], rows[1:]


def serve_csv(monkeypatch, content=None):
    def fake(url, headers=None, timeout=None):
        return Resp(content if content is not None else read(BULK))
    monkeypatch.setattr(sebi.requests, "get", fake)


def test_latest_matches_independent_parse_of_raw_csv(monkeypatch):
    serve_csv(monkeypatch)
    header, rows = independent_csv_rows()
    df = sebi.fetch_latest_bulk_deals()
    assert len(df) == len(rows), "no row may be lost or invented (identical rows are kept)"
    qty_i = header.index("Quantity Traded")
    assert int(df["quantity"].sum()) == sum(int(r[qty_i].replace(",", "")) for r in rows)
    assert df["symbol"].tolist() == [r[header.index("Symbol")] for r in rows]
    assert df["client_name"].tolist() == [r[header.index("Client Name")] for r in rows]


def test_latest_schema_label_and_deal_types(monkeypatch):
    serve_csv(monkeypatch)
    df = sebi.fetch_latest_bulk_deals()
    assert list(df.columns) == list(SCHEMAS["bulk_deals"])
    for c, t in SCHEMAS["bulk_deals"].items():
        assert str(df[c].dtype) == t, c
    assert set(df["source"]) == {"SEBI_NSE_CSV"}
    assert set(df["deal_type"]) <= {"BUY", "SELL"}
    assert "Remarks" not in df.columns


def test_latest_passes_validate_and_normalize_is_a_no_op(monkeypatch):
    serve_csv(monkeypatch)
    df = sebi.fetch_latest_bulk_deals()
    assert validate(df, "bulk_deals").ok
    pd.testing.assert_frame_equal(normalize(df, "bulk_deals"), df)


def test_latest_changed_header_is_rejected(monkeypatch, caplog):
    serve_csv(monkeypatch, read(BULK).replace(b"Quantity Traded", b"Qty"))
    with caplog.at_level(logging.ERROR, logger="pyequity.connectors.sebi"):
        assert sebi.fetch_latest_bulk_deals().empty
    assert any("layout" in r.message for r in caplog.records)


# ------------------- bulk deals API: per-day mode on real data ---------------

def api_records():
    path = glob.glob(str(FIX / "sebi_bulk_deals_2026-09-21.*"))
    assert len(path) == 1, "expected the existing API fixture from PR 2"
    payload = json.loads(Path(path[0]).read_text(encoding="utf-8"))
    return payload.get("data", []) if isinstance(payload, dict) else payload


class DaySession:
    def __init__(self, by_day):
        self.by_day = by_day

    def get(self, url, params=None, timeout=None):
        class R:
            status_code = 200

            def __init__(s, data):
                s._d = data

            def raise_for_status(s):
                pass

            def json(s):
                return {"data": s._d}
        return R(self.by_day.get(params["from"], []))


def test_by_day_on_real_api_fixture_flags_cap_and_matches_single_query(monkeypatch):
    records = api_records()
    session = DaySession({"21-09-2026": records})
    monkeypatch.setattr(sebi, "create_nse_session", lambda *a, **k: session)

    by_day = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-21")
    single = sebi.fetch_bulk_deals("2026-09-21", "2026-09-21")

    assert len(records) == 70, "the real single-day response is at NSE's cap"
    assert len(by_day) == len(records)
    assert by_day.attrs["capped_days"] == ["2026-09-21"]
    pd.testing.assert_frame_equal(by_day.reset_index(drop=True), single.reset_index(drop=True))
    single.attrs = {}
    plain = by_day.copy()
    plain.attrs = {}
    assert validate(plain, "bulk_deals").ok
    pd.testing.assert_frame_equal(normalize(plain, "bulk_deals"), plain)