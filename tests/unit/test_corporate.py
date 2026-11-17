"""Offline tests for pyequity.connectors.corporate (no internet needed)."""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import requests

from pyequity.connectors import corporate
from pyequity.connectors.corporate import extract_action_type, fetch_corporate_actions
from pyequity.processing.normalize import normalize, validate
from pyequity.schema import SCHEMAS, columns_of, empty_frame

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "corporate_actions_reliance.json"
LOGGER = "pyequity.connectors.corporate"


class FakeResponse:
    def __init__(self, payload, status=200, bad_json=False):
        self._payload, self.status_code, self._bad_json = payload, status, bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.response


def install(monkeypatch, payload, status=200, bad_json=False):
    session = FakeSession(FakeResponse(payload, status, bad_json))
    monkeypatch.setattr(corporate, "create_nse_session", lambda *a, **k: session)
    return session


def row(symbol="RELIANCE", ex="05-Jun-2026", subject="Dividend - Rs 6 Per Share",
        rec="05-Jun-2026", isin="INE002A01018", **extra):
    base = {"symbol": symbol, "exDate": ex, "subject": subject, "recDate": rec,
            "isin": isin, "faceVal": "10", "series": "EQ", "comp": "X Ltd"}
    base.update(extra)
    return base


def fixture_rows():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def is_empty_standard(df):
    return df.empty and list(df.columns) == columns_of("corporate_actions")


# ---------------------------------------------------------------------------
# Real response (20 RELIANCE rows)
# ---------------------------------------------------------------------------

def test_real_fixture_gives_exact_schema_dtypes_and_validates(monkeypatch):
    install(monkeypatch, fixture_rows())
    df = fetch_corporate_actions("RELIANCE")
    assert len(df) == 20
    assert list(df.columns) == list(SCHEMAS["corporate_actions"])
    for col, dtype in SCHEMAS["corporate_actions"].items():
        assert str(df[col].dtype) == dtype, col
    assert validate(df, "corporate_actions").ok


def test_real_fixture_values_and_parsing(monkeypatch):
    raw = fixture_rows()
    install(monkeypatch, raw)
    df = fetch_corporate_actions("RELIANCE")
    # raw text kept exactly, including NSE's leading spaces; API order kept
    assert df["action_detail"].tolist() == [r["subject"] for r in raw]
    assert (df["symbol"] == "RELIANCE").all() and (df["source"] == "NSE").all()
    assert (df["isin"] == "INE002A01018").all()
    # legacy label is unchanged for every row
    assert df["action_type"].tolist() == [extract_action_type(r["subject"]) for r in raw]

    rights = df[df["action_detail"].str.strip() == "Rights 1:15 @ Premium Rs 1247"].iloc[0]
    assert (rights["rights_new"], rights["rights_held"], rights["rights_premium"]) == (1, 15, 1247)
    assert rights["action_categories"] == "RIGHTS" and rights["parse_status"] == "ok"
    assert rights["ex_date"] == pd.Timestamp("2020-05-13")

    bonus = df[df["action_detail"].str.strip() == "Bonus 1:1"]
    assert len(bonus) == 2 and (bonus["bonus_new"] == 1).all() and (bonus["bonus_held"] == 1).all()

    demerger = df[df["action_detail"] == "Demerger"].iloc[0]
    assert demerger["action_categories"] == "DEMERGER" and demerger["parse_status"] == "needs_review"
    assert demerger["action_type"] == "MERGER"            # legacy label, unchanged on purpose
    assert pd.isna(demerger["bonus_new"]) and pd.isna(demerger["rights_new"])

    # "-" in NSE's recDate becomes NaT, never an error
    assert df["record_date"].isna().sum() == 5


def test_real_fixture_survives_normalize_unchanged_apart_from_order(monkeypatch):
    install(monkeypatch, fixture_rows())
    df = fetch_corporate_actions("RELIANCE")
    out = normalize(df, "corporate_actions")
    assert len(out) == len(df)
    assert out["ex_date"].dropna().is_monotonic_increasing
    key = list(SCHEMAS["corporate_actions"])
    pd.testing.assert_frame_equal(out.sort_values(key, kind="mergesort").reset_index(drop=True),
                                  df.sort_values(key, kind="mergesort").reset_index(drop=True))


# ---------------------------------------------------------------------------
# Request parameters
# ---------------------------------------------------------------------------

def test_default_request_asks_for_full_history_not_just_latest_rows(monkeypatch):
    session = install(monkeypatch, [row()])
    fetch_corporate_actions("reliance")
    url, params = session.calls[0]
    assert url == "https://www.nseindia.com/api/corporates-corporateActions"
    assert params["symbol"] == "RELIANCE" and params["index"] == "equities"
    assert params["from_date"] == "01-01-1990"
    ahead = datetime.strptime(params["to_date"], "%d-%m-%Y") - datetime.now()
    assert timedelta(days=363) < ahead < timedelta(days=366)


def test_explicit_dates_are_converted_to_nse_format(monkeypatch):
    session = install(monkeypatch, [row()])
    fetch_corporate_actions("RELIANCE", "2026-06-01", "2026-06-10")
    params = session.calls[0][1]
    assert (params["from_date"], params["to_date"]) == ("01-06-2026", "10-06-2026")


@pytest.mark.parametrize("args", [
    ("RELIANCE", "01/06/2026", "2026-06-10"),
    ("RELIANCE", "2026-06-01", "10-06-2026"),
    ("RELIANCE", "2026-06-10", "2026-06-01"),       # from after to
    ("", "2026-06-01", "2026-06-10"),
    ("   ", None, None),
    (None, None, None),
])
def test_bad_arguments_return_empty_without_a_request(monkeypatch, args):
    session = install(monkeypatch, [row()])
    assert is_empty_standard(fetch_corporate_actions(*args))
    assert session.calls == []


# ---------------------------------------------------------------------------
# Failures never raise
# ---------------------------------------------------------------------------

def test_session_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("dropped")
    monkeypatch.setattr(corporate, "create_nse_session", boom)
    assert is_empty_standard(fetch_corporate_actions("RELIANCE"))


def test_http_error_returns_empty(monkeypatch):
    install(monkeypatch, [], status=500)
    assert is_empty_standard(fetch_corporate_actions("RELIANCE"))


def test_non_json_body_returns_empty(monkeypatch):
    install(monkeypatch, None, bad_json=True)
    assert is_empty_standard(fetch_corporate_actions("RELIANCE"))


@pytest.mark.parametrize("payload", [{"message": "blocked"}, "oops", 5, None, [1, 2], [{"symbol": "X"}]])
def test_unexpected_shapes_return_empty(monkeypatch, payload):
    install(monkeypatch, payload)
    assert is_empty_standard(fetch_corporate_actions("RELIANCE"))


def test_empty_answer_is_logged_and_says_it_cannot_tell_why(monkeypatch, caplog):
    install(monkeypatch, [])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        df = fetch_corporate_actions("NOTASYMBOL123")
    assert is_empty_standard(df)
    assert any("unknown or renamed" in r.message for r in caplog.records)


def test_wrapped_data_list_is_accepted(monkeypatch):
    install(monkeypatch, {"data": [row()]})
    assert len(fetch_corporate_actions("RELIANCE")) == 1


# ---------------------------------------------------------------------------
# Safety nets
# ---------------------------------------------------------------------------

def test_rows_for_another_symbol_are_dropped_with_a_warning(monkeypatch, caplog):
    install(monkeypatch, [row(), row(symbol="OTHERCO", subject="Bonus 1:1")])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        df = fetch_corporate_actions("RELIANCE")
    assert df["symbol"].tolist() == ["RELIANCE"]
    assert any("different symbol" in r.message for r in caplog.records)


def test_symbol_comes_from_the_api_not_from_the_request(monkeypatch):
    install(monkeypatch, [row(symbol="reliance ")])          # stray case/space from the API
    assert fetch_corporate_actions("RELIANCE")["symbol"].tolist() == ["RELIANCE"]


def test_rows_outside_the_requested_window_are_dropped(monkeypatch, caplog):
    install(monkeypatch, [row(ex="05-Jun-2026"), row(ex="05-Jun-2024")])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        df = fetch_corporate_actions("RELIANCE", "2026-01-01", "2026-12-31")
    assert df["ex_date"].tolist() == [pd.Timestamp("2026-06-05")]
    assert any("outside" in r.message for r in caplog.records)


def test_missing_optional_fields_become_missing_values(monkeypatch):
    rec = row()
    del rec["isin"], rec["recDate"]
    install(monkeypatch, [rec])
    df = fetch_corporate_actions("RELIANCE")
    assert df["isin"].isna().all() and df["record_date"].isna().all()
    assert list(df.columns) == columns_of("corporate_actions")


def test_missing_required_fields_return_empty(monkeypatch):
    rec = row()
    del rec["subject"]
    install(monkeypatch, [rec])
    assert is_empty_standard(fetch_corporate_actions("RELIANCE"))


def test_unparseable_ex_date_is_kept_as_nat_and_logged(monkeypatch, caplog):
    install(monkeypatch, [row(ex="31/12/2025"), row(ex="-")])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        df = fetch_corporate_actions("RELIANCE")
    assert len(df) == 2 and df["ex_date"].isna().all()
    assert sum("could not be parsed" in r.message for r in caplog.records) == 1   # "-" is not reported


# ---------------------------------------------------------------------------
# Same-day actions are separate rows (BAJFINANCE, 2025-06-16)
# ---------------------------------------------------------------------------

def test_bonus_and_split_on_the_same_day_stay_two_rows_through_normalize(monkeypatch):
    install(monkeypatch, [
        row("BAJFINANCE", "16-Jun-2025", "Bonus 4:1", "16-Jun-2025"),
        row("BAJFINANCE", "16-Jun-2025",
            "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share", "16-Jun-2025"),
    ])
    df = fetch_corporate_actions("BAJFINANCE")
    out = normalize(df, "corporate_actions")
    assert len(df) == len(out) == 2
    bonus = out[out["action_categories"] == "BONUS"].iloc[0]
    split = out[out["action_categories"] == "SPLIT"].iloc[0]
    assert (bonus["bonus_new"], bonus["bonus_held"]) == (4, 1)
    assert (split["split_old_fv"], split["split_new_fv"]) == (2, 1)


# ---------------------------------------------------------------------------
# The legacy label is deliberately unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,legacy", [
    ("Dividend - Rs 6 Per Share", "DIVIDEND"),
    ("Bonus 1:1", "BONUS"),
    ("Face Value Split From Rs.10/- To Rs.2/-", "SPLIT"),
    ("Rights 1:15 @ Premium Rs 1247", "RIGHTS"),
    ("Buyback", "BUYBACK"),
    ("Annual General Meeting", "OTHER"),
    ("Demerger", "MERGER"),                              # known legacy flaw
    ("Bonus 1:1/Face Value Split From Rs 10 To Rs 2", "BONUS"),   # known legacy flaw
    (None, ""),
    (float("nan"), ""),
])
def test_legacy_action_type_is_unchanged(text, legacy):
    assert extract_action_type(text) == legacy
