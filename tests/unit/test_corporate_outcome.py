"""Offline tests for fetch_corporate_actions_outcome (no internet needed)."""
import pandas as pd
import pytest
import requests

from pyequity.connectors import corporate
from pyequity.connectors.corporate import (
    fetch_corporate_actions,
    fetch_corporate_actions_outcome,
)
from pyequity.reliability.outcome import FetchOutcome, Status
from pyequity.schema import columns_of, empty_frame


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
        rec="05-Jun-2026", isin="INE002A01018"):
    return {"symbol": symbol, "exDate": ex, "subject": subject, "recDate": rec,
            "isin": isin, "faceVal": "10", "series": "EQ", "comp": "X Ltd"}


def assert_empty_standard(df):
    assert df.empty and list(df.columns) == columns_of("corporate_actions")


# --- OK ---------------------------------------------------------------------

def test_valid_response_is_ok_with_processed_records(monkeypatch):
    install(monkeypatch, [row(), row(ex="01-Jan-2026", subject="Bonus 1:1")])
    out = fetch_corporate_actions_outcome("reliance")
    assert isinstance(out, FetchOutcome)
    assert out.status is Status.OK and out.ok and out.reason == ""
    df = out.data
    assert list(df.columns) == columns_of("corporate_actions")
    assert df["symbol"].tolist() == ["RELIANCE", "RELIANCE"]
    assert df["action_type"].tolist() == ["DIVIDEND", "BONUS"]
    assert df["ex_date"].tolist() == [pd.Timestamp("2026-06-05"), pd.Timestamp("2026-01-01")]
    assert (df["source"] == "NSE").all() and (df["isin"] == "INE002A01018").all()
    bonus = df.iloc[1]
    assert (bonus["bonus_new"], bonus["bonus_held"]) == (1, 1)


def test_wrapped_data_list_is_ok(monkeypatch):
    install(monkeypatch, {"data": [row()]})
    assert fetch_corporate_actions_outcome("RELIANCE").status is Status.OK


# --- EMPTY ------------------------------------------------------------------

def test_empty_list_is_empty_not_failed(monkeypatch):
    install(monkeypatch, [])
    out = fetch_corporate_actions_outcome("NOTASYMBOL123")
    assert out.status is Status.EMPTY and out.empty and not out.failed
    assert_empty_standard(out.data)


def test_empty_wrapped_list_is_empty(monkeypatch):
    install(monkeypatch, {"data": []})
    assert fetch_corporate_actions_outcome("RELIANCE").status is Status.EMPTY


def test_rows_for_other_symbol_only_is_empty(monkeypatch):
    install(monkeypatch, [row(symbol="OTHERCO")])
    out = fetch_corporate_actions_outcome("RELIANCE")
    assert out.status is Status.EMPTY
    assert_empty_standard(out.data)


def test_rows_outside_date_window_only_is_empty(monkeypatch):
    install(monkeypatch, [row(ex="05-Jun-2024")])
    out = fetch_corporate_actions_outcome("RELIANCE", "2026-01-01", "2026-12-31")
    assert out.status is Status.EMPTY
    assert_empty_standard(out.data)


def test_some_rows_filtered_still_ok(monkeypatch):
    install(monkeypatch, [row(), row(symbol="OTHERCO"), row(ex="05-Jun-2024")])
    out = fetch_corporate_actions_outcome("RELIANCE", "2026-01-01", "2026-12-31")
    assert out.status is Status.OK and len(out.data) == 1


# --- FAILED -----------------------------------------------------------------

def test_session_failure_is_failed(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("dropped")
    monkeypatch.setattr(corporate, "create_nse_session", boom)
    out = fetch_corporate_actions_outcome("RELIANCE")
    assert out.status is Status.FAILED and out.failed and out.data is None
    assert "dropped" in out.reason


def test_http_error_is_failed(monkeypatch):
    install(monkeypatch, [], status=500)
    out = fetch_corporate_actions_outcome("RELIANCE")
    assert out.failed and out.data is None and "500" in out.reason


def test_non_json_body_is_failed(monkeypatch):
    install(monkeypatch, None, bad_json=True)
    assert fetch_corporate_actions_outcome("RELIANCE").failed


@pytest.mark.parametrize("payload", [{"message": "blocked"}, "oops", 5, None, [1, 2]])
def test_invalid_structure_is_failed(monkeypatch, payload):
    install(monkeypatch, payload)
    out = fetch_corporate_actions_outcome("RELIANCE")
    assert out.failed and out.data is None and out.reason


def test_missing_ex_date_is_failed(monkeypatch):
    rec = row()
    del rec["exDate"]
    install(monkeypatch, [rec])
    out = fetch_corporate_actions_outcome("RELIANCE")
    assert out.failed and "exDate" in out.reason


def test_missing_subject_is_failed(monkeypatch):
    rec = row()
    del rec["subject"]
    install(monkeypatch, [rec])
    out = fetch_corporate_actions_outcome("RELIANCE")
    assert out.failed and "subject" in out.reason


def test_processing_error_is_failed_and_logged(monkeypatch, caplog):
    install(monkeypatch, [row()])

    def boom(_):
        raise RuntimeError("parser exploded")
    monkeypatch.setattr(corporate, "parse_action_details", boom)
    with caplog.at_level("ERROR", logger=corporate.__name__):
        out = fetch_corporate_actions_outcome("RELIANCE")
    assert out.failed and out.data is None and "parser exploded" in out.reason
    assert any("parser exploded" in r.message for r in caplog.records)


@pytest.mark.parametrize("args", [
    ("", None, None), (None, None, None),
    ("RELIANCE", "01/06/2026", "2026-06-10"),
    ("RELIANCE", "2026-06-10", "2026-06-01"),
])
def test_bad_arguments_are_failed_without_a_request(monkeypatch, args):
    session = install(monkeypatch, [row()])
    out = fetch_corporate_actions_outcome(*args)
    assert out.failed and out.reason and session.calls == []


# --- Backward compatibility -------------------------------------------------

def test_wrapper_returns_same_frame_as_outcome_data(monkeypatch):
    install(monkeypatch, [row(), row(ex="01-Jan-2026", subject="Bonus 1:1")])
    df = fetch_corporate_actions("RELIANCE")
    assert isinstance(df, pd.DataFrame)
    pd.testing.assert_frame_equal(df, fetch_corporate_actions_outcome("RELIANCE").data)


@pytest.mark.parametrize("payload,status,bad_json", [
    ([], 200, False),                       # EMPTY
    ([row(symbol="OTHERCO")], 200, False),  # EMPTY after filtering
    ({"message": "x"}, 200, False),         # FAILED: structure
    ([], 500, False),                       # FAILED: HTTP
    (None, 200, True),                      # FAILED: not JSON
])
def test_wrapper_gives_standard_empty_frame_for_empty_and_failed(
        monkeypatch, payload, status, bad_json):
    install(monkeypatch, payload, status, bad_json)
    df = fetch_corporate_actions("RELIANCE")
    assert isinstance(df, pd.DataFrame)
    assert_empty_standard(df)
    pd.testing.assert_frame_equal(df, empty_frame("corporate_actions"))


def test_wrapper_signature_unchanged():
    import inspect
    sig = inspect.signature(fetch_corporate_actions)
    assert list(sig.parameters) == ["symbol", "from_date", "to_date"]
    assert sig.parameters["from_date"].default is None
    assert sig.parameters["to_date"].default is None


# --- Extra checks mirroring the F&O outcome tests ----------------------------

def test_ok_outcome_defaults(monkeypatch):
    install(monkeypatch, [row()])
    out = fetch_corporate_actions_outcome("RELIANCE")
    assert out.reason == "" and out.attempts == 1
    assert not out.empty and not out.failed


def test_empty_outcome_has_reason_and_is_not_failed(monkeypatch):
    install(monkeypatch, [])
    out = fetch_corporate_actions_outcome("RELIANCE")
    assert out.reason and not out.ok and not out.failed and out.data is not None


def test_failed_reason_names_the_exception_type(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("dropped")
    monkeypatch.setattr(corporate, "create_nse_session", boom)
    out = fetch_corporate_actions_outcome("RELIANCE")
    assert "ConnectionError" in out.reason and "dropped" in out.reason


def test_outcome_makes_one_request_and_does_not_retry(monkeypatch):
    session = install(monkeypatch, [], status=500)
    assert fetch_corporate_actions_outcome("RELIANCE").failed
    assert len(session.calls) == 1


@pytest.mark.parametrize("payload,status", [([row()], 200), ([], 200), ([], 500)])
def test_wrapper_makes_the_same_requests_as_the_outcome(monkeypatch, payload, status):
    s1 = install(monkeypatch, payload, status)
    fetch_corporate_actions_outcome("RELIANCE", "2026-01-01", "2026-12-31")
    s2 = install(monkeypatch, payload, status)
    fetch_corporate_actions("RELIANCE", "2026-01-01", "2026-12-31")
    assert s1.calls == s2.calls and len(s2.calls) == 1