"""Offline tests: reliability toolkit wired into amfi.py and sebi.py.

No network: requests.get and create_nse_session are replaced, sleeps are
recorded instead of waited.
"""
import json
import logging
from pathlib import Path

import pandas as pd
import pytest
import requests

from pyequity.connectors import amfi, sebi
from pyequity.reliability.outcome import Status
from pyequity.schema import SCHEMAS, empty_frame

FIX = Path(__file__).resolve().parents[1] / "fixtures"
NAV_COLS = list(SCHEMAS["nav"])
BULK_COLS = list(SCHEMAS["bulk_deals"])


def response(status=200, payload=None, headers=None):
    r = requests.Response()
    r.status_code = status
    r.headers.update(headers or {})
    r._payload = payload
    r.json = lambda: payload  # type: ignore[method-assign]
    return r


@pytest.fixture
def sleeps(monkeypatch):
    """Record sleeps instead of sleeping (amfi and sebi share the time module)."""
    recorded = []
    monkeypatch.setattr(amfi.time, "sleep", lambda s: recorded.append(s))
    return recorded


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("unexpected real network call")
    monkeypatch.setattr(amfi.requests, "get", boom)
    monkeypatch.setattr(sebi, "create_nse_session", boom)


def mfapi(monkeypatch, steps):
    """Make requests.get play ``steps`` in order (exception -> raised)."""
    calls = []
    items = list(steps)

    def fake_get(url, params=None, headers=None, timeout=None):
        step = items[min(len(calls), len(items) - 1)]
        calls.append(url)
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(amfi.requests, "get", fake_get)
    return calls


def real_payload():
    return json.loads((FIX / "amfi_nav_119551.json").read_text(encoding="utf-8"))


# ------------------------- fetch_nav_outcome --------------------------------

def test_outcome_ok_equals_fetch_nav_on_real_fixture(monkeypatch, sleeps):
    payload = real_payload()
    mfapi(monkeypatch, [response(200, payload)])
    outcome = amfi.fetch_nav_outcome(119551)
    mfapi(monkeypatch, [response(200, payload)])
    legacy = amfi.fetch_nav(119551)
    assert outcome.status is Status.OK
    pd.testing.assert_frame_equal(outcome.data, legacy)
    assert list(outcome.data.columns) == NAV_COLS and len(outcome.data) == 30


def test_outcome_applies_date_filter_like_fetch_nav(monkeypatch, sleeps):
    payload = real_payload()
    mfapi(monkeypatch, [response(200, payload)])
    outcome = amfi.fetch_nav_outcome(119551, "2026-10-01", "2026-10-31")
    mfapi(monkeypatch, [response(200, payload)])
    pd.testing.assert_frame_equal(outcome.data, amfi.fetch_nav(119551, "2026-10-01", "2026-10-31"))


def test_outcome_empty_for_unknown_scheme_is_not_retried(monkeypatch, sleeps):
    invalid = json.loads((FIX / "amfi_invalid_scheme.json").read_text(encoding="utf-8"))
    calls = mfapi(monkeypatch, [response(200, invalid)])
    outcome = amfi.fetch_nav_outcome(999999999)
    assert outcome.status is Status.EMPTY and len(calls) == 1 and sleeps == []
    assert list(outcome.data.columns) == NAV_COLS


def test_outcome_empty_when_range_has_no_rows(monkeypatch, sleeps):
    mfapi(monkeypatch, [response(200, real_payload())])
    outcome = amfi.fetch_nav_outcome(119551, "2001-01-01", "2001-01-31")
    assert outcome.empty


def test_outcome_retries_server_error_then_succeeds(monkeypatch, sleeps):
    calls = mfapi(monkeypatch, [response(503), response(200, real_payload())])
    outcome = amfi.fetch_nav_outcome(119551)
    assert outcome.ok and outcome.attempts == 2 and len(calls) == 2 and len(sleeps) == 1


def test_outcome_connection_error_is_failed_not_empty(monkeypatch, sleeps):
    calls = mfapi(monkeypatch, [requests.ConnectionError("down")])
    outcome = amfi.fetch_nav_outcome(119551)
    assert outcome.failed and not outcome.empty and len(calls) == 3
    assert "ConnectionError" in outcome.reason


def test_outcome_max_attempts_is_respected(monkeypatch, sleeps):
    calls = mfapi(monkeypatch, [requests.Timeout("slow")])
    outcome = amfi.fetch_nav_outcome(119551, max_attempts=2)
    assert outcome.failed and len(calls) == 2


def test_outcome_client_error_is_failed_without_retry(monkeypatch, sleeps):
    calls = mfapi(monkeypatch, [response(404)])
    outcome = amfi.fetch_nav_outcome(119551)
    assert outcome.failed and len(calls) == 1 and sleeps == []


def test_outcome_unparseable_body_is_retried(monkeypatch, sleeps):
    bad = response(200)
    bad.json = lambda: (_ for _ in ()).throw(json.JSONDecodeError("Expecting value", "", 0))
    calls = mfapi(monkeypatch, [bad, response(200, real_payload())])
    assert amfi.fetch_nav_outcome(119551).ok and len(calls) == 2


@pytest.mark.parametrize("code", ["abc", None, [1]])
def test_outcome_bad_scheme_code_fails_without_request(monkeypatch, code):
    outcome = amfi.fetch_nav_outcome(code)  # no_network fixture would explode on any request
    assert outcome.failed and "invalid scheme code" in outcome.reason


@pytest.mark.parametrize("bad", ["2026/09/01", "yesterday"])
def test_outcome_bad_date_fails_without_request(bad):
    assert amfi.fetch_nav_outcome(119551, bad, None).failed
    assert amfi.fetch_nav_outcome(119551, None, bad).failed


def test_outcome_payload_that_is_not_a_dict_is_empty(monkeypatch, sleeps):
    mfapi(monkeypatch, [response(200, ["unexpected"])])
    assert amfi.fetch_nav_outcome(119551).empty


def test_fetch_nav_unchanged_on_server_error(monkeypatch, sleeps):
    """The legacy function keeps its behaviour: one try, empty frame, no raise."""
    calls = mfapi(monkeypatch, [response(503)])
    df = amfi.fetch_nav(119551)
    assert df.empty and list(df.columns) == NAV_COLS and len(calls) == 1


# -------------------- fetch_nav_official failed_chunks ----------------------

def test_official_reports_failed_chunks_in_attrs(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        r = response(500)
        r.content = b""
        return r

    monkeypatch.setattr(amfi.requests, "get", fake_get)
    df = amfi.fetch_nav_official(119551, "2026-09-15", "2026-09-23")
    assert df.empty and df.attrs["failed_chunks"] == ["2026-09-15 to 2026-09-23"]


# ------------------------- sebi by-day wiring -------------------------------

def rec(day, n, client="C"):
    return [{"BD_DT_DATE": day, "BD_SYMBOL": "ABC", "BD_SCRIP_NAME": "Abc Ltd",
             "BD_CLIENT_NAME": f"{client}{i}", "BD_BUY_SELL": "BUY" if i % 2 else "SELL",
             "BD_QTY_TRD": 100 + i, "BD_TP_WATP": 10.5} for i in range(n)]


class Session:
    """Fake NSE session. ``plan`` maps dd-mm-YYYY to a list of steps per call."""

    def __init__(self, plan):
        self.plan = {k: list(v) for k, v in plan.items()}
        self.calls = []
        self.closed = False

    def get(self, url, params=None, timeout=None):
        day = params["from"]
        self.calls.append(day)
        steps = self.plan.get(day, [[]])
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, BaseException):
            raise step
        return response(200, {"data": step})

    def close(self):
        self.closed = True


def use_sessions(monkeypatch, *sessions):
    made = []
    queue = list(sessions)

    def factory(*a, **k):
        s = queue.pop(0) if len(queue) > 1 else queue[0]
        made.append(s)
        return s

    monkeypatch.setattr(sebi, "create_nse_session", factory)
    return made


def test_by_day_retries_transient_error_and_keeps_the_day(monkeypatch, sleeps):
    s = Session({"21-09-2026": [requests.ConnectionError("blip"), rec("21-Sep-2026", 3)]})
    use_sessions(monkeypatch, s)
    df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-21")
    assert len(df) == 3 and df.attrs["failed_days"] == []
    assert s.calls == ["21-09-2026", "21-09-2026"]


def test_by_day_gives_up_after_retries_and_marks_day_failed(monkeypatch, sleeps, caplog):
    s = Session({"21-09-2026": [requests.ConnectionError("down")]})
    use_sessions(monkeypatch, s)
    with caplog.at_level(logging.WARNING):
        df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-21")
    assert df.empty and df.attrs["failed_days"] == ["2026-09-21"]
    assert len(s.calls) == 3
    assert any("Could not fetch 1 day" in r.message for r in caplog.records)


def test_by_day_empty_day_is_not_retried_and_not_failed(monkeypatch, sleeps):
    s = Session({"21-09-2026": [[]]})
    use_sessions(monkeypatch, s)
    df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-21")
    assert df.empty and df.attrs["failed_days"] == [] and len(s.calls) == 1


def test_by_day_replaces_session_after_a_failed_day(monkeypatch, sleeps):
    first = Session({"21-09-2026": [requests.ConnectionError("blocked")],
                     "22-09-2026": [rec("22-Sep-2026", 2)]})
    second = Session({"22-09-2026": [rec("22-Sep-2026", 2)]})
    made = use_sessions(monkeypatch, first, second)
    df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-22")
    assert df.attrs["failed_days"] == ["2026-09-21"] and len(df) == 2
    assert made == [first, second] and first.closed
    assert second.calls == ["22-09-2026"]


def test_by_day_reuses_one_session_when_all_days_succeed(monkeypatch, sleeps):
    s = Session({"21-09-2026": [rec("21-Sep-2026", 1)], "22-09-2026": [rec("22-Sep-2026", 1)]})
    made = use_sessions(monkeypatch, s)
    df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-22")
    assert len(made) == 1 and len(df) == 2


def test_by_day_spaces_requests_with_rate_limiter(monkeypatch, sleeps):
    s = Session({d: [rec(d.replace("09-2026", "Sep-2026"), 1)]
                 for d in ("21-09-2026", "22-09-2026", "23-09-2026")})
    use_sessions(monkeypatch, s)
    sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-23")
    # sleeps are faked, so the real clock barely moves: the limiter reserves
    # slots one second apart, i.e. the 2nd request waits ~1s and the 3rd ~2s
    waits = [w for w in sleeps if w > 0.5]
    assert len(waits) == 2
    assert waits[0] == pytest.approx(1.0, abs=0.2) and waits[1] == pytest.approx(2.0, abs=0.2)


def test_by_day_session_failure_stops_early_and_reports_remaining_days(monkeypatch, sleeps):
    def broken(*a, **k):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(sebi, "create_nse_session", broken)
    df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-25")
    assert df.empty and list(df.columns) == BULK_COLS
    assert df.attrs["failed_days"] == ["2026-09-21", "2026-09-22", "2026-09-23",
                                      "2026-09-24", "2026-09-25"]
    # session creation was retried a bounded number of times, not once per day
    assert len(sleeps) <= 4


def test_by_day_malformed_records_mark_day_failed_not_raise(monkeypatch, sleeps):
    s = Session({"21-09-2026": [[{"unexpected": "shape"}]], "22-09-2026": [rec("22-Sep-2026", 2)]})
    use_sessions(monkeypatch, s)
    df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-22")
    assert df.attrs["failed_days"] == ["2026-09-21"] and len(df) == 2


def test_by_day_cap_flagging_still_works(monkeypatch, sleeps):
    s = Session({"21-09-2026": [rec("21-Sep-2026", 70)]})
    use_sessions(monkeypatch, s)
    df = sebi.fetch_bulk_deals_by_day("2026-09-21", "2026-09-21")
    assert df.attrs["capped_days"] == ["2026-09-21"] and len(df) == 70