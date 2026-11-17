"""Offline, deterministic tests for pyequity.connectors.indices.

A fake NSE server replaces the network. It models what was observed live:
newest-first records, an HTTP 422 for bad requests, case-insensitive names,
long names accepted while some short names return nothing, and a row cap that
keeps the OLDEST rows (the cap cuts the newest data).
"""
import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import pytest
import requests

from pyequity.connectors import indices
from pyequity.connectors.indices import (
    _fetch_indices_chunk,
    fetch_index_history,
    list_indices,
)
from pyequity.processing.normalize import validate
from pyequity.schema import SCHEMAS, empty_frame

LOGGER = "pyequity.connectors.indices"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
HISTORY = "historicalOR/indicesHistory"
NAMES = "index-names"

# Shaped like the real /api/index-names response (stn = [short, long] pairs).
NAME_PAIRS = [
    ["NIFTY 50", "NIFTY 50"],
    ["NIFTY BANK", "NIFTY BANK"],
    ["NIFTY FIN SERVICE", "NIFTY FINANCIAL SERVICES"],
    ["NIFTY CONSUMPTION", "NIFTY INDIA CONSUMPTION"],
    ["NIFTY50 VALUE 20", "NIFTY50 VALUE 20"],
    ["Nifty 50 Futures Index", "NIFTY 50 FUTURES INDEX"],
    ["Securities in F&O", "Securities in F&O"],
    ["Permitted to Trade", "Permitted to Trade"],
    ["BHARATBOND-APR25", "NIFTY BHARAT BOND INDEX - APRIL 2025"],
]


class FakeResponse:
    def __init__(self, body, status: int = 200):
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.text = self.content.decode("utf-8", "replace")
        self.status_code = status

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")


def weekdays(start: date, end: date) -> List[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def dmy(text: str) -> date:
    return datetime.strptime(text, "%d-%m-%Y").date()


class FakeNSE:
    """Fake NSE session. ``accepted`` maps a lower-case name to (EOD name, first day)."""

    def __init__(self, accepted: Optional[Dict[str, tuple]] = None, cap: int = 70):
        self.accepted = accepted or {
            "nifty 50": ("NIFTY 50", date(2020, 1, 1)),
            "nifty bank": ("NIFTY BANK", date(2020, 1, 1)),
            "nifty financial services": ("NIFTY FIN SERVICE", date(2020, 1, 1)),
            "nifty50 value 20": ("NIFTY50 VALUE 20", date(2026, 1, 1)),
        }
        self.cap = cap
        self.names_body = {"stn": NAME_PAIRS, "nts": [[b, a] for a, b in NAME_PAIRS]}
        self.names_fails = False
        self.history_override: Optional[Callable[[dict], FakeResponse]] = None
        self.fail_when: Optional[Callable[[date, date], Optional[str]]] = None
        self.copies_per_day = 1
        self.calls: List[tuple] = []

    # -- helpers for assertions -------------------------------------------
    def history_calls(self) -> List[dict]:
        return [p for u, p in self.calls if HISTORY in u]

    def names_calls(self) -> int:
        return sum(1 for u, _ in self.calls if NAMES in u)

    @staticmethod
    def record(eod_name: str, day: date) -> dict:
        base = 1000 + day.toordinal() % 500
        return {
            "EOD_INDEX_NAME": eod_name,
            "EOD_OPEN_INDEX_VAL": float(base),
            "EOD_HIGH_INDEX_VAL": float(base + 5),
            "EOD_CLOSE_INDEX_VAL": float(base + 1),
            "EOD_LOW_INDEX_VAL": float(base - 5),
            "HIT_TURN_OVER": 123.45,
            "HIT_TRADED_QTY": 1000 + day.toordinal() % 100,
            "EOD_TIMESTAMP": day.strftime("%d-%b-%Y").upper(),
            "HI_TIMESTAMP": "2026-01-01T00:00:00.000Z",
        }

    # -- the session API -----------------------------------------------------
    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if NAMES in url:
            if self.names_fails:
                raise requests.ConnectionError("names down")
            return FakeResponse(self.names_body)
        assert HISTORY in url, url
        if self.history_override is not None:
            return self.history_override(params)
        lo, hi = dmy(params["from"]), dmy(params["to"])
        if self.fail_when:
            kind = self.fail_when(lo, hi)
            if kind == "422":
                return FakeResponse({"error": True, "message": "invalid params"}, status=422)
            if kind == "conn":
                raise requests.ConnectionError("dropped")
        entry = self.accepted.get(params["indexType"].lower())
        if entry is None:
            return FakeResponse({"data": []})
        eod_name, first = entry
        days = [d for d in weekdays(max(lo, first), hi)]
        rows = []
        for d in days:
            rows.extend([self.record(eod_name, d)] * self.copies_per_day)
        rows = rows[: self.cap]          # the cap keeps the OLDEST rows ...
        return FakeResponse({"data": rows[::-1]})   # ... and the reply is newest first


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("a test tried to use the real network")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    indices._reset_index_name_cache()
    yield
    indices._reset_index_name_cache()


class Env:
    def __init__(self, monkeypatch, server: FakeNSE):
        self.server = server
        self.sessions = 0

        def factory(*args, **kwargs):
            self.sessions += 1
            return server

        monkeypatch.setattr("pyequity.connectors.indices.create_nse_session", factory)


@pytest.fixture
def env(monkeypatch):
    return Env(monkeypatch, FakeNSE())


def error_messages(caplog) -> List[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def warning_messages(caplog) -> List[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def assert_standard_empty(df: pd.DataFrame) -> None:
    assert df.empty
    assert list(df.columns) == list(SCHEMAS["index"])
    assert {c: str(t) for c, t in df.dtypes.items()} == SCHEMAS["index"]


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_matches_shared_schema_exactly(env):
    df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-13")
    assert len(df) == 10
    assert list(df.columns) == list(SCHEMAS["index"])
    assert {c: str(t) for c, t in df.dtypes.items()} == SCHEMAS["index"]
    assert (df["source"] == "NSE").all()
    assert (df["index_name"] == "NIFTY 50").all()


def test_output_passes_validate_and_is_sorted_oldest_first(env):
    df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-13")
    assert validate(df, "index").errors == []
    assert df["date"].is_monotonic_increasing
    assert not df["date"].duplicated().any()


def test_values_match_the_raw_records(env):
    df = fetch_index_history("NIFTY BANK", "2026-03-02", "2026-03-06")
    raw = [FakeNSE.record("NIFTY BANK", d) for d in weekdays(date(2026, 3, 2), date(2026, 3, 6))]
    assert df["close"].tolist() == [r["EOD_CLOSE_INDEX_VAL"] for r in raw]
    assert df["open"].tolist() == [r["EOD_OPEN_INDEX_VAL"] for r in raw]
    assert df["volume"].tolist() == [r["HIT_TRADED_QTY"] for r in raw]


def test_turnover_is_not_added_to_the_output(env):
    df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert "HIT_TURN_OVER" not in df.columns and "turnover" not in df.columns


def test_real_fixture_chunk_parses_to_seven_rows():
    body = (FIXTURES / "indices_nifty50.json").read_bytes()

    class Session:
        def get(self, *a, **k):
            return FakeResponse(body)

    df = _fetch_indices_chunk(Session(), "NIFTY 50", "15-09-2026", "23-09-2026")
    assert len(df) == 7
    assert {c: str(t) for c, t in df.dtypes.items()} == SCHEMAS["index"]
    assert (df["index_name"] == "NIFTY 50").all()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_name", [None, "", "   ", "\t\n", 123, ["NIFTY 50"]])
def test_bad_index_names_return_empty_without_touching_nse(env, caplog, bad_name):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history(bad_name, "2026-03-02", "2026-03-06")
    assert_standard_empty(df)
    assert env.sessions == 0 and env.server.calls == []
    assert error_messages(caplog)


@pytest.mark.parametrize("start,end", [
    ("2026/03/02", "2026-03-06"), ("2026-03-02", "06-03-2026"), ("2026-02-30", "2026-03-06"),
    ("", "2026-03-06"), (None, "2026-03-06"), ("2026-03-02", None), (20260302, 20260306),
])
def test_bad_dates_return_empty_without_touching_nse(env, caplog, start, end):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", start, end)
    assert_standard_empty(df)
    assert env.sessions == 0
    assert error_messages(caplog)


def test_start_after_end_returns_empty(env, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-06", "2026-03-02")
    assert_standard_empty(df)
    assert env.sessions == 0 and error_messages(caplog)


def test_single_trading_day(env):
    df = fetch_index_history("NIFTY 50", "2026-03-03", "2026-03-03")
    assert len(df) == 1 and df["date"].iloc[0] == pd.Timestamp("2026-03-03")


def test_weekend_only_range_is_empty_without_error_log(env, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-07", "2026-03-08")
    assert_standard_empty(df)
    assert not error_messages(caplog)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_short_range_uses_one_request(env):
    fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-13")
    assert len(env.server.history_calls()) == 1


def test_long_range_is_complete_contiguous_and_not_overlapping(env):
    df = fetch_index_history("NIFTY 50", "2025-01-01", "2026-06-30")
    expected = weekdays(date(2025, 1, 1), date(2026, 6, 30))
    assert df["date"].dt.date.tolist() == expected
    windows = [(dmy(p["from"]), dmy(p["to"])) for p in env.server.history_calls()]
    assert windows[0][0] == date(2025, 1, 1) and windows[-1][1] == date(2026, 6, 30)
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == prev_end + timedelta(days=1)
    assert all((b - a).days <= 60 for a, b in windows)


@pytest.mark.parametrize("start,end", [
    ("2026-01-01", "2026-02-28"),   # two months
    ("2026-01-01", "2026-03-31"),   # three months
    ("2025-09-23", "2026-09-23"),   # one year, several chunks
])
def test_ranges_return_every_trading_day(env, start, end):
    df = fetch_index_history("NIFTY 50", start, end)
    assert df["date"].dt.date.tolist() == weekdays(date.fromisoformat(start), date.fromisoformat(end))


def test_one_session_is_reused_for_all_chunks(env):
    fetch_index_history("NIFTY 50", "2025-01-01", "2026-06-30")
    assert env.sessions == 1
    assert len(env.server.history_calls()) > 3


def test_requests_are_paced_between_chunks(env, monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    fetch_index_history("NIFTY 50", "2025-01-01", "2026-06-30")
    assert len(sleeps) == len(env.server.history_calls()) - 1


# ---------------------------------------------------------------------------
# Failure policy: never partial history
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["422", "conn"])
def test_failed_middle_chunk_returns_empty_and_names_the_range(env, caplog, kind):
    bad_from = date(2025, 1, 1) + timedelta(days=61 * 2)
    env.server.fail_when = lambda lo, hi: kind if lo == bad_from else None
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2025-01-01", "2026-06-30")
    assert_standard_empty(df)
    text = " ".join(error_messages(caplog))
    assert "incomplete" in text.lower() and str(bad_from) in text
    assert len(env.server.history_calls()) == 3      # stops at the first failure


def test_failed_first_chunk_returns_empty(env, caplog):
    env.server.fail_when = lambda lo, hi: "422"
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-13")
    assert_standard_empty(df)
    assert error_messages(caplog)


def test_422_does_not_trigger_alias_lookup(env):
    env.server.fail_when = lambda lo, hi: "422"
    fetch_index_history("NIFTY FIN SERVICE", "2026-03-02", "2026-03-13")
    assert env.server.names_calls() == 0


def _override(env, body, status=200):
    env.server.history_override = lambda params: FakeResponse(body, status)


def _good(day=date(2026, 3, 2), **changes):
    rec = FakeNSE.record("NIFTY 50", day)
    rec.update(changes)
    return rec


def _without(field, day=date(2026, 3, 2)):
    rec = _good(day)
    del rec[field]
    return rec


BAD_BODIES = {
    "http_422": ({"error": True, "message": "invalid params"}, 422),
    "not_json": (b"<html>Access denied</html>", 200),
    "error_payload": ({"error": True, "message": "invalid params"}, 200),
    "data_not_list": ({"data": "oops"}, 200),
    "no_data_key": ({"foo": []}, 200),
    "top_level_list": ([_good()], 200),
    "records_not_objects": ({"data": [1, 2]}, 200),
    "missing_close": ({"data": [_without("EOD_CLOSE_INDEX_VAL")]}, 200),
    "missing_timestamp": ({"data": [_without("EOD_TIMESTAMP")]}, 200),
    "bad_date": ({"data": [_good(EOD_TIMESTAMP="99-XXX-2026")]}, 200),
    "dash_close": ({"data": [_good(EOD_CLOSE_INDEX_VAL="-")]}, 200),
    "null_open": ({"data": [_good(EOD_OPEN_INDEX_VAL=None)]}, 200),
}


@pytest.mark.parametrize("case", list(BAD_BODIES))
def test_bad_responses_return_empty_and_log_an_error(env, caplog, case):
    body, status = BAD_BODIES[case]
    _override(env, body, status)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert_standard_empty(df)
    assert error_messages(caplog)


def test_empty_data_is_not_an_error(env, caplog):
    _override(env, {"data": []})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert_standard_empty(df)
    assert not error_messages(caplog)


def test_session_creation_failure_does_not_raise(monkeypatch, caplog):
    def boom(*a, **k):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr("pyequity.connectors.indices.create_nse_session", boom)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert_standard_empty(df)
    assert error_messages(caplog)


# ---------------------------------------------------------------------------
# Field handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("volume", [None, "-", "abc"])
def test_missing_volume_becomes_zero_and_is_logged(env, caplog, volume):
    days = weekdays(date(2026, 3, 2), date(2026, 3, 4))
    _override(env, {"data": [_good(d, HIT_TRADED_QTY=volume) for d in days]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-04")
    assert len(df) == 3 and (df["volume"] == 0).all()
    assert str(df["volume"].dtype) == "int64"
    assert any("3 volume" in m for m in warning_messages(caplog))


def test_absent_volume_field_gives_zero_and_a_warning(env, caplog):
    _override(env, {"data": [_without("HIT_TRADED_QTY")]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-02")
    assert len(df) == 1 and df["volume"].iloc[0] == 0
    assert any("volume" in m for m in warning_messages(caplog))


@pytest.mark.parametrize("name_value", [None, "", "   "])
def test_blank_eod_index_name_falls_back_to_requested_name_with_warning(env, caplog, name_value):
    _override(env, {"data": [_good(EOD_INDEX_NAME=name_value)]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-02")
    assert df["index_name"].tolist() == ["NIFTY 50"]
    assert any("1 record(s) had no EOD_INDEX_NAME" in m and "'NIFTY 50'" in m
               for m in warning_messages(caplog))


def test_absent_eod_index_name_falls_back_to_requested_name_with_warning(env, caplog):
    _override(env, {"data": [_without("EOD_INDEX_NAME")]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-02")
    assert df["index_name"].tolist() == ["NIFTY 50"]
    assert any("no EOD_INDEX_NAME" in m for m in warning_messages(caplog))


def test_name_fallback_warning_counts_all_chunks_once(env, caplog):
    days = weekdays(date(2026, 3, 2), date(2026, 3, 6))
    _override(env, {"data": [_good(d, EOD_INDEX_NAME="") for d in days]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    hits = [m for m in warning_messages(caplog) if "no EOD_INDEX_NAME" in m]
    assert len(hits) == 1 and hits[0].startswith("[Indices] 5 record(s)")


def test_present_eod_index_name_gives_no_fallback_warning(env, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert not any("EOD_INDEX_NAME" in m for m in warning_messages(caplog))


def test_rows_outside_the_window_are_dropped_with_a_warning(env, caplog):
    inside = _good(date(2026, 3, 3))
    outside = _good(date(2026, 4, 1))
    _override(env, {"data": [outside, inside]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert df["date"].tolist() == [pd.Timestamp("2026-03-03")]
    assert any("outside" in m for m in warning_messages(caplog))


def test_only_out_of_window_rows_gives_empty(env):
    _override(env, {"data": [_good(date(2026, 4, 1))]})
    assert_standard_empty(fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06"))


def test_identical_duplicate_records_collapse(env):
    rec = _good(date(2026, 3, 3))
    _override(env, {"data": [rec, dict(rec)]})
    df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert len(df) == 1


def test_conflicting_same_date_records_are_an_error(env, caplog):
    a = _good(date(2026, 3, 3))
    b = _good(date(2026, 3, 3), EOD_CLOSE_INDEX_VAL=9999.0)
    _override(env, {"data": [a, b]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert_standard_empty(df)
    assert any("Conflicting" in m for m in error_messages(caplog))


# ---------------------------------------------------------------------------
# Index names and aliases
# ---------------------------------------------------------------------------

def test_name_whitespace_is_cleaned_and_case_is_left_alone(env):
    df = fetch_index_history("  nifty \t  50 ", "2026-03-02", "2026-03-06")
    assert env.server.history_calls()[0]["indexType"] == "nifty 50"
    assert df["index_name"].unique().tolist() == ["NIFTY 50"]     # NSE's label


def test_normal_path_never_requests_the_name_list(env):
    fetch_index_history("NIFTY 50", "2025-01-01", "2026-06-30")
    assert env.server.names_calls() == 0


def test_long_name_returns_nses_short_label(env):
    df = fetch_index_history("NIFTY FINANCIAL SERVICES", "2026-03-02", "2026-03-06")
    assert df["index_name"].unique().tolist() == ["NIFTY FIN SERVICE"]
    assert env.server.names_calls() == 0


@pytest.mark.parametrize("alias", ["NIFTY FIN SERVICE", "nifty fin service", " Nifty  Fin Service "])
def test_short_alias_is_retried_as_long_name_for_the_whole_call(env, alias):
    df = fetch_index_history(alias, "2025-01-01", "2026-06-30")
    assert df["date"].dt.date.tolist() == weekdays(date(2025, 1, 1), date(2026, 6, 30))
    assert df["index_name"].unique().tolist() == ["NIFTY FIN SERVICE"]
    sent = [p["indexType"] for p in env.server.history_calls()]
    assert sent[0].lower() == "nifty fin service"
    assert set(s.lower() for s in sent[1:]) == {"nifty financial services"}
    assert env.server.names_calls() == 1


def test_name_list_is_fetched_once_per_process(env):
    fetch_index_history("NIFTY FIN SERVICE", "2026-03-02", "2026-03-06")
    fetch_index_history("NIFTY CONSUMPTION", "2026-03-02", "2026-03-06")
    assert env.server.names_calls() == 1


def test_same_name_pair_is_not_an_alias_and_is_not_retried(env):
    # NIFTY50 VALUE 20 has no data before 2026-01-01, so the first chunk is empty.
    df = fetch_index_history("NIFTY50 VALUE 20", "2025-01-02", "2025-01-10")
    assert_standard_empty(df)
    assert len(env.server.history_calls()) == 1


def test_case_only_pair_is_not_treated_as_an_alias(env):
    env.server.accepted["nifty 50 futures index"] = ("NIFTY 50 FUTURES INDEX", date(2026, 6, 1))
    assert_standard_empty(fetch_index_history("NIFTY 50 FUTURES INDEX", "2025-01-02", "2025-01-10"))
    assert len(env.server.history_calls()) == 1


def test_unknown_name_logs_error_with_close_suggestions(env, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY FINANCIAL SERVICE", "2026-03-02", "2026-03-06")
    assert_standard_empty(df)
    text = " ".join(error_messages(caplog))
    assert "not in NSE's index list" in text and "NIFTY FINANCIAL SERVICES" in text


def test_unknown_name_still_fetches_in_case_the_list_is_stale(env, caplog):
    # Not in the list, no data in the first chunk, data later: keep fetching.
    env.server.accepted["old index"] = ("OLD INDEX", date(2026, 4, 1))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("OLD INDEX", "2026-01-01", "2026-04-30")
    assert len(df) == len(weekdays(date(2026, 4, 1), date(2026, 4, 30)))
    assert any("stale" in m for m in error_messages(caplog))


def test_unknown_name_with_data_never_requests_the_name_list(env):
    env.server.accepted["old index"] = ("OLD INDEX", date(2020, 1, 1))
    assert len(fetch_index_history("OLD INDEX", "2026-03-02", "2026-03-06")) == 5
    assert env.server.names_calls() == 0


def test_name_list_failure_warns_and_continues(env, caplog):
    env.server.names_fails = True
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY FIN SERVICE", "2026-03-02", "2026-03-06")
    assert_standard_empty(df)
    assert any("index-name list" in m for m in warning_messages(caplog))


def test_malformed_name_list_is_ignored_and_not_cached(env, caplog):
    env.server.names_body = {"stn": "garbage"}
    assert_standard_empty(fetch_index_history("NIFTY FIN SERVICE", "2026-03-02", "2026-03-06"))
    env.server.names_body = {"stn": NAME_PAIRS}
    assert len(fetch_index_history("NIFTY FIN SERVICE", "2026-03-02", "2026-03-06")) == 5


def test_ambiguous_alias_is_not_resolved(env, caplog):
    env.server.names_body = {"stn": [["DUP", "NIFTY 50"], ["DUP", "NIFTY BANK"]]}
    env.server.accepted["dup"] = ("DUP", date(2030, 1, 1))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert_standard_empty(fetch_index_history("DUP", "2026-03-02", "2026-03-06"))
    assert len(env.server.history_calls()) == 1
    assert any("several long names" in m for m in warning_messages(caplog))


# ---------------------------------------------------------------------------
# Row cap
# ---------------------------------------------------------------------------

def test_capped_chunk_is_split_and_result_is_complete(env, monkeypatch):
    monkeypatch.setattr(indices, "_ROW_CAP", 10)
    env.server.cap = 10
    df = fetch_index_history("NIFTY 50", "2026-01-01", "2026-03-02")
    assert df["date"].dt.date.tolist() == weekdays(date(2026, 1, 1), date(2026, 3, 2))
    assert len(env.server.history_calls()) > 1


def test_chunk_with_exactly_cap_rows_is_not_trusted(env, monkeypatch):
    monkeypatch.setattr(indices, "_ROW_CAP", 10)
    env.server.cap = 10
    df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-13")   # exactly 10 trading days
    assert len(df) == 10
    assert len(env.server.history_calls()) == 3      # full range, then two halves


def test_server_that_drops_newest_rows_cannot_cause_a_silent_gap(env, monkeypatch):
    monkeypatch.setattr(indices, "_ROW_CAP", 70)
    env.server.cap = 5                                # server cuts silently below our cap
    df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-13")
    # 5 < 70 so the client cannot know: this documents the limit of the guard.
    assert len(df) == 5


def test_single_day_at_the_cap_fails_the_whole_call_and_terminates(env, monkeypatch, caplog):
    monkeypatch.setattr(indices, "_ROW_CAP", 10)
    env.server.cap = 1000
    env.server.copies_per_day = 11
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert_standard_empty(df)
    assert any("single day" in m for m in error_messages(caplog))
    assert len(env.server.history_calls()) <= 2 * 5 + 5      # bounded, no runaway


def test_request_budget_backstop(env, monkeypatch, caplog):
    monkeypatch.setattr(indices, "_ROW_CAP", 1)       # everything looks capped
    env.server.cap = 1000
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-13")
    assert_standard_empty(df)
    assert len(env.server.history_calls()) < 100


# ---------------------------------------------------------------------------
# list_indices
# ---------------------------------------------------------------------------

def test_list_indices_returns_long_names_without_non_index_entries(env):
    names = list_indices()
    assert "NIFTY FINANCIAL SERVICES" in names and "NIFTY 50" in names
    assert "NIFTY FIN SERVICE" not in names                     # short alias not listed
    assert "Securities in F&O" not in names and "Permitted to Trade" not in names
    assert "NIFTY 50 FUTURES INDEX" in names
    assert len(names) == len({n.lower() for n in names})
    assert names[0] == "NIFTY 50"                               # NSE order kept


def test_list_indices_is_cached(env):
    list_indices()
    list_indices()
    assert env.server.names_calls() == 1 and env.sessions == 1


def test_list_indices_returns_empty_list_on_network_failure(env, caplog):
    env.server.names_fails = True
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert list_indices() == []
    assert warning_messages(caplog)


def test_list_indices_failure_is_not_cached(env):
    env.server.names_fails = True
    assert list_indices() == []
    env.server.names_fails = False
    assert "NIFTY 50" in list_indices()


def test_list_indices_returns_empty_list_when_session_creation_fails(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr("pyequity.connectors.indices.create_nse_session", boom)
    assert list_indices() == []


def test_list_indices_returns_empty_list_on_unexpected_shape(env):
    env.server.names_body = ["not", "a", "dict"]
    assert list_indices() == []


def test_list_indices_docstring_states_the_availability_limitation():
    doc = list_indices.__doc__.lower()
    assert "not that history exists" in doc and "historical availability" in doc


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------

def test_source_has_no_print_and_no_old_project_name():
    source = Path(indices.__file__).read_text(encoding="utf-8")
    assert not re.search(r"(?<![\w.])print\(", source)
    assert ("py" + "capital") not in source.lower()
    assert all(ord(ch) < 128 for ch in source), "plain-text logs and docs only"

# ---------------------------------------------------------------------------
# Volume validation: nothing is truncated or rounded
# ---------------------------------------------------------------------------

INT64_MAX = 2 ** 63 - 1

INVALID_VOLUMES = [
    # negative
    -1, -250000, "-5", -2 ** 53 - 1,
    # fractional
    -0.5, 12.5, 1000000.25, "12.5", "1e-1", "0.000001",
    # above int64 (cannot be stored without rounding or overflow)
    2 ** 63, 2 ** 63 + 1, 10 ** 30, "9223372036854775808", "1e30", 1e30,
    # floats above 2**53 may already have been rounded by the JSON decoder
    9007199254740994.0, 2.0 ** 60,
    # absurd exponent must be rejected, not expanded
    "1e999999999",
]

# (raw JSON value, exact expected int64). 2**53 + 1 is the classic float64 casualty.
VALID_VOLUMES = [
    (0, 0), (12.0, 12), ("12", 12), (" 12 ", 12), ("12.0", 12), ("1e3", 1000), ("-0", 0),
    (237608688, 237608688),
    (2 ** 53 - 1, 2 ** 53 - 1),
    (2 ** 53, 2 ** 53),
    (2.0 ** 53, 2 ** 53),
    (2 ** 53 + 1, 9007199254740993),
    (2 ** 53 + 2, 9007199254740994),
    ("9007199254740993", 9007199254740993),
    (2 ** 62 + 1, 4611686018427387905),
    (INT64_MAX, INT64_MAX),
    (str(INT64_MAX), INT64_MAX),
]

MISSING_VOLUMES = [None, "-", "abc", "", "   ", True, False, [], {}, float("nan"),
                   float("inf"), float("-inf"), "NaN", "Infinity"]


@pytest.mark.parametrize("volume", INVALID_VOLUMES, ids=repr)
def test_invalid_volume_quantities_fail_the_whole_call(env, caplog, volume):
    days = weekdays(date(2026, 3, 2), date(2026, 3, 4))
    records = [_good(d) for d in days]
    records[1]["HIT_TRADED_QTY"] = volume
    _override(env, {"data": records})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-04")
    assert_standard_empty(df)
    assert any("invalid HIT_TRADED_QTY" in m for m in error_messages(caplog))


@pytest.mark.parametrize("volume,expected", VALID_VOLUMES, ids=lambda v: repr(v))
def test_valid_integer_volumes_are_kept_exactly(env, caplog, volume, expected):
    _override(env, {"data": [_good(HIT_TRADED_QTY=volume)]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-02")
    assert df["volume"].tolist() == [expected]
    assert str(df["volume"].dtype) == "int64"
    assert not any("volume" in m for m in warning_messages(caplog))   # a real 0 is not "missing"


@pytest.mark.parametrize("volume", MISSING_VOLUMES, ids=repr)
def test_missing_or_non_numeric_volume_becomes_zero_and_is_counted(env, caplog, volume):
    days = weekdays(date(2026, 3, 2), date(2026, 3, 4))
    records = [_good(d) for d in days]
    records[0]["HIT_TRADED_QTY"] = volume
    _override(env, {"data": records})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-04")
    assert df["volume"].tolist()[0] == 0 and len(df) == 3
    assert any("1 volume value(s)" in m for m in warning_messages(caplog))


def test_big_int_next_to_a_missing_value_is_not_rounded_by_pandas(env):
    """The pandas pitfall: a column of ints plus None becomes float64 and rounds."""
    days = weekdays(date(2026, 3, 2), date(2026, 3, 4))
    records = [_good(d) for d in days]
    records[0]["HIT_TRADED_QTY"] = 2 ** 53 + 1
    records[1]["HIT_TRADED_QTY"] = None
    records[2]["HIT_TRADED_QTY"] = 2 ** 62 + 1
    # sanity: this is exactly how pandas would have corrupted it
    assert pd.DataFrame(records)["HIT_TRADED_QTY"].tolist()[0] == float(2 ** 53)
    _override(env, {"data": records})
    df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-04")
    assert df["volume"].tolist() == [2 ** 53 + 1, 0, 2 ** 62 + 1]
    assert str(df["volume"].dtype) == "int64"


def test_two_adjacent_big_ints_stay_distinct(env):
    days = weekdays(date(2026, 3, 2), date(2026, 3, 3))
    records = [_good(days[0], HIT_TRADED_QTY=2 ** 53 + 1), _good(days[1], HIT_TRADED_QTY=2 ** 53 + 2)]
    _override(env, {"data": records})
    df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-03")
    assert df["volume"].tolist() == [9007199254740993, 9007199254740994]


def test_exact_volume_helper_classification():
    f = indices._exact_volume
    assert f(2 ** 53 + 1) == ("ok", 9007199254740993)
    assert f("9007199254740993") == ("ok", 9007199254740993)
    assert f(2 ** 63) == ("invalid", 0)
    assert f(9007199254740994.0) == ("invalid", 0)
    assert f(-1) == ("invalid", 0)
    assert f(0.5) == ("invalid", 0)
    assert f(None) == ("missing", 0)
    assert f(True) == ("missing", 0)


def test_one_invalid_volume_in_a_later_chunk_fails_the_whole_call(env, caplog):
    def handler(params):
        lo = dmy(params["from"])
        day = lo if lo.weekday() < 5 else lo + timedelta(days=7 - lo.weekday())
        rec = _good(day)
        if lo > date(2026, 3, 1):
            rec["HIT_TRADED_QTY"] = -5
        return FakeResponse({"data": [rec]})

    env.server.history_override = handler
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-01-01", "2026-06-30")
    assert_standard_empty(df)
    assert any("invalid HIT_TRADED_QTY" in m for m in error_messages(caplog))


# ---------------------------------------------------------------------------
# Duplicates across chunk boundaries
# ---------------------------------------------------------------------------

def test_exact_duplicate_dates_across_chunks_are_removed(env):
    rec = _good(date(2026, 1, 15))
    _override(env, {"data": [rec]})                     # every chunk returns the same record
    df = fetch_index_history("NIFTY 50", "2026-01-01", "2026-06-30")
    assert len(env.server.history_calls()) > 1
    assert df["date"].tolist() == [pd.Timestamp("2026-01-15")]


def test_conflicting_duplicate_dates_across_chunks_fail_the_whole_call(env, caplog):
    def handler(params):
        close = 1000.0 if dmy(params["from"]) == date(2026, 1, 1) else 2000.0
        return FakeResponse({"data": [_good(date(2026, 1, 15), EOD_CLOSE_INDEX_VAL=close)]})

    env.server.history_override = handler
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", "2026-01-01", "2026-06-30")
    assert_standard_empty(df)
    assert any("Conflicting" in m for m in error_messages(caplog))


def test_exact_duplicates_survive_next_to_distinct_rows(env):
    a, b = _good(date(2026, 3, 2)), _good(date(2026, 3, 3))
    _override(env, {"data": [b, dict(b), a]})
    df = fetch_index_history("NIFTY 50", "2026-03-02", "2026-03-06")
    assert df["date"].dt.date.tolist() == [date(2026, 3, 2), date(2026, 3, 3)]


# ---------------------------------------------------------------------------
# Request budget
# ---------------------------------------------------------------------------

def _chunk_sizes(total_days: int):
    sizes, left = [], total_days
    while left > 0:
        size = min(indices._CHUNK_DAYS + 1, left)
        sizes.append(size)
        left -= size
    return sizes


@pytest.mark.parametrize("total_days", [1, 2, 7, 61, 62, 122, 365, 1096, 7305])
def test_budget_covers_full_splitting_plus_one_alias_retry(total_days):
    sizes = _chunk_sizes(total_days)
    worst_splitting = sum(2 * size - 1 for size in sizes)          # binary tree to single days
    worst_alias = (2 * sizes[0] - 1) + 1                           # re-walk first chunk + name list
    assert indices._max_requests(total_days, sizes[0]) >= worst_splitting + worst_alias


def test_adversarial_alias_case_stays_inside_the_budget(env, caplog):
    """First chunk fully splits and sees no data, then the alias retry splits it again."""
    cap_rows = [FakeNSE.record("X", date(2026, 3, 2))] * 70

    def handler(params):
        multi_day = params["from"] != params["to"]
        return FakeResponse({"data": cap_rows if multi_day else []})

    env.server.history_override = handler
    first_chunk_days = indices._CHUNK_DAYS + 1
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY FIN SERVICE", "2026-01-01", "2026-03-02")
    assert_standard_empty(df)
    assert not any("budget" in m for m in error_messages(caplog))
    expected = (2 * first_chunk_days - 1) * 2 + 1                  # two full walks + name list
    assert len(env.server.history_calls()) + env.server.names_calls() == expected
    assert expected <= indices._max_requests(first_chunk_days, first_chunk_days)


def test_long_range_with_heavy_splitting_completes_inside_the_budget(env, monkeypatch, caplog):
    monkeypatch.setattr(indices, "_ROW_CAP", 10)
    env.server.cap = 10
    start, end = date(2023, 1, 1), date(2026, 1, 1)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", start.isoformat(), end.isoformat())
    assert df["date"].dt.date.tolist() == weekdays(start, end)
    total_days = (end - start).days + 1
    assert len(env.server.history_calls()) <= indices._max_requests(total_days, indices._CHUNK_DAYS + 1)
    assert not any("budget" in m for m in error_messages(caplog))
