"""Offline tests for pyequity.api (mutual_funds, sebi, shared checks).

The connectors are replaced by fakes; nothing touches the network.
"""
import datetime as dt
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pyequity.api import _args, mutual_funds, sebi
from pyequity.api._errors import (
    DataQualityWarning, DataUnavailableError, EmptyResultWarning,
    FallbackWarning, IncompleteDataWarning, InvalidArgumentError,
    PyEquityError, PyEquityWarning)
from pyequity.connectors import amfi
from pyequity.connectors import sebi as connector
from pyequity.reliability.outcome import FetchOutcome, Status
from pyequity.schema import SCHEMAS, empty_frame

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def nav_frame(source="AMFI", dates=("2026-09-15", "2026-09-16", "2026-09-17"), nav=100.0):
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)),
        "scheme_code": pd.Series([119551] * len(dates), dtype="int64"),
        "scheme_name": "Test Fund",
        "nav": [nav + i for i in range(len(dates))],
        "source": source,
    })


def check_schema(df, name):
    assert list(df.columns) == list(SCHEMAS[name])
    for c, t in SCHEMAS[name].items():
        assert str(df[c].dtype) == t, c


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("unexpected connector call")
    for name in ("fetch_nav_outcome", "fetch_nav_official"):
        monkeypatch.setattr(amfi, name, boom)
    for name in ("fetch_latest_bulk_deals", "fetch_bulk_deals_by_day"):
        monkeypatch.setattr(connector, name, boom)


def fake_outcome(monkeypatch, outcome):
    calls = []
    monkeypatch.setattr(amfi, "fetch_nav_outcome",
                        lambda *a, **k: calls.append(a) or outcome)
    return calls


def fake_official(monkeypatch, frame):
    calls = []
    monkeypatch.setattr(amfi, "fetch_nav_official",
                        lambda *a, **k: calls.append(a) or frame)
    return calls




# ------------------------------ _args / _errors -----------------------------

def test_error_hierarchy():
    assert issubclass(InvalidArgumentError, ValueError)
    assert issubclass(InvalidArgumentError, PyEquityError)
    assert issubclass(DataUnavailableError, PyEquityError)
    assert not issubclass(DataUnavailableError, ValueError)
    for w in (FallbackWarning, IncompleteDataWarning, EmptyResultWarning, DataQualityWarning):
        assert issubclass(w, PyEquityWarning) and issubclass(w, UserWarning)


@pytest.mark.parametrize("value,expected", [
    ("2026-09-23", "2026-09-23"),
    (" 2026-09-23 ", "2026-09-23"),
    (dt.date(2026, 9, 23), "2026-09-23"),
    (dt.datetime(2026, 9, 23, 15, 30), "2026-09-23"),
    (pd.Timestamp("2026-09-23"), "2026-09-23"),
])
def test_check_date_accepts(value, expected):
    assert _args.check_date(value, "start") == expected


@pytest.mark.parametrize("bad", ["23-09-2026", "2026/09/23", "2026-13-01", "", None, 20260923, "today"])
def test_check_date_rejects(bad):
    with pytest.raises(InvalidArgumentError, match="start"):
        _args.check_date(bad, "start")


def test_check_range_rejects_reversed_and_returns_strings():
    assert _args.check_range("2026-01-01", dt.date(2026, 1, 31)) == ("2026-01-01", "2026-01-31")
    assert _args.check_range("2026-01-01", "2026-01-01") == ("2026-01-01", "2026-01-01")
    with pytest.raises(InvalidArgumentError, match="cannot be after"):
        _args.check_range("2026-02-01", "2026-01-01")


# --------------------------- mutual_funds.get_nav ---------------------------

@pytest.mark.parametrize("code", ["abc", "", "-5", 0, -3, 12.5, None, True, [119551], "11 9"])
def test_get_nav_rejects_bad_scheme_codes(code):
    with pytest.raises(InvalidArgumentError, match="scheme_code"):
        mutual_funds.get_nav(code, "2026-09-15", "2026-09-17")


@pytest.mark.parametrize("code", [119551, "119551", " 119551 ", 119551.0])
def test_get_nav_accepts_code_forms(monkeypatch, code):
    calls = fake_outcome(monkeypatch, FetchOutcome(Status.OK, nav_frame()))
    mutual_funds.get_nav(code, "2026-09-15", "2026-09-17")
    assert calls[0][0] == 119551


def test_get_nav_rejects_bad_dates_range_and_source():
    with pytest.raises(InvalidArgumentError):
        mutual_funds.get_nav(119551, "15-09-2026", "2026-09-17")
    with pytest.raises(InvalidArgumentError):
        mutual_funds.get_nav(119551, "2026-09-17", "2026-09-15")
    with pytest.raises(InvalidArgumentError, match="source"):
        mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17", source="yahoo")


def test_get_nav_ok_path_no_warning_and_labelled(monkeypatch):
    calls = fake_outcome(monkeypatch, FetchOutcome(Status.OK, nav_frame()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17")
    check_schema(df, "nav")
    assert len(df) == 3
    assert df.attrs["source"] == "AMFI" and df.attrs["fallback_used"] is False
    assert calls == [(119551, "2026-09-15", "2026-09-17")]


def test_get_nav_accepts_date_objects(monkeypatch):
    calls = fake_outcome(monkeypatch, FetchOutcome(Status.OK, nav_frame()))
    mutual_funds.get_nav(119551, dt.date(2026, 9, 15), pd.Timestamp("2026-09-17"))
    assert calls[0][1:] == ("2026-09-15", "2026-09-17")


def test_get_nav_empty_warns_and_does_not_fall_back(monkeypatch):
    fake_outcome(monkeypatch, FetchOutcome(Status.EMPTY, empty_frame("nav")))
    # fetch_nav_official is a "boom" fake: calling it would fail the test
    with pytest.warns(EmptyResultWarning, match="119551"):
        df = mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17")
    assert df.empty and list(df.columns) == list(SCHEMAS["nav"])


def test_get_nav_failed_falls_back_with_warning_and_label(monkeypatch):
    fake_outcome(monkeypatch, FetchOutcome(Status.FAILED, None, "ConnectionError: down", 3))
    official = nav_frame(source="AMFI_OFFICIAL")
    official.attrs["failed_chunks"] = []
    calls = fake_official(monkeypatch, official)
    with pytest.warns(FallbackWarning, match="AMFI_OFFICIAL"):
        df = mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17")
    assert (df["source"] == "AMFI_OFFICIAL").all()
    assert df.attrs["fallback_used"] is True and df.attrs["source"] == "AMFI_OFFICIAL"
    assert calls == [(119551, "2026-09-15", "2026-09-17")]
    check_schema(df, "nav")


def test_get_nav_fallback_with_failed_chunks_warns_incomplete(monkeypatch):
    fake_outcome(monkeypatch, FetchOutcome(Status.FAILED, None, "boom", 3))
    official = nav_frame(source="AMFI_OFFICIAL")
    official.attrs["failed_chunks"] = ["2026-09-01 to 2026-09-30"]
    fake_official(monkeypatch, official)
    with pytest.warns(PyEquityWarning) as rec:
        df = mutual_funds.get_nav(119551, "2026-09-01", "2026-09-30")
    kinds = {type(w.message) for w in rec}
    assert FallbackWarning in kinds and IncompleteDataWarning in kinds
    assert df.attrs["failed_chunks"] == ["2026-09-01 to 2026-09-30"]


def test_get_nav_both_sources_fail_raises(monkeypatch):
    fake_outcome(monkeypatch, FetchOutcome(Status.FAILED, None, "ConnectionError: down", 3))
    empty = empty_frame("nav")
    empty.attrs["failed_chunks"] = ["2026-09-15 to 2026-09-17"]
    fake_official(monkeypatch, empty)
    with pytest.warns(FallbackWarning):
        with pytest.raises(DataUnavailableError, match="ConnectionError"):
            mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17")


def test_get_nav_source_mfapi_never_falls_back(monkeypatch):
    fake_outcome(monkeypatch, FetchOutcome(Status.FAILED, None, "HTTPError 503", 3))
    with pytest.raises(DataUnavailableError, match="503"):
        mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17", source="mfapi")


def test_get_nav_source_official_skips_mfapi(monkeypatch):
    official = nav_frame(source="AMFI_OFFICIAL")
    official.attrs["failed_chunks"] = []
    calls = fake_official(monkeypatch, official)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17", source="official")
    assert len(calls) == 1 and df.attrs["fallback_used"] is False
    assert df.attrs["requested_source"] == "official"


def test_get_nav_source_official_empty_warns(monkeypatch):
    fake_official(monkeypatch, empty_frame("nav"))
    with pytest.warns(EmptyResultWarning):
        df = mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17", source="official")
    assert df.empty


def test_get_nav_integrity_errors_are_warned(monkeypatch):
    bad = nav_frame()
    bad.loc[1, "nav"] = float("nan")
    fake_outcome(monkeypatch, FetchOutcome(Status.OK, bad))
    with pytest.warns(DataQualityWarning, match="integrity"):
        df = mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17")
    assert len(df) == 3  # reported, never silently dropped


def test_get_nav_output_is_sorted_and_deduplicated(monkeypatch):
    messy = pd.concat([nav_frame(dates=("2026-09-17", "2026-09-15")),
                       nav_frame(dates=("2026-09-15",))], ignore_index=True)
    fake_outcome(monkeypatch, FetchOutcome(Status.OK, messy))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = mutual_funds.get_nav(119551, "2026-09-15", "2026-09-17")
    assert df["date"].is_monotonic_increasing and df["date"].is_unique


def test_get_nav_on_real_fixture_end_to_end(monkeypatch):
    """Real mfapi payload -> fetch_nav_outcome -> get_nav, through the real parser."""
    import requests
    payload = json.loads((FIX / "amfi_nav_119551.json").read_text(encoding="utf-8"))
    monkeypatch.undo()
    r = requests.Response()
    r.status_code = 200
    r.json = lambda: payload  # type: ignore[method-assign]
    monkeypatch.setattr(amfi.requests, "get", lambda *a, **k: r)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = mutual_funds.get_nav(119551, "2026-10-01", "2026-11-13")
    check_schema(df, "nav")
    assert df["date"].min() >= pd.Timestamp("2026-10-01") and len(df) > 10
    assert (df["source"] == "AMFI").all()


# ------------------------------- sebi.get_bulk_deals -------------------------

def deals(day, n, source="SEBI_NSE"):
    return pd.DataFrame({
        "date": pd.to_datetime([day] * n),
        "symbol": "ABC", "scrip_name": "Abc Ltd",
        "client_name": [f"C{i}" for i in range(n)],
        "deal_type": ["BUY" if i % 2 else "SELL" for i in range(n)],
        "quantity": pd.Series(range(100, 100 + n), dtype="int64"),
        "price": 10.5, "source": source,
    })


def by_day_result(frame, capped=(), failed=()):
    frame = frame.copy()
    frame.attrs["capped_days"] = list(capped)
    frame.attrs["failed_days"] = list(failed)
    return frame


def fake_by_day(monkeypatch, frame):
    calls = []
    monkeypatch.setattr(connector, "fetch_bulk_deals_by_day",
                        lambda a, b: calls.append((a, b)) or frame)
    return calls


def fake_latest(monkeypatch, frame):
    calls = []
    monkeypatch.setattr(connector, "fetch_latest_bulk_deals",
                        lambda: calls.append(1) or frame)
    return calls


def test_bulk_latest_when_no_dates(monkeypatch):
    fake_latest(monkeypatch, deals("2026-11-16", 5, "SEBI_NSE_CSV"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = sebi.get_bulk_deals()
    check_schema(df, "bulk_deals")
    assert len(df) == 5 and (df["source"] == "SEBI_NSE_CSV").all()
    assert df.attrs["capped_days"] == [] and df.attrs["failed_days"] == []


def test_bulk_latest_empty_warns(monkeypatch):
    fake_latest(monkeypatch, empty_frame("bulk_deals"))
    with pytest.warns(EmptyResultWarning, match="latest trading day"):
        df = sebi.get_bulk_deals()
    assert df.empty


def test_bulk_needs_both_dates_or_neither():
    with pytest.raises(InvalidArgumentError, match="both"):
        sebi.get_bulk_deals(start="2026-09-21")
    with pytest.raises(InvalidArgumentError, match="both"):
        sebi.get_bulk_deals(end="2026-09-21")


def test_bulk_rejects_bad_dates_reversed_and_too_long():
    with pytest.raises(InvalidArgumentError):
        sebi.get_bulk_deals("21-09-2026", "2026-09-22")
    with pytest.raises(InvalidArgumentError):
        sebi.get_bulk_deals("2026-09-22", "2026-09-21")
    with pytest.raises(InvalidArgumentError, match="too long"):
        sebi.get_bulk_deals("2025-01-01", "2026-12-31")


def test_bulk_range_ok_no_warning(monkeypatch):
    calls = fake_by_day(monkeypatch, by_day_result(deals("2026-09-21", 3)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = sebi.get_bulk_deals("2026-09-21", dt.date(2026, 9, 21))
    check_schema(df, "bulk_deals")
    assert len(df) == 3 and calls == [("2026-09-21", "2026-09-21")]
    assert df.attrs["capped_days"] == [] and df.attrs["failed_days"] == []


def test_bulk_capped_day_is_warned_and_listed(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(deals("2026-09-21", 70), capped=["2026-09-21"]))
    fake_latest(monkeypatch, empty_frame("bulk_deals"))
    with pytest.warns(IncompleteDataWarning, match="2026-09-21"):
        df = sebi.get_bulk_deals("2026-09-21", "2026-09-21")
    assert df.attrs["capped_days"] == ["2026-09-21"] and len(df) == 70


def test_bulk_capped_latest_day_is_completed_from_csv(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(
        pd.concat([deals("2026-11-13", 70), deals("2026-11-16", 70)], ignore_index=True),
        capped=["2026-11-13", "2026-11-16"]))
    fake_latest(monkeypatch, deals("2026-11-16", 134, "SEBI_NSE_CSV"))
    with pytest.warns(IncompleteDataWarning) as rec:
        df = sebi.get_bulk_deals("2026-11-13", "2026-11-16")
    assert df.attrs["capped_days"] == ["2026-11-13"]  # only the older day stays flagged
    day16 = df[df["date"] == "2026-11-16"]
    assert len(day16) == 134 and (day16["source"] == "SEBI_NSE_CSV").all()
    assert len(df[df["date"] == "2026-11-13"]) == 70
    assert "2026-11-16" not in str(rec[0].message)
    assert df["date"].is_monotonic_increasing


def test_bulk_completion_fully_resolves_when_only_latest_day_capped(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(deals("2026-11-16", 70), capped=["2026-11-16"]))
    fake_latest(monkeypatch, deals("2026-11-16", 134, "SEBI_NSE_CSV"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = sebi.get_bulk_deals("2026-11-16", "2026-11-16")
    assert len(df) == 134 and df.attrs["capped_days"] == []


def test_bulk_csv_that_is_smaller_never_replaces_api_rows(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(deals("2026-11-16", 70), capped=["2026-11-16"]))
    fake_latest(monkeypatch, deals("2026-11-16", 10, "SEBI_NSE_CSV"))
    with pytest.warns(IncompleteDataWarning):
        df = sebi.get_bulk_deals("2026-11-16", "2026-11-16")
    assert len(df) == 70 and (df["source"] == "SEBI_NSE").all()


def test_bulk_csv_for_a_different_day_is_ignored(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(deals("2026-09-21", 70), capped=["2026-09-21"]))
    fake_latest(monkeypatch, deals("2026-11-16", 134, "SEBI_NSE_CSV"))
    with pytest.warns(IncompleteDataWarning):
        df = sebi.get_bulk_deals("2026-09-21", "2026-09-21")
    assert len(df) == 70 and df.attrs["capped_days"] == ["2026-09-21"]


def test_bulk_complete_latest_false_makes_no_extra_request(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(deals("2026-11-16", 70), capped=["2026-11-16"]))
    # fetch_latest_bulk_deals is a "boom" fake: any call fails the test
    with pytest.warns(IncompleteDataWarning):
        df = sebi.get_bulk_deals("2026-11-16", "2026-11-16", complete_latest=False)
    assert len(df) == 70


def test_bulk_latest_not_fetched_when_nothing_capped(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(deals("2026-09-21", 3)))
    sebi.get_bulk_deals("2026-09-21", "2026-09-21")  # boom fake guards the latest call


def test_bulk_failed_days_warn_but_keep_rows(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(deals("2026-09-22", 3), failed=["2026-09-21"]))
    with pytest.warns(IncompleteDataWarning, match="2026-09-21"):
        df = sebi.get_bulk_deals("2026-09-21", "2026-09-22")
    assert len(df) == 3 and df.attrs["failed_days"] == ["2026-09-21"]


def test_bulk_everything_failed_raises(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(empty_frame("bulk_deals"),
                                           failed=["2026-09-21", "2026-09-22"]))
    with pytest.raises(DataUnavailableError, match="2026-09-21"):
        sebi.get_bulk_deals("2026-09-21", "2026-09-22")


def test_bulk_genuinely_empty_range_warns_not_raises(monkeypatch):
    fake_by_day(monkeypatch, by_day_result(empty_frame("bulk_deals")))
    with pytest.warns(EmptyResultWarning):
        df = sebi.get_bulk_deals("2026-09-26", "2026-09-27")
    check_schema(df, "bulk_deals")
    assert df.attrs["failed_days"] == []


def test_bulk_identical_rows_are_never_dropped(monkeypatch):
    twin = pd.concat([deals("2026-09-21", 1), deals("2026-09-21", 1)], ignore_index=True)
    fake_by_day(monkeypatch, by_day_result(twin))
    assert len(sebi.get_bulk_deals("2026-09-21", "2026-09-21")) == 2