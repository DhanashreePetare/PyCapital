"""Offline tests: replay real raw API responses through the real connectors.

The files in tests/fixtures/ are raw responses captured by
tools/capture_fixtures.py. Here each connector's public function is run on
them with the network replaced by the saved body, so these tests need no
internet. They check four things per dataset:

1. the connector output has exactly the schema columns/dtypes and passes
   ``validate()`` with no errors,
2. no row is lost or invented (row count equals the raw record count),
3. values in the output match the raw record they came from,
4. ``normalize()`` leaves the output unchanged (the "no change to existing
   output" guarantee, proven on real data).

Corporate actions are the one exception to (4): the connector returns the
newest action first while ``normalize()`` sorts oldest-first, so that test
compares the same rows ignoring order. If normalize is ever wired into the
connectors, the order of corporate actions will flip; the team has to decide
on one order and document it.
"""
import json
import logging
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import pandas as pd
import pytest
import requests
from pandas.testing import assert_frame_equal

from pyequity.connectors.amfi import fetch_nav
from pyequity.connectors.bse import fetch_bse_historical
from pyequity.connectors.corporate import fetch_corporate_actions
from pyequity.connectors.fno import fetch_fno_data
from pyequity.connectors.indices import fetch_index_history
from pyequity.connectors.nse import fetch_nse_historical
from pyequity.connectors.sebi import fetch_bulk_deals
from pyequity.processing.normalize import normalize, validate
from pyequity.schema import SCHEMAS, empty_frame

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class NetworkBlocked(BaseException):
    """BaseException on purpose: connectors catch Exception and would hide it."""


class FakeResponse:
    """Just enough of requests.Response for the connectors."""

    def __init__(self, body: bytes, status: int = 200):
        self.content = body
        self.text = body.decode("utf-8")
        self.status_code = status
        self.url = "fixture"
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    def __init__(self, fixture: str):
        self._body = _read(fixture)

    def get(self, *args, **kwargs):
        return FakeResponse(self._body)


def _read(fixture: str) -> bytes:
    return (FIXTURES / f"{fixture}.json").read_bytes()


def _records(fixture: str) -> list:
    """The list of raw records inside a fixture, whatever its wrapper."""
    payload = json.loads(_read(fixture))
    if isinstance(payload, list):
        return payload
    for key in ("StockData", "data"):
        if key in payload:
            return payload[key]
    raise AssertionError(f"{fixture}: no record list found")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No real HTTP and no real sleeping in any test in this file."""
    def blocked(*args, **kwargs):
        raise NetworkBlocked("a test tried to use the network")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def _use_session(mp, module: str, fixture: str) -> None:
    mp.setattr(f"pyequity.connectors.{module}.create_nse_session",
               lambda *a, **k: FakeSession(fixture))


def _use_requests(mp, fixture: str) -> None:
    body = _read(fixture)
    mp.setattr("requests.get", lambda *a, **k: FakeResponse(body))


# ---------------------------------------------------------------------------
# Runners: call the public connector function on a fixture
# ---------------------------------------------------------------------------

def _run_nse(mp):
    _use_session(mp, "nse", "nse_equity_reliance")
    return fetch_nse_historical("RELIANCE", "2026-09-15", "2026-09-23")


def _run_bse(mp):
    _use_requests(mp, "bse_reliance")
    return fetch_bse_historical("500325", "2026-09-15", "2026-09-23")


def _run_indices(mp):
    _use_session(mp, "indices", "indices_nifty50")
    return fetch_index_history("NIFTY 50", "2026-09-15", "2026-09-23")


def _run_fno(mp):
    _use_session(mp, "fno", "fno_reliance_futstk")
    return fetch_fno_data("RELIANCE", "2026-09-15", "2026-09-23", "27-Oct-2026", "FUTSTK")


def _run_corporate(mp):
    _use_session(mp, "corporate", "corporate_actions_reliance")
    return fetch_corporate_actions("RELIANCE")


def _run_sebi(mp):
    _use_session(mp, "sebi", "sebi_bulk_deals_2026-09-21")
    return fetch_bulk_deals("2026-09-21", "2026-09-21")


def _run_amfi(mp):
    _use_requests(mp, "amfi_nav_119551")
    return fetch_nav(119551)


# ---------------------------------------------------------------------------
# Spot checks: output values equal the raw record they came from
# ---------------------------------------------------------------------------

def _number(value) -> float:
    return float(str(value).replace(",", ""))


def _spot_nse(df, recs):
    expected = {pd.to_datetime(r["mtimestamp"], format="%d-%b-%Y"): _number(r["chClosingPrice"])
                for r in recs}
    assert df.set_index("date")["close"].to_dict() == expected


def _spot_bse(df, recs):
    expected = {pd.to_datetime(r["RealDate"]): _number(r["qe_close"]) for r in recs}
    assert df.set_index("date")["close"].to_dict() == expected


def _spot_indices(df, recs):
    expected = {pd.to_datetime(r["EOD_TIMESTAMP"], format="%d-%b-%Y"): _number(r["EOD_CLOSE_INDEX_VAL"])
                for r in recs}
    assert df.set_index("date")["close"].to_dict() == expected


def _spot_fno(df, recs):
    expected = {pd.to_datetime(r["FH_TIMESTAMP"], format="%d-%b-%Y"): _number(r["FH_CLOSING_PRICE"])
                for r in recs}
    assert df.set_index("date")["close"].to_dict() == expected


def _spot_corporate(df, recs):
    assert sorted(df["action_detail"]) == sorted(r["subject"] for r in recs)


def _spot_sebi(df, recs):
    # A silently unparseable quantity/price would show up here as 0 / NaN.
    assert (df["quantity"] > 0).all()
    assert df["price"].notna().all()
    assert int(df["quantity"].sum()) == sum(int(_number(r["BD_QTY_TRD"])) for r in recs)
    assert df["price"].sum() == pytest.approx(sum(_number(r["BD_TP_WATP"]) for r in recs))


def _spot_amfi(df, recs):
    expected = {pd.to_datetime(r["date"], format="%d-%m-%Y"): _number(r["nav"]) for r in recs}
    assert df.set_index("date")["nav"].to_dict() == expected


class Case(NamedTuple):
    id: str
    dataset: str
    fixture: str
    run: Callable
    start: Optional[str]
    end: Optional[str]
    spot: Callable
    order_changes: bool = False


CASES = [
    Case("nse_equity", "equity", "nse_equity_reliance", _run_nse,
         "2026-09-15", "2026-09-23", _spot_nse),
    Case("bse_equity", "equity", "bse_reliance", _run_bse,
         "2026-09-15", "2026-09-23", _spot_bse),
    Case("indices", "index", "indices_nifty50", _run_indices,
         "2026-09-15", "2026-09-23", _spot_indices),
    Case("fno", "fno", "fno_reliance_futstk", _run_fno,
         "2026-09-15", "2026-09-23", _spot_fno),
    Case("corporate_actions", "corporate_actions", "corporate_actions_reliance",
         _run_corporate, None, None, _spot_corporate, order_changes=True),
    Case("sebi_bulk_deals", "bulk_deals", "sebi_bulk_deals_2026-09-21", _run_sebi,
         "2026-09-21", "2026-09-21", _spot_sebi),
    Case("amfi_nav", "nav", "amfi_nav_119551", _run_amfi, None, None, _spot_amfi),
]
IDS = [c.id for c in CASES]


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_output_has_exact_schema_and_validates(case, monkeypatch):
    df = case.run(monkeypatch)
    assert not df.empty
    assert list(df.columns) == list(SCHEMAS[case.dataset])
    report = validate(df, case.dataset, start=case.start, end=case.end)
    assert report.ok, [f"{i.code}: {i.message}" for i in report.errors]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_no_row_lost_or_invented(case, monkeypatch):
    df = case.run(monkeypatch)
    assert len(df) == len(_records(case.fixture))


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_values_match_raw_records(case, monkeypatch):
    df = case.run(monkeypatch)
    case.spot(df, _records(case.fixture))


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_normalize_leaves_connector_output_unchanged(case, monkeypatch):
    df = case.run(monkeypatch)
    normalized = normalize(df, case.dataset)
    if case.order_changes:
        key = list(SCHEMAS[case.dataset])
        df = df.sort_values(key, kind="mergesort").reset_index(drop=True)
        normalized = normalized.sort_values(key, kind="mergesort").reset_index(drop=True)
    assert_frame_equal(normalized, df)


def test_normalize_sorts_corporate_actions_oldest_first(monkeypatch):
    df = _run_corporate(monkeypatch)
    assert normalize(df, "corporate_actions")["ex_date"].dropna().is_monotonic_increasing


def test_nse_and_bse_fixtures_share_the_equity_schema(monkeypatch):
    nse, bse = _run_nse(monkeypatch), _run_bse(monkeypatch)
    assert list(nse.columns) == list(bse.columns)
    assert {c: str(t) for c, t in nse.dtypes.items()} == {c: str(t) for c, t in bse.dtypes.items()}


# ---------------------------------------------------------------------------
# Failure cases: bad input must give the standard empty frame, never raise
# ---------------------------------------------------------------------------

def _run_nse_invalid(mp):
    _use_session(mp, "nse", "nse_equity_invalid_symbol")
    return fetch_nse_historical("WRONGSYMBOL123", "2026-09-15", "2026-09-23")


def _run_bse_invalid(mp):
    _use_requests(mp, "bse_invalid_scripcode")
    return fetch_bse_historical("999999", "2026-09-15", "2026-09-23")


def _run_amfi_invalid(mp):
    _use_requests(mp, "amfi_invalid_scheme")
    return fetch_nav(999999999)


@pytest.mark.parametrize("dataset, run", [
    ("equity", _run_nse_invalid),
    ("equity", _run_bse_invalid),
    ("nav", _run_amfi_invalid),
], ids=["nse_invalid_symbol", "bse_invalid_scripcode", "amfi_invalid_scheme"])
def test_invalid_input_returns_standard_empty_frame(dataset, run, monkeypatch):
    df = run(monkeypatch)
    assert_frame_equal(df, empty_frame(dataset))
    assert validate(df, dataset).ok
    assert_frame_equal(normalize(df, dataset), empty_frame(dataset))


# ---------------------------------------------------------------------------
# Known limitation: NSE caps bulk deals at 70 rows per query
# ---------------------------------------------------------------------------

def test_bulk_deals_cap_is_visible_in_fixture_and_logged(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        df = _run_sebi(monkeypatch)
    assert len(df) == 70, "fixture no longer shows the 70-row cap; revisit the SEBI docs"
    assert df["date"].nunique() == 1
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# Fixture bookkeeping
# ---------------------------------------------------------------------------

def test_manifest_matches_fixture_files():
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["fixtures"]
    assert entries
    for name, entry in entries.items():
        path = FIXTURES / entry["file"]
        assert path.exists(), f"{name}: {path.name} missing"
        assert path.stat().st_size == entry["bytes"], f"{name}: file changed since capture"
    on_disk = {p.name for p in FIXTURES.iterdir()
               if p.suffix in (".json", ".txt") and p.name != "manifest.json"}
    assert on_disk == {e["file"] for e in entries.values()}


def test_every_case_has_a_fixture_in_the_manifest():
    names = set(json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"])
    needed = {c.fixture for c in CASES} | {
        "nse_equity_invalid_symbol", "bse_invalid_scripcode", "amfi_invalid_scheme"}
    assert needed <= names