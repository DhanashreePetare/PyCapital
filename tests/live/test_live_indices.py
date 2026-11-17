"""Live checks for pyequity.connectors.indices (hit the real NSE, run manually).

Run:  python -m pytest tests/live/test_live_indices.py -v

Not part of the default offline run (pyproject testpaths is tests/unit). The
alias check uses the window 2026-11-03 to 2026-11-16 where the short/long name
behaviour was probed; alias behaviour outside that window is unverified.
"""
import pandas as pd
import pytest

from pyequity.connectors.indices import fetch_index_history, list_indices
from pyequity.schema import SCHEMAS

pytestmark = pytest.mark.live


def _assert_schema(df: pd.DataFrame) -> None:
    assert list(df.columns) == list(SCHEMAS["index"])
    assert {c: str(t) for c, t in df.dtypes.items()} == SCHEMAS["index"]


def test_nifty50_known_week_matches_the_captured_fixture():
    df = fetch_index_history("NIFTY 50", "2026-09-15", "2026-09-23")
    _assert_schema(df)
    assert len(df) == 7
    assert df.loc[df["date"] == "2026-09-23", "close"].iloc[0] == 23446.8


def test_nifty_bank_one_year_has_no_silent_truncation():
    df = fetch_index_history("NIFTY BANK", "2025-09-23", "2026-09-23")
    _assert_schema(df)
    assert 235 <= len(df) <= 260, f"unexpected row count {len(df)}"
    assert df["date"].is_unique
    assert df["date"].diff().dropna().max() <= pd.Timedelta(days=6)
    assert df["date"].min() <= pd.Timestamp("2025-09-26")
    assert df["date"].max() >= pd.Timestamp("2026-09-18")


def test_short_alias_is_resolved_and_labelled_with_nses_name():
    df = fetch_index_history("NIFTY FIN SERVICE", "2026-11-03", "2026-11-16")
    _assert_schema(df)
    assert len(df) > 0
    assert set(df["index_name"]) == {"NIFTY FIN SERVICE"}


def test_invalid_name_returns_empty_frame():
    df = fetch_index_history("NOT A REAL INDEX", "2026-09-15", "2026-09-23")
    _assert_schema(df)
    assert df.empty


def test_list_indices_contains_core_indices_and_no_non_index_entries():
    names = list_indices()
    assert len(names) > 100
    assert "NIFTY 50" in names and "NIFTY BANK" in names
    assert "Securities in F&O" not in names and "Permitted to Trade" not in names
