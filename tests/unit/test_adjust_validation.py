"""Validation harness for pyequity.processing.adjust.

Written against the contract agreed with C:

* compute_factors(actions) -> DataFrame with one row per ex-date and the
  columns ex_date, factor, applied, reason.
* adjust_prices(prices, actions) -> DataFrame in the equity schema (same
  columns, dtypes and order as the input).
* Price factor of a bonus a:b is b / (a + b); of a split old -> new face
  value is new / old. Actions on one ex-date multiply.
* A row dated d is scaled by the product of the factors of all applied
  actions with ex_date > d. Open/high/low/close are multiplied, volume is
  divided and rounded, trades and rows on/after the ex-date are untouched.
* Not applied (applied=False, reason set, warning logged): needs_review
  rows and rights issues (v0.1). Rows with an empty ex-date cannot appear
  in the factor table. An ex-date after the last price date is reported by
  compute_factors, but adjust_prices leaves the prices untouched.
* Dividends are ignored. Exact duplicate action rows count once.

The four real cases use raw NSE prices saved by tools/capture_adjust_cases.py.
"""
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyequity.connectors._action_parser import parse_action_detail
from pyequity.processing.normalize import validate
from pyequity.schema import SCHEMAS, columns_of, empty_frame

adjust = pytest.importorskip("pyequity.processing.adjust",
                             reason="processing/adjust.py is not merged yet")

GOLDEN = Path(__file__).parent / "golden" / "adjust"
PRICE_COLS = ["open", "high", "low", "close"]

BONUS_11 = "Bonus 1:1"
BONUS_41 = "Bonus 4:1"
SPLIT_2_1 = "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share"
SPLIT_2_1_HDFC = " Face Value Split (Sub-Division) - From Rs 2 Per Share To Rs 1 Per Share"
SPLIT_10_2 = "Face Value Split From Rs.10/- To Rs.2/-"
COMBINED = " Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
RIGHTS = " Rights 1:15 @ Premium Rs 1247"
DEMERGER = "Demerger"
DIVIDEND = "Dividend - Rs 6 Per Share"


# ---------------------------------------------------------------- helpers
def make_actions(symbol, items):
    """Build an actions frame in the corporate_actions schema.

    items: list of (ex_date string or None, NSE description text). The
    structured columns come from the real parser, so the harness also checks
    that parser and adjuster agree.
    """
    rows = []
    for ex, text in items:
        rows.append({
            "ex_date": pd.Timestamp(ex) if ex else pd.NaT,
            "symbol": symbol, "action_type": "", "action_detail": text,
            "record_date": pd.NaT, "source": "NSE", "isin": None,
            **parse_action_detail(text),
        })
    df = pd.DataFrame(rows)[columns_of("corporate_actions")]
    return df.astype(SCHEMAS["corporate_actions"])


def synthetic_prices(n=20, price=100.0, volume=1000):
    df = pd.DataFrame({
        "date": pd.bdate_range("2026-01-05", periods=n),
        "open": price, "high": price + 2, "low": price - 2, "close": price,
        "volume": volume, "trades": 10, "symbol": "TEST", "source": "NSE",
    })
    return df.astype(SCHEMAS["equity"])


def synthetic_prices_for_actions(actions, n=20, price=100.0, volume=1000):
    """Generate raw prices that become continuous after applying actions."""
    prices = synthetic_prices(n=n, price=price, volume=volume)
    factors = adjust.compute_factors(actions)
    last_date = prices["date"].max()

    for idx, date in prices["date"].items():
        applicable = factors.loc[
            factors["applied"]
            & factors["ex_date"].notna()
            & (factors["ex_date"] > date)
            & (factors["ex_date"] <= last_date),
            "factor",
        ]

        cumulative_factor = applicable.prod() if not applicable.empty else 1.0

        for col in PRICE_COLS:
            prices.at[idx, col] /= cumulative_factor

    return prices.astype(SCHEMAS["equity"])


def load_prices(filename):
    df = pd.read_csv(GOLDEN / filename, parse_dates=["date"])
    return df.astype(SCHEMAS["equity"])


def ordered(df):
    return df.sort_values("date").reset_index(drop=True)


def factor_row(factors, ex):
    rows = factors[factors["ex_date"] == pd.Timestamp(ex)]
    assert len(rows) == 1, f"expected exactly one factor row for {ex}, got {len(rows)}"
    return rows.iloc[0]


def assert_unchanged(prices, adjusted):
    assert_frame_equal(ordered(adjusted), ordered(prices))


# ------------------------------------------------------------- real cases
class Case(NamedTuple):
    symbol: str
    ex: str
    file: str
    items: list
    factor: float


CASES = [
    Case("RELIANCE", "2024-10-28", "case_RELIANCE_2024-10-28.csv",
         [("2024-10-28", BONUS_11)], 0.5),
    Case("HDFCBANK", "2019-09-19", "case_HDFCBANK_2019-09-19.csv",
         [("2019-09-19", SPLIT_2_1_HDFC)], 0.5),
    Case("INFY", "2018-09-04", "case_INFY_2018-09-04.csv",
         [("2018-09-04", BONUS_11)], 0.5),
    Case("BAJFINANCE", "2025-06-16", "case_BAJFINANCE_2025-06-16.csv",
         [("2025-06-16", BONUS_41), ("2025-06-16", SPLIT_2_1)], 0.1),
]
BY_SYMBOL = lambda c: c.symbol  # noqa: E731


@pytest.mark.parametrize("case", CASES, ids=BY_SYMBOL)
def test_real_case_factor(case):
    factors = adjust.compute_factors(make_actions(case.symbol, case.items))
    row = factor_row(factors, case.ex)
    assert row["applied"]
    assert row["factor"] == pytest.approx(case.factor)


@pytest.mark.parametrize("case", CASES, ids=BY_SYMBOL)
def test_real_case_prices_before_ex_date_are_scaled_and_volume_inverse(case):
    prices = load_prices(case.file)
    raw = ordered(prices)
    adj = ordered(adjust.adjust_prices(prices, make_actions(case.symbol, case.items)))
    assert adj["date"].tolist() == raw["date"].tolist()
    before = raw["date"] < pd.Timestamp(case.ex)
    assert before.any()
    for col in PRICE_COLS:
        assert adj.loc[before, col].to_numpy() == pytest.approx(
            raw.loc[before, col].to_numpy() * case.factor, rel=1e-9)
    assert ((adj.loc[before, "volume"] - raw.loc[before, "volume"] / case.factor).abs() <= 1).all()
    assert (adj["trades"] == raw["trades"]).all()


@pytest.mark.parametrize("case", CASES, ids=BY_SYMBOL)
def test_real_case_ex_date_and_later_rows_are_untouched(case):
    prices = load_prices(case.file)
    adj = ordered(adjust.adjust_prices(prices, make_actions(case.symbol, case.items)))
    raw = ordered(prices)
    after = raw["date"] >= pd.Timestamp(case.ex)
    assert_frame_equal(adj[after].reset_index(drop=True), raw[after].reset_index(drop=True))


@pytest.mark.parametrize("case", CASES, ids=BY_SYMBOL)
def test_real_case_no_artificial_jump_at_the_ex_date(case):
    prices = load_prices(case.file)
    raw = ordered(prices)
    ex = pd.Timestamp(case.ex)
    assert (raw["date"] == ex).any(), "ex-date missing from the saved prices"
    raw_move = (raw.loc[raw["date"] >= ex, "close"].iloc[0]
                / raw.loc[raw["date"] < ex, "close"].iloc[-1] - 1)
    assert abs(raw_move) > 0.35, "saved prices are not raw; regenerate them"
    adj = ordered(adjust.adjust_prices(prices, make_actions(case.symbol, case.items)))
    adj_move = (adj.loc[adj["date"] >= ex, "close"].iloc[0]
                / adj.loc[adj["date"] < ex, "close"].iloc[-1] - 1)
    assert abs(adj_move) < 0.12


@pytest.mark.parametrize("case", CASES, ids=BY_SYMBOL)
def test_real_case_price_times_volume_is_preserved(case):
    prices = load_prices(case.file)
    raw = ordered(prices)
    adj = ordered(adjust.adjust_prices(prices, make_actions(case.symbol, case.items)))
    before = raw["date"] < pd.Timestamp(case.ex)
    value_raw = raw.loc[before, "close"] * raw.loc[before, "volume"]
    value_adj = adj.loc[before, "close"] * adj.loc[before, "volume"]
    assert value_adj.to_numpy() == pytest.approx(value_raw.to_numpy(), rel=1e-4)


@pytest.mark.parametrize("case", CASES, ids=BY_SYMBOL)
def test_real_case_output_matches_schema_and_validates(case):
    prices = load_prices(case.file)
    adj = adjust.adjust_prices(prices, make_actions(case.symbol, case.items))
    assert list(adj.columns) == list(SCHEMAS["equity"])
    assert {c: str(t) for c, t in adj.dtypes.items()} == SCHEMAS["equity"]
    assert validate(adj, "equity").ok


# ------------------------------------------------------ synthetic: maths
def test_two_actions_multiply_cumulatively():
    prices = synthetic_prices()
    d = prices["date"]

    actions = make_actions("TEST", [
        (str(d[5].date()), BONUS_11),
        (str(d[12].date()), BONUS_11)
    ])

    # Generate realistic raw prices based on the corporate actions
    prices = synthetic_prices_for_actions(actions)

    adj = ordered(adjust.adjust_prices(prices, actions))

    # All adjusted closing prices should now be continuous
    assert adj["close"].to_numpy() == pytest.approx(100.0)

    assert (adj.loc[:4, "volume"] - 4000).abs().max() <= 1
    assert (adj.loc[5:11, "volume"] - 2000).abs().max() <= 1
    assert (adj.loc[12:, "volume"] == 1000).all()
    
def test_bonus_and_split_in_one_description_multiply():
    prices = synthetic_prices()
    ex = str(prices["date"][10].date())
    actions = make_actions("TEST", [(ex, COMBINED)])

    row = factor_row(adjust.compute_factors(actions), ex)
    assert row["applied"] and row["factor"] == pytest.approx(0.1)

    prices = synthetic_prices_for_actions(actions)
    adj = ordered(adjust.adjust_prices(prices, actions))

    assert adj["close"].to_numpy() == pytest.approx(100.0)


def test_separate_rows_on_the_same_day_give_one_combined_factor():
    prices = synthetic_prices()
    ex = str(prices["date"][10].date())
    actions = make_actions("TEST", [(ex, BONUS_11), (ex, SPLIT_10_2)])
    row = factor_row(adjust.compute_factors(actions), ex)
    assert row["applied"] and row["factor"] == pytest.approx(0.1)


def test_exact_duplicate_action_rows_count_once():
    prices = synthetic_prices()
    ex = str(prices["date"][10].date())
    actions = make_actions(
        "TEST",
        [(ex, BONUS_11), (ex, BONUS_11)],
    )

    factors = adjust.compute_factors(actions)
    row = factor_row(factors, ex)

    assert row["applied"]
    assert row["factor"] == pytest.approx(0.5)

    prices = synthetic_prices_for_actions(actions)
    adj = ordered(adjust.adjust_prices(prices, actions))

    assert adj["close"].to_numpy() == pytest.approx(100.0)


# ---------------------------------------------- synthetic: things to skip
def test_empty_actions_return_prices_unchanged():
    prices = synthetic_prices()
    assert_unchanged(prices, adjust.adjust_prices(prices, empty_frame("corporate_actions")))


def test_dividends_do_not_change_prices():
    prices = synthetic_prices()
    ex = str(prices["date"][10].date())
    assert_unchanged(prices, adjust.adjust_prices(prices, make_actions("TEST", [(ex, DIVIDEND)])))


@pytest.mark.parametrize("text", [DEMERGER, RIGHTS], ids=["demerger", "rights_v01"])
def test_unsupported_actions_are_skipped_with_a_reason_and_a_warning(text, caplog):
    prices = synthetic_prices()
    ex = str(prices["date"][8].date())
    actions = make_actions("TEST", [(ex, text)])
    with caplog.at_level(logging.WARNING):
        factors = adjust.compute_factors(actions)
        with pytest.warns(UserWarning, match="price-affecting corporate action") as warning:
          adj = adjust.adjust_prices(prices, actions)
    row = factor_row(factors, ex)
    assert not row["applied"]
    assert isinstance(row["reason"], str) and row["reason"]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert_unchanged(prices, adj)
    warning_text = " ".join(str(w.message) for w in warning)
    assert text.strip().upper().split()[0] in warning_text.upper()


def test_action_without_an_ex_date_is_skipped_and_others_still_apply(caplog):
    prices = synthetic_prices()

    ex = str(prices["date"][5].date())
    actions = make_actions("TEST", [(None, BONUS_11), (ex, BONUS_11)])

    # Generate realistic raw prices for the dated corporate action.
    prices = synthetic_prices_for_actions(actions)

    with caplog.at_level(logging.WARNING):
        factors = adjust.compute_factors(actions)

        with pytest.warns(UserWarning, match="no ex-date") as warning:
            adj = ordered(adjust.adjust_prices(prices, actions))

    # The action without an ex-date must not be applied.
    assert not factors[factors["ex_date"].isna()]["applied"].any()

    # The dated bonus action must still be applied.
    row = factor_row(factors, ex)
    assert row["applied"]
    assert row["factor"] == pytest.approx(0.5)

    # Verify that the expected warning was raised.
    warning_text = " ".join(str(w.message) for w in warning)
    assert "no ex-date" in warning_text

    # Adjusted prices should be continuous across the ex-date.
    assert adj["close"].to_numpy() == pytest.approx(100.0)


def test_action_after_the_last_price_date_changes_no_prices():
    # compute_factors sees only actions, so it reports the factor the text
    # means. Deciding that the ex-date lies after the last price row is
    # adjust_prices' job: those prices must stay untouched.
    prices = synthetic_prices()
    ex = str((prices["date"].max() + pd.Timedelta(days=3)).date())
    actions = make_actions("TEST", [(ex, BONUS_11)])
    row = factor_row(adjust.compute_factors(actions), ex)
    assert row["applied"] and row["factor"] == pytest.approx(0.5)
    assert_unchanged(prices, adjust.adjust_prices(prices, actions))


def test_action_before_the_first_price_date_changes_nothing():
    prices = synthetic_prices()
    ex = str((prices["date"].min() - pd.Timedelta(days=10)).date())
    assert_unchanged(prices, adjust.adjust_prices(prices, make_actions("TEST", [(ex, BONUS_11)])))


# ------------------------------------------------------------ hygiene
def test_inputs_are_never_modified():
    prices = synthetic_prices()

    ex = str(prices["date"][5].date())
    actions = make_actions(
        "TEST",
        [(ex, BONUS_11), (None, BONUS_11), (ex, DEMERGER)],
    )

    # Generate realistic prices for the supported, dated action.
    prices = synthetic_prices_for_actions(actions)

    # Keep copies to verify that neither input is modified.
    prices_before = prices.copy()
    actions_before = actions.copy()

    adjust.compute_factors(actions)

    # Unsupported and undated actions intentionally raise warnings.
    with pytest.warns(UserWarning) as warning:
        adjust.adjust_prices(prices, actions)

    warning_text = " ".join(str(w.message) for w in warning)

    assert "DEMERGER" in warning_text
    assert "no ex-date" in warning_text

    # Verify that the original inputs remain unchanged.
    assert_frame_equal(prices, prices_before)
    assert_frame_equal(actions, actions_before)
