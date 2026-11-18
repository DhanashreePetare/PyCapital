"""Offline tests for pyequity.processing.adjust (no internet needed)."""
import copy
import json
import logging
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyequity.connectors import corporate
from pyequity.connectors.corporate import fetch_corporate_actions
from pyequity.processing import adjust
from pyequity.processing.adjust import FACTOR_COLUMNS, adjust_prices, compute_factors
from pyequity.processing.normalize import validate
from pyequity.schema import SCHEMAS, empty_frame

LOGGER = "pyequity.processing.adjust"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "corporate_actions_reliance.json"
NAN = float("nan")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def act(ex="2024-10-28", cats="BONUS", symbol="TEST", status="ok", bonus=(NAN, NAN),
        split=(NAN, NAN), rights=(NAN, NAN, NAN), detail=""):
    return {"ex_date": ex, "symbol": symbol, "action_type": "", "action_detail": detail,
            "record_date": pd.NaT, "source": "NSE", "isin": "INE000000000",
            "action_categories": cats, "bonus_new": bonus[0], "bonus_held": bonus[1],
            "split_old_fv": split[0], "split_new_fv": split[1], "rights_new": rights[0],
            "rights_held": rights[1], "rights_premium": rights[2], "parse_status": status}


def bonus(new, held, ex="2024-10-28", **kw):
    kw.setdefault("detail", f"Bonus {new}:{held}")
    return act(ex=ex, cats="BONUS", bonus=(new, held), **kw)


def split(old, new, ex="2024-10-28", **kw):
    kw.setdefault("detail", f"Split {old}->{new}")
    return act(ex=ex, cats="SPLIT", split=(old, new), **kw)


def dividend(ex="2024-08-19", **kw):
    return act(ex=ex, cats="DIVIDEND", detail="Dividend - Rs 10 Per Share", **kw)


def actions_df(*rows):
    df = pd.DataFrame(list(rows))
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    return df


def raw_prices(start, end, events=(), symbol="TEST", base=100.0, base_volume=1000):
    """Weekday RAW prices that fall/jump exactly as the events say they should.

    close(d) = base * product of factors with ex <= d; volume the inverse. Once
    adjusted correctly the close and volume are constant for every row.
    """
    days = pd.bdate_range(start, end)
    price_factor = np.ones(len(days))
    for ex, factor in events:
        price_factor = np.where(days >= pd.Timestamp(ex), price_factor * factor, price_factor)
    close = base * price_factor
    return pd.DataFrame({
        "date": days.astype("datetime64[ns]"), "open": close * 1.01, "high": close * 1.02,
        "low": close * 0.98, "close": close,
        "volume": np.rint(base_volume / price_factor).astype("int64"),
        "trades": np.arange(len(days), dtype="int64") + 10,
        "symbol": symbol, "source": "NSE"}).astype({"symbol": object, "source": object})


def reliance_actions(monkeypatch):
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))

    class Response:
        def json(self):
            return rows

        def raise_for_status(self):
            pass

    class Session:
        def get(self, *a, **k):
            return Response()

    monkeypatch.setattr(corporate, "create_nse_session", lambda *a, **k: Session())
    return fetch_corporate_actions("RELIANCE", "2010-01-01", "2026-12-31")


def no_user_warnings():
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("error", UserWarning)
    return ctx


def messages(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


# ===========================================================================
# compute_factors
# ===========================================================================

@pytest.mark.parametrize("row,expected", [
    (bonus(1, 1), Fraction(1, 2)),
    (bonus(4, 1), Fraction(1, 5)),
    (bonus(1, 3), Fraction(3, 4)),
    (bonus(2, 1), Fraction(1, 3)),
    (bonus(1, 2), Fraction(2, 3)),
    (split(2, 1), Fraction(1, 2)),
    (split(10, 1), Fraction(1, 10)),
    (split(10, 2), Fraction(1, 5)),
    (split(5, 2), Fraction(2, 5)),
], ids=lambda v: v if isinstance(v, Fraction) else None)
def test_ratio_maths(row, expected):
    out = compute_factors(actions_df(row))
    assert len(out) == 1 and bool(out["applied"].iloc[0]) is True
    assert out["factor"].iloc[0] == float(expected)


def test_combined_bonus_and_split_in_one_row_multiply():
    row = act(cats="BONUS|SPLIT", bonus=(1, 1), split=(2, 1))
    assert compute_factors(actions_df(row))["factor"].tolist() == [0.25]


def test_bajfinance_separate_rows_on_one_date_multiply_to_one_row():
    out = compute_factors(actions_df(split(2, 1, ex="2025-06-16"), bonus(4, 1, ex="2025-06-16")))
    assert len(out) == 1
    assert out["factor"].iloc[0] == 0.1 and bool(out["applied"].iloc[0])
    assert "BONUS 4:1" in out["reason"].iloc[0] and "SPLIT 2->1" in out["reason"].iloc[0]


def test_factors_are_exact_fractions_converted_once():
    # float(2/3) * float(7/8) differs from float(7/12): exact arithmetic must win
    f1, f2 = Fraction(2, 3), Fraction(7, 8)
    assert float(f1) * float(f2) != float(f1 * f2)
    out = compute_factors(actions_df(bonus(1, 2, ex="2024-01-10"), bonus(1, 7, ex="2024-01-10")))
    assert out["factor"].iloc[0] == float(Fraction(7, 12))


def test_decimal_ratio_is_exact_not_binary_float():
    out = compute_factors(actions_df(bonus(0.1, 0.9)))        # 0.9 / (0.1 + 0.9) = 0.9
    assert out["factor"].iloc[0] == 0.9


def test_output_schema_sorting_and_dtypes():
    out = compute_factors(actions_df(bonus(1, 1, ex="2024-10-28"), bonus(1, 1, ex="2017-09-07")))
    assert list(out.columns) == list(FACTOR_COLUMNS)
    assert {c: str(t) for c, t in out.dtypes.items()} == FACTOR_COLUMNS
    assert out["ex_date"].is_monotonic_increasing and len(out) == 2


def test_string_ex_dates_are_accepted():
    df = actions_df(bonus(1, 1))
    df["ex_date"] = "28-Oct-2024"
    assert compute_factors(df)["ex_date"].iloc[0] == pd.Timestamp("2024-10-28")


# -- duplicates ----------------------------------------------------------------

def test_duplicates_with_different_whitespace_count_once(caplog):
    rows = [bonus(1, 1, detail=" Bonus 1:1"), bonus(1, 1, detail="Bonus 1:1"),
            bonus(1, 1, detail="Bonus 1:1  ")]
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        out = compute_factors(actions_df(*rows))
    assert out["factor"].tolist() == [0.5]
    assert any("3" not in m and "2 duplicate" in m for m in messages(caplog, logging.INFO))


def test_different_events_on_the_same_date_are_not_duplicates():
    out = compute_factors(actions_df(bonus(1, 1), bonus(1, 2)))
    assert out["factor"].iloc[0] == float(Fraction(1, 2) * Fraction(2, 3))


def test_duplicate_unparsed_rows_with_different_text_are_kept_apart():
    rows = [act(cats="OTHER", status="needs_review", detail="Mystery action A"),
            act(cats="OTHER", status="needs_review", detail="Mystery action B")]
    out = compute_factors(actions_df(*rows))
    assert out["reason"].iloc[0].count("OTHER") == 2


# -- real Reliance rows ----------------------------------------------------------

def test_reliance_fixture_replay(monkeypatch):
    out = compute_factors(reliance_actions(monkeypatch))
    assert out["ex_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2017-09-07", "2020-05-13", "2023-07-20", "2024-10-28"]
    assert out["factor"].tolist() == [0.5, 1.0, 1.0, 0.5]
    assert out["applied"].tolist() == [True, False, False, True]
    assert "RIGHTS" in out["reason"].iloc[1] and "DEMERGER" in out["reason"].iloc[2]


def test_reliance_fixture_warns_for_each_skipped_event(monkeypatch, caplog):
    actions = reliance_actions(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        compute_factors(actions)
    text = " ".join(messages(caplog, logging.WARNING))
    assert "2020-05-13" in text and "RIGHTS" in text
    assert "2023-07-20" in text and "DEMERGER" in text
    assert len(messages(caplog, logging.WARNING)) == 2     # dividends and AGMs stay silent


# -- skip paths --------------------------------------------------------------------

def test_rights_are_skipped_with_a_warning(caplog):
    row = act(cats="RIGHTS", rights=(1, 15, 1247), detail="Rights 1:15 @ Premium Rs 1247")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = compute_factors(actions_df(row))
    assert out["factor"].tolist() == [1.0] and out["applied"].tolist() == [False]
    assert any("RIGHTS" in m and "Rights 1:15" in m for m in messages(caplog, logging.WARNING))


@pytest.mark.parametrize("cats", ["DEMERGER", "MERGER", "SCHEME_OF_ARRANGEMENT", "OTHER"])
def test_structural_and_unknown_actions_are_skipped(cats):
    out = compute_factors(actions_df(act(cats=cats, status="needs_review")))
    assert out["factor"].tolist() == [1.0] and out["applied"].tolist() == [False]
    assert cats in out["reason"].iloc[0]


def test_needs_review_is_never_applied_even_with_numbers_present():
    out = compute_factors(actions_df(act(cats="BONUS", status="needs_review", bonus=(1, 1))))
    assert out["factor"].tolist() == [1.0] and out["applied"].tolist() == [False]
    assert "needs_review" in out["reason"].iloc[0]


def test_needs_review_row_with_only_non_price_categories_still_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = compute_factors(actions_df(act(cats="DIVIDEND", status="needs_review", detail="odd")))
    assert out["applied"].tolist() == [False]
    assert messages(caplog, logging.WARNING)


def test_missing_ex_date_is_skipped_with_a_warning(caplog):
    df = actions_df(bonus(1, 1), bonus(1, 1, ex="2024-10-28"))
    df.loc[0, "ex_date"] = pd.NaT
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = compute_factors(df)
    assert len(out) == 1 and out["factor"].tolist() == [0.5]
    assert any("without an ex-date" in m for m in messages(caplog, logging.WARNING))


def test_missing_symbol_row_is_skipped_with_a_warning(caplog):
    df = actions_df(bonus(1, 1), bonus(1, 1, ex="2020-01-02"))
    df.loc[1, "symbol"] = None
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = compute_factors(df)
    assert out["ex_date"].tolist() == [pd.Timestamp("2024-10-28")]
    assert any("without a symbol" in m for m in messages(caplog, logging.WARNING))


# -- P1: bonus plus rights on one date ----------------------------------------------

def test_p1_bonus_and_rights_in_one_row_apply_bonus_skip_rights():
    row = act(cats="BONUS|RIGHTS", bonus=(1, 1), rights=(1, 15, 1247))
    out = compute_factors(actions_df(row))
    assert out["factor"].tolist() == [0.5] and out["applied"].tolist() == [True]
    reason = out["reason"].iloc[0]
    assert "applied: BONUS 1:1" in reason and "skipped: RIGHTS" in reason


def test_p1_bonus_and_rights_in_separate_rows():
    rights = act(cats="RIGHTS", rights=(1, 15, 1247))
    out = compute_factors(actions_df(bonus(1, 1), rights))
    assert len(out) == 1 and out["factor"].tolist() == [0.5] and out["applied"].tolist() == [True]
    assert "RIGHTS" in out["reason"].iloc[0]


# -- P2: invalid ratios ----------------------------------------------------------------

@pytest.mark.parametrize("new,held", [(0, 1), (1, 0), (-1, 1), (1, -1), (NAN, 1), (1, NAN),
                                      (float("inf"), 1), (1, float("inf"))])
def test_p2_invalid_bonus_ratio_is_skipped_never_one(new, held):
    out = compute_factors(actions_df(act(cats="BONUS", bonus=(new, held))))
    assert out["applied"].tolist() == [False] and out["factor"].tolist() == [1.0]
    assert "invalid ratio" in out["reason"].iloc[0] and "skipped" in out["reason"].iloc[0]


@pytest.mark.parametrize("old,new", [(1, 2), (1, 1), (0, 1), (2, 0), (-2, 1), (NAN, 1), (2, NAN)])
def test_p2_invalid_split_is_skipped_never_one(old, new):
    out = compute_factors(actions_df(act(cats="SPLIT", split=(old, new))))
    assert out["applied"].tolist() == [False]
    assert "invalid face values" in out["reason"].iloc[0]


def test_p2_valid_part_survives_next_to_an_invalid_part():
    row = act(cats="BONUS|SPLIT", bonus=(1, 1), split=(1, 2))
    out = compute_factors(actions_df(row))
    assert out["factor"].tolist() == [0.5] and out["applied"].tolist() == [True]
    assert "skipped: SPLIT" in out["reason"].iloc[0]


# -- P5: non-price dates --------------------------------------------------------------

@pytest.mark.parametrize("cats", ["DIVIDEND", "BUYBACK", "AGM", "EVOTING"])
def test_p5_non_price_actions_are_omitted(cats):
    assert compute_factors(actions_df(act(cats=cats))).empty


def test_p5_dividend_on_a_bonus_date_does_not_change_the_row():
    out = compute_factors(actions_df(dividend(ex="2024-10-28"), bonus(1, 1)))
    assert len(out) == 1 and out["factor"].tolist() == [0.5]
    assert "DIVIDEND" not in out["reason"].iloc[0]


def test_p5_empty_inputs_give_an_empty_typed_frame():
    for df in (pd.DataFrame(), actions_df(bonus(1, 1)).iloc[0:0]):
        out = compute_factors(df)
        assert out.empty and {c: str(t) for c, t in out.dtypes.items()} == FACTOR_COLUMNS


# -- errors and purity --------------------------------------------------------------------

def test_missing_columns_raise_value_error():
    df = actions_df(bonus(1, 1)).drop(columns=["parse_status"])
    with pytest.raises(ValueError, match="parse_status"):
        compute_factors(df)


@pytest.mark.parametrize("bad", [None, [1, 2], {"a": 1}, "text", 5])
def test_non_dataframe_raises_type_error(bad):
    with pytest.raises(TypeError):
        compute_factors(bad)
    with pytest.raises(TypeError):
        adjust_prices(raw_prices("2024-01-01", "2024-01-05"), bad)
    with pytest.raises(TypeError):
        adjust_prices(bad, actions_df(bonus(1, 1)))


def test_p6_mixed_symbols_raise_value_error():
    with pytest.raises(ValueError, match="single symbol"):
        compute_factors(actions_df(bonus(1, 1, symbol="AAA"), bonus(1, 1, symbol="BBB")))


def test_p6_symbol_case_and_spaces_are_one_symbol():
    out = compute_factors(actions_df(bonus(1, 1, symbol="reliance"),
                                     bonus(1, 2, symbol=" RELIANCE ", ex="2020-01-02")))
    assert len(out) == 2


def test_mixed_symbols_detected_even_when_one_has_only_dividends():
    with pytest.raises(ValueError):
        compute_factors(actions_df(bonus(1, 1, symbol="AAA"), dividend(symbol="BBB")))


def test_compute_factors_does_not_mutate_its_input():
    df = actions_df(bonus(1, 1), split(2, 1, ex="2020-01-02"), dividend())
    before = df.copy(deep=True)
    compute_factors(df)
    pd.testing.assert_frame_equal(df, before)


# ===========================================================================
# adjust_prices: scaling
# ===========================================================================

def test_bonus_scales_rows_before_the_ex_date_and_leaves_the_rest():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    before = px["date"] < pd.Timestamp("2024-10-28")
    for col in ("open", "high", "low", "close"):
        assert (out.loc[before, col] == px.loc[before, col] * 0.5).all()
        assert (out.loc[~before, col] == px.loc[~before, col]).all()
    assert (out.loc[before, "volume"] == px.loc[before, "volume"] * 2).all()
    assert (out.loc[~before, "volume"] == px.loc[~before, "volume"]).all()


def test_the_ex_date_row_itself_is_untouched():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    row = px["date"] == pd.Timestamp("2024-10-28")
    pd.testing.assert_frame_equal(out[row], px[row])


def test_adjusted_series_is_continuous():
    px = raw_prices("2024-10-01", "2024-11-30", events=[("2024-10-28", 0.5)])
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert np.allclose(out["close"], 50.0) and (out["volume"] == 2000).all()


def test_price_times_volume_is_preserved_before_the_ex_date():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    before = px["date"] < pd.Timestamp("2024-10-28")
    assert np.allclose(out.loc[before, "close"] * out.loc[before, "volume"],
                       px.loc[before, "close"] * px.loc[before, "volume"])


def test_chained_events_multiply_for_rows_before_both():
    events = [("2017-09-07", 0.5), ("2024-10-28", 0.5)]
    px = raw_prices("2017-01-02", "2025-03-31", events=events)
    out = adjust_prices(px, actions_df(bonus(1, 1, ex="2017-09-07"), bonus(1, 1, ex="2024-10-28")))
    assert np.allclose(out["close"], 25.0)
    first = px["date"] < pd.Timestamp("2017-09-07")
    middle = (px["date"] >= pd.Timestamp("2017-09-07")) & (px["date"] < pd.Timestamp("2024-10-28"))
    assert np.allclose(out.loc[first, "close"] / px.loc[first, "close"], 0.25)
    assert np.allclose(out.loc[middle, "close"] / px.loc[middle, "close"], 0.5)


@pytest.mark.filterwarnings("ignore:adjusted prices fail the continuity check")
def test_cumulative_factor_uses_exact_fractions():
    f1, f2 = Fraction(2, 3), Fraction(7, 8)
    assert float(f1) * float(f2) != float(f1 * f2)
    px = raw_prices("2024-01-01", "2024-03-29", events=[("2024-02-01", 2 / 3), ("2024-03-01", 7 / 8)])
    px["close"] = 1.0
    out = adjust_prices(px, actions_df(bonus(1, 2, ex="2024-02-01"), bonus(1, 7, ex="2024-03-01")))
    first = px["date"] < pd.Timestamp("2024-02-01")
    assert (out.loc[first, "close"] == float(Fraction(7, 12))).all()


@pytest.mark.filterwarnings("ignore:adjusted prices fail the continuity check")
def test_ohlc_are_not_rounded():
    px = raw_prices("2024-10-21", "2024-11-01")
    px.loc[0, "close"] = 100.33
    out = adjust_prices(px, actions_df(bonus(2, 1, ex="2024-10-29")))   # factor 1/3
    assert out.loc[0, "close"] == 100.33 * float(Fraction(1, 3))


@pytest.mark.filterwarnings("ignore:adjusted prices fail the continuity check")
def test_volume_rounds_half_to_even():
    px = raw_prices("2024-10-21", "2024-11-01")
    px["volume"] = np.int64(1)
    px.loc[px.index[:5], "volume"] = np.array([1, 3, 5, 7, 9], dtype="int64")
    out = adjust_prices(px, actions_df(split(5, 2, ex="2024-10-28")))      # factor 0.4
    assert out["volume"].iloc[:5].tolist() == [2, 8, 12, 18, 22]   # 2.5, 7.5, 12.5, 17.5, 22.5
    assert out["volume"].iloc[5:].tolist() == [1] * 5               # on/after the ex-date: untouched


@pytest.mark.filterwarnings("ignore:adjusted prices fail the continuity check")
def test_volume_stays_exact_for_untouched_rows_even_when_huge():
    px = raw_prices("2024-10-21", "2024-11-01")
    px["volume"] = np.where(px["date"] < pd.Timestamp("2024-10-23"), 100, 2 ** 60 + 1).astype("int64")
    out = adjust_prices(px, actions_df(bonus(1, 1, ex="2024-10-23")))
    assert out["volume"].iloc[:2].tolist() == [200, 200]                  # scaled
    assert (out["volume"].iloc[2:] == 2 ** 60 + 1).all()                  # not routed through float


def test_adjusted_volume_that_cannot_be_exact_fails_safe(caplog):
    px = raw_prices("2024-10-21", "2024-11-01")
    px["volume"] = np.int64(2 ** 52)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = adjust_prices(px, actions_df(split(10, 1, ex="2024-10-28")))
    assert out.empty and list(out.columns) == list(SCHEMAS["equity"])
    assert messages(caplog, logging.ERROR)


def test_other_columns_and_index_and_row_order_are_preserved():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    px["note"] = "keep me"
    px = px.iloc[::-1].copy()                      # reversed row order
    px.index = np.arange(500, 500 + len(px))       # non-default index
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.index.equals(px.index)
    assert out["date"].tolist() == px["date"].tolist()
    assert list(out.columns) == list(px.columns)
    for col in ("trades", "symbol", "source", "note"):
        assert out[col].tolist() == px[col].tolist()


def test_output_dtypes_match_the_equity_schema_even_for_microsecond_dates():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    px["date"] = px["date"].astype("datetime64[us]")            # what pandas 3 gives today
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    for col, dtype in SCHEMAS["equity"].items():
        assert str(out[col].dtype) == dtype, col


def test_adjusted_output_passes_validate():
    px = raw_prices("2024-10-01", "2024-11-29", events=[("2024-10-28", 0.5)])
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert validate(out, "equity").errors == []


def test_inputs_are_never_mutated():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    acts = actions_df(bonus(1, 1), dividend())
    px_before, acts_before = px.copy(deep=True), acts.copy(deep=True)
    adjust_prices(px, acts)
    pd.testing.assert_frame_equal(px, px_before)
    pd.testing.assert_frame_equal(acts, acts_before)


def test_output_is_a_new_object_even_when_nothing_changes():
    px = raw_prices("2024-10-21", "2024-11-01")
    out = adjust_prices(px, actions_df(dividend()))
    assert out is not px
    out.loc[out.index[0], "close"] = -1
    assert px["close"].iloc[0] != -1


# ---------------------------------------------------------------------------
# adjust_prices: events outside the data, duplicates, P1, P3
# ---------------------------------------------------------------------------

def test_p3_event_after_the_last_price_row_is_not_applied(caplog):
    px = raw_prices("2024-10-21", "2024-11-01")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = adjust_prices(px, actions_df(bonus(1, 1, ex="2025-01-15")))
    pd.testing.assert_frame_equal(out, px)
    assert any("after the last price row" in m for m in messages(caplog, logging.INFO))


def test_p3_future_event_does_not_trigger_a_user_warning():
    px = raw_prices("2024-10-21", "2024-11-01")
    ctx = no_user_warnings()
    try:
        adjust_prices(px, actions_df(bonus(1, 1, ex="2025-01-15")))
    finally:
        ctx.__exit__(None, None, None)


def test_p3_event_exactly_on_the_last_price_date_is_applied():
    px = raw_prices("2024-10-21", "2024-10-25", events=[("2024-10-25", 0.5)])
    out = adjust_prices(px, actions_df(bonus(1, 1, ex="2024-10-25")))
    assert np.allclose(out["close"], 50.0)


def test_events_before_or_on_the_first_price_date_have_no_effect():
    px = raw_prices("2024-10-21", "2024-11-01")
    acts = actions_df(bonus(1, 1, ex="2020-01-02"), bonus(1, 1, ex="2024-10-21"))
    pd.testing.assert_frame_equal(adjust_prices(px, acts), px)


def test_duplicate_actions_are_applied_once():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    acts = actions_df(bonus(1, 1, detail=" Bonus 1:1"), bonus(1, 1, detail="Bonus 1:1"))
    assert np.allclose(adjust_prices(px, acts)["close"], 50.0)


def test_p1_bonus_applied_and_rights_skipped_on_one_date():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    rights = act(cats="RIGHTS", rights=(1, 15, 1247))
    with pytest.warns(UserWarning, match="RIGHTS"):
        out = adjust_prices(px, actions_df(bonus(1, 1), rights))
    assert np.allclose(out["close"], 50.0)            # bonus applied


# ---------------------------------------------------------------------------
# P4: skipped price-affecting events
# ---------------------------------------------------------------------------

def test_p4_skipped_event_inside_the_range_warns_and_leaves_prices_alone(caplog):
    px = raw_prices("2024-10-21", "2024-11-01")
    rights = act(cats="RIGHTS", rights=(1, 15, 1247), ex="2024-10-28", detail="Rights 1:15")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.warns(UserWarning, match=r"NOT applied.*TEST 2024-10-28.*RIGHTS"):
            out = adjust_prices(px, actions_df(rights))
    pd.testing.assert_frame_equal(out, px)
    assert any("NOT applied" in m for m in messages(caplog, logging.WARNING))


def test_p4_only_one_user_warning_per_call_even_with_several_skips():
    px = raw_prices("2024-10-21", "2024-11-12")
    acts = actions_df(act(cats="DEMERGER", status="needs_review", ex="2024-10-28"),
                      act(cats="MERGER", status="needs_review", ex="2024-11-05"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adjust_prices(px, acts)
    mine = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(mine) == 1
    assert "DEMERGER" in str(mine[0].message) and "MERGER" in str(mine[0].message)


def test_p4_skipped_event_outside_the_price_range_is_silent():
    px = raw_prices("2024-10-21", "2024-11-01")
    acts = actions_df(act(cats="DEMERGER", status="needs_review", ex="2023-07-20"),
                      act(cats="RIGHTS", rights=(1, 15, 1), ex="2025-03-03"))
    ctx = no_user_warnings()
    try:
        out = adjust_prices(px, acts)
    finally:
        ctx.__exit__(None, None, None)
    pd.testing.assert_frame_equal(out, px)


def test_p4_invalid_ratio_inside_the_range_warns():
    px = raw_prices("2024-10-21", "2024-11-01")
    with pytest.warns(UserWarning, match="invalid ratio"):
        out = adjust_prices(px, actions_df(act(cats="BONUS", bonus=(0, 1))))
    pd.testing.assert_frame_equal(out, px)


def test_p4_action_without_ex_date_warns_for_a_symbol_in_the_prices():
    px = raw_prices("2024-10-21", "2024-11-01")
    df = actions_df(bonus(1, 1))
    df.loc[0, "ex_date"] = pd.NaT
    with pytest.warns(UserWarning, match="no ex-date"):
        out = adjust_prices(px, df)
    pd.testing.assert_frame_equal(out, px)


def test_p4_undated_action_of_another_symbol_is_ignored():
    px = raw_prices("2024-10-21", "2024-11-01")
    df = actions_df(bonus(1, 1, symbol="OTHER"))
    df.loc[0, "ex_date"] = pd.NaT
    ctx = no_user_warnings()
    try:
        adjust_prices(px, df)
    finally:
        ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# P6: symbols
# ---------------------------------------------------------------------------

def test_p6_actions_are_matched_per_symbol_in_multi_symbol_prices():
    a = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)], symbol="AAA")
    b = raw_prices("2024-10-21", "2024-11-01", symbol="BBB")
    px = pd.concat([a, b], ignore_index=True)
    out = adjust_prices(px, actions_df(bonus(1, 1, symbol="AAA")))
    assert np.allclose(out.loc[out["symbol"] == "AAA", "close"], 50.0)
    pd.testing.assert_frame_equal(out[out["symbol"] == "BBB"], px[px["symbol"] == "BBB"])


def test_p6_each_symbol_gets_its_own_events():
    a = raw_prices("2024-01-01", "2024-12-31", events=[("2024-03-01", 0.5)], symbol="AAA")
    b = raw_prices("2024-01-01", "2024-12-31", events=[("2024-09-02", 0.2)], symbol="BBB")
    px = pd.concat([a, b], ignore_index=True)
    out = adjust_prices(px, actions_df(bonus(1, 1, symbol="AAA", ex="2024-03-01"),
                                       bonus(4, 1, symbol="BBB", ex="2024-09-02")))
    assert np.allclose(out.loc[out["symbol"] == "AAA", "close"], 50.0)
    assert np.allclose(out.loc[out["symbol"] == "BBB", "close"], 20.0)


def test_p6_symbol_match_ignores_case_and_spaces():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)], symbol="RELIANCE")
    out = adjust_prices(px, actions_df(bonus(1, 1, symbol=" reliance ")))
    assert np.allclose(out["close"], 50.0)


def test_p6_actions_for_symbols_not_in_the_prices_are_ignored():
    px = raw_prices("2024-10-21", "2024-11-01")
    out = adjust_prices(px, actions_df(bonus(1, 1, symbol="NOPE")))
    pd.testing.assert_frame_equal(out, px)


def test_p6_prices_without_actions_for_their_symbol_come_back_unchanged():
    px = raw_prices("2024-10-21", "2024-11-01", symbol="AAA")
    pd.testing.assert_frame_equal(adjust_prices(px, actions_df(dividend(symbol="AAA"))), px)


# ---------------------------------------------------------------------------
# P7: continuity sanity check
# ---------------------------------------------------------------------------

def test_p7_correct_raw_prices_do_not_warn_or_log(caplog):
    px = raw_prices("2024-10-01", "2024-11-29", events=[("2024-10-28", 0.5)])
    ctx = no_user_warnings()
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            adjust_prices(px, actions_df(bonus(1, 1)))
    finally:
        ctx.__exit__(None, None, None)
    assert not messages(caplog, logging.WARNING)


def test_p7_already_adjusted_prices_warn_but_are_still_scaled(caplog):
    px = raw_prices("2024-10-01", "2024-11-29")                   # smooth: already adjusted
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.warns(UserWarning, match=r"continuity.*TEST 2024-10-28.*0\.5"):
            out = adjust_prices(px, actions_df(bonus(1, 1)))
    before = px["date"] < pd.Timestamp("2024-10-28")
    assert (out.loc[before, "close"] == px.loc[before, "close"] * 0.5).all()   # data not altered by the check
    assert any("Sanity check failed" in m for m in messages(caplog, logging.WARNING))


def test_p7_wrong_ratio_warns():
    px = raw_prices("2024-10-01", "2024-11-29", events=[("2024-10-28", 0.5)])
    with pytest.warns(UserWarning, match="continuity"):
        adjust_prices(px, actions_df(bonus(4, 1)))                # claims 0.2, data fell by 0.5


@pytest.mark.parametrize("ratio_close,warns", [(0.72, False), (1.38, False), (0.69, True), (1.45, True)])
def test_p7_threshold_edges(ratio_close, warns):
    # previous close 100, ex-date close 100 / ratio_close, adjusted previous = 100 -> ratio = ratio_close
    px = raw_prices("2024-10-24", "2024-10-31")
    px["close"] = np.where(px["date"] >= pd.Timestamp("2024-10-28"), 100.0 / ratio_close, 100.0)
    # factor 1.0 is impossible, so use a tiny split and compare with the real ratio instead
    acts = actions_df(split(10_000_000, 9_999_999, ex="2024-10-28"))
    ctx = warnings.catch_warnings(record=True)
    caught = ctx.__enter__()
    warnings.simplefilter("always")
    try:
        adjust_prices(px, acts)
    finally:
        ctx.__exit__(None, None, None)
    got = any(issubclass(w.category, UserWarning) and "continuity" in str(w.message) for w in caught)
    assert got is warns


def test_p7_check_needs_rows_on_both_sides_of_the_ex_date():
    px = raw_prices("2024-10-28", "2024-11-01")                   # nothing before the ex-date
    ctx = no_user_warnings()
    try:
        adjust_prices(px, actions_df(bonus(1, 1, ex="2024-10-28")))
    finally:
        ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Real data: the four live probe cases (numbers from the 2026-11-17 NSE probe)
# ---------------------------------------------------------------------------

PROBE = [
    ("RELIANCE", "2024-10-28", [(2024, 10, 23, 2677.05), (2024, 10, 24, 2679.60), (2024, 10, 25, 2655.70)],
     [(2024, 10, 28, 1334.35), (2024, 10, 29, 1340.00)], [bonus(1, 1, ex="2024-10-28", symbol="RELIANCE")], 0.5),
    ("HDFCBANK", "2019-09-19", [(2019, 9, 16, 2244.15), (2019, 9, 17, 2211.35), (2019, 9, 18, 2187.75)],
     [(2019, 9, 19, 1101.05), (2019, 9, 20, 1199.60)], [split(2, 1, ex="2019-09-19", symbol="HDFCBANK")], 0.5),
    ("INFY", "2018-09-04", [(2018, 8, 30, 1416.20), (2018, 8, 31, 1441.10), (2018, 9, 3, 1434.25)],
     [(2018, 9, 4, 737.15), (2018, 9, 5, 729.90)], [bonus(1, 1, ex="2018-09-04", symbol="INFY")], 0.5),
    ("BAJFINANCE", "2025-06-16", [(2025, 6, 11, 9425.0), (2025, 6, 12, 9368.5), (2025, 6, 13, 9331.0)],
     [(2025, 6, 16, 938.0), (2025, 6, 17, 923.0)],
     [split(2, 1, ex="2025-06-16", symbol="BAJFINANCE"), bonus(4, 1, ex="2025-06-16", symbol="BAJFINANCE")], 0.1),
]


@pytest.mark.parametrize("symbol,ex,before,after,rows,factor", PROBE, ids=[p[0] for p in PROBE])
def test_probe_cases_adjust_to_a_continuous_series_without_warnings(symbol, ex, before, after, rows, factor):
    data = [(pd.Timestamp(y, m, d), c) for y, m, d, c in before + after]
    px = pd.DataFrame({"date": [d for d, _ in data], "open": [c for _, c in data],
                       "high": [c for _, c in data], "low": [c for _, c in data],
                       "close": [c for _, c in data], "volume": 1_000_000, "trades": 1,
                       "symbol": symbol, "source": "NSE"})
    ctx = no_user_warnings()                      # the sanity check must accept real data
    try:
        out = adjust_prices(px, actions_df(*rows))
    finally:
        ctx.__exit__(None, None, None)
    n_before = len(before)
    assert np.allclose(out["close"].iloc[:n_before], px["close"].iloc[:n_before] * factor)
    assert (out["close"].iloc[n_before:] == px["close"].iloc[n_before:]).all()
    ratio = out["close"].iloc[n_before - 1] / out["close"].iloc[n_before]
    assert 0.95 < ratio < 1.05                    # real market moves (INFY +2.8%) fit the band
    assert out["volume"].iloc[0] == round(1_000_000 / factor)


def test_reliance_replay_with_real_actions_scales_by_era(monkeypatch):
    acts = reliance_actions(monkeypatch)
    events = [("2017-09-07", 0.5), ("2024-10-28", 0.5)]
    px = raw_prices("2016-01-04", "2025-12-31", events=events, symbol="RELIANCE")
    with pytest.warns(UserWarning) as caught:
        out = adjust_prices(px, acts)
    text = " ".join(str(w.message) for w in caught)
    assert "RIGHTS" in text and "DEMERGER" in text and "2020-05-13" in text and "2023-07-20" in text
    era1 = px["date"] < pd.Timestamp("2017-09-07")
    era2 = (px["date"] >= pd.Timestamp("2017-09-07")) & (px["date"] < pd.Timestamp("2024-10-28"))
    era3 = px["date"] >= pd.Timestamp("2024-10-28")
    assert np.allclose(out.loc[era1, "close"] / px.loc[era1, "close"], 0.25)
    assert np.allclose(out.loc[era2, "close"] / px.loc[era2, "close"], 0.5)
    assert (out.loc[era3, "close"] == px.loc[era3, "close"]).all()
    assert np.allclose(out["close"], 25.0)                         # continuous end to end
    assert validate(out, "equity").errors == []


def test_reliance_replay_over_a_window_without_skipped_events_is_silent(monkeypatch):
    acts = reliance_actions(monkeypatch)
    px = raw_prices("2024-10-01", "2024-11-29", events=[("2024-10-28", 0.5)], symbol="RELIANCE")
    ctx = no_user_warnings()
    try:
        out = adjust_prices(px, acts)
    finally:
        ctx.__exit__(None, None, None)
    assert np.allclose(out["close"], 50.0)


# ---------------------------------------------------------------------------
# Fail-safe behaviour and edge inputs
# ---------------------------------------------------------------------------

def test_empty_or_columnless_actions_return_prices_unchanged():
    px = raw_prices("2024-10-21", "2024-11-01")
    for acts in (pd.DataFrame(), actions_df(bonus(1, 1)).iloc[0:0], pd.DataFrame(index=[0, 1])):
        out = adjust_prices(px, acts)
        pd.testing.assert_frame_equal(out, px)


def test_actions_missing_parsed_columns_fail_safe_not_unadjusted(caplog):
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    acts = actions_df(bonus(1, 1)).drop(columns=["bonus_new", "parse_status"])
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = adjust_prices(px, acts)
    assert out.empty and list(out.columns) == list(SCHEMAS["equity"])
    assert any("bonus_new" in m for m in messages(caplog, logging.ERROR))


@pytest.mark.parametrize("drop", ["date", "open", "high", "low", "close", "volume", "symbol"])
def test_prices_missing_a_required_column_fail_safe(drop, caplog):
    px = raw_prices("2024-10-21", "2024-11-01").drop(columns=[drop])
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.empty and list(out.columns) == list(SCHEMAS["equity"])
    assert any(drop in m for m in messages(caplog, logging.ERROR))


def test_empty_prices_with_the_right_columns_come_back_empty():
    px = raw_prices("2024-10-21", "2024-11-01").iloc[0:0]
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.empty and list(out.columns) == list(px.columns)
    assert str(out["volume"].dtype) == "int64" and str(out["date"].dtype) == "datetime64[ns]"


def test_empty_standard_equity_frame_is_accepted():
    out = adjust_prices(empty_frame("equity"), actions_df(bonus(1, 1)))
    assert out.empty and list(out.columns) == list(SCHEMAS["equity"])


def test_missing_dates_fail_safe(caplog):
    px = raw_prices("2024-10-21", "2024-11-01")
    px.loc[2, "date"] = pd.NaT
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.empty and messages(caplog, logging.ERROR)


@pytest.mark.parametrize("bad", [np.nan, 12.5, "abc"])
def test_bad_volume_fails_safe(bad, caplog):
    px = raw_prices("2024-10-21", "2024-11-01")
    px["volume"] = px["volume"].astype("object")
    px.loc[1, "volume"] = bad
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.empty and messages(caplog, logging.ERROR)


def test_non_numeric_price_fails_safe(caplog):
    px = raw_prices("2024-10-21", "2024-11-01")
    px["close"] = px["close"].astype("object")
    px.loc[0, "close"] = "n/a"
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.empty


def test_nan_prices_stay_nan_and_are_not_invented():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    px.loc[1, "close"] = np.nan
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert np.isnan(out.loc[1, "close"]) and not np.isnan(out.loc[0, "close"])


def test_rows_with_a_blank_price_symbol_are_left_alone():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    px.loc[0, "symbol"] = ""
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.loc[0, "close"] == px.loc[0, "close"]
    assert out.loc[1, "close"] == px.loc[1, "close"] * 0.5


def test_logs_use_the_adjust_prefix_and_nothing_is_printed(capsys, caplog):
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        adjust_prices(px, actions_df(bonus(1, 1)))
    assert capsys.readouterr().out == ""
    assert all(m.startswith("[Adjust]") for m in messages(caplog, logging.INFO))


# ---------------------------------------------------------------------------
# Schema dtypes: trades (and every other schema column) come back in schema dtype
# ---------------------------------------------------------------------------

def _prices_with_trades(trades, **kw):
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)], **kw)
    px["trades"] = trades if not callable(trades) else trades(len(px))
    return px


def test_int32_trades_become_int64_and_pass_validate():
    """The live failure: the NSE connector returned trades as int32 on pandas 3."""
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    px["trades"] = np.arange(len(px), dtype="int32") + 100
    assert str(px["trades"].dtype) == "int32"
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert str(out["trades"].dtype) == "int64"
    assert out["trades"].tolist() == px["trades"].tolist()                 # values unchanged
    assert validate(out, "equity").errors == []


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32"])
def test_every_integer_width_of_trades_becomes_int64(dtype):
    px = _prices_with_trades(lambda n: np.arange(n, dtype=dtype) + 1)
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert str(out["trades"].dtype) == "int64"
    assert out["trades"].tolist() == px["trades"].astype("int64").tolist()


@pytest.mark.parametrize("make", [
    lambda n: np.arange(n, dtype="float64") + 5.0,                    # whole floats
    lambda n: pd.Series([str(i + 7) for i in range(n)], dtype="object"),   # numeric text
    lambda n: pd.Series(np.arange(n, dtype="int64") + 3).astype("object"),
], ids=["float64", "text", "object_int"])
def test_whole_valued_trades_of_other_dtypes_convert_exactly(make):
    px = _prices_with_trades(make)
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert str(out["trades"].dtype) == "int64"
    assert out["trades"].tolist() == [int(float(v)) for v in px["trades"]]


def test_large_int64_trades_are_preserved_exactly():
    px = _prices_with_trades(lambda n: np.full(n, 2 ** 62 + 1, dtype="int64"))
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert (out["trades"] == 2 ** 62 + 1).all()


@pytest.mark.parametrize("bad", [np.nan, 12.5, "abc", np.inf], ids=["nan", "fractional", "text", "inf"])
def test_trades_that_cannot_convert_exactly_fail_safe_not_truncate(bad, caplog):
    px = _prices_with_trades(lambda n: pd.Series(np.arange(n, dtype="float64") + 1).astype("object"))
    px.loc[1, "trades"] = bad
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.empty and list(out.columns) == list(SCHEMAS["equity"])
    assert any("trades" in m for m in messages(caplog, logging.ERROR))


def test_uint64_trades_above_the_int64_range_fail_safe(caplog):
    px = _prices_with_trades(lambda n: np.full(n, 2 ** 63 + 5, dtype="uint64"))
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.empty and any("trades" in m for m in messages(caplog, logging.ERROR))


def test_trades_dtype_is_fixed_on_every_return_path():
    px = raw_prices("2024-10-21", "2024-11-01")
    px["trades"] = np.arange(len(px), dtype="int32")
    cases = {
        "no actions (empty frame)": pd.DataFrame(),
        "no actions (zero rows)": actions_df(bonus(1, 1)).iloc[0:0],
        "future event only": actions_df(bonus(1, 1, ex="2030-01-01")),
        "other symbol only": actions_df(bonus(1, 1, symbol="NOPE")),
        "applied event": actions_df(bonus(1, 1, ex="2024-10-28")),
    }
    for label, acts in cases.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out = adjust_prices(px, acts)
        assert str(out["trades"].dtype) == "int64", label
        assert out["trades"].tolist() == px["trades"].tolist(), label


def test_empty_prices_with_int32_trades_come_back_int64():
    px = raw_prices("2024-10-21", "2024-11-01").iloc[0:0].copy()
    px["trades"] = px["trades"].astype("int32")
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.empty and str(out["trades"].dtype) == "int64"


def test_prices_without_a_trades_column_are_still_accepted():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)]).drop(columns=["trades"])
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert "trades" not in out.columns and np.allclose(out["close"], 50.0)


def test_every_schema_column_gets_its_schema_dtype_from_awkward_inputs():
    """pandas-3 style input: us dates, float32 prices, int32 volume/trades, str symbol/source."""
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    px = px.astype({"date": "datetime64[us]", "open": "float32", "high": "float32",
                    "low": "float32", "close": "float32", "volume": "int32",
                    "trades": "int32", "symbol": "str", "source": "str"})
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    for col, dtype in SCHEMAS["equity"].items():
        assert str(out[col].dtype) == dtype, col
    assert validate(out, "equity").errors == []


def test_dtype_fix_keeps_index_order_values_and_extra_columns():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    px["trades"] = np.arange(len(px), dtype="int32")
    px["note"] = np.arange(len(px), dtype="int16")                # an extra column: dtype must stay
    px = px.iloc[::-1].copy()
    px.index = np.arange(900, 900 + len(px))
    out = adjust_prices(px, actions_df(bonus(1, 1)))
    assert out.index.equals(px.index) and out["date"].tolist() == px["date"].tolist()
    assert list(out.columns) == list(px.columns)
    assert out["trades"].tolist() == px["trades"].tolist()
    assert out["note"].tolist() == px["note"].tolist() and str(out["note"].dtype) == "int16"


def test_the_input_trades_column_itself_is_never_changed():
    px = raw_prices("2024-10-21", "2024-11-01", events=[("2024-10-28", 0.5)])
    px["trades"] = np.arange(len(px), dtype="int32")
    adjust_prices(px, actions_df(bonus(1, 1)))
    assert str(px["trades"].dtype) == "int32"


def test_source_has_no_print_and_no_old_project_name():
    import re
    source = Path(adjust.__file__).read_text(encoding="utf-8")
    assert not re.search(r"(?<![\w.])print\(", source)
    assert ("py" + "capital") not in source.lower()
    assert all(ord(ch) < 128 for ch in source)

def test_negative_volume_fails_safe(caplog):
    px = raw_prices("2024-10-21", "2024-11-01")
    px["volume"] = -5

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        out = adjust_prices(px, actions_df(bonus(1, 1)))

    assert out.empty
    assert list(out.columns) == list(SCHEMAS["equity"])
    assert messages(caplog, logging.ERROR)