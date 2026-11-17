"""Offline tests for the corporate action text parser.

The data file tests/unit/golden/corporate_action_descriptions.csv holds the 392
distinct descriptions collected from real NSE responses (29 symbols, 2006-2026).
Spaces at the start of a line are part of the real data.
"""
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from pyequity.connectors._action_parser import (
    CATEGORY_ORDER, parse_action_detail as parse, parse_action_details,
)

DATA = Path(__file__).parent / "golden" / "corporate_action_descriptions.csv"
NAN = float("nan")


def real_descriptions():
    return pd.read_csv(DATA, keep_default_na=False)["action_detail"].tolist()


def same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


def check(text, cats, status, bonus=(NAN, NAN), split=(NAN, NAN), rights=(NAN, NAN, NAN)):
    r = parse(text)
    assert r["action_categories"] == cats, text
    assert r["parse_status"] == status, text
    assert same(r["bonus_new"], bonus[0]) and same(r["bonus_held"], bonus[1]), text
    assert same(r["split_old_fv"], split[0]) and same(r["split_new_fv"], split[1]), text
    assert same(r["rights_new"], rights[0]) and same(r["rights_held"], rights[1]), text
    assert same(r["rights_premium"], rights[2]), text


# ---------------------------------------------------------------------------
# Real descriptions, hand-checked (every non-dividend format in the data)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,bonus", [
    ("Bonus 1:1", (1, 1)),
    (" Bonus 1:1", (1, 1)),               # leading space is real NSE data
    ("Bonus 1: 1", (1, 1)),               # space after the colon (KOTAKBANK)
    ("Bonus 1:3", (1, 3)),                # POWERGRID: 1 new for 3 held
    (" Bonus 1:3", (1, 3)),
    ("Bonus 4:1", (4, 1)),                # BAJFINANCE: 4 new for 1 held
])
def test_bonus(text, bonus):
    check(text, "BONUS", "ok", bonus=bonus)


@pytest.mark.parametrize("text,old,new", [
    ("Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share", 5, 1),
    ("Face Value Split From Rs.10/- To Rs.5/-", 10, 5),
    ("Face Value Split From Rs.10/- To Rs.2/-", 10, 2),
    ("Face Value Split From Rs 10/- To Rs 2/- Per Share", 10, 2),
    ("Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share", 2, 1),
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", 10, 2),
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", 10, 1),
    (" Face Value Split (Sub-Division) - From Rs 2 Per Share To Rs 1 Per Share", 2, 1),
    (" Fv Splt Frm Rs 10 To Re 1", 10, 1),   # JSWSTEEL abbreviation
])
def test_split(text, old, new):
    check(text, "SPLIT", "ok", split=(old, new))


@pytest.mark.parametrize("text,new,held,premium", [
    ("Rights 3:25 @ Premium Rs 1799/-", 3, 25, 1799),
    ("Rights 3:19 @ Premium Rs 1090 Per Share", 3, 19, 1090),
    (" Rights 19:67 @ Premium Rs 215 Per Share", 19, 67, 215),
    (" Rights 1:14 @ Premium Rs 530/-", 1, 14, 530),
    (" Rights 1:15 @ Premium Rs 1247", 1, 15, 1247),   # RELIANCE 2020
])
def test_rights_single_tranche(text, new, held, premium):
    check(text, "RIGHTS", "ok", rights=(new, held, premium))


def test_rights_two_tranches_need_review_and_give_no_numbers():
    text = (" Rights - 4:25 Fully Paid Up Shares @ Premium Rs 500/- Per Share / "
            "2:25 Partly Paid Up Shares @ Premium Rs 605/- Per Share")
    check(text, "RIGHTS", "needs_review")


@pytest.mark.parametrize("text,split", [
    ("Bonus 1:1 / Face Value Split From Rs.10/- To Re.1/-", (10, 1)),               # TITAN 2011
    (" Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", (10, 2)),  # BAJFINANCE 2016
])
def test_bonus_and_split_in_one_description_keep_both_parts(text, split):
    check(text, "BONUS|SPLIT", "ok", bonus=(1, 1), split=split)


def test_dividend_with_hidden_bonus_icici_2017():
    # The legacy classifier labelled this DIVIDEND and lost the bonus.
    # (Revised) alone must not force a review.
    check(" Annual General Meeting/Dividend - Rs 2.50 Per Share/Bonus 1:10 (Revised)",
          "BONUS|DIVIDEND", "ok", bonus=(1, 10))


def test_bonus_debentures_are_not_a_share_bonus():
    check("Scheme Of Arrangement - Bonus Debentures 6:1", "SCHEME_OF_ARRANGEMENT", "needs_review")


@pytest.mark.parametrize("text,cats", [
    ("Demerger", "DEMERGER"),               # legacy said MERGER
    (" Scheme Of Arrangement", "SCHEME_OF_ARRANGEMENT"),
    ("Scheme Of Arrangement", "SCHEME_OF_ARRANGEMENT"),
    ("De-merger", "DEMERGER"),
    ("Merger", "MERGER"),
    ("Amalgamation", "MERGER"),
])
def test_structural_actions_are_classified_but_never_given_a_ratio(text, cats):
    check(text, cats, "needs_review")


@pytest.mark.parametrize("text", ["Buy Back", " Buyback", " Buy-Back", "Buyback"])
def test_buyback_spellings_are_one_category(text):
    check(text, "BUYBACK", "ok")


def test_agm_and_evoting_alone():
    check(" Annual General Meeting", "AGM", "ok")
    check("Annual General Meeting", "AGM", "ok")
    check("E-Voting", "EVOTING", "ok")


def test_garbled_face_value_text_is_not_guessed():
    # BHARTIARTL 2009: contains "Fv Rs10tors5". Dividend is clear, the rest is not.
    check("Agm/Div-Rs.2/Fv Rs10tors5", "DIVIDEND", "needs_review")


@pytest.mark.parametrize("text", [
    "Dividend - Rs 6 Per Share", " Dividend - Rs 7 Per Share", "Div-Rs.21.50 Per Share",
    "Agm/Div-Rs.6.25 Per Share", "Agm/Div-Re.0.75 Per Share", "Agm/Dividend - 140%",
    " Annual General Meeting/ Divdend - Re 0.50/- Per Share",     # real typo
    "Interim Dividend - Rs 12 Per Share Special Dividend - Rs 6 Per Share",
    "Dividend-Rs.8.50 Per Share (Including Special Dividend Of Rs.2/- Per Share).",
    "Interim Dividend - Rs 5 Per Share (Date And Purpose Revised)",
    "Special Interim Dividend Re.0.80 Per Share",
    "Annual General Meeting And Dividend Rs.14/- Per Share",
])
def test_dividend_shapes(text):
    check(text, "DIVIDEND", "ok")


# ---------------------------------------------------------------------------
# Safety: unknown or risky text must never be silently "ok"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Capital Reduction", "Consolidation Of Shares", "Redemption Of Preference Shares",
    "Interest Payment", "Something We Have Never Seen",
])
def test_unknown_text_needs_review(text):
    check(text, "OTHER", "needs_review")


@pytest.mark.parametrize("text", [
    "Bonus 1:1 (Subject To Approval)",     # unexplained suffix
    "Bonus 0:1", "Bonus 1:0",              # implausible
    "Bonus One For One",
])
def test_bad_bonus_text_needs_review(text):
    assert parse(text)["parse_status"] == "needs_review", text
    assert "BONUS" in parse(text)["action_categories"]


def test_reverse_split_is_not_treated_as_a_forward_split():
    r = parse("Face Value Split From Re 1 To Rs 10")
    assert r["parse_status"] == "needs_review"


def test_rights_without_premium_format_is_not_guessed():
    r = parse("Rights 1:15")
    assert r["parse_status"] == "needs_review"
    assert math.isnan(r["rights_new"])


def test_categories_come_out_in_fixed_order_whatever_the_text_order():
    a = parse("Face Value Split From Rs 10 To Rs 2/Bonus 1:1")
    b = parse("Bonus 1:1/Face Value Split From Rs 10 To Rs 2")
    assert a["action_categories"] == b["action_categories"] == "BONUS|SPLIT"
    assert CATEGORY_ORDER.index("BONUS") < CATEGORY_ORDER.index("SPLIT") < CATEGORY_ORDER.index("DIVIDEND")


@pytest.mark.parametrize("bad", [None, float("nan"), 123, 4.5, "", "   ", b"bytes", ["x"], pd.NaT])
def test_never_raises_on_odd_input(bad):
    r = parse(bad)
    assert r["action_categories"] == "OTHER"
    assert r["parse_status"] == "needs_review"


def test_parse_action_details_keeps_order_and_length():
    out = parse_action_details(["Bonus 1:1", None, "Buyback"])
    assert [o["action_categories"] for o in out] == ["BONUS", "OTHER", "BUYBACK"]


# ---------------------------------------------------------------------------
# All 392 real descriptions
# ---------------------------------------------------------------------------

def test_data_file_is_the_real_set():
    texts = real_descriptions()
    assert len(texts) == 392
    assert sum(t.startswith(" ") for t in texts) == 138
    assert len({t.strip() for t in texts}) == 364


def test_every_real_description_parses_with_a_valid_result():
    for text in real_descriptions():
        r = parse(text)
        assert r["parse_status"] in ("ok", "needs_review"), text
        cats = r["action_categories"].split("|")
        assert cats and all(c in CATEGORY_ORDER for c in cats), text
        assert cats == sorted(cats, key=CATEGORY_ORDER.index), text


def test_only_the_known_real_descriptions_need_review():
    flagged = sorted(t.strip() for t in real_descriptions() if parse(t)["parse_status"] == "needs_review")
    assert flagged == sorted([
        "Scheme Of Arrangement", "Scheme Of Arrangement",
        "Scheme Of Arrangement - Bonus Debentures 6:1",
        "Demerger",
        "Rights - 4:25 Fully Paid Up Shares @ Premium Rs 500/- Per Share / "
        "2:25 Partly Paid Up Shares @ Premium Rs 605/- Per Share",
        "Agm/Div-Rs.2/Fv Rs10tors5",
    ])


def test_keyword_invariants_hold_on_every_real_description():
    """Independent re-check with simple rules: no price-affecting keyword is lost."""
    for text in real_descriptions():
        low = re.sub(r"\s+", " ", text.lower())
        cats = parse(text)["action_categories"].split("|")
        if re.search(r"\bbonus\b(?!\s+debenture)", low):
            assert "BONUS" in cats, text
        if re.search(r"\bsplit\b|\bsplt\b|sub-?division", low):
            assert "SPLIT" in cats, text
        if re.search(r"\brights?\b", low):
            assert "RIGHTS" in cats, text
        if re.search(r"\bde-?merger\b", low):
            assert "DEMERGER" in cats and "MERGER" not in cats, text
        if re.search(r"\bbuy[\s-]?back\b", low):
            assert "BUYBACK" in cats, text


def test_numbers_exist_exactly_when_the_category_is_present_and_ok():
    for text in real_descriptions():
        r = parse(text)
        cats = r["action_categories"].split("|")
        if r["parse_status"] == "ok":
            assert (not math.isnan(r["bonus_new"])) == ("BONUS" in cats), text
            assert (not math.isnan(r["split_old_fv"])) == ("SPLIT" in cats), text
            assert (not math.isnan(r["rights_new"])) == ("RIGHTS" in cats), text
