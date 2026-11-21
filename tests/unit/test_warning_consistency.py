"""Warning consistency across PyEquity (Step 4, Part 5).

Every warning the library issues belongs to the shared hierarchy in
``pyequity.api._errors``, which derives from ``UserWarning`` so that code that
filters or catches ``UserWarning`` keeps working.
"""
import ast
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyequity.api import _errors
from pyequity.api._errors import (
    DataQualityWarning, EmptyResultWarning, FallbackWarning,
    IncompleteDataWarning, PyEquityWarning)
from pyequity.processing.adjust import adjust_prices

SRC = Path(__file__).resolve().parents[2] / "src" / "pyequity"
NAN = float("nan")


def _hierarchy():
    return {name: cls for name, cls in vars(_errors).items()
            if inspect.isclass(cls) and issubclass(cls, Warning)}


# ------------------------------ hierarchy ------------------------------------

def test_every_warning_class_derives_from_pyequity_warning_and_user_warning():
    classes = _hierarchy()
    assert set(classes) == {"PyEquityWarning", "FallbackWarning", "IncompleteDataWarning",
                            "EmptyResultWarning", "DataQualityWarning"}
    for cls in classes.values():
        assert issubclass(cls, PyEquityWarning) and issubclass(cls, UserWarning)


def test_no_warning_class_is_defined_outside_the_shared_module():
    for path in SRC.rglob("*.py"):
        if path.name == "_errors.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
                assert not any(b.endswith("Warning") for b in bases), \
                    f"{path.relative_to(SRC)}: {node.name} duplicates the shared hierarchy"


def test_every_warnings_warn_in_src_uses_a_shared_category():
    allowed = set(_hierarchy())
    found = 0
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "warn"
                    and getattr(node.func.value, "id", "") == "warnings"):
                continue
            found += 1
            category = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg == "category"), None)
            name = getattr(category, "id", getattr(category, "attr", None))
            shown = name or "the default UserWarning"
            where = f"{path.relative_to(SRC)}:{node.lineno}"
            assert name in allowed, (
                f"{where} warns with {shown}; use a class from pyequity.api._errors")
    assert found > 10  # the audit really looked at the library's warnings


# ------------------------------ adjust_prices --------------------------------

def act(cats, ex="2024-10-28", status="ok", bonus=(NAN, NAN), rights=(NAN, NAN, NAN),
        detail=""):
    return {"ex_date": pd.Timestamp(ex), "symbol": "TEST", "action_type": "",
            "action_detail": detail, "record_date": pd.NaT, "source": "NSE",
            "isin": "INE000000000", "action_categories": cats, "bonus_new": bonus[0],
            "bonus_held": bonus[1], "split_old_fv": NAN, "split_new_fv": NAN,
            "rights_new": rights[0], "rights_held": rights[1], "rights_premium": rights[2],
            "parse_status": status}


def bonus_1_1():
    return act("BONUS", bonus=(1, 1), detail="Bonus 1:1")


def rights():
    return act("RIGHTS", rights=(1, 15, 1247), detail="Rights 1:15")


def raw_prices(start="2024-10-01", end="2024-11-29", halve_on=None):
    days = pd.bdate_range(start, end)
    factor = np.where(days >= pd.Timestamp(halve_on), 0.5, 1.0) if halve_on else np.ones(len(days))
    close = 100.0 * factor
    return pd.DataFrame({
        "date": days.astype("datetime64[ns]"), "open": close * 1.01, "high": close * 1.02,
        "low": close * 0.98, "close": close,
        "volume": np.rint(1000 / factor).astype("int64"),
        "trades": np.arange(len(days), dtype="int64") + 10,
        "symbol": "TEST", "source": "NSE"}).astype({"symbol": object, "source": object})


def run(prices, *action_rows):
    """adjust_prices with every warning recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = adjust_prices(prices, pd.DataFrame(list(action_rows)))
    return out, [w for w in caught if issubclass(w.category, UserWarning)]


def test_skipped_action_warns_with_incomplete_data_warning():
    prices = raw_prices()
    out, caught = run(prices, rights())
    assert [w.category for w in caught] == [IncompleteDataWarning]
    assert "NOT applied" in str(caught[0].message) and "RIGHTS" in str(caught[0].message)
    pd.testing.assert_frame_equal(out, prices)  # a warning never changes the prices


def test_continuity_failure_warns_with_data_quality_warning():
    out, caught = run(raw_prices(), bonus_1_1())  # smooth series: already adjusted
    assert [w.category for w in caught] == [DataQualityWarning]
    assert "continuity" in str(caught[0].message)


def test_both_conditions_give_one_warning_each_in_their_own_category():
    # A skipped RIGHTS row and a bonus applied to an already-smooth series.
    out, caught = run(raw_prices(), bonus_1_1(), act("RIGHTS", ex="2024-11-12",
                                                     rights=(1, 15, 1247), detail="Rights"))
    assert sorted(w.category.__name__ for w in caught) == [
        "DataQualityWarning", "IncompleteDataWarning"]


def test_the_two_adjust_categories_can_be_filtered_independently():
    prices = raw_prices()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        with pytest.raises(IncompleteDataWarning):
            warnings.simplefilter("error", IncompleteDataWarning)
            adjust_prices(prices, pd.DataFrame([rights()]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncompleteDataWarning)
        warnings.simplefilter("error", DataQualityWarning)
        with pytest.raises(DataQualityWarning):
            adjust_prices(prices, pd.DataFrame([bonus_1_1()]))


@pytest.mark.parametrize("category", [UserWarning, PyEquityWarning])
@pytest.mark.parametrize("action", [rights, bonus_1_1])
def test_filters_written_for_userwarning_still_catch_them(category, action):
    # Backward compatibility: code that used to handle a plain UserWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", category)
        with pytest.raises(category):
            adjust_prices(raw_prices(), pd.DataFrame([action()]))
    with pytest.warns(category):
        adjust_prices(raw_prices(), pd.DataFrame([action()]))


def test_clean_adjustment_issues_no_warning_under_error_filter():
    prices = raw_prices(halve_on="2024-10-28")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = adjust_prices(prices, pd.DataFrame([bonus_1_1()]))
    assert np.allclose(out["close"], 50.0)


def test_adjust_warnings_are_distinct_from_the_fetch_warnings():
    # Callers can tell "adjustment not applied" from "no rows returned".
    assert not issubclass(IncompleteDataWarning, EmptyResultWarning)
    assert not issubclass(DataQualityWarning, FallbackWarning)
    assert not issubclass(IncompleteDataWarning, DataQualityWarning)