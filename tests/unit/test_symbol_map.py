import pytest
from pyequity.connectors.symbol_map import get_bse_code


def test_known_symbol_returns_bse_code():
    assert get_bse_code("RELIANCE") == "500325"


def test_lookup_is_case_insensitive():
    assert get_bse_code("reliance") == "500325"


def test_unknown_symbol_raises_value_error():
    with pytest.raises(ValueError):
        get_bse_code("NOTASYMBOL")