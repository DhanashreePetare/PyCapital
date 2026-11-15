import pytest
from pyequity.connectors.symbol_map import get_bse_code, SYMBOL_TO_BSE


def test_known_symbol_returns_bse_code():
    assert get_bse_code("RELIANCE") == "500325"


def test_lookup_is_case_insensitive():
    assert get_bse_code("reliance") == "500325"


def test_unknown_symbol_raises_value_error():
    with pytest.raises(ValueError):
        get_bse_code("NOTASYMBOL")

def test_all_codes_are_six_digit_strings():
    assert all(isinstance(c, str) and c.isdigit() and len(c) == 6
               for c in SYMBOL_TO_BSE.values())

def test_no_duplicate_codes():
    codes = list(SYMBOL_TO_BSE.values())
    assert len(codes) == len(set(codes))

def test_original_mappings_unchanged():
    original = {"RELIANCE": "500325", "TCS": "532540", "HDFCBANK": "500180",
                "INFY": "500209", "ICICIBANK": "532174", "SBIN": "500112",
                "BHARTIARTL": "532454", "HINDUNILVR": "500696", "ITC": "500875",
                "BAJFINANCE": "500034", "WIPRO": "507685", "AXISBANK": "532215",
                "KOTAKBANK": "500247", "LT": "500510", "DMART": "540376"}
    for s, c in original.items():
        assert SYMBOL_TO_BSE[s] == c

def test_map_is_large():
    assert len(SYMBOL_TO_BSE) > 2000