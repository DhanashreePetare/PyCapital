"""
Shared symbol reference data for PyCapital.

Purpose:
    Maps a standard NSE stock symbol (e.g. "RELIANCE") to the
    equivalent BSE numeric scripcode (e.g. "500325").

    This lives outside any single connector because it is shared
    reference data — the future reliability layer (NSE<->BSE
    automatic fallback) and other modules (e.g. corporate actions)
    will also need this mapping, not just the BSE connector.
"""

# Maps NSE symbol -> BSE scripcode
SYMBOL_TO_BSE = {
    "RELIANCE":   "500325",
    "TCS":        "532540",
    "HDFCBANK":   "500180",
    "INFY":       "500209",
    "ICICIBANK":  "532174",
    "SBIN":       "500112",
    "BHARTIARTL": "532454",
    "HINDUNILVR": "500696",
    "ITC":        "500875",
    "BAJFINANCE": "500034",
    "WIPRO":      "507685",
    "AXISBANK":   "532215",
    "KOTAKBANK":  "500247",
    "LT":         "500510",
    "DMART":      "540376",
}


def get_bse_code(symbol: str) -> str:
    """
    Convert an NSE symbol to its BSE scripcode.

    Args:
        symbol: NSE stock symbol, e.g. "RELIANCE" (case-insensitive)

    Returns:
        BSE scripcode as a string, e.g. "500325"

    Raises:
        ValueError: if the symbol is not yet mapped
    """
    code = SYMBOL_TO_BSE.get(symbol.upper())
    if not code:
        raise ValueError(
            f"BSE scripcode not found for symbol: {symbol}. "
            f"Add it to connectors/symbol_map.py"
        )
    return code