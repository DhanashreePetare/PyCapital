"""Standard column schemas for every PyCapital dataset."""
import pandas as pd

SCHEMAS = {
    "equity": {"date": "datetime64[ns]", "open": "float64", "high": "float64",
               "low": "float64", "close": "float64", "volume": "int64",
               "trades": "int64", "symbol": "object", "source": "object"},
    "index": {"date": "datetime64[ns]", "index_name": "object", "open": "float64",
              "high": "float64", "low": "float64", "close": "float64",
              "volume": "int64", "source": "object"},
    "fno": {"date": "datetime64[ns]", "symbol": "object", "instrument_type": "object",
            "expiry_date": "datetime64[ns]", "option_type": "object",
            "strike_price": "float64", "open": "float64", "high": "float64",
            "low": "float64", "close": "float64", "settle_price": "float64",
            "volume": "int64", "value": "int64", "open_interest": "int64",
            "change_in_oi": "int64", "underlying_value": "float64", "source": "object"},
    "corporate_actions": {"ex_date": "datetime64[ns]", "symbol": "object",
                          "action_type": "object", "action_detail": "object",
                          "record_date": "datetime64[ns]", "source": "object"},
    "nav": {"date": "datetime64[ns]", "scheme_code": "int64",
            "scheme_name": "object", "nav": "float64", "source": "object"},
    "bulk_deals": {"date": "datetime64[ns]", "symbol": "object", "scrip_name": "object",
                   "client_name": "object", "deal_type": "object",
                   "quantity": "int64", "price": "float64", "source": "object"},
}


def columns_of(name: str) -> list:
    return list(SCHEMAS[name])


def empty_frame(name: str) -> pd.DataFrame:
    """Empty DataFrame with the standard columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMAS[name].items()})