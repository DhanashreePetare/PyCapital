"""Live end-to-end checks for pyequity.processing.adjust (hit the real NSE, run manually).

Run:  python -m pytest tests/live/test_live_adjust.py -v

Not part of the default offline run (pyproject testpaths is tests/unit). The
cases are the four ex-dates verified by the 2026-11-17 probe, where NSE
returned RAW prices that fell by the expected factor on the ex-date.
"""
import warnings

import pandas as pd
import pytest

from pyequity.connectors.corporate import fetch_corporate_actions
from pyequity.connectors.nse import fetch_nse_historical
from pyequity.processing.adjust import adjust_prices
from pyequity.processing.normalize import validate

pytestmark = pytest.mark.live

CASES = [
    ("RELIANCE", "2024-10-28", "2024-10-14", "2024-11-08", 0.5),
    ("HDFCBANK", "2019-09-19", "2019-09-05", "2019-10-03", 0.5),
    ("INFY", "2018-09-04", "2018-08-21", "2018-09-18", 0.5),
    ("BAJFINANCE", "2025-06-16", "2025-06-02", "2025-06-30", 0.1),
]


@pytest.mark.parametrize("symbol,ex,start,end,factor", CASES, ids=[c[0] for c in CASES])
def test_adjusted_series_has_no_jump_at_the_ex_date(symbol, ex, start, end, factor):
    prices = fetch_nse_historical(symbol, start, end)
    actions = fetch_corporate_actions(symbol, start, end)
    assert not prices.empty and not actions.empty, "NSE returned nothing; cannot judge"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)     # the sanity check must accept real data
        out = adjust_prices(prices, actions)
    out = out.sort_values("date").reset_index(drop=True)
    ex_ts = pd.Timestamp(ex)
    before, after = out[out["date"] < ex_ts], out[out["date"] >= ex_ts]
    assert not before.empty and not after.empty
    jump = after["close"].iloc[0] / before["close"].iloc[-1]
    assert 0.9 < jump < 1.1, f"adjusted close jumped by {jump:.3f} at the ex-date"
    raw = prices.sort_values("date").reset_index(drop=True)
    raw_before = raw[raw["date"] < ex_ts]["close"].iloc[-1]
    assert before["close"].iloc[-1] == pytest.approx(raw_before * factor, rel=1e-9)
    assert validate(out, "equity").errors == []
