"""Adjust raw equity prices for bonus issues and stock splits.

NSE's historical price API returns RAW prices: a 1:1 bonus halves the close on
the ex-date. This module scales the history before each ex-date so the series
is continuous. It is pure (no network, inputs never modified).

Two public functions:

* ``compute_factors(actions)``  one row per price-relevant ex-date.
* ``adjust_prices(prices, actions)``  the adjusted equity frame.

Which actions are applied
-------------------------
* BONUS and SPLIT rows with ``parse_status == "ok"`` and valid numbers.
  Bonus ``a:b`` (a new per b held) gives a price factor ``b / (a + b)``; a split
  from face value ``old`` to ``new`` gives ``new / old``.
* Skipped, with a warning: RIGHTS (not adjusted in v0.1), DEMERGER, MERGER,
  SCHEME_OF_ARRANGEMENT, OTHER, every ``needs_review`` row, invalid ratios
  (never replaced by a factor of 1) and rows without an ex-date.
* Ignored (debug log only): DIVIDEND, BUYBACK, AGM, EVOTING. Dividends are not
  adjusted.
* Bonus and rights on one date: the bonus is applied, the rights part skipped.
* Several applied parts on one ex-date, in one row (``BONUS|SPLIT``) or in
  separate rows, multiply (BAJFINANCE 2025: split 0.5 x bonus 0.2 = 0.1).
* Duplicate rows are collapsed on their parsed content, not their raw text.

Rules for ``adjust_prices``
---------------------------
* A price row dated ``d`` is scaled by the product of the factors of all
  applied events with ``ex_date > d``. Rows on or after an ex-date are
  untouched. Ratios are exact fractions converted to float once.
* Events dated after the last price row are NOT applied (they have not
  happened yet in the data); they are logged.
* Open, high, low, close are multiplied (not rounded). Volume is divided by the
  same factor and rounded with ``np.rint`` (half to even), so price x volume is
  preserved. Every equity-schema column present is returned in its schema
  dtype (``trades`` as int64, ``symbol`` and ``source`` as object) with its
  values unchanged; a ``trades`` value that cannot convert exactly (missing,
  fractional, non-numeric) makes the call fail safe. Extra columns and the
  index are unchanged.
* Prices of several symbols are matched to actions per symbol, ignoring case.
* A skipped price-affecting event inside the price range triggers a logger
  warning and one ``UserWarning`` per call. Prices are never altered because
  of a warning.
* After adjusting, the adjusted previous close is compared with the ex-date
  close at every applied ex-date. A ratio outside 0.7 to 1.4 only warns (it
  suggests already-adjusted prices or a wrong ratio).

Fail-safe errors: not a DataFrame raises ``TypeError``. A prices frame missing
required columns, or an actions frame missing the parsed columns, is logged and
returns ``empty_frame("equity")``: handing back unadjusted prices there would
pass raw data off as adjusted. An actions frame with no rows or no columns
means "no actions" and returns the prices unchanged.

Adjusting twice double-adjusts. Pass RAW prices, once.

Known limitation: a failed corporate-actions fetch and "no actions" both reach
this module as an empty frame. The caller must treat a failed fetch as an error.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pyequity.schema import empty_frame

logger = logging.getLogger(__name__)

FACTOR_COLUMNS: Dict[str, str] = {
    "ex_date": "datetime64[ns]",
    "factor": "float64",
    "applied": "bool",
    "reason": "object",
}

SANITY_RATIO_LOW = 0.7
SANITY_RATIO_HIGH = 1.4

_REQUIRED_ACTION_COLUMNS = ("ex_date", "symbol", "action_categories", "parse_status",
                            "bonus_new", "bonus_held", "split_old_fv", "split_new_fv")
_NUMERIC_ACTION_COLUMNS = ("bonus_new", "bonus_held", "split_old_fv", "split_new_fv",
                           "rights_new", "rights_held", "rights_premium")
_REQUIRED_PRICE_COLUMNS = ("date", "open", "high", "low", "close", "volume", "symbol")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_IGNORED_CATEGORIES = frozenset({"DIVIDEND", "BUYBACK", "AGM", "EVOTING"})
_SKIP_REASONS = {
    "RIGHTS": "rights are not adjusted in v0.1",
    "DEMERGER": "demerger cannot be adjusted from a ratio",
    "MERGER": "merger cannot be adjusted from a ratio",
    "SCHEME_OF_ARRANGEMENT": "scheme of arrangement cannot be adjusted from a ratio",
    "OTHER": "unrecognised price-affecting text",
}
_MAX_EXACT_VOLUME = 2 ** 53


# ---------------------------------------------------------------------------
# Row evaluation
# ---------------------------------------------------------------------------

@dataclass
class _Event:
    symbol: str
    ex_date: pd.Timestamp
    factor: Fraction = Fraction(1)
    applied_parts: List[str] = field(default_factory=list)
    skipped_parts: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return bool(self.applied_parts)

    @property
    def reason(self) -> str:
        parts = []
        if self.applied_parts:
            parts.append("applied: " + "; ".join(self.applied_parts))
        if self.skipped_parts:
            parts.append("skipped: " + "; ".join(self.skipped_parts))
        return " | ".join(parts)


@dataclass
class _Prepared:
    events: Dict[str, List[_Event]]
    undated: List[Tuple[str, str]]            # (symbol, description) of price rows with no ex-date
    symbols_seen: set
    duplicates: int = 0


def _fmt(value: float) -> str:
    return f"{value:g}"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _fraction(value: float) -> Fraction:
    """Exact decimal value of a parsed number (1.0 -> 1, 0.5 -> 1/2)."""
    return Fraction(str(value))


def _categories(value: Any) -> List[str]:
    if not isinstance(value, str):
        return []
    return [c.strip().upper() for c in value.split("|") if c.strip()]


def _norm_symbol(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _evaluate_row(row: Dict[str, Any]) -> Tuple[Fraction, List[str], List[str], bool]:
    """Return (factor, applied parts, skipped parts, price_relevant) for one action row."""
    cats = _categories(row.get("action_categories")) or ["OTHER"]
    status = row.get("parse_status")
    relevant = [c for c in cats if c not in _IGNORED_CATEGORIES]
    ok = isinstance(status, str) and status.strip().lower() == "ok"
    if not relevant and ok:
        return Fraction(1), [], [], False

    if not ok:
        parts = [f"{c} (needs_review: {_SKIP_REASONS.get(c, 'text not fully understood')})"
                 for c in relevant] or [f"unparsed text (parse_status={status!r})"]
        return Fraction(1), [], parts, True

    factor = Fraction(1)
    applied: List[str] = []
    skipped: List[str] = []
    for category in relevant:
        if category == "BONUS":
            new, held = _number(row.get("bonus_new")), _number(row.get("bonus_held"))
            if new is None or held is None or new <= 0 or held <= 0:
                skipped.append(f"BONUS (invalid ratio {row.get('bonus_new')!r}:{row.get('bonus_held')!r})")
                continue
            part = _fraction(held) / (_fraction(new) + _fraction(held))
            applied.append(f"BONUS {_fmt(new)}:{_fmt(held)} (x{_fmt(float(part))})")
            factor *= part
        elif category == "SPLIT":
            old, new = _number(row.get("split_old_fv")), _number(row.get("split_new_fv"))
            if old is None or new is None or old <= 0 or new <= 0 or old <= new:
                skipped.append(f"SPLIT (invalid face values {row.get('split_old_fv')!r} -> "
                               f"{row.get('split_new_fv')!r})")
                continue
            part = _fraction(new) / _fraction(old)
            applied.append(f"SPLIT {_fmt(old)}->{_fmt(new)} (x{_fmt(float(part))})")
            factor *= part
        else:
            skipped.append(f"{category} ({_SKIP_REASONS.get(category, 'not adjusted')})")
    return factor, applied, skipped, True


def _dedupe_key(row: Dict[str, Any], ex_date: Any, symbol: str) -> tuple:
    status = row.get("parse_status")
    needs_text = not (isinstance(status, str) and status.strip().lower() == "ok")
    text = " ".join(str(row.get("action_detail", "")).split()).lower() if needs_text else ""
    return (None if pd.isna(ex_date) else ex_date, symbol,
            "|".join(_categories(row.get("action_categories"))), str(status).strip().lower(),
            tuple(_number(row.get(c)) for c in _NUMERIC_ACTION_COLUMNS), text)


def _prepare(actions: pd.DataFrame) -> _Prepared:
    """Parse, de-duplicate and aggregate action rows into one event per (symbol, ex-date)."""
    missing = [c for c in _REQUIRED_ACTION_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(f"actions frame is missing parsed columns {missing}; "
                         "pass the output of fetch_corporate_actions()")
    ex_dates = pd.to_datetime(actions["ex_date"], errors="coerce")
    if getattr(ex_dates.dt, "tz", None) is not None:
        ex_dates = ex_dates.dt.tz_localize(None)
    ex_dates = ex_dates.dt.normalize()

    columns = [c for c in (*_REQUIRED_ACTION_COLUMNS, "action_detail", *_NUMERIC_ACTION_COLUMNS)
               if c in actions.columns]
    rows = actions[list(dict.fromkeys(columns))].to_dict("records")

    prepared = _Prepared(events={}, undated=[], symbols_seen=set())
    seen: set = set()
    merged: Dict[Tuple[str, pd.Timestamp], _Event] = {}
    for row, ex_date in zip(rows, ex_dates):
        symbol = _norm_symbol(row.get("symbol"))
        if symbol:
            prepared.symbols_seen.add(symbol)
        key = _dedupe_key(row, ex_date, symbol)
        if key in seen:
            prepared.duplicates += 1
            logger.debug("[Adjust] Duplicate action row ignored: %s %s", symbol, ex_date)
            continue
        seen.add(key)

        factor, applied, skipped, relevant = _evaluate_row(row)
        text = " ".join(str(row.get("action_detail", "")).split())
        if not relevant:
            logger.debug("[Adjust] Ignoring non-price action: %s %s %s", symbol, ex_date, text)
            continue
        if not symbol:
            logger.warning("[Adjust] Price-affecting action without a symbol skipped: %s", text or row)
            continue
        if pd.isna(ex_date):
            description = f"{text or row.get('action_categories')} (no ex-date): " + "; ".join(skipped or applied)
            logger.warning("[Adjust] %s: action without an ex-date skipped: %s", symbol, description)
            prepared.undated.append((symbol, description))
            continue

        event = merged.setdefault((symbol, ex_date), _Event(symbol=symbol, ex_date=ex_date))
        event.factor *= factor
        event.applied_parts.extend(applied)
        event.skipped_parts.extend(skipped)
        if text:
            event.texts.append(text)

    for (symbol, _), event in sorted(merged.items(), key=lambda item: (item[0][0], item[0][1])):
        prepared.events.setdefault(symbol, []).append(event)
        for part in event.skipped_parts:
            logger.debug("[Adjust] %s %s: not applied: %s", symbol, event.ex_date.date(), part)
    if prepared.duplicates:
        logger.info("[Adjust] %d duplicate action row(s) collapsed", prepared.duplicates)
    return prepared


def _check_frame(value: Any, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(value).__name__}")


def _empty_factors() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in FACTOR_COLUMNS.items()})


def _no_actions(actions: pd.DataFrame) -> bool:
    return actions.shape[0] == 0 or actions.shape[1] == 0


# ---------------------------------------------------------------------------
# compute_factors
# ---------------------------------------------------------------------------

def compute_factors(actions: pd.DataFrame) -> pd.DataFrame:
    """Price-adjustment factor for every price-relevant ex-date of ONE symbol.

    Args:
        actions: Output of ``fetch_corporate_actions()`` for a single symbol
            (columns ex_date, symbol, action_categories, parse_status and the
            parsed ratio columns).

    Returns:
        DataFrame with columns ``ex_date`` (datetime64[ns]), ``factor``
        (float64, the multiplier for prices BEFORE the ex-date), ``applied``
        (bool, True when at least one part was applied) and ``reason`` (what
        was applied and what was skipped), sorted by ex-date, one row per
        ex-date. Dates that only carry dividends, buybacks, AGMs or e-voting
        are omitted, so ``applied=False`` always means "price-affecting but
        not adjusted" (factor 1.0). Rows without an ex-date cannot appear in
        the table; they are logged. Empty frame if there are no actions.

    Raises:
        TypeError: ``actions`` is not a DataFrame.
        ValueError: required parsed columns are missing, or the actions belong
            to more than one symbol. An empty factor table would be
            indistinguishable from "no actions", so this is not hidden.
    """
    _check_frame(actions, "actions")
    if _no_actions(actions):
        return _empty_factors()
    prepared = _prepare(actions)
    if len(prepared.symbols_seen) > 1:
        raise ValueError("compute_factors needs actions for a single symbol, got "
                         f"{sorted(prepared.symbols_seen)}")
    events = [event for per_symbol in prepared.events.values() for event in per_symbol]
    if not events:
        return _empty_factors()
    events.sort(key=lambda e: e.ex_date)
    for event in events:
        for part in event.skipped_parts:
            logger.warning("[Adjust] %s %s: NOT applied: %s%s", event.symbol, event.ex_date.date(), part,
                           f" [{'; '.join(event.texts)}]" if event.texts else "")
    out = pd.DataFrame({
        "ex_date": pd.Series([e.ex_date for e in events]).astype("datetime64[ns]"),
        "factor": pd.Series([float(e.factor) for e in events], dtype="float64"),
        "applied": pd.Series([e.applied for e in events], dtype="bool"),
        "reason": pd.Series([e.reason for e in events], dtype="object"),
    })
    logger.info("[Adjust] %d price-relevant ex-date(s), %d with an applied factor",
                len(out), int(out["applied"].sum()))
    return out


# ---------------------------------------------------------------------------
# adjust_prices
# ---------------------------------------------------------------------------

def _fail(message: str) -> pd.DataFrame:
    logger.error("[Adjust] %s", message)
    return empty_frame("equity")


def _exact_int64(series: pd.Series, name: str) -> np.ndarray:
    """Convert a column to int64 without ever truncating or rounding a value.

    Integer dtypes convert directly (exact, any width). Anything else must be
    numeric, finite and whole, and no larger than 2**53 so that it survived
    float64 exactly. Raises ValueError otherwise.
    """
    if series.dtype.kind in "iu":
        if series.dtype.kind == "u" and len(series) and int(series.max()) > np.iinfo("int64").max:
            raise ValueError(f"{name} has values above the int64 range")
        return series.astype("int64").to_numpy()
    try:
        values = pd.to_numeric(series, errors="raise").astype("float64").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} has non-numeric values ({exc})") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} has missing values")
    if (values != np.floor(values)).any():
        raise ValueError(f"{name} must be whole numbers")
    if (np.abs(values) > _MAX_EXACT_VOLUME).any():
        raise ValueError(f"{name} has values too large to convert exactly")
    return values.astype("int64")


def _finish(df: pd.DataFrame, dates: pd.Series, ohlc: Dict[str, np.ndarray],
            volume: np.ndarray, trades: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Copy df and set every equity-schema column present to its schema dtype.

    Values, index, row order and all other columns are kept as they are.
    """
    out = df.copy()
    out["date"] = dates.astype("datetime64[ns]")
    for column, values in ohlc.items():
        out[column] = pd.Series(values, index=df.index, dtype="float64")
    out["volume"] = pd.Series(volume, index=df.index, dtype="int64")
    if trades is not None:
        out["trades"] = pd.Series(trades, index=df.index, dtype="int64")
    for column in ("symbol", "source"):           # pandas 3 would otherwise hand back "str"
        if column in out.columns:
            out[column] = out[column].astype(object)
    return out


def adjust_prices(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Adjust RAW equity prices for bonus issues and splits.

    Pass raw prices exactly once: adjusting an adjusted series double-adjusts.

    Args:
        prices: Equity frame (``fetch_nse_historical`` output). Needs date,
            open, high, low, close, volume and symbol; other columns are kept.
            May hold several symbols.
        actions: Corporate actions (``fetch_corporate_actions`` output), for
            one or several symbols. Matched to prices per symbol, ignoring case.

    Returns:
        A new frame with the same index, row order and columns. OHLC are
        multiplied by the cumulative factor (float64, unrounded); volume is
        divided by it and rounded to int64 (half to even); date is
        datetime64[ns]; trades is int64; symbol and source are object dtype. Rows on or after the last applicable ex-date are
        unchanged. If ``actions`` has no rows or no columns the prices are
        returned unchanged (a copy). If a required column is missing, returns
        ``empty_frame("equity")`` and logs an error.

    Warns:
        UserWarning: a price-affecting action inside the price range was not
        applied, or an applied ex-date fails the 0.7 to 1.4 sanity check. The
        prices are not changed because of a warning.

    Raises:
        TypeError: an argument is not a DataFrame.
    """
    _check_frame(prices, "prices")
    _check_frame(actions, "actions")
    missing = [c for c in _REQUIRED_PRICE_COLUMNS if c not in prices.columns]
    if missing:
        return _fail(f"prices frame is missing required column(s) {missing}")

    dates = pd.to_datetime(prices["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    if dates.isna().any():
        return _fail("prices frame has missing or unparseable dates")
    try:
        ohlc_in = {c: prices[c].astype("float64").to_numpy() for c in _PRICE_COLUMNS}
        volume_f = pd.to_numeric(prices["volume"], errors="raise").astype("float64").to_numpy()
    except (TypeError, ValueError) as exc:
        return _fail(f"prices frame has non-numeric price or volume values ({exc})")
    if not np.isfinite(volume_f).all() or (volume_f != np.floor(volume_f)).any():
        return _fail("prices volume must be whole numbers without missing values")
    volume_in = prices["volume"].astype("int64").to_numpy()
    trades_in = None
    if "trades" in prices.columns:
        try:
            trades_in = _exact_int64(prices["trades"], "trades")
        except ValueError as exc:
            return _fail(f"prices frame has invalid trades values: {exc}")

    if len(prices) == 0 or _no_actions(actions):
        logger.info("[Adjust] No actions or no prices: returning prices unchanged")
        return _finish(prices, dates, ohlc_in, volume_in, trades_in)

    try:
        prepared = _prepare(actions)
    except ValueError as exc:
        return _fail(str(exc))

    day = dates.dt.normalize().to_numpy("datetime64[ns]")
    symbols = prices["symbol"].map(_norm_symbol).to_numpy()
    cumulative = np.ones(len(prices), dtype="float64")
    touched = np.zeros(len(prices), dtype=bool)
    skipped_notes: List[str] = []
    checks: List[Tuple[str, pd.Timestamp, np.ndarray]] = []   # (symbol, ex_date, positions sorted by date)
    applied_count = future_count = 0

    for symbol in sorted(set(symbols) - {""}):
        positions = np.flatnonzero(symbols == symbol)
        symbol_days = day[positions]
        first, last = symbol_days.min(), symbol_days.max()
        events = prepared.events.get(symbol, [])
        active: List[_Event] = []
        for event in events:
            ex = np.datetime64(event.ex_date, "ns")
            if event.skipped_parts and first < ex <= last:
                skipped_notes.append(f"{symbol} {event.ex_date.date()}: {event.reason}")
            if not event.applied:
                continue
            if ex > last:
                future_count += 1
                logger.info("[Adjust] %s %s is after the last price row (%s); not applied: %s",
                            symbol, event.ex_date.date(), pd.Timestamp(last).date(), event.reason)
                continue
            active.append(event)
        if not active:
            continue
        applied_count += len(active)

        ex_array = np.array([np.datetime64(e.ex_date, "ns") for e in active], dtype="datetime64[ns]")
        suffix = [Fraction(1)] * (len(active) + 1)
        for i in range(len(active) - 1, -1, -1):
            suffix[i] = suffix[i + 1] * active[i].factor
        suffix_float = np.array([float(x) for x in suffix], dtype="float64")
        first_event_after = np.searchsorted(ex_array, symbol_days, side="right")
        cumulative[positions] = suffix_float[first_event_after]
        touched[positions] = first_event_after < len(active)
        order = positions[np.argsort(symbol_days, kind="stable")]
        for event in active:
            checks.append((symbol, event.ex_date, order))

    for symbol, description in prepared.undated:
        if symbol in set(symbols):
            skipped_notes.append(f"{symbol}: {description}")

    ohlc_out = {c: values * cumulative for c, values in ohlc_in.items()}
    volume_out = volume_in.copy()
    if touched.any():
        scaled = np.rint(volume_f[touched] / cumulative[touched])
        if (np.abs(scaled) > _MAX_EXACT_VOLUME).any():
            return _fail("adjusted volume would exceed 2**53 and could not be exact")
        volume_out[touched] = scaled.astype("int64")

    result = _finish(prices, dates, ohlc_out, volume_out, trades_in)

    sanity_notes: List[str] = []
    close = ohlc_out["close"]
    for symbol, ex_date, order in checks:
        ordered_days = day[order]
        ex = np.datetime64(ex_date, "ns")
        split_at = int(np.searchsorted(ordered_days, ex, side="left"))
        if split_at == 0 or split_at >= len(order):
            continue
        previous, current = close[order[split_at - 1]], close[order[split_at]]
        if not (np.isfinite(previous) and np.isfinite(current)) or previous <= 0 or current <= 0:
            continue
        ratio = previous / current
        if not SANITY_RATIO_LOW <= ratio <= SANITY_RATIO_HIGH:
            sanity_notes.append(f"{symbol} {ex_date.date()}: adjusted previous close / ex-date close "
                                f"= {ratio:.3f} (expected {SANITY_RATIO_LOW} to {SANITY_RATIO_HIGH})")

    for note in sanity_notes:
        logger.warning("[Adjust] Sanity check failed, prices left as computed: %s", note)
    for note in skipped_notes:
        logger.warning("[Adjust] Price-affecting action NOT applied: %s", note)
    if skipped_notes:
        warnings.warn("price-affecting corporate action(s) were NOT applied, so prices before "
                      "these dates are still unadjusted for them: " + " || ".join(skipped_notes),
                      UserWarning, stacklevel=2)
    if sanity_notes:
        warnings.warn("adjusted prices fail the continuity check (already-adjusted prices or a wrong "
                      "ratio?): " + " || ".join(sanity_notes), UserWarning, stacklevel=2)

    logger.info("[Adjust] %d event(s) applied, %d future event(s) not applied, %d price-affecting "
                "action(s) skipped, %d of %d rows scaled",
                applied_count, future_count, len(skipped_notes), int(touched.sum()), len(prices))
    return result
