"""
Shared normalization and validation for PyEquity datasets.

Connectors keep their own source-specific renaming and parsing. This module
does the shared, source-independent work on top of their output:

* ``normalize()``  fixes *structure*: schema column order, dtypes, sort order,
  duplicate rows, and the ``source`` tag. It never judges the data and never
  drops a row because its values look wrong.
* ``validate()``   checks *content* and returns a ``ValidationReport``. It never
  modifies the data. Every record is preserved; problems are reported and logged.

Both functions take the dataset name used in ``pyequity.schema.SCHEMAS``
("equity", "index", "fno", "corporate_actions", "nav", "bulk_deals"). The
schema in ``schema.py`` is the single source of truth for columns and dtypes.

Error policy
------------
* Unknown dataset name -> ``ValueError``; input that is not a DataFrame ->
  ``TypeError``. Both are developer mistakes, not normal data failures.
* Anything wrong with the data itself never raises. A frame that is missing
  required schema columns is logged as an error and replaced by
  ``empty_frame(dataset)``; everything else is reported by ``validate()``.

Known limitation: missing integer values become 0
-------------------------------------------------
Integer schema columns (volume, trades, open_interest, quantity, ...) cannot
hold nulls. To stay compatible with ``indices.py`` and ``fno.py``, ``normalize()``
replaces missing or unparseable integer values with 0 and logs a warning with
the count. **A missing volume is not the same thing as a zero volume.** Callers
who need to tell them apart must inspect the log or the raw connector output.
This is never applied to price columns: missing prices stay NaN and are
reported by ``validate()`` as errors.

Row order: sorting is by date only and is stable, so rows that share a date
keep the order they arrived in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from pyequity.schema import SCHEMAS, empty_frame

logger = logging.getLogger(__name__)

# Column holding the observation date of each dataset.
DATE_COLUMN: Dict[str, str] = {
    "equity": "date",
    "index": "date",
    "fno": "date",
    "corporate_actions": "ex_date",
    "nav": "date",
    "bulk_deals": "date",
}

# Columns that identify one row. normalize() drops later rows with the same key.
# None means "never drop rows": identical bulk deals can be genuine, so
# validate() only warns about them.
DEDUPE_KEYS: Dict[str, Optional[List[str]]] = {
    "equity": ["date", "symbol", "source"],
    "index": ["date", "index_name", "source"],
    "fno": ["date", "symbol", "instrument_type", "expiry_date",
            "option_type", "strike_price", "source"],
    "corporate_actions": ["ex_date", "symbol", "action_type",
                          "action_detail", "record_date", "source"],
    "nav": ["date", "scheme_code"],
    "bulk_deals": None,
}

# Price-like columns that must never be null.
PRICE_COLUMNS: Dict[str, List[str]] = {
    "equity": ["open", "high", "low", "close"],
    "index": ["open", "high", "low", "close"],
    "fno": ["open", "high", "low", "close"],
    "corporate_actions": [],
    "nav": ["nav"],
    "bulk_deals": ["price"],
}

# Columns that can never legitimately be negative.
NON_NEGATIVE_COLUMNS: Dict[str, List[str]] = {
    "equity": ["volume", "trades"],
    "index": ["volume"],
    "fno": ["volume", "value", "open_interest"],
}

# Other columns that should normally be filled (null -> warning).
REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "equity": ["symbol", "volume"],
    "index": ["index_name", "volume"],
    "fno": ["symbol", "instrument_type", "expiry_date", "volume", "open_interest"],
    "corporate_actions": ["symbol", "action_detail"],
    "nav": ["scheme_code"],
    "bulk_deals": ["symbol", "client_name", "deal_type", "quantity"],
}

# Datasets that are a continuous daily series, so date gaps are meaningful.
GAP_DATASETS = ("equity", "index", "fno", "nav")
# Datasets where first/last date should sit near the requested range.
COVERAGE_DATASETS = ("equity", "index", "fno", "nav", "bulk_deals")

DEFAULT_MAX_GAP_DAYS = 5
DEFAULT_EDGE_TOLERANCE_DAYS = 7

_EPOCH = pd.Timestamp("1970-01-01")


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Issue:
    """One problem found by ``validate()``.

    Attributes:
        severity: "error" (data is wrong or unusable) or "warning" (suspicious).
        code:     Short machine-readable identifier, e.g. "high_lt_low".
        message:  Human-readable description.
        count:    Number of affected rows (0 when not row-based).
    """
    severity: str
    code: str
    message: str
    count: int = 0


@dataclass
class ValidationReport:
    """Result of ``validate()``. Holds issues only; never holds data.

    Attributes:
        dataset: Dataset name that was validated.
        rows:    Number of rows in the validated frame.
        issues:  All issues found, in the order they were detected.
    """
    dataset: str
    rows: int
    issues: List[Issue] = field(default_factory=list)

    @property
    def errors(self) -> List[Issue]:
        """Issues with severity "error"."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[Issue]:
        """Issues with severity "warning"."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings are allowed)."""
        return not self.errors

    def has(self, code: str) -> bool:
        """Return True if an issue with this code was reported."""
        return any(i.code == code for i in self.issues)

    def summary(self) -> str:
        """One-line summary, e.g. 'equity: 248 rows, 0 errors, 1 warnings'."""
        return (f"{self.dataset}: {self.rows} rows, "
                f"{len(self.errors)} errors, {len(self.warnings)} warnings")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_args(df, dataset: str) -> None:
    if dataset not in SCHEMAS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Expected one of: {sorted(SCHEMAS)}"
        )
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")


def _numeric(series: pd.Series) -> pd.Series:
    """Parse to numbers; anything unparseable or infinite becomes NaN."""
    out = pd.to_numeric(series, errors="coerce")
    return out.replace([float("inf"), float("-inf")], float("nan"))


def _dates(series: pd.Series) -> pd.Series:
    """Parse to tz-naive datetimes; anything unparseable becomes NaT."""
    out = pd.to_datetime(series, errors="coerce")
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_localize(None)
    return out


def _parse_bound(value: Optional[str], name: str) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a YYYY-MM-DD string, got {value!r}")


def _coerce(series: pd.Series, dtype: str, label: str) -> pd.Series:
    """Coerce one column to its schema dtype, logging anything lossy."""
    if dtype == "datetime64[ns]":
        out = _dates(series)
        bad = int((out.isna() & series.notna()).sum())
        if bad:
            logger.warning("[Normalize] %s: %d value(s) could not be parsed as "
                           "dates and became NaT", label, bad)
        return out.astype("datetime64[ns]")

    if dtype == "float64":
        out = _numeric(series)
        bad = int((out.isna() & series.notna()).sum())
        if bad:
            logger.warning("[Normalize] %s: %d value(s) could not be parsed as "
                           "numbers and became NaN", label, bad)
        return out.astype("float64")

    if dtype == "int64":
        out = _numeric(series)
        unparsed = int((out.isna() & series.notna()).sum())
        missing = int(out.isna().sum())
        if missing:
            logger.warning(
                "[Normalize] %s: %d missing/unparseable value(s) (%d unparseable) "
                "replaced with 0. Note: a missing value is not the same as a "
                "real zero.", label, missing, unparsed)
            out = out.fillna(0)
        if bool((out != out.round()).any()):
            logger.warning("[Normalize] %s: non-integer values were truncated "
                           "toward zero", label)
        return out.astype("int64")

    if dtype == "object":
        return series.astype(object)

    raise ValueError(f"Unsupported schema dtype {dtype!r} for {label}")


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def normalize(df: pd.DataFrame, dataset: str, *,
              source: Optional[str] = None) -> pd.DataFrame:
    """Enforce the PyEquity schema on a connector's output.

    Structure only: columns are ordered and typed per ``schema.py``, rows are
    sorted by date (stable) and de-duplicated on the dataset's key, and the
    index is reset. Rows are never dropped because their values look wrong; use
    ``validate()`` for that. Extra columns not in the schema are kept after the
    schema columns. Running it twice gives the same result as running it once.

    Missing or unparseable integer values become 0 with a logged warning (see
    the module docstring: missing is not the same as zero). Price columns are
    never filled.

    Args:
        df:      DataFrame produced by a connector.
        dataset: Dataset name in ``pyequity.schema.SCHEMAS``.
        source:  Optional source tag. Used only to fill a missing ``source``
                 column or null ``source`` values; existing values are never
                 overwritten.

    Returns:
        A new DataFrame with the schema columns first and in schema order.
        Returns ``empty_frame(dataset)`` if ``df`` has no columns at all, if
        it has duplicate column names, or if any required schema column is
        missing (logged as an error).

    Raises:
        ValueError: If ``dataset`` is not a known dataset name.
        TypeError:  If ``df`` is not a DataFrame.
    """
    _check_args(df, dataset)
    schema = SCHEMAS[dataset]

    if len(df.columns) == 0:
        logger.debug("[Normalize] %s: input has no columns, returning empty frame", dataset)
        return empty_frame(dataset)

    if df.columns.duplicated().any():
        dup = sorted(set(df.columns[df.columns.duplicated()]))
        logger.error("[Normalize] %s: duplicate column names %s; "
                     "returning empty frame", dataset, dup)
        return empty_frame(dataset)

    out = df.reset_index(drop=True)  # also gives us a private copy

    if source is not None:
        if "source" not in out.columns:
            out["source"] = source
        else:
            out["source"] = out["source"].where(out["source"].notna(), source)

    missing = [c for c in schema if c not in out.columns]
    if missing:
        logger.error("[Normalize] %s: required column(s) %s missing; "
                     "returning empty frame", dataset, missing)
        return empty_frame(dataset)

    for col, dtype in schema.items():
        out[col] = _coerce(out[col], dtype, f"{dataset}.{col}")

    extras = [c for c in out.columns if c not in schema]
    out = out[list(schema) + extras]

    out = out.sort_values(DATE_COLUMN[dataset], kind="mergesort", na_position="last")

    keys = DEDUPE_KEYS[dataset]
    if keys:
        before = len(out)
        out = out.drop_duplicates(subset=keys, keep="first")
        dropped = before - len(out)
        if dropped:
            logger.info("[Normalize] %s: dropped %d duplicate row(s)", dataset, dropped)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, dataset: str, *,
             start: Optional[str] = None, end: Optional[str] = None,
             max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
             edge_tolerance_days: int = DEFAULT_EDGE_TOLERANCE_DAYS
             ) -> ValidationReport:
    """Check a DataFrame against the schema and basic data-integrity rules.

    Never modifies ``df`` and never drops anything. Each issue is also logged
    (errors at ERROR, warnings at WARNING). Works on both raw connector output
    and normalized frames.

    Errors: missing columns, wrong dtypes, null dates, null prices, duplicate
    keys, ``high < low``, negative volumes/counts, rows outside the requested
    range. Warnings: close outside the high-low range, non-positive prices,
    unsorted dates, other nulls, calendar gaps, first/last date far from the
    requested range, suspicious bulk-deal rows.

    Gap and coverage checks are warnings only: there is no holiday calendar
    yet, so a long exchange closure or a late listing looks the same as lost
    data. Treat them as "look here", not proof.

    Args:
        df:      DataFrame to check.
        dataset: Dataset name in ``pyequity.schema.SCHEMAS``.
        start:   Optional requested start date, YYYY-MM-DD.
        end:     Optional requested end date, YYYY-MM-DD.
        max_gap_days: Calendar-day gap between consecutive dates of one series
                 above which a warning is raised. Default 5.
        edge_tolerance_days: How far the first/last date may sit from
                 ``start``/``end`` before a coverage warning. Default 7.

    Returns:
        A ``ValidationReport``. ``report.ok`` is True when there are no errors.

    Raises:
        ValueError: If ``dataset`` is unknown or ``start``/``end`` is not a
                    YYYY-MM-DD string.
        TypeError:  If ``df`` is not a DataFrame.
    """
    _check_args(df, dataset)
    start_ts = _parse_bound(start, "start")
    end_ts = _parse_bound(end, "end")

    schema = SCHEMAS[dataset]
    report = ValidationReport(dataset=dataset, rows=len(df))

    def add(severity: str, code: str, message: str, count: int = 0) -> None:
        report.issues.append(Issue(severity, code, message, count))

    def finish() -> ValidationReport:
        for i in report.issues:
            level = logging.ERROR if i.severity == "error" else logging.WARNING
            logger.log(level, "[Validate:%s] %s: %s", dataset, i.code, i.message)
        return report

    if df.columns.duplicated().any():
        add("error", "duplicate_columns", "duplicate column names present")
        return finish()

    missing = [c for c in schema if c not in df.columns]
    if missing:
        add("error", "missing_columns", f"missing required column(s): {missing}")
        return finish()

    # dtypes
    bad_types = []
    for col, expected in schema.items():
        actual = df[col].dtype
        if expected == "object":
            ok = pd.api.types.is_object_dtype(actual) or isinstance(actual, pd.StringDtype)
        else:
            ok = str(actual) == expected
        if not ok:
            bad_types.append(f"{col} should be {expected}, got {actual}")
    if bad_types:
        add("error", "bad_dtype", "; ".join(bad_types), len(bad_types))

    if df.empty:
        return finish()

    date_col = DATE_COLUMN[dataset]
    dates = _dates(df[date_col])
    valid_dates = dates.dropna()

    # null dates
    n_null_dates = int(dates.isna().sum())
    if n_null_dates:
        # Corporate actions legitimately contain rows without an ex-date.
        sev = "warning" if dataset == "corporate_actions" else "error"
        add(sev, "null_dates", f"{n_null_dates} row(s) have a null {date_col}", n_null_dates)

    # requested range
    day = dates.dt.normalize()
    if start_ts is not None or end_ts is not None:
        outside = pd.Series(False, index=df.index)
        if start_ts is not None:
            outside |= day < start_ts
        if end_ts is not None:
            outside |= day > end_ts
        n_out = int(outside.sum())
        if n_out:
            add("error", "rows_outside_range",
                f"{n_out} row(s) fall outside the requested range {start}..{end}", n_out)

    if not valid_dates.empty and dataset in COVERAGE_DATASETS:
        first, last = valid_dates.min(), valid_dates.max()
        if start_ts is not None and (first - start_ts).days > edge_tolerance_days:
            add("warning", "coverage_start",
                f"first date {first.date()} is {(first - start_ts).days} days after "
                f"requested start {start}; data may be missing, or the instrument "
                f"may not have existed then")
        if end_ts is not None and (end_ts - last).days > edge_tolerance_days:
            add("warning", "coverage_end",
                f"last date {last.date()} is {(end_ts - last).days} days before "
                f"requested end {end}; data may be missing")

    # sort order
    if not valid_dates.is_monotonic_increasing:
        add("warning", "unsorted_dates", f"{date_col} is not in ascending order")

    # duplicates
    keys = DEDUPE_KEYS[dataset]
    try:
        if keys:
            n_dup = int(df.assign(**{date_col: dates}).duplicated(subset=keys).sum())
            if n_dup:
                if dataset == "corporate_actions":
                    add("warning", "duplicate_rows",
                        f"{n_dup} exact duplicate row(s)", n_dup)
                else:
                    add("error", "duplicate_keys",
                        f"{n_dup} row(s) repeat an existing key {keys}", n_dup)
        else:
            n_dup = int(df[list(schema)].duplicated().sum())
            if n_dup:
                add("warning", "duplicate_rows",
                    f"{n_dup} exact duplicate row(s); kept because identical "
                    f"records can be genuine", n_dup)
    except TypeError:
        logger.debug("[Validate:%s] duplicate check skipped (unhashable values)", dataset)

    # prices
    price_cols = PRICE_COLUMNS[dataset]
    if price_cols:
        nums = pd.DataFrame({c: _numeric(df[c]) for c in price_cols})
        n_null = int(nums.isna().any(axis=1).sum())
        if n_null:
            add("error", "null_prices",
                f"{n_null} row(s) have a null/unparseable value in {price_cols}", n_null)
        n_nonpos = int((nums <= 0).any(axis=1).sum())
        if n_nonpos:
            add("warning", "nonpositive_price",
                f"{n_nonpos} row(s) have a price <= 0 (can be genuine for "
                f"contracts with no trades)", n_nonpos)
        if {"open", "high", "low", "close"} <= set(price_cols):
            n_hl = int((nums["high"] < nums["low"]).sum())
            if n_hl:
                add("error", "high_lt_low", f"{n_hl} row(s) have high < low", n_hl)
            n_c = int(((nums["close"] < nums["low"]) | (nums["close"] > nums["high"])).sum())
            if n_c:
                add("warning", "close_outside_range",
                    f"{n_c} row(s) have close outside the high-low range", n_c)

    # negative counts
    for col in NON_NEGATIVE_COLUMNS.get(dataset, []):
        n_neg = int((_numeric(df[col]) < 0).sum())
        if n_neg:
            add("error", "negative_values", f"{n_neg} row(s) have negative {col}", n_neg)

    # other nulls
    null_report = {}
    for col in REQUIRED_COLUMNS.get(dataset, []):
        s = df[col]
        n = int((_numeric(s) if schema[col] in ("int64", "float64") else s).isna().sum())
        if n:
            null_report[col] = n
    if null_report:
        add("warning", "null_values", f"null values found: {null_report}",
            sum(null_report.values()))

    # bulk deals
    if dataset == "bulk_deals":
        n_q = int((_numeric(df["quantity"]) <= 0).sum())
        if n_q:
            add("warning", "nonpositive_quantity",
                f"{n_q} row(s) have quantity <= 0 (a missing quantity is "
                f"turned into 0 by normalize())", n_q)
        deal = df["deal_type"].dropna().astype(str).str.strip().str.upper()
        odd = deal[~deal.isin(["BUY", "SELL"])]
        if not odd.empty:
            add("warning", "unexpected_deal_type",
                f"{len(odd)} row(s) have deal_type other than BUY/SELL, "
                f"e.g. {sorted(set(odd))[:3]}", len(odd))

    # gaps in continuous series
    if dataset in GAP_DATASETS and not valid_dates.empty:
        group_cols = [c for c in keys if c != date_col]
        tmp = df.assign(**{date_col: dates}).dropna(subset=[date_col])
        tmp = tmp.sort_values(date_col, kind="mergesort")
        tmp["_day"] = (tmp[date_col] - _EPOCH).dt.days
        diff = tmp.groupby(group_cols, dropna=False, sort=False)["_day"].diff()
        big = diff > max_gap_days
        n_gaps = int(big.sum())
        if n_gaps:
            examples = [str(d.date()) for d in tmp.loc[big, date_col].head(3)]
            add("warning", "date_gaps",
                f"{n_gaps} gap(s) longer than {max_gap_days} calendar days "
                f"(largest {int(diff.max())} days), first ending on {examples}; "
                f"could be a holiday stretch or missing data", n_gaps)

    return finish()