"""Public API for SEBI-mandated disclosures (bulk deals, published via NSE).

Example:
    >>> from pyequity.api import sebi
    >>> latest = sebi.get_bulk_deals()                       # complete latest day
    >>> week = sebi.get_bulk_deals("2026-09-14", "2026-09-18")  # capped days flagged
"""

import warnings
from typing import Optional, Tuple

import pandas as pd

from pyequity.api._args import DateLike, check_range
from pyequity.api._errors import (
    DataUnavailableError, EmptyResultWarning, IncompleteDataWarning,
    InvalidArgumentError)
from pyequity.connectors import sebi as _connector
from pyequity.processing.normalize import normalize
from pyequity.schema import empty_frame

# Each day is a separate, rate-limited request (about 2 seconds), so very
# long ranges are refused instead of silently taking many minutes.
MAX_RANGE_DAYS = 366


def _day_strings(frame: pd.DataFrame) -> pd.Series:
    return frame["date"].dt.strftime("%Y-%m-%d")


def _complete_latest_day(df: pd.DataFrame, capped: list) -> Tuple[pd.DataFrame, list]:
    """Replace capped days by the complete latest-day file where it covers them.

    Args:
        df: Frame from the per-day fetch.
        capped: Days that hit NSE's row cap.

    Returns:
        ``(frame, remaining_capped_days)``.
    """
    latest = _connector.fetch_latest_bulk_deals()
    if latest.empty:
        return df, capped
    latest_days = _day_strings(latest)
    remaining = list(capped)
    for day in capped:
        full = latest[latest_days == day]
        have = df[_day_strings(df) == day]
        if full.empty or len(full) < len(have):
            continue
        df = pd.concat([df[_day_strings(df) != day], full], ignore_index=True)
        remaining.remove(day)
    return df.sort_values("date", kind="stable").reset_index(drop=True), remaining


def get_bulk_deals(start: Optional[DateLike] = None, end: Optional[DateLike] = None, *,
                   complete_latest: bool = True) -> pd.DataFrame:
    """Get bulk deals disclosed on NSE.

    - No dates: the COMPLETE list for the latest trading day (NSE's
      ``bulk.csv``; ``source`` is ``"SEBI_NSE_CSV"``).
    - With ``start`` and ``end``: one request per weekday. NSE returns at most
      70 rows per request, so busy days are cut off. Every such day is named
      in an ``IncompleteDataWarning`` and listed in ``df.attrs["capped_days"]``.
      If a capped day is the latest trading day (and ``complete_latest`` is
      true), it is replaced by the complete list. Older capped days cannot be
      completed; NSE offers no archive for them.

    ``df.attrs`` also holds ``failed_days`` (days that could not be fetched).
    attrs may be dropped by later pandas operations.

    Args:
        start: First date, ``"YYYY-MM-DD"`` or a date object (with ``end``).
        end:   Last date (with ``start``).
        complete_latest: Replace a capped latest trading day by the complete
            file (one extra request, only made if a day was capped).

    Returns:
        pandas DataFrame in the standard ``bulk_deals`` schema.

    Raises:
        InvalidArgumentError: Only one of start/end given, bad dates, a
            reversed range, or a range longer than 366 days.
        DataUnavailableError: Nothing could be fetched and some days failed,
            so it is unknown whether deals exist.
    """
    if start is None and end is None:
        df = _connector.fetch_latest_bulk_deals()
        if df.empty:
            warnings.warn(
                "No bulk deals returned for the latest trading day: either none were "
                "published, or NSE's bulk.csv could not be read.",
                EmptyResultWarning, stacklevel=2)
        out = normalize(df, "bulk_deals")
        out.attrs["capped_days"] = []
        out.attrs["failed_days"] = []
        return out

    if start is None or end is None:
        raise InvalidArgumentError("Give both start and end, or neither for the latest day.")
    start_s, end_s = check_range(start, end)
    span = (pd.Timestamp(end_s) - pd.Timestamp(start_s)).days + 1
    if span > MAX_RANGE_DAYS:
        raise InvalidArgumentError(
            f"Range of {span} days is too long (maximum {MAX_RANGE_DAYS}); "
            f"each day is a separate request. Split it into shorter ranges.")

    df = _connector.fetch_bulk_deals_by_day(start_s, end_s)
    capped = list(df.attrs.get("capped_days", []))
    failed = list(df.attrs.get("failed_days", []))

    if df.empty:
        if failed:
            raise DataUnavailableError(
                f"Could not fetch bulk deals for {start_s} to {end_s}; "
                f"failed days: {', '.join(failed)}.")
        warnings.warn(f"No bulk deals found between {start_s} and {end_s}.",
                      EmptyResultWarning, stacklevel=2)
        out = normalize(empty_frame("bulk_deals"), "bulk_deals")
        out.attrs["capped_days"], out.attrs["failed_days"] = [], []
        return out

    if capped and complete_latest:
        df, capped = _complete_latest_day(df, capped)

    if capped:
        warnings.warn(
            f"{len(capped)} day(s) hit NSE's 70-row cap and are INCOMPLETE: "
            f"{', '.join(capped)}. Rows are missing; only the latest trading day can "
            f"be completed.", IncompleteDataWarning, stacklevel=2)
    if failed:
        warnings.warn(
            f"Could not fetch {len(failed)} day(s); the result is incomplete: "
            f"{', '.join(failed)}.", IncompleteDataWarning, stacklevel=2)

    out = normalize(df, "bulk_deals")
    out.attrs["capped_days"] = capped
    out.attrs["failed_days"] = failed
    return out