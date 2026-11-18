"""
AMFI Mutual Fund NAV connector for PyEquity.

Two sources are available:

1. ``fetch_nav``: https://api.mfapi.in (community-maintained, sourced from
   official AMFI NAV disclosures). Plain public JSON API, no session needed.
   Output ``source`` is ``"AMFI"``.
2. ``fetch_nav_official``: AMFI's own published text files
   (``NAVAll.txt`` for names, ``DownloadNAVHistoryReport_Po.aspx`` for
   history). Output ``source`` is ``"AMFI_OFFICIAL"`` so data from the two
   sources can always be told apart. It is NOT wired into ``fetch_nav`` as an
   automatic fallback; that belongs to the reliability layer.

Note: scheme_code -> scheme_name is NOT hardcoded anywhere in this file.
scheme_name is always read live from the response itself, since AMFI scheme
codes can map to different funds than expected: verify via list_schemes()
before assuming a code matches a fund name.

Verified on real responses (November 2026, see docs/research):
- AMFI's text files are semicolon separated and their column layout has
  changed before, so they are parsed by header NAME, never by position.
  If a required header is missing, the chunk is rejected and logged.
- The history file is requested without an AMC id, which returns every
  scheme for the requested dates (large: roughly 1 MB per day), so requests
  are kept to short chunks.
- Official and mfapi.in NAVs agreed on every date tested for scheme 119551
  (15-23 Sep 2026). That is a spot check, not proof for all schemes.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import requests

from pyequity.schema import SCHEMAS, empty_frame

BASE_URL = "https://api.mfapi.in/mf"

NAVALL_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
HISTORY_URL = "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"

# AMFI's download page documents a 90 day maximum per request. We stay well
# below it to bound memory: a request without an AMC id returns every scheme.
# NOTE: the no-AMC-id request was only verified on a 3 day range.
_OFFICIAL_CHUNK_DAYS = 30

_OFFICIAL_SOURCE = "AMFI_OFFICIAL"
_HISTORY_REQUIRED = ("Scheme Code", "NAV Name", "Net Asset Value", "Date")
_NAVALL_REQUIRED = ("Scheme Code", "Scheme Name", "Net Asset Value", "Date")
_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PyEquity)"}

logger = logging.getLogger(__name__)


def list_schemes() -> pd.DataFrame:
    """
    Fetch the full list of AMFI scheme codes and names.

    Returns:
        pandas DataFrame with columns: schemeCode, schemeName
        Empty DataFrame on failure.
    """
    try:
        response = requests.get(BASE_URL, timeout=15)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        logger.error(f"[AMFI] Error fetching scheme list: {e}")
        return pd.DataFrame(columns=["schemeCode", "schemeName"])


def fetch_nav(scheme_code, from_date=None, to_date=None) -> pd.DataFrame:
    """
    Fetch NAV history for a single AMFI mutual fund scheme (mfapi.in).

    A single NAV value that cannot be read as a number (or a date that
    cannot be parsed) drops only that row and logs a warning with the
    count; it no longer discards the whole result.

    Args:
        scheme_code : AMFI scheme code, e.g. 119551
        from_date   : Start date in YYYY-MM-DD format (optional, matches
                      the project convention used by nse.py/bse.py).
                      No lower bound if None.
        to_date     : End date in YYYY-MM-DD format (optional).
                      No upper bound if None.

    Returns:
        pandas DataFrame with columns:
        date, scheme_code, scheme_name, nav, source
        Sorted oldest to newest. Empty DataFrame on any failure,
        never raises, matching every other connector in this project.
    """
    columns = ["date", "scheme_code", "scheme_name", "nav", "source"]

    try:
        url = f"{BASE_URL}/{scheme_code}"
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            logger.error(f"[AMFI] Error: status code {response.status_code} for scheme {scheme_code}")
            return empty_frame("nav")

        payload = response.json()

        if not payload.get("data"):
            logger.warning(f"[AMFI] No NAV data returned for scheme code {scheme_code}")
            return empty_frame("nav")

        scheme_name = payload.get("meta", {}).get("scheme_name", "")

        df = pd.DataFrame(payload["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df["date"] = df["date"].astype("datetime64[ns]")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

        unreadable = df["date"].isna() | df["nav"].isna()
        if unreadable.any():
            logger.warning(
                f"[AMFI] Dropped {int(unreadable.sum())} of {len(df)} rows with an "
                f"unreadable date or NAV for scheme {scheme_code}"
            )
            df = df[~unreadable].copy()
            if df.empty:
                return empty_frame("nav")

        df["nav"] = df["nav"].astype("float64")
        df["scheme_code"] = int(scheme_code)
        df["scheme_name"] = scheme_name
        df["source"] = "AMFI"

        df = df[columns]

        # Convert standard YYYY-MM-DD input to filter against the data
        if from_date:
            df = df[df["date"] >= datetime.strptime(from_date, "%Y-%m-%d")]
        if to_date:
            df = df[df["date"] <= datetime.strptime(to_date, "%Y-%m-%d")]

        df = df.sort_values("date").reset_index(drop=True)

        return df.astype(SCHEMAS["nav"])

    except Exception as e:
        logger.error(f"[AMFI] Exception for scheme {scheme_code}: {e}")
        return empty_frame("nav")


# ---------------------------------------------------------------------------
# Official AMFI files
# ---------------------------------------------------------------------------

def _split_with_header(text: str, required: Tuple[str, ...]) -> Optional[Tuple[List[str], List[List[str]]]]:
    """
    Split an AMFI semicolon text file into (header, data rows).

    The header is the first line that has a "Scheme Code" column, wherever
    that column sits. Data rows
    are the lines below it with exactly as many fields as the header; the
    category and fund-house heading lines AMFI interleaves are skipped.

    Returns None if no header is found or a required column is missing, so
    callers can reject the file instead of guessing at its layout.
    """
    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    for line in text.splitlines():
        if header is None:
            candidate = [h.strip() for h in line.split(";")]
            if "Scheme Code" in candidate:
                header = candidate
                if any(col not in header for col in required):
                    return None
            continue
        fields = line.split(";")
        if len(fields) == len(header):
            rows.append([f.strip() for f in fields])
    if header is None:
        return None
    return header, rows


def _parse_official_history(text: str, scheme_code: int) -> Optional[Tuple[pd.DataFrame, int, str]]:
    """
    Parse one AMFI history file for a single scheme.

    Returns None if the file layout is not as expected. Otherwise returns
    (frame, unreadable_row_count, nav_name) where frame has columns
    date and nav only (float64), one row per date found for the scheme.
    """
    split = _split_with_header(text, _HISTORY_REQUIRED)
    if split is None:
        return None
    header, rows = split
    i_code = header.index("Scheme Code")
    i_name = header.index("NAV Name")
    i_nav = header.index("Net Asset Value")
    i_date = header.index("Date")

    wanted = str(int(scheme_code))
    mine = [r for r in rows if r[i_code] == wanted]
    nav_name = mine[0][i_name] if mine else ""

    df = pd.DataFrame({
        "date": pd.to_datetime([r[i_date] for r in mine], format="%d-%b-%Y", errors="coerce"),
        "nav": pd.to_numeric([r[i_nav] for r in mine], errors="coerce"),
    })
    df["date"] = df["date"].astype("datetime64[ns]")
    unreadable = df["date"].isna() | df["nav"].isna()
    bad = int(unreadable.sum())
    df = df[~unreadable].copy()
    df["nav"] = df["nav"].astype("float64")
    return df.reset_index(drop=True), bad, nav_name


def _parse_navall_name(text: str, scheme_code: int) -> Optional[str]:
    """
    Build a scheme name from the latest-NAV file: "Scheme Name - Plan - Option".

    Matches the name mfapi.in reports for the one scheme checked (119551).
    Empty Plan/Option parts are left out. Returns None if the layout is not
    as expected or the scheme is not listed.
    """
    split = _split_with_header(text, _NAVALL_REQUIRED)
    if split is None:
        return None
    header, rows = split
    i_code = header.index("Scheme Code")
    i_name = header.index("Scheme Name")
    wanted = str(int(scheme_code))
    for r in rows:
        if r[i_code] == wanted:
            parts = [r[i_name]]
            for extra in ("Plan", "Option"):
                if extra in header and r[header.index(extra)]:
                    parts.append(r[header.index(extra)])
            return " - ".join(parts)
    return None


def _decode(response) -> str:
    """Decode a response body as UTF-8 ourselves; requests guesses wrongly for these files."""
    return response.content.decode("utf-8-sig", errors="replace")


def _official_scheme_name(scheme_code: int) -> Optional[str]:
    """Fetch NAVAll.txt and return the scheme name, or None on any problem."""
    try:
        response = requests.get(NAVALL_URL, headers=_REQUEST_HEADERS, timeout=60)
        if response.status_code != 200:
            logger.warning(f"[AMFI] NAVAll.txt returned status {response.status_code}")
            return None
        name = _parse_navall_name(_decode(response), scheme_code)
        if name is None:
            logger.warning(f"[AMFI] Scheme {scheme_code} not found in NAVAll.txt (or its layout changed)")
        return name
    except Exception as e:
        logger.warning(f"[AMFI] Could not read NAVAll.txt: {e}")
        return None


def fetch_nav_official(scheme_code, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch NAV history for one scheme from AMFI's own published files.

    Unlike fetch_nav, a date range is required because AMFI serves history
    per date range. The range is fetched in short chunks. The files cover
    every scheme, so each request is large (about 1 MB per day requested):
    use this for cross-checking and filling gaps over short windows, not
    for bulk history.

    If a chunk cannot be fetched or its layout is not recognised, that
    chunk is skipped and a warning naming the date range is logged, so the
    result may be incomplete. A chunk with no rows for the scheme (for
    example before the scheme existed) is normal and is not a failure.

    Args:
        scheme_code : AMFI scheme code, e.g. 119551
        from_date   : Start date in YYYY-MM-DD format
        to_date     : End date in YYYY-MM-DD format

    Returns:
        pandas DataFrame with columns:
        date, scheme_code, scheme_name, nav, source ("AMFI_OFFICIAL")
        Sorted oldest to newest. Empty DataFrame on any failure; never raises.
    """
    try:
        code = int(scheme_code)
    except (TypeError, ValueError):
        logger.error(f"[AMFI] Invalid scheme code: {scheme_code!r}")
        return empty_frame("nav")

    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        logger.error("[AMFI] Invalid date format. Use YYYY-MM-DD")
        return empty_frame("nav")

    if start > end:
        logger.error("[AMFI] from_date cannot be after to_date")
        return empty_frame("nav")

    try:
        frames: List[pd.DataFrame] = []
        failed: List[str] = []
        unreadable_total = 0
        history_name = ""

        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=_OFFICIAL_CHUNK_DAYS - 1), end)
            label = f"{chunk_start.date()} to {chunk_end.date()}"
            params = {
                "frmdt": chunk_start.strftime("%d-%b-%Y"),
                "todt": chunk_end.strftime("%d-%b-%Y"),
            }
            try:
                response = requests.get(HISTORY_URL, params=params,
                                        headers=_REQUEST_HEADERS, timeout=120)
                if response.status_code != 200:
                    logger.error(f"[AMFI] History request failed with status "
                                 f"{response.status_code} for {label}")
                    failed.append(label)
                else:
                    parsed = _parse_official_history(_decode(response), code)
                    if parsed is None:
                        logger.error(f"[AMFI] Unrecognised history file layout for {label}; "
                                     f"AMFI may have changed its format")
                        failed.append(label)
                    else:
                        df_chunk, bad, nav_name = parsed
                        unreadable_total += bad
                        if nav_name and not history_name:
                            history_name = nav_name
                        if not df_chunk.empty:
                            frames.append(df_chunk)
            except Exception as e:
                logger.error(f"[AMFI] History request error for {label}: {e}")
                failed.append(label)
            chunk_start = chunk_end + timedelta(days=1)

        if unreadable_total:
            logger.warning(f"[AMFI] Dropped {unreadable_total} official rows with an "
                           f"unreadable date or NAV for scheme {code}")
        if failed:
            logger.warning(f"[AMFI] Result for scheme {code} may be incomplete; "
                           f"could not read: {', '.join(failed)}")

        if not frames:
            logger.warning(f"[AMFI] No official NAV rows found for scheme {code} "
                           f"from {from_date} to {to_date}")
            return empty_frame("nav")

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        name = _official_scheme_name(code) or history_name
        df["scheme_code"] = code
        df["scheme_name"] = name
        df["source"] = _OFFICIAL_SOURCE
        df = df[["date", "scheme_code", "scheme_name", "nav", "source"]]
        return df.astype(SCHEMAS["nav"])

    except Exception as e:
        logger.error(f"[AMFI] Exception for official NAV of scheme {scheme_code}: {e}")
        return empty_frame("nav")