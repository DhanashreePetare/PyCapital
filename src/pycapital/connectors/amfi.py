"""
AMFI Mutual Fund NAV connector for PyCapital.

Data source: https://api.mfapi.in (community-maintained, sourced from
official AMFI NAV disclosures). No API key, no session/auth needed —
this is a plain public JSON API, unlike NSE/BSE which need session
warm-up.

Note: scheme_code -> scheme_name is NOT hardcoded anywhere in this file.
scheme_name is always read live from the API response itself, since
AMFI scheme codes can map to different funds than expected — verify
via list_schemes() before assuming a code matches a fund name.
"""

import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://api.mfapi.in/mf"


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
        print(f"[AMFI] Error fetching scheme list: {e}")
        return pd.DataFrame(columns=["schemeCode", "schemeName"])


def fetch_nav(scheme_code, from_date=None, to_date=None) -> pd.DataFrame:
    """
    Fetch NAV history for a single AMFI mutual fund scheme.

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
        Sorted oldest to newest. Empty DataFrame on any failure —
        never raises, matching every other connector in this project.
    """
    columns = ["date", "scheme_code", "scheme_name", "nav", "source"]

    try:
        url = f"{BASE_URL}/{scheme_code}"
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            print(f"[AMFI] Error: status code {response.status_code} for scheme {scheme_code}")
            return pd.DataFrame(columns=columns)

        payload = response.json()

        if not payload.get("data"):
            print(f"[AMFI] No NAV data returned for scheme code {scheme_code}")
            return pd.DataFrame(columns=columns)

        scheme_name = payload.get("meta", {}).get("scheme_name", "")

        df = pd.DataFrame(payload["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["nav"] = df["nav"].astype(float)
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

        return df

    except Exception as e:
        print(f"[AMFI] Exception for scheme {scheme_code}: {e}")
        return pd.DataFrame(columns=columns)