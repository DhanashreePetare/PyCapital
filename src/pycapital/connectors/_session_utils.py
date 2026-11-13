import requests
import time


def create_nse_session():
    """
    Create and warm up a generic NSE session.
    Shared by any NSE connector that doesn't need a symbol-specific
    quote-page warmup (corporate actions, F&O, indices).
    fetch_nse_historical() in nse.py uses its own symbol-specific
    variant and is intentionally left untouched.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.nseindia.com",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "Connection": "keep-alive"
    })
    session.get("https://www.nseindia.com", timeout=15)
    time.sleep(3)
    return session