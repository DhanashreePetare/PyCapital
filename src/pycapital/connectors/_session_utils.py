import requests
import time

def create_nse_session(symbol=None):
    """
    Create and warm up an NSE session.

    Args:
        symbol: Optional NSE symbol. If given, the session is warmed up
                through that stock's quote page (needed by the equity
                historical endpoint). If None, a generic homepage-only
                warm-up is used (corporate actions, F&O, indices, bulk deals).
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
        "Accept-Encoding": "gzip, deflate, br" if symbol else "gzip, deflate",
        "Referer": (
            f"https://www.nseindia.com/get-quote/equity/{symbol}"
            if symbol else "https://www.nseindia.com"
        ),
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "Connection": "keep-alive"
    })

    session.get("https://www.nseindia.com", timeout=15)
    time.sleep(3)

    if symbol:
        session.get(f"https://www.nseindia.com/get-quote/equity/{symbol}", timeout=15)
        time.sleep(2)

    return session