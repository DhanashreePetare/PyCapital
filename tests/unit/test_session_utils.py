from unittest.mock import patch
from pyequity.connectors._session_utils import create_nse_session


def test_session_does_not_request_brotli():
    # requests cannot decode brotli unless an extra package is installed
    with patch("requests.Session.get"), patch("time.sleep"):
        for symbol in (None, "RELIANCE"):
            s = create_nse_session(symbol)
            assert "br" not in s.headers["Accept-Encoding"]