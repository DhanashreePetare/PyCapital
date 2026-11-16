"""Capture real raw API responses as offline test fixtures.

Run manually, from the repository root, with internet access:

    python tools/capture_fixtures.py

Not part of the library. For each endpoint it calls the connector's own fetch
function once, records the raw HTTP body exactly as received (no endpoint URLs
or parameters are repeated here, so this cannot drift from the connectors),
and saves it under tests/fixtures/ together with a manifest.json describing
where and when it was captured.

Only fixed, short, historical date ranges are used, so the fixtures are small
and the same connectors can be replayed on them offline. The one exception is
the AMFI response, which covers 13 years: it is cut to its first AMFI_KEEP
entries (structure unchanged) and the manifest says so.

Exit code is 0 only if every case was captured as expected.
"""
import json
import logging
import sys
import time
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from pyequity.connectors._session_utils import create_nse_session
from pyequity.connectors.amfi import fetch_nav
from pyequity.connectors.bse import _fetch_bse_chunk
from pyequity.connectors.corporate import fetch_corporate_actions
from pyequity.connectors.fno import _fetch_fno_chunk
from pyequity.connectors.indices import _fetch_indices_chunk
from pyequity.connectors.nse import _fetch_nse_chunk
from pyequity.connectors.sebi import fetch_bulk_deals

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
AMFI_KEEP = 30
PAUSE_SECONDS = 1


class Recorder:
    """Forwards a GET call to the real function and remembers every response."""

    def __init__(self, real_get):
        self._real_get = real_get
        self.responses = []

    def get(self, *args, **kwargs):
        response = self._real_get(*args, **kwargs)
        self.responses.append(response)
        return response


def capture(rec, name, call_desc, call, *, expect_empty=False, keep_data=None):
    """Run one connector call, check it behaved, save the raw body.

    Returns the manifest entry. Raises RuntimeError if the capture is not
    trustworthy (wrong status, not JSON, unexpected empty/non-empty result).
    """
    rec.responses.clear()
    result = call()
    if len(rec.responses) != 1:
        raise RuntimeError(f"expected exactly 1 HTTP response, got {len(rec.responses)}")
    response = rec.responses[0]
    body = response.content

    try:
        payload = json.loads(body)
        is_json = True
    except ValueError:
        payload = None
        is_json = False

    if expect_empty:
        if not result.empty:
            raise RuntimeError("expected an empty result but the connector returned rows")
    else:
        if response.status_code != 200:
            raise RuntimeError(f"HTTP status {response.status_code}")
        if not is_json:
            raise RuntimeError("body is not valid JSON (first bytes: %r)" % body[:60])
        if result.empty:
            raise RuntimeError("connector returned no rows; fixture would be useless")

    original_entries = truncated_to = None
    if keep_data is not None:
        entries = payload["data"]
        original_entries, truncated_to = len(entries), keep_data
        if original_entries <= keep_data:
            raise RuntimeError("nothing to truncate; response already small")
        payload["data"] = entries[:keep_data]
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    filename = f"{name}.{'json' if is_json else 'txt'}"
    (FIXTURE_DIR / filename).write_bytes(body)
    return {
        "file": filename,
        "connector_call": call_desc,
        "source_url": response.url,
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type"),
        "bytes": len(body),
        "expect_empty": expect_empty,
        "live_rows": len(result),
        "original_data_entries": original_entries,
        "truncated_to_entries": truncated_to,
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    real_requests_get = requests.get  # taken before it is patched below
    rec_http = Recorder(real_requests_get)

    print("Opening NSE sessions (takes a few seconds)...")
    eq_session = create_nse_session("RELIANCE")
    gen_session = create_nse_session()
    rec_eq = Recorder(eq_session.get)
    rec_gen = Recorder(gen_session.get)
    eq_proxy = SimpleNamespace(get=rec_eq.get)
    gen_proxy = SimpleNamespace(get=rec_gen.get)

    # (recorder, fixture name, description, callable, options)
    cases = [
        (rec_eq, "nse_equity_reliance",
         "_fetch_nse_chunk(session, 'RELIANCE', '15-09-2026', '23-09-2026', 'EQ')",
         lambda: _fetch_nse_chunk(eq_proxy, "RELIANCE", "15-09-2026", "23-09-2026", "EQ"), {}),
        (rec_eq, "nse_equity_invalid_symbol",
         "_fetch_nse_chunk(session, 'WRONGSYMBOL123', ...) (session warmed for RELIANCE)",
         lambda: _fetch_nse_chunk(eq_proxy, "WRONGSYMBOL123", "15-09-2026", "23-09-2026", "EQ"),
         {"expect_empty": True}),
        (rec_http, "bse_reliance",
         "_fetch_bse_chunk('500325', '2026-09-15', '2026-09-23')",
         lambda: _fetch_bse_chunk("500325", "2026-09-15", "2026-09-23"), {}),
        (rec_http, "bse_invalid_scripcode",
         "_fetch_bse_chunk('999999', '2026-09-15', '2026-09-23')",
         lambda: _fetch_bse_chunk("999999", "2026-09-15", "2026-09-23"), {"expect_empty": True}),
        (rec_gen, "indices_nifty50",
         "_fetch_indices_chunk(session, 'NIFTY 50', '15-09-2026', '23-09-2026')",
         lambda: _fetch_indices_chunk(gen_proxy, "NIFTY 50", "15-09-2026", "23-09-2026"), {}),
        (rec_gen, "fno_reliance_futstk",
         "_fetch_fno_chunk(session, 'RELIANCE', '15-09-2026', '23-09-2026', '27-Oct-2026', 'FUTSTK')",
         lambda: _fetch_fno_chunk(gen_proxy, "RELIANCE", "15-09-2026", "23-09-2026",
                                  "27-Oct-2026", "FUTSTK"), {}),
        (rec_gen, "corporate_actions_reliance",
         "fetch_corporate_actions('RELIANCE')",
         lambda: fetch_corporate_actions("RELIANCE"), {}),
        (rec_gen, "sebi_bulk_deals_2026-09-21",
         "fetch_bulk_deals('2026-09-21', '2026-09-21')",
         lambda: fetch_bulk_deals("2026-09-21", "2026-09-21"), {}),
        (rec_http, "amfi_nav_119551",
         "fetch_nav(119551)",
         lambda: fetch_nav(119551), {"keep_data": AMFI_KEEP}),
        (rec_http, "amfi_invalid_scheme",
         "fetch_nav(999999999)",
         lambda: fetch_nav(999999999), {"expect_empty": True}),
    ]

    entries, failures = {}, []
    with ExitStack() as stack:
        stack.enter_context(patch("requests.get", side_effect=rec_http.get))
        stack.enter_context(patch("pyequity.connectors.corporate.create_nse_session",
                                  return_value=gen_proxy))
        stack.enter_context(patch("pyequity.connectors.sebi.create_nse_session",
                                  return_value=gen_proxy))
        for rec, name, desc, call, options in cases:
            try:
                entries[name] = capture(rec, name, desc, call, **options)
                e = entries[name]
                print(f"  OK    {name:32s} status={e['http_status']} bytes={e['bytes']:>7} "
                      f"live_rows={e['live_rows']}")
            except Exception as exc:  # report and carry on with the other cases
                failures.append(name)
                print(f"  FAIL  {name:32s} {exc}")
            time.sleep(PAUSE_SECONDS)

    manifest = {
        "captured_on": date.today().isoformat(),
        "generated_by": "tools/capture_fixtures.py",
        "note": ("Raw response bodies exactly as received, except where "
                 "truncated_to_entries is set. Do not edit by hand; re-run the tool."),
        "fixtures": entries,
    }
    (FIXTURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {len(entries)} fixture(s) and manifest.json to {FIXTURE_DIR}")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())