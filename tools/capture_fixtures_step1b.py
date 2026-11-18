"""
Capture raw responses for the AMFI official files and NSE bulk.csv.

Run manually (needs internet); not part of the library. Writes to
tests/fixtures/step1b/ and a manifest (manifest.json) with URL, parameters,
size and sha256 of every saved file, so tests can prove the fixtures were not
edited by hand.

The AMFI files cover every scheme (megabytes), so they are SHRUNK before
saving: the header line, the first few non-data lines (category and fund
house headings) and the data lines of a few chosen schemes are kept,
byte-for-byte; every other line is removed. The manifest says so. bulk.csv is
saved whole.

Usage:  python tools/capture_fixtures_step1b.py
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from pyequity.connectors import amfi, sebi

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "step1b"
SCHEME = "119551"            # scheme the offline tests assert on
HISTORY_FROM, HISTORY_TO = "15-Sep-2026", "23-Sep-2026"
EXTRA_SCHEMES = 3            # other schemes kept so the file is not single-scheme
HEADING_LINES_KEPT = 3


def shrink(text, keep_codes_first_extra, required_header):
    """Keep header, a few heading lines, and data lines of chosen schemes."""
    lines = text.splitlines(keepends=True)
    header = None
    out, headings, extra_codes = [], 0, []
    for line in lines:
        fields = [f.strip() for f in line.rstrip("\r\n").split(";")]
        if header is None:
            if "Scheme Code" in fields:
                header = fields
                out.append(line)
            continue
        if len(fields) == len(header) and fields[0].isdigit():
            code = fields[0]
            if code == SCHEME:
                out.append(line)
            elif code in extra_codes:
                out.append(line)
            elif len(extra_codes) < keep_codes_first_extra:
                extra_codes.append(code)
                out.append(line)
        elif line.strip() and headings < HEADING_LINES_KEPT:
            headings += 1
            out.append(line)
    if header is None or any(c not in header for c in required_header):
        raise ValueError("expected header columns not found: " + str(required_header))
    return "".join(out), extra_codes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "files": {}}
    failures = 0

    def save(name, data: bytes, url, params, status, note):
        (OUT / name).write_bytes(data)
        manifest["files"][name] = {"url": url, "params": params, "status": status,
                                   "bytes": len(data),
                                   "sha256": hashlib.sha256(data).hexdigest(), "note": note}

    jobs = [
        ("amfi_history_119551_2026-09-15_2026-09-23.txt", amfi.HISTORY_URL,
         {"frmdt": HISTORY_FROM, "todt": HISTORY_TO},
         ("Scheme Code", "NAV Name", "Net Asset Value", "Date")),
        ("amfi_navall_sample.txt", amfi.NAVALL_URL, None,
         ("Scheme Code", "Scheme Name", "Net Asset Value", "Date")),
    ]
    for name, url, params, required in jobs:
        try:
            r = requests.get(url, params=params, headers=amfi._REQUEST_HEADERS, timeout=180)
            if r.status_code != 200:
                raise ValueError(f"status {r.status_code}")
            text = r.content.decode("utf-8-sig", errors="replace")
            small, extra = shrink(text, EXTRA_SCHEMES, required)
            if (SCHEME + ";") not in small:
                raise ValueError(f"scheme {SCHEME} not in response")
            save(name, small.encode("utf-8"), url, params, r.status_code,
                 f"SHRUNK from {len(r.content)} bytes: header + {HEADING_LINES_KEPT} heading "
                 f"lines + data lines of schemes {SCHEME} and {extra}; kept lines unchanged")
            print(f"  OK    {name:52s} bytes={len(small.encode("utf-8")):7d} (from {len(r.content)})")
        except Exception as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")

    try:
        got = None
        for url in sebi._BULK_CSV_URLS:
            r = requests.get(url, headers=sebi._CSV_HEADERS, timeout=60)
            if r.status_code == 200:
                got = (url, r)
                break
        if got is None:
            raise ValueError("no bulk.csv address answered 200")
        url, r = got
        parsed = sebi._parse_bulk_csv(r.content.decode("utf-8-sig", errors="replace"))
        if parsed is None or parsed.empty:
            raise ValueError("bulk.csv did not parse to any rows")
        save("bulk_latest.csv", r.content, url, None, r.status_code,
             f"saved whole; {len(parsed)} rows parsed")
        print(f"  OK    {'bulk_latest.csv':52s} bytes={len(r.content):7d} rows={len(parsed)}")
    except Exception as e:
        failures += 1
        print(f"  FAIL  bulk_latest.csv: {e}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(manifest['files'])} fixture(s) and manifest.json to {OUT}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())