# Step 1 research findings (member B)

Measured on real responses from the author's machine, 16 to 17 November 2026.
Each finding says what was verified and what was not. Nothing here comes from memory.
The probe scripts used are not part of the library.

## 1. AMFI official NAV files (fallback to mfapi.in)

### Latest-NAV file
- URLs that worked: `https://www.amfiindia.com/spages/NAVAll.txt` and the same path on `portal.amfiindia.com`. Both returned identical content (1,530,437 bytes).
- Semicolon-separated text with 8 columns: `Scheme Code; ISIN Div Payout/ ISIN Growth; ISIN Div Reinvestment; Scheme Name; Plan; Option; Net Asset Value; Date`.
- Older documentation describes 6 columns, so the format has changed before. Parse by header name, never by position, and fail loudly if a required column is missing.
- Category and fund-house heading lines sit between data rows and must be skipped.
- Dates look like `13-Nov-2026`. The file holds the latest NAV per scheme only, so it spans 836 distinct dates (wound-up schemes keep their last NAV, some from 2008).
- `requests` guessed the wrong encoding (apostrophes came out as `â`). Decode the bytes as UTF-8 explicitly.
- Scheme 119551: `Scheme Name + " - " + Plan + " - " + Option` equals mfapi's `scheme_name` exactly (one scheme checked).

### History download
- Endpoint: `https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx` with `frmdt` and `todt` (`DD-Mon-YYYY`), optional `mf` (AMC id) and `tp` (scheme type filter).
- Column order differs from the latest-NAV file: `Scheme Code; NAV Name; Plan; Option; ISIN Div Payout/ISIN Growth; ISIN Div Reinvestment; Net Asset Value; Date`. The `NAV Name` text is formatted differently from mfapi's name, so take names from the latest-NAV file where possible.
- Without `mf`, one request returned every fund house (8,738 schemes for 3 dates, 3.4 MB). With `mf=3` it returned 401 schemes for 9 dates (about 378 KB). No scheme-to-AMC mapping is needed.
- Depth: January 2018 returned 31 dates, so deep history exists.
- AMFI's page documents a 90-day maximum. A 92-day request still returned all 92 dates, so the limit was not enforced, but the code does not rely on that and requests 30-day chunks.
- NAVs are published on weekends for some schemes (9 distinct dates for 9 calendar days).
- Cost: roughly 1 MB per day of data, because every scheme is returned. Realistic for gap-filling and cross-checking short windows, not for 13 years of one scheme.

### Cross-check against mfapi.in
Scheme 119551, 15 to 23 September 2026: 7 dates on each side, same dates, 0 differing NAVs.
This is one scheme and one week. It is a first check, not proof.

### Data-quality notes
- 0 non-numeric NAV values in either file in these samples. Handle them defensively anyway (drop the row, log a count).

### Not verified
Other schemes, long ranges, any `N.A.` value in practice, rate limits or blocking under heavy use, and whether `tp` values are stable.

## 2. NSE bulk deals

- `https://archives.nseindia.com/content/equities/bulk.csv` (and the `nsearchives` host) holds the **latest trading day only**: 136 rows (13 Nov 2026) and 134 rows (16 Nov 2026), with one date each. It is complete for that day.
- Columns: `Date, Symbol, Security Name, Client Name, Buy/Sell, Quantity Traded, Trade Price / Wght. Avg. Price, Remarks`.
- The historical API (`/api/historicalOR/bulk-block-short-deals`) returned **exactly 70 rows for every single day tried** (10, 21, 22, 23 September 2026), while a complete day holds about 134 to 136. So the cap applies per day.
- A two-day query returned 70 rows from the first day only, and nothing from the second.
- All 70 API rows for 16 Nov were present in the CSV (70 of 70, none invented), but they were not simply the first 70 CSV rows. Which rows are kept is not predictable.
- NSE's own page text says 50 or 70 records depending on the page version. The measured value is 70.
- The earlier wording in `sebi.py` ("most recent ~70 kept, older missing") was wrong and has been corrected.
- No archive file or endpoint with bulk-deals history was found.

### Conclusion
Only the latest trading day can be obtained completely. Older days can only be obtained capped, so the library flags every capped day instead of presenting it as complete.

### Not verified
BSE's bulk-deals page as a second source, any date-parameterised CSV, and why NSE's cap differs from its page text.

## 3. What the code does about this

| Finding | Code |
|---|---|
| Official AMFI files are usable and agree with mfapi on the one scheme checked | `fetch_nav_official` (source `AMFI_OFFICIAL`), not yet wired in as an automatic fallback (Step 3) |
| One unreadable NAV used to empty the whole result | `fetch_nav` now drops only that row and logs a count |
| `bulk.csv` is complete for the latest day | `fetch_latest_bulk_deals` (source `SEBI_NSE_CSV`) |
| The API is capped on every day | `fetch_bulk_deals_by_day` flags every capped day in the log and `df.attrs` |
| Old cap wording was wrong | corrected in the `sebi.py` module docstring |

## 4. Open items for the team
- Step 3: wire `fetch_nav_official` as the fallback to `fetch_nav`.
- Step 4: the public API should warn when bulk deals contain capped days.
- Re-check every URL above before release. These are public endpoints and can change without notice.