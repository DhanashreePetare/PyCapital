# NSE to BSE fallback: design (draft, for team agreement)

Status: proposal. Owner: A. Lives in `reliability/fallback.py`, called by the
public `stocks.get_history(..., source="auto")` once that exists.

## Goal
If NSE cannot give us a stock's history, return BSE's data for the same stock
instead, say so clearly, and never mix the two exchanges inside one result.

## Decisions
1. **Whole range from one exchange.** NSE and BSE closes differ by about
   0.1 to 0.6 percent and their volumes are different (each exchange trades
   separately). Gap-filling one series from the other would create fake jumps.
2. **Trigger: NSE failed, not NSE empty.** Fall back only when the NSE request
   errored (network, HTTP error, bad response, dropped session). An empty but
   valid answer is not a failure: the symbol may have no trades in the range.
   This needs B's retry layer to report "failed" and "empty" separately, and the
   `nse.py` change that stops silently skipping failed chunks.
3. **Retry first.** B's retry with backoff runs before any fallback is tried.
4. **Mapping needed.** BSE needs a scripcode. If `symbol_map.py` has none, we do
   not guess: log it and return an empty frame.
5. **Marking, two ways, no schema change.**
   - the `source` column already says "NSE" or "BSE" on every row;
   - one WARNING is logged per fallback ("NSE failed for X (reason); using BSE").
   - Proposed: also set `df.attrs["fallback"] = {"requested": "NSE", "used": "BSE", "reason": ...}`.
6. **Opt out.** `source="NSE"` or `source="BSE"` forces one exchange and never
   falls back. `source="auto"` (default) is NSE first, then BSE.
7. **Adjustment.** Corporate actions come from NSE only. If prices came from BSE
   and the actions fetch also failed, adjusted data cannot be guaranteed: return
   unadjusted prices and log a warning.
8. **Both fail:** empty `equity` frame and an error log. Nothing raises.

## Open questions
- Also emit a Python `warnings.warn` (visible by default) besides the log line?
- Do we want a "cross-check" mode that fetches both exchanges and reports
  disagreements above a threshold? It is a selling point, but it doubles load.
- Should an empty NSE answer for a symbol that exists in `symbol_map` ever
  trigger a fallback? Current answer: no.

## Needs from others
- B: retry returning "failed" versus "empty", rate limiter, session manager.
- C: nothing, except corporate actions staying NSE-based.