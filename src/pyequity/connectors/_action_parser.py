"""
Parser for NSE corporate action descriptions (the "subject" field).

NSE describes each action as free text, for example ``Bonus 1:1``,
``Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share``
or ``Rights 1:15 @ Premium Rs 1247``. This module turns that text into
structured values so price adjustment does not have to read English.

The rules were written against 392 distinct real descriptions (2006-2026,
29 NSE symbols) and checked against them one by one. Principles:

* Parse a normalised copy of the text. The caller keeps the raw text unchanged.
* Detect every category independently, so combined actions such as
  ``Bonus 1:1/Face Value Split ...`` keep all their parts.
* Never guess. A number is only returned when the text matches a known format
  completely. Anything else is flagged ``needs_review``.
* ``needs_review`` also covers actions that cannot be adjusted from a ratio
  alone (demerger, merger, scheme of arrangement) and every unknown text.

Ratios are recorded exactly as written: the first number is the NEW shares,
the second number is the shares HELD (``Bonus 1:3`` = 1 new share for every
3 held). This direction was verified against NSE circulars for POWERGRID
(bonus 1:3), BAJFINANCE (bonus 4:1) and RELIANCE (rights 1:15).

``rights_premium`` is the premium per share as written, NOT the issue price.
The issue price is premium plus face value.
"""
import math
import re
from typing import Dict, Iterable, List

# Fixed order of categories in the pipe-delimited output.
CATEGORY_ORDER = (
    "BONUS", "SPLIT", "RIGHTS", "DIVIDEND", "BUYBACK", "DEMERGER", "MERGER",
    "SCHEME_OF_ARRANGEMENT", "AGM", "EVOTING", "OTHER",
)

STATUS_OK = "ok"
STATUS_REVIEW = "needs_review"

_NUM = r"(\d+(?:\.\d+)?)"
_RS = r"r[se]\s?\.?"  # Rs, Rs., Re, Re.

# Share bonus only: "bonus debentures", "bonus preference shares" etc. are not.
_BONUS = re.compile(r"\bbonus\b(?!\s+(?:debenture|preference|pref\b|ncd|warrant|redeemable))")
_BONUS_FULL = re.compile(r"^bonus\s*(\d+)\s*:\s*(\d+)$")

_SPLIT = re.compile(r"\bsplit\b|\bsplt\b|sub-?\s?division")
_SPLIT_FULL = re.compile(
    rf"^(?:face value split|fv splt|fv split)?\s*(?:\(sub-?\s?division\))?\s*-?\s*"
    rf"(?:from|frm)\s*{_RS}\s*{_NUM}\s*(?:per share)?\s*to\s*{_RS}\s*{_NUM}\s*(?:per share)?$"
)

_RIGHTS = re.compile(r"\brights?\b")
_RIGHTS_FULL = re.compile(
    rf"^rights\s*(\d+)\s*:\s*(\d+)\s*@\s*premium\s*{_RS}\s*{_NUM}\s*(?:per share)?$"
)

_DIVIDEND = re.compile(r"\bdiv(?:idend|dend)?\b")  # "Divdend" is a real NSE typo
_BUYBACK = re.compile(r"\bbuy[\s-]?back\b")
_DEMERGER = re.compile(r"\bde-?\s?merger\b")
_MERGER = re.compile(r"\bmerger\b|\bamalgamation\b")
_SCHEME = re.compile(r"\bscheme\b")
_AGM = re.compile(r"\bagm\b|annual general meeting")
_AGM_ONLY = re.compile(r"^(?:agm|annual general meeting)$")
_EVOTING = re.compile(r"\be-?\s?voting\b")
_REVISED = re.compile(r"\((?:[a-z ]*)revised\)")
_RATIO = re.compile(r"\d+\s*:\s*\d+")

# Words that signal a price-affecting event we have no rule for.
_RISKY = re.compile(
    r"\bfv\b|face value|consolidat|reduction|redempt|warrant|spin|conversion|"
    r"debenture|preference"
)


def _is_nan(value: object) -> bool:
    """True for a float NaN (the 'not parsed yet' marker used below)."""
    return isinstance(value, float) and math.isnan(value)


def _empty_result() -> Dict[str, object]:
    nan = float("nan")
    return {
        "action_categories": "OTHER",
        "bonus_new": nan, "bonus_held": nan,
        "split_old_fv": nan, "split_new_fv": nan,
        "rights_new": nan, "rights_held": nan, "rights_premium": nan,
        "parse_status": STATUS_REVIEW,
    }


def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace, drop markers that carry no meaning."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = _REVISED.sub("", t)
    t = t.replace("/-", "")
    t = re.sub(r"\b(r[se])\.", r"\1 ", t)  # "Rs.10" -> "rs 10"
    return re.sub(r"\s+", " ", t).strip(" .")


def parse_action_detail(text: object) -> Dict[str, object]:
    """
    Turn one NSE corporate action description into structured fields.

    Never raises. ``None``, NaN, empty or non-text input gives category
    ``OTHER`` with status ``needs_review``.

    Args:
        text: The raw ``subject`` text from NSE, e.g. ``" Bonus 1:1"``.
            Leading and trailing spaces are ignored for parsing.

    Returns:
        Dict with these keys:

        * ``action_categories``: pipe-delimited categories in a fixed order,
          e.g. ``"BONUS|SPLIT"``. One of BONUS, SPLIT, RIGHTS, DIVIDEND,
          BUYBACK, DEMERGER, MERGER, SCHEME_OF_ARRANGEMENT, AGM, EVOTING,
          OTHER. AGM and EVOTING appear only when nothing else is found.
        * ``bonus_new``, ``bonus_held``: from ``Bonus a:b`` (NaN if absent).
        * ``split_old_fv``, ``split_new_fv``: face values from a split.
        * ``rights_new``, ``rights_held``, ``rights_premium``: from
          ``Rights a:b @ Premium Rs P``.
        * ``parse_status``: ``"ok"`` when everything detected was parsed
          cleanly, ``"needs_review"`` for demerger, merger, scheme, unparsed
          or ambiguous price-affecting text, and unknown text.
    """
    result = _empty_result()
    if not isinstance(text, str):
        return result
    t = _normalise(text)
    if not t:
        return result

    cats = set()
    problems: List[str] = []

    without_demerger = _DEMERGER.sub(" ", t)
    if _DEMERGER.search(t):
        cats.add("DEMERGER")
    if _MERGER.search(without_demerger):
        cats.add("MERGER")
    if _SCHEME.search(t):
        cats.add("SCHEME_OF_ARRANGEMENT")
    if _BUYBACK.search(t):
        cats.add("BUYBACK")
    if _DIVIDEND.search(t):
        cats.add("DIVIDEND")

    # "/" separates actions in one description, e.g. "Bonus 1:1/Face Value Split ..."
    segments = [s.strip(" -.") for s in re.split(r"\s*/\s*", t) if s.strip(" -.")]
    for seg in segments:
        if _BONUS.search(seg):
            cats.add("BONUS")
            m = _BONUS_FULL.match(seg)
            if m and _is_nan(result["bonus_new"]):  # still NaN
                result["bonus_new"], result["bonus_held"] = float(m.group(1)), float(m.group(2))
            else:
                problems.append("bonus not parsed")
        elif _SPLIT.search(seg):
            cats.add("SPLIT")
            m = _SPLIT_FULL.match(seg)
            if m and _is_nan(result["split_old_fv"]):
                result["split_old_fv"], result["split_new_fv"] = float(m.group(1)), float(m.group(2))
            else:
                problems.append("split not parsed")
        elif _RIGHTS.search(seg):
            cats.add("RIGHTS")
            m = _RIGHTS_FULL.match(seg)
            if m and _is_nan(result["rights_new"]):
                result["rights_new"] = float(m.group(1))
                result["rights_held"] = float(m.group(2))
                result["rights_premium"] = float(m.group(3))
            else:
                problems.append("rights not parsed")
        elif (_DIVIDEND.search(seg) or _BUYBACK.search(seg) or _AGM_ONLY.match(seg)
              or _EVOTING.search(seg) or _SCHEME.search(seg) or _DEMERGER.search(seg)
              or _MERGER.search(seg)):
            continue
        elif cats & {"BONUS", "SPLIT", "RIGHTS"}:
            problems.append("unexplained text next to a price-affecting action")

    price_affecting = bool(cats & {"BONUS", "SPLIT", "RIGHTS"})
    if not price_affecting and "SCHEME_OF_ARRANGEMENT" not in cats and _RISKY.search(t):
        problems.append("price-affecting keyword not parsed")

    if not cats:
        if _AGM.search(t):
            cats.add("AGM")
        elif _EVOTING.search(t):
            cats.add("EVOTING")
        else:
            cats.add("OTHER")

    if len(_RATIO.findall(t)) > 1:
        problems.append("more than one ratio")

    def _bad(a: str, b: str) -> bool:
        x, y = result[a], result[b]
        return not _is_nan(x) and not (x > 0 and y > 0)

    if _bad("bonus_new", "bonus_held"):
        problems.append("implausible bonus ratio")
    if _bad("rights_new", "rights_held"):
        problems.append("implausible rights ratio")
    old, new = result["split_old_fv"], result["split_new_fv"]
    if not _is_nan(old) and not (old > new > 0):
        problems.append("split is not a forward split")

    if cats & {"DEMERGER", "MERGER", "SCHEME_OF_ARRANGEMENT"}:
        problems.append("cannot be adjusted from a ratio")
    if "OTHER" in cats:
        problems.append("unknown format")

    result["action_categories"] = "|".join(c for c in CATEGORY_ORDER if c in cats)
    result["parse_status"] = STATUS_REVIEW if problems else STATUS_OK
    return result


def parse_action_details(texts: Iterable[object]) -> List[Dict[str, object]]:
    """
    Parse many descriptions.

    Args:
        texts: Any iterable of raw descriptions (e.g. a pandas Series).

    Returns:
        List of result dicts, one per input, in the same order.
    """
    return [parse_action_detail(t) for t in texts]
