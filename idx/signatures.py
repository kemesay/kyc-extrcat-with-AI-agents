"""
Value-signature registry: patterns that identify a field by the SHAPE of
its value alone, with no label required -- CLAUDE.md's `locate()`
pseudocode step B ("Value signature -- finds fields whose LABEL is
illegible or absent"). Always tier 5 (regex sweep, last resort) per
CLAUDE.md's Precedence table: an unlabeled shape match is real evidence,
but weaker than anything a label anchored it to.

Every pattern below is IMPORTED from wherever it was already defined and
proven against a real document, not redefined here -- a second, slightly
different copy of "what a date looks like" is exactly the kind of drift
this consolidation exists to prevent. See docs/GLOBAL_COVERAGE.md's "Value
signatures" table for the full target list; this is the subset this
project actually has confirmed patterns for today (postal_code and
iso3_country have no existing pattern anywhere in this codebase to import
-- not added here rather than guessed).
"""

from __future__ import annotations

import re

from .aamva import _HEIGHT_SHAPE
from .generic_ocr import _ALNUM_VALUE, _DATE_VALUE, _NAME_VALUE
from .mrz_ocr import MRZ_CHARSET

SIGNATURES: dict[str, re.Pattern[str]] = {
    "date": re.compile(_DATE_VALUE),
    "sex": re.compile(r"[MF]|Male|Female"),
    "height": re.compile(_HEIGHT_SHAPE),
    "document_number": re.compile(_ALNUM_VALUE),
    "name": re.compile(_NAME_VALUE),
    "mrz_line": re.compile(r"[" + re.escape(MRZ_CHARSET) + r"]{20,44}"),
}


def matches(signature: str, text: str) -> bool:
    """True when `text`, taken as a WHOLE value (not a substring), has
    the shape of `signature`. Unknown signature name -> False, never a
    guess about what that concept might look like.
    """
    pattern = SIGNATURES.get(signature)
    return bool(pattern and pattern.fullmatch(text.strip()))
