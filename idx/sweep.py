"""
Field-first per-field sweep -- CLAUDE.md's `locate()`. The naive loop is
"identify the template, then read its known positions"; this inverts it:
for one field, search the ENTIRE page for it, by label anchor first, then
by value signature alone, regardless of what any other field's search
found.

Wired in TODAY as the engine behind `generic_ocr.py`'s last-resort tier
ONLY -- see docs/ARCHITECTURE.md's "additive fallback layer" decision
(confirmed with the user before this was built): barcode/MRZ/AAMVA/
Ethiopia-ID's own parsers are untouched. This replaces generic_ocr.py's
whole-text-blob regex search (a bounded CHARACTER window after wherever a
label's regex happened to match in one long joined string) with genuine
per-token POSITIONAL search -- real label-anchor geometry (right, then
below), not a character count that has no idea where the actual next
printed row starts.

Requires per-token bounding boxes (`ocr_engine.ocr_words`), unlike every
other OCR call in this project (`ocr_engine.ocr_text`, a plain text blob
with no position at all) -- this is the one new prerequisite the
field-first design needs, and the reason it's introduced as its own
module rather than retrofitted into the existing whole-text callers.

Step D (rotated regions) is deliberately NOT a case inside `locate` itself
-- it's the caller's job to re-invoke `locate` against a rotated variant's
own tokens, the same rotation-retry idiom already proven in mrz_ocr.py's
`find_mrz_candidates` and aamva.py's `_ocr_texts_rotated` (try upright
first, only pay for a rotated re-OCR if nothing was found).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

from . import ontology


@dataclass
class OcrToken:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 0.0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def cy(self) -> int:
        return self.top + self.height // 2


@dataclass
class Candidate:
    value: str
    tier: int   # CLAUDE.md's Precedence table -- 3 (label_anchor) or 5 (signature_sweep) from this module
    source: str   # "label_anchor" | "signature_sweep"
    tokens: list[OcrToken]
    anchor: Optional[OcrToken] = None


def tokens_from_image(pil_image: Any) -> list[OcrToken]:
    """Positioned OCR tokens for one image -- the field-first sweep's
    input. Blank-text detections (pure whitespace/noise boxes an OCR
    engine still emits a box for) are dropped; nothing else is filtered
    here, so a low-confidence real word is still a candidate rather than
    silently discarded before it ever reaches label-anchor/signature
    matching.

    Delegates to `ocr_engine.ocr_words` -- see that module's docstring for
    which engine actually runs and why. RapidOCR detects whole LINES
    rather than individual words, so `ocr_engine._line_to_words` splits
    each line into word-shaped boxes proportional to character span; this
    function stays engine-agnostic and just adapts the shared `OcrWord`
    into this module's own `OcrToken`.
    """
    from .ocr_engine import ocr_words

    return [
        OcrToken(
            text=w.text, left=w.left, top=w.top, width=w.width, height=w.height,
            confidence=getattr(w, "confidence", 0.0) or 0.0,
        )
        for w in ocr_words(pil_image)
    ]


def _same_row(anchor: OcrToken, tok: OcrToken) -> bool:
    """Vertically overlapping enough to be considered the same printed
    row -- tolerant of a few pixels' baseline jitter between adjacent
    tokens rather than requiring exact top/height equality.
    """
    return abs(tok.cy - anchor.cy) <= max(anchor.height, tok.height) * 0.6


def _overlaps_x(anchor: OcrToken, tok: OcrToken) -> bool:
    """Share any horizontal span with the label -- the printed column
    under 15/Sex, not the date sitting in the next column to the left
    on the same value row (confirmed on a real AAMVA numbered front:
    DOB | SEX | EYES share one value row, so reading 'below' left-to-
    right hits the date tokens first and never reaches the F under Sex).
    """
    return tok.left < anchor.right and tok.right > anchor.left


def _ordered_candidates(anchor: OcrToken, tokens: list[OcrToken]) -> list[OcrToken]:
    """CLAUDE.md's `locate()` step A: "right, then below" from a matched
    label token. Same-row tokens strictly to the right of the anchor,
    left-to-right, THEN tokens on rows below the anchor (within a
    generous vertical range -- a handful of row heights, covering a
    label whose value sits a line or two down). Below-tokens that share
    the label's column are tried before the rest of that row: a
    three-column AAMVA band prints independent values on one line, and
    left-to-right would otherwise attribute the leftmost column's value
    to every label on the row above.
    """
    same_row = sorted(
        (t for t in tokens if t is not anchor and _same_row(anchor, t) and t.left >= anchor.right),
        key=lambda t: t.left,
    )
    below = [
        t for t in tokens
        if t is not anchor and t.cy > anchor.bottom
        and t.top <= anchor.bottom + anchor.height * 6
    ]
    below.sort(key=lambda t: (0 if _overlaps_x(anchor, t) else 1, t.top, t.left))
    return same_row + below


def search_from_anchor(
    anchor: OcrToken, tokens: list[OcrToken], value_pattern: re.Pattern[str],
    max_tokens: int = 6, max_skip: int = 3,
) -> tuple[Optional[str], list[OcrToken]]:
    """Tries joining the LONGEST prefix first (`max_tokens` down to 1) of
    the ordered candidate tokens after `anchor` into one phrase, taking
    the first (longest) count whose ENTIRE joined phrase matches
    `value_pattern` -- handles both single-token values (a date, a
    document number) and multi-token ones (a name spanning several
    words) without needing to know in advance how many words the value
    has. Longest-first matters: a pattern like "1 to 4 space-separated
    words" would otherwise stop at the first single word ("ANNA") and
    never try the fuller two-word phrase ("ANNA MARIA") that's actually
    the real value, since the optional trailing words make the short
    prefix ALSO a technically-complete match. Returns (value,
    matched_tokens) or (None, []) if no prefix ever matches.

    If NO prefix starting at the very next candidate matches at all, this
    retries starting one candidate further along (up to `max_skip` times)
    before giving up entirely -- confirmed necessary against a real Dutch
    ID card: `_same_row`'s tolerance can legitimately misjudge an
    unrelated, larger-font token (the document number, in a much bigger
    typeface elsewhere on the same visual band) as "same row, to the
    right" of a label whose real value sits below it instead. Without
    retrying from the next candidate, that one misclassified token
    poisons the ENTIRE search -- every phrase length still includes it as
    the first word, so nothing ever matches and the real value one
    position later is never even attempted. `max_skip` is small and
    bounded on purpose: this is a recovery from one wrong leading
    candidate, not a license to wander arbitrarily far down the page
    looking for something that happens to fit the shape.

    Known limitation: greedy longest-first can still overshoot into the
    NEXT field's own label if it happens to share the same value shape
    (e.g. an all-caps word) and sits within `max_tokens` of a skip
    position that DOES produce a match -- there is no boundary check
    against "the next token is itself a label anchor for some other
    field" yet. Acceptable for the tier-5 last-resort role this is wired
    into today (see docs/ARCHITECTURE.md); worth revisiting before this
    ever anchors a higher-precedence tier.
    """
    ordered = _ordered_candidates(anchor, tokens)
    for start in range(min(max_skip + 1, len(ordered))):
        remaining = ordered[start:]
        for n in range(min(max_tokens, len(remaining)), 0, -1):
            candidate_tokens = remaining[:n]
            phrase = " ".join(t.text for t in candidate_tokens)
            m = value_pattern.match(phrase)
            if m and m.end() == len(phrase):
                return phrase, candidate_tokens
    return None, []


_MAX_LABEL_PHRASE_WORDS = 4


def _label_phrase_candidates(start: OcrToken, tokens: list[OcrToken]) -> list[list[OcrToken]]:
    """Same-row tokens from `start` rightward, as increasingly long
    candidate label phrases (longest first) -- many real aliases are
    themselves multi-word ("given names", "date of birth"), and an OCR
    engine commonly recognizes each word of a printed label as its OWN
    positioned token rather than recombining them; a single-token check
    alone would never see the whole label. Confirmed necessary against a
    real Dutch ID card: "given names" prints as two separate same-row
    tokens, and neither one alone is a known alias by itself.
    """
    row = [start] + sorted(
        (t for t in tokens if t is not start and _same_row(start, t) and t.left >= start.right),
        key=lambda t: t.left,
    )
    return [row[:n] for n in range(min(_MAX_LABEL_PHRASE_WORDS, len(row)), 0, -1)]


def _union_box(phrase_tokens: list[OcrToken]) -> OcrToken:
    """A synthetic token spanning the WHOLE label phrase's bounding box --
    the effective anchor `search_from_anchor` should search from. Using
    just the phrase's LAST token instead (e.g. "no." at the tail of
    "documentnummer/document no.") is wrong whenever the real value sits
    below the label's own START rather than its end: confirmed against a
    real Dutch ID card, where the document number prints below-and-left
    of "documentnummer/document", not below the short "no." that happens
    to trail it -- anchoring on "no." alone put the value outside both
    the same-row and below search ranges entirely.
    """
    left = min(t.left for t in phrase_tokens)
    top = min(t.top for t in phrase_tokens)
    right = max(t.right for t in phrase_tokens)
    bottom = max(t.bottom for t in phrase_tokens)
    return OcrToken(
        text=" ".join(t.text for t in phrase_tokens), left=left, top=top, width=right - left, height=bottom - top,
    )


def locate(
    field: str, tokens: list[OcrToken], value_pattern: re.Pattern[str], allow_unlabeled: bool = True,
) -> list[Candidate]:
    """CLAUDE.md's `locate()`, steps A and B (C -- embedded in free text
    -- has no caller yet; D is the caller's own rotation-retry loop, see
    module docstring). Never raises on tokens/patterns that find nothing;
    an empty list is a legitimate answer, not a defect.

    `allow_unlabeled=False` skips step B (value signature) entirely --
    for a field whose "signature" is really just a generic name shape
    ("1-4 capitalized words"), practically ANY prominent capitalized text
    satisfies it, which isn't a real signature in CLAUDE.md's sense at
    all (a genuine one is narrow: a document-ID format, a date shape). A
    real Dutch ID card's own bilingual header ("NEDERLANDSE ... IDENTITY
    CARD") matched this shape before every OTHER field's label-anchor
    search even ran, and kept winning by coincidence for `full_name`/
    `nationality`/`place_of_birth` -- fields this module's OWN sibling,
    `extract_generic_fields`, already refuses to populate without a real
    nearby label, per its module docstring's "deliberately conservative"
    rule. This makes `sweep.locate` keep that same discipline for the
    fields whose value pattern is this generic, rather than silently
    being laxer than the whole-text search it's meant to complement.
    """
    out: list[Candidate] = []

    # A. Label anchor -- every alias this field's ontology entry has,
    # normalized/compound-split by `ontology.match_label` itself. Tries
    # the token alone AND joined with however many same-row tokens follow
    # it (longest phrase first) -- see `_label_phrase_candidates`.
    for tok in tokens:
        for phrase_tokens in _label_phrase_candidates(tok, tokens):
            phrase_text = " ".join(t.text for t in phrase_tokens)
            if ontology.match_label(phrase_text, field):
                # Search right/below from the WHOLE matched label phrase's
                # bounding box, not just its first or last word -- see
                # `_union_box`'s docstring for the real card that requires
                # this.
                value, matched = search_from_anchor(_union_box(phrase_tokens), tokens, value_pattern)
                if value:
                    out.append(Candidate(
                        value=value, tier=3, source="label_anchor", tokens=matched,
                        anchor=_union_box(phrase_tokens),
                    ))
                break

    # B. Value signature -- unlabeled, capped at tier 5 regardless of how
    # confident the shape match looks (CLAUDE.md: "unlabelled is real
    # evidence but capped at tier 5").
    if allow_unlabeled:
        for tok in tokens:
            if value_pattern.fullmatch(tok.text):
                out.append(Candidate(value=tok.text, tier=5, source="signature_sweep", tokens=[tok]))

    return out
