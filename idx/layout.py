"""
Layout understanding — reconstruct reading order and neighborhood from
OCR boxes. Line-by-line scanning of a joined text blob loses the
relationship between a label and its value on stacked, two-column, and
bilingual cards (Virginia name above value, Georgia DOB top-right,
Ethiopian Digital ID issue date up the left margin).

Every token on a Page knows its nearest left / right / above / below
neighbor, its reading-order line, a coarse block, and a script hint.
Extractors consume this; they do not re-implement geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .sweep import OcrToken, _same_row


def detect_script(text: str) -> str:
    """Coarse script of a token, not a country guess. Used to prefer
    alias sets; never to infer nationality or document type.
    """
    if any("\u1200" <= ch <= "\u137F" or "\u1380" <= ch <= "\u139F" or "\u2D80" <= ch <= "\u2DDF" for ch in text):
        return "ethiopic"
    if any("\u0600" <= ch <= "\u06FF" for ch in text):
        return "arabic"
    if any("\u4E00" <= ch <= "\u9FFF" for ch in text):
        return "han"
    if any("\u3040" <= ch <= "\u30FF" for ch in text):
        return "japanese"
    return "latin"


def _overlap_x(a: OcrToken, b: OcrToken) -> bool:
    return a.left < b.right and b.left < a.right


def _reading_order_lines(tokens: list[OcrToken]) -> list[int]:
    """Assign a line index in top-to-bottom reading order. Tokens that
    share a row (see sweep._same_row) get the same line number.
    """
    if not tokens:
        return []
    indexed = sorted(enumerate(tokens), key=lambda it: (it[1].top, it[1].left))
    line_of: list[int] = [0] * len(tokens)
    groups: list[list[int]] = []
    for orig_i, tok in indexed:
        placed = False
        for group in groups:
            if _same_row(tokens[group[0]], tok):
                group.append(orig_i)
                placed = True
                break
        if not placed:
            groups.append([orig_i])
    for line_n, group in enumerate(groups):
        for orig_i in group:
            line_of[orig_i] = line_n
    return line_of


def _block_for(tok: OcrToken, page_height: int, page_width: int) -> str:
    """Coarse region only — a spatial prior that may reorder, never
    exclude, candidates (CLAUDE.md). Header / personal / footer bands.
    """
    if page_height <= 0:
        return "personal_information"
    cy = tok.cy / page_height
    if cy < 0.18:
        return "header"
    if cy > 0.82:
        return "footer"
    if page_width > 0 and tok.left / page_width > 0.72:
        return "margin"
    return "personal_information"


@dataclass(frozen=True)
class Neighborhood:
    left: str
    right: str
    above: str
    below: str
    line: int
    block: str
    language: str
    bbox: tuple[int, int, int, int]


@dataclass
class Page:
    """One OCR page with neighbor links. `tokens[i]` pairs with
    `neighborhoods[i]`.
    """

    tokens: list[OcrToken]
    neighborhoods: list[Neighborhood]
    languages: tuple[str, ...]

    def neighborhood_of(self, tok: OcrToken) -> Optional[Neighborhood]:
        for candidate, nb in zip(self.tokens, self.neighborhoods, strict=True):
            if candidate is tok or (
                candidate.text == tok.text
                and candidate.left == tok.left
                and candidate.top == tok.top
            ):
                return nb
        return None

    def as_token_records(self) -> list[dict[str, object]]:
        """The shape a bank-grade layout layer should expose: every OCR
        token knows its neighborhood, not just its text.
        """
        out: list[dict[str, object]] = []
        for tok, nb in zip(self.tokens, self.neighborhoods, strict=True):
            out.append(
                {
                    "text": tok.text,
                    "bbox": [nb.bbox[0], nb.bbox[1], nb.bbox[2], nb.bbox[3]],
                    "line": nb.line,
                    "block": nb.block,
                    "left": nb.left,
                    "right": nb.right,
                    "above": nb.above,
                    "below": nb.below,
                    "language": nb.language,
                }
            )
        return out


def _nearest(
    tok: OcrToken,
    tokens: list[OcrToken],
    *,
    direction: str,
) -> Optional[OcrToken]:
    best: Optional[OcrToken] = None
    best_key: Optional[int] = None
    for other in tokens:
        if other is tok:
            continue
        if direction == "right":
            if not _same_row(tok, other) or other.left < tok.right:
                continue
            key = other.left
        elif direction == "left":
            if not _same_row(tok, other) or other.right > tok.left:
                continue
            key = -other.right
        elif direction == "below":
            if not _overlap_x(tok, other) or other.cy <= tok.bottom:
                continue
            key = other.top
        elif direction == "above":
            if not _overlap_x(tok, other) or other.bottom >= tok.cy:
                continue
            key = -other.bottom
        else:
            continue
        if best is None or best_key is None or key < best_key:
            best = other
            best_key = key
    return best


def build_page(tokens: list[OcrToken]) -> Page:
    """Compute neighbor links, reading-order lines, and coarse blocks."""
    if not tokens:
        return Page(tokens=[], neighborhoods=[], languages=())
    page_height = max(t.bottom for t in tokens)
    page_width = max(t.right for t in tokens)
    lines = _reading_order_lines(tokens)
    neighborhoods: list[Neighborhood] = []
    scripts: list[str] = []
    for i, tok in enumerate(tokens):
        left = _nearest(tok, tokens, direction="left")
        right = _nearest(tok, tokens, direction="right")
        above = _nearest(tok, tokens, direction="above")
        below = _nearest(tok, tokens, direction="below")
        script = detect_script(tok.text)
        scripts.append(script)
        neighborhoods.append(
            Neighborhood(
                left=left.text if left else "",
                right=right.text if right else "",
                above=above.text if above else "",
                below=below.text if below else "",
                line=lines[i],
                block=_block_for(tok, page_height, page_width),
                language=script,
                bbox=(tok.left, tok.top, tok.right, tok.bottom),
            )
        )
    unique_scripts = tuple(dict.fromkeys(scripts))
    return Page(tokens=list(tokens), neighborhoods=neighborhoods, languages=unique_scripts)
