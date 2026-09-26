"""
Candidate scoring — collect every plausible reading for a field, then
rank. First-match ("find DOB, take the next word") loses on two-column
cards and on documents that print several dates.

Features are reusable and field-agnostic: label proximity, same row,
expected type, competing-label repulsion, date plausibility, OCR
confidence. The validator (idx/validation.py) still runs later on the
chosen values; this module only ranks extractors' proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import re
from typing import Optional

from . import ontology
from .layout import Page
from .sweep import Candidate, OcrToken, _same_row

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


@dataclass
class ScoredCandidate:
    value: str
    score: int
    reasons: list[str] = field(default_factory=list)
    source: str = ""
    tier: int = 5
    tokens: list[OcrToken] = field(default_factory=list)


def _years_in(value: str) -> list[int]:
    return [int(m.group(1)) for m in _YEAR_RE.finditer(value)]


def _competing_fields(field: str) -> tuple[str, ...]:
    if field == "date_of_birth":
        return ("date_of_issue", "date_of_expiry")
    if field == "date_of_issue":
        return ("date_of_birth", "date_of_expiry")
    if field == "date_of_expiry":
        return ("date_of_birth", "date_of_issue")
    if field == "surname":
        return ("given_names", "full_name")
    if field == "given_names":
        return ("surname", "full_name", "sex", "nationality")
    if field == "permit_type":
        return ("date_of_expiry", "date_of_birth")
    return ()


def _neighbor_texts(page: Optional[Page], tok: Optional[OcrToken]) -> list[str]:
    if page is None or tok is None:
        return []
    nb = page.neighborhood_of(tok)
    if nb is None:
        return []
    return [t for t in (nb.left, nb.right, nb.above, nb.below) if t]


def score_candidate(
    field: str,
    candidate: Candidate,
    page: Optional[Page] = None,
    *,
    today: Optional[date] = None,
) -> ScoredCandidate:
    """Return a score and human-readable reasons. Higher is better.
    Negative features (future birth, sitting next to ISS while scoring
    date_of_birth) lower the score but never delete the candidate —
    disagreement is for the validator / human review, not silent drop.
    """
    today = today or date.today()
    profile = ontology.field_profile(field)
    reasons: list[str] = []
    score = 0

    if candidate.source == "label_anchor":
        score += 50
        reasons.append("next to matching label")
    else:
        score += 5
        reasons.append("unlabeled value signature")

    anchor = candidate.anchor
    value_tok = candidate.tokens[0] if candidate.tokens else None
    if anchor is not None and value_tok is not None:
        if _same_row(anchor, value_tok):
            score += 20
            reasons.append("same row as label")
        elif value_tok.cy > anchor.bottom:
            score += 12
            reasons.append("below label")

    if page is not None and value_tok is not None:
        nb = page.neighborhood_of(value_tok)
        if nb is not None and nb.block == "personal_information":
            score += 8
            reasons.append("personal information block")

    conf = 0.0
    if candidate.tokens:
        conf = max((getattr(t, "confidence", 0.0) or 0.0) for t in candidate.tokens)
    if conf >= 0.9:
        score += 10
        reasons.append("high OCR confidence")
    elif conf >= 0.5:
        score += 4

    years = _years_in(candidate.value)
    if profile.expected_type == "date" and years:
        year = years[0]
        min_year = profile.min_year or 1900
        if field == "date_of_birth":
            oldest = today.year - (profile.max_age or 120)
            if year > today.year:
                score -= 80
                reasons.append("future birth impossible")
            elif year < oldest or year < min_year:
                score -= 40
                reasons.append("implausible birth year")
            else:
                score += 15
                reasons.append("valid birth year")
        elif field == "date_of_issue":
            if year > today.year + 1:
                score -= 50
                reasons.append("issue date too far in the future")
            else:
                score += 8
                reasons.append("plausible issue year")
        elif field == "date_of_expiry":
            if year >= today.year:
                score += 12
                reasons.append("expiry in the future")
            else:
                score += 2
                reasons.append("expiry in the past (may still be the printed date)")

    if profile.expected_type == "enum" and profile.enum_values:
        if any(candidate.value.casefold() == v.casefold() for v in profile.enum_values):
            score += 15
            reasons.append("allowed enum value")

    if field == "document_number":
        compact = re.sub(r"[^A-Za-z0-9]", "", candidate.value)
        has_letter = bool(re.search(r"[A-Za-z]", compact))
        has_digit = bool(re.search(r"\d", compact))
        if has_letter and has_digit and 8 <= len(compact) <= 10:
            score += 25
            reasons.append("mixed letter-digit identifier")
        elif compact.isdigit() and len(compact) <= 7:
            score -= 20
            reasons.append("short all-digit token is a weak unlabeled document number")

    for other in _competing_fields(field):
        texts = _neighbor_texts(page, value_tok)
        if anchor is not None:
            texts.extend(_neighbor_texts(page, anchor))
        if any(ontology.match_label(t, other) for t in texts):
            score -= 35
            reasons.append(f"near competing {other} label")
            break

    return ScoredCandidate(
        value=candidate.value,
        score=score,
        reasons=reasons,
        source=candidate.source,
        tier=candidate.tier,
        tokens=list(candidate.tokens),
    )


def pick_best(
    field: str,
    candidates: list[Candidate],
    page: Optional[Page] = None,
    *,
    today: Optional[date] = None,
) -> Optional[ScoredCandidate]:
    """Highest-scoring candidate, with labeled readings preferred over
    unlabeled ones when scores tie. Returns None when the list is empty.
    """
    if not candidates:
        return None
    scored = [score_candidate(field, c, page, today=today) for c in candidates]
    scored.sort(key=lambda s: (-s.score, s.tier, s.value))
    return scored[0]
