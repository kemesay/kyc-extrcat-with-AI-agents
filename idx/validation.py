"""
Universal per-field plausibility validation -- "reject an implausible
reading outright rather than report it as fact" (already the established
convention for height/weight in `aamva.py`'s `_plausible_weight_lb` and
Field 16/17 handling), generalized here into ONE place that runs for
every document family, instead of being reinvented per-family the way
height/weight's own bound was.

This is not a status/eligibility inference (CLAUDE.md's own prohibition,
see `pipeline._apply_validity_check`'s docstring for the same discipline
applied to `document_currently_valid`) -- it never asks "is this person
eligible", only "is this VALUE physically possible" (a birth date in the
future, an expiry before an issue date). Those are facts about arithmetic,
not judgments about the holder.

Deliberately conservative in scope: NEVER touches a barcode- or MRZ-
sourced value (`source_tier` 1-2) -- those are checksum-verified and
structurally authoritative regardless of how the underlying date looks,
so an implausible-looking but checksum-correct value is left exactly as
read (rejecting it would be worse than the OCR error this module exists
to catch). Only `label_anchor`/`vlm`/`signature_sweep`-sourced values
(tier 3+, never checksum-verified) are subject to rejection -- exactly
the tiers CLAUDE.md's Precedence table already treats as the ones a
higher-tier channel is allowed to override.

Rejecting a single field reverts it to `unreadable` (CLAUDE.md invariant
#4 -- located but its value can't be trusted), never silently leaving
`state="read"` with a value already known to be wrong. A cross-field
contradiction (issue date after expiry, expiry before birth) has no
single field to blame -- both values may be individually plausible, just
mutually inconsistent -- so that case is reported as a warning and
`human_review_required` is set, the same "flag it, never silently
resolve it" discipline as every other cross-check in this project,
rather than guessing which one is wrong.
"""

from __future__ import annotations

from datetime import date, timedelta

from .schema import ExtractedIDDocument, FieldValue, unreadable_field

_MAX_PLAUSIBLE_AGE_YEARS = 120
# A freshly-issued document can be dated a short while into the future
# relative to when it's actually scanned (clock skew, a card issued
# ahead of an already-scheduled pickup date) -- a generous grace window
# rather than a hard "today" cutoff, so a genuine near-term issue date
# isn't rejected as if it were a `2088`-style OCR misread.
_ISSUE_DATE_FUTURE_GRACE_DAYS = 60


def _is_checksum_verified(field: FieldValue) -> bool:
    return field.source_tier is not None and field.source_tier <= 2


def _parse_iso(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _validate_date_of_birth(field: FieldValue, today: date) -> FieldValue:
    if not field.value or _is_checksum_verified(field):
        return field
    dob = _parse_iso(field.value)
    if dob is None:
        return field
    oldest_plausible = today.replace(year=today.year - _MAX_PLAUSIBLE_AGE_YEARS)
    if dob > today or dob < oldest_plausible:
        return unreadable_field(source=field.source)
    return field


def _validate_issue_date(field: FieldValue, today: date) -> FieldValue:
    if not field.value or _is_checksum_verified(field):
        return field
    issue = _parse_iso(field.value)
    if issue is None:
        return field
    if issue > today + timedelta(days=_ISSUE_DATE_FUTURE_GRACE_DAYS):
        return unreadable_field(source=field.source)
    return field


# AAMVA/MRZ normalize to the single-letter code, but the AI-vision
# crosscheck path deliberately preserves the value "as printed" (see
# vision.py's own prompt schema) rather than normalizing it -- a real
# Ethiopian Digital ID's own printed English label is the full word
# ("Male"), confirmed a real false rejection here once this check first
# shipped. "X" is ICAO 9303's own third MRZ sex code (unspecified/other).
_PLAUSIBLE_SEX_VALUES = {"M", "F", "X", "MALE", "FEMALE"}


def _validate_sex(field: FieldValue) -> FieldValue:
    if not field.value or _is_checksum_verified(field):
        return field
    if field.value.upper() not in _PLAUSIBLE_SEX_VALUES:
        return unreadable_field(source=field.source)
    return field


def apply_plausibility_checks(doc: ExtractedIDDocument, today: date | None = None) -> None:
    """Single entry point, called once in `pipeline.extract_id_document`
    for every document family (see `_apply_validity_check` for the
    identical "one universal post-processing step, not threaded through
    each `_from_*` builder" pattern this follows). `today` is only ever
    overridden by tests -- production callers always use the real date.
    """
    today = today or date.today()

    doc.date_of_birth = _validate_date_of_birth(doc.date_of_birth, today)
    doc.date_of_issue = _validate_issue_date(doc.date_of_issue, today)
    doc.sex = _validate_sex(doc.sex)

    dob = _parse_iso(doc.date_of_birth.value) if doc.date_of_birth.value else None
    issue = _parse_iso(doc.date_of_issue.value) if doc.date_of_issue.value else None
    expiry = _parse_iso(doc.date_of_expiry.value) if doc.date_of_expiry.value else None

    if dob and issue and dob > issue:
        doc.warnings.append(
            f"date_of_birth ({doc.date_of_birth.value}) is after date_of_issue "
            f"({doc.date_of_issue.value}) -- these two dates are mutually inconsistent. "
            f"Neither value was individually implausible enough to reject outright; "
            f"verify both against the physical document."
        )
        doc.human_review_required = True

    if issue and expiry and issue > expiry:
        doc.warnings.append(
            f"date_of_issue ({doc.date_of_issue.value}) is after date_of_expiry "
            f"({doc.date_of_expiry.value}) -- these two dates are mutually inconsistent. "
            f"Neither value was individually implausible enough to reject outright; "
            f"verify both against the physical document."
        )
        doc.human_review_required = True

    if dob and expiry and dob > expiry:
        doc.warnings.append(
            f"date_of_birth ({doc.date_of_birth.value}) is after date_of_expiry "
            f"({doc.date_of_expiry.value}) -- these two dates are mutually inconsistent. "
            f"Neither value was individually implausible enough to reject outright; "
            f"verify both against the physical document."
        )
        doc.human_review_required = True
