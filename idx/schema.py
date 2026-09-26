"""Unified output schema — same shape regardless of which extraction path fired.

Per-field provenance (CLAUDE.md invariant #6: "Provenance and confidence
are per-field, never per-document") — every identity/document/descriptive/
permit field that can plausibly be READ OFF THE DOCUMENT is a `FieldValue`,
not a bare string: it carries its own `state` (CLAUDE.md invariant #4's
five-state model — `read`/`unreadable`/`printed_blank`/`absent_from_
specimen`/`not_applicable`), which CHANNEL produced it and at what
Precedence tier, and a confidence specific to that one field. A document
can have a checksum-verified surname and a genuinely illegible expiry;
document-level `confidence`/`extraction_method` are a coarse SUMMARY for
convenience, never the authority — see `pipeline.py`'s `_coverage_tier` and
`derive_document_confidence` for how they're derived FROM the per-field
results, never asserted as a constant by a builder.

Deliberately NOT wrapped in `FieldValue`: fields that are document-level
facts rather than a value read off the page (`document_category`,
`mrz_format`, `extraction_method`, `confidence`, `coverage_tier`,
`document_currently_valid`/`validity_checked_at` — computed once against
the wall clock, not "read"), boolean/tri-state VERIFICATION flags rather
than printed values (`real_id_compliant`, `restrictions_asserted_none`,
`endorsements_asserted_none`), the OCR-cross-pass-agreement metadata
(`personal_number_ocr_agreement` — itself a confidence signal ABOUT
personal_number, not a value), and the structural collections
(`mrz_raw`/`mrz_checks`/`verification_checks`/`extra`/`warnings`/
`source_image`).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

# CLAUDE.md invariant #4's five states, in the order a caller should treat
# them: `read` is the only state with a trustworthy `value`; every other
# state means "do not treat `value`/`raw` as data" (they're forced blank
# by `field_value`/`unreadable_field`/`blank_field`/`not_applicable_field`
# below, never left to a caller's assumption).
FieldState = Literal["read", "unreadable", "printed_blank", "absent_from_specimen", "not_applicable"]

# CLAUDE.md's Precedence table, tiers 1-5 -- which CHANNEL produced this
# field, independent of which document family it came from. "crosscheck"
# covers a VLM call that only ENRICHES an already-verified field (place_of_
# birth/issuing_authority/date_of_issue on top of a checksum-verified MRZ)
# rather than being the field's sole source -- still tier 4 (VLM), flagged
# separately so a caller can tell "this ENTIRE document came from a VLM
# guess" apart from "one gap-filler field did".
FieldSource = Literal["barcode", "mrz", "label_anchor", "vlm", "vlm_crosscheck", "signature_sweep"]

_SOURCE_TIER: dict[FieldSource, int] = {
    "barcode": 1,
    "mrz": 2,
    "label_anchor": 3,
    "vlm": 4,
    "vlm_crosscheck": 4,
    "signature_sweep": 5,
}


class FieldValue(BaseModel):
    value: str = ""      # Normalized value. Only meaningful when state == "read".
    raw: str = ""         # Exactly as printed/decoded. Only meaningful when state == "read".
    state: FieldState = "absent_from_specimen"
    confidence: float = 0.0   # 0.0-1.0. Only meaningful when state == "read".
    source: Optional[FieldSource] = None
    source_tier: Optional[int] = None    # 1-5, derived from `source` -- never set independently.
    # Other channels that independently produced the SAME value for this
    # field -- e.g. Ethiopia's FAN cross-verified between barcode and QR,
    # or an MRZ field corroborated by AI crosscheck. Empty when only one
    # channel ever attempted this field. This is what CLAUDE.md's
    # Precedence table means by "agreement across tiers raises confidence" --
    # recorded here per-field, never folded into document-level confidence
    # directly.
    agreeing_sources: list[FieldSource] = []


def field_value(
    value: str, *, source: FieldSource, confidence: float, raw: Optional[str] = None,
) -> FieldValue:
    """The normal constructor: a value was actually read. `state` is
    derived from whether `value` is non-blank -- a builder that read a
    field as an explicit empty string (`printed_blank`) must use
    `blank_field`/set state itself, never claim `state="read"` with an
    empty value (that's exactly the null/"" collapse invariant #4
    prohibits).
    """
    return FieldValue(
        value=value, raw=raw if raw is not None else value,
        state="read" if value else "absent_from_specimen",
        confidence=confidence if value else 0.0,
        source=source if value else None,
        source_tier=_SOURCE_TIER[source] if value else None,
    )


def unreadable_field(source: Optional[FieldSource] = None) -> FieldValue:
    """Field located, value illegible -- e.g. glare over a printed date.
    Distinct from `absent_from_specimen` (no label/signature found at
    all) -- see CLAUDE.md invariant #4."""
    return FieldValue(state="unreadable", source=source, source_tier=_SOURCE_TIER.get(source) if source else None)


def blank_field(source: Optional[FieldSource] = None) -> FieldValue:
    """Field located, printed empty -- e.g. a barcode explicitly encoding
    an empty restrictions element. VERIFIED empty, not "we didn't check."

    That verification strength is not uniform across `source` values,
    which is exactly why `source`/`source_tier` are still recorded here
    rather than dropped once `state` is decided. A barcode-sourced blank
    (tier 1) is a byte-level fact: the element decoded to empty, full
    stop. A `label_anchor`-sourced blank (tier 3) is weaker -- it means a
    plain OCR pass found the field's label crisply legible with nothing
    captured beside it, which is usually a genuinely blank value region
    but can also be a value that printed too faintly to survive OCR at
    the capture resolution given. Confirmed against a real Washington
    license's "9 CLASS" row. Treat an OCR-sourced `printed_blank` as an
    observation about this image, not a certified claim about the
    jurisdiction's layout spec -- a caller that needs the stronger
    guarantee should check `source_tier` before trusting blankness as
    fact rather than best-effort.
    """
    return FieldValue(state="printed_blank", source=source, source_tier=_SOURCE_TIER.get(source) if source else None)


def not_applicable_field() -> FieldValue:
    """This layout does not print this field -- requires a profile
    (authored or learned) behind the claim; see CLAUDE.md invariant #4 and
    docs/GLOBAL_COVERAGE.md. Not currently asserted anywhere in this
    codebase (no learned-profile system exists to back it) -- present so
    the state exists in the schema ahead of that system landing, per
    GLOBAL_COVERAGE.md's own resolution-order design."""
    return FieldValue(state="not_applicable")


_ABSENT = FieldValue()   # state="absent_from_specimen" is the class default -- shared, immutable-by-convention


class ExtractedIDDocument(BaseModel):
    document_category: str = ""     # "passport" | "national_id" | "drivers_license" | "residence_permit" | "unknown"
    mrz_format: Optional[str] = None  # "TD3" | "TD2" | "TD1" | None
    extraction_method: str = ""     # "mrz" | "mrz+ai" | "ai_vision" | "mrz_low_confidence"
    confidence: str = "low"         # "high" (checksum-verified) | "medium" | "low"
    # Honest, document-level "how was this actually obtained" signal --
    # see GLOBAL_COVERAGE.md's "Coverage tiers" table. Derived from
    # extraction_method/confidence in pipeline.py's `_coverage_tier`, not
    # settable independently, so it can never drift out of sync with the
    # method that actually produced this result.
    #   A: MRZ or a decoded barcode/QR -- checksum- or cross-verified.
    #   B: a recognized document family read without that verification
    #      (label-anchored OCR, or only one of two machine sources).
    #   D: an unrecognized layout read by AI vision -- flexible, unverifiable.
    #   E: last-resort label-alias/regex sweep, or nothing else applied.
    # Tier C (a promoted learned profile) has no equivalent in this
    # codebase -- no learned-profile system exists -- and is never assigned.
    coverage_tier: str = ""

    country_code: FieldValue = _ABSENT
    country_name: FieldValue = _ABSENT

    surname: FieldValue = _ABSENT
    given_names: FieldValue = _ABSENT
    full_name: FieldValue = _ABSENT
    # Native-script rendering of the same name (e.g. Amharic on an
    # Ethiopian document) -- see docs/GLOBAL_COVERAGE.md's "universal
    # component set": often the legally authoritative form, and the only
    # form that matches domestic records. Never discarded in favor of a
    # transliteration.
    name_in_native_script: FieldValue = _ABSENT
    name_script: str = ""   # e.g. "Latin", "Ethiopic" -- blank when native-script wasn't captured.
    # What the document itself asserts about name order, never a guess --
    # see docs/GLOBAL_COVERAGE.md.
    name_order_hint: Literal[
        "surname_first", "given_first", "patronymic_chain", "single_field", "unknown"
    ] = "unknown"
    sex: FieldValue = _ABSENT
    date_of_birth: FieldValue = _ABSENT     # .value is ISO YYYY-MM-DD when resolvable
    date_of_issue: FieldValue = _ABSENT
    date_of_expiry: FieldValue = _ABSENT
    # Tri-state (None when date_of_expiry.value isn't a resolvable ISO date) rather than
    # baked into `confidence` or `warnings` alone -- "is this document valid" is a
    # distinct question from "how well did extraction read it", and a caller filtering/
    # sorting on validity shouldn't have to parse warning strings to get it. Computed
    # once, at extraction time, against the wall-clock date -- NOT re-derived later,
    # since "valid" is a fact about a specific moment, not a property of the document
    # itself. `validity_checked_at` is stored alongside for exactly that reason: read
    # back six months later, `document_currently_valid=True` alone would be silently
    # wrong, but paired with the date it was computed it's still fully interpretable.
    document_currently_valid: Optional[bool] = None
    validity_checked_at: Optional[str] = None   # ISO date the check above was performed
    document_number: FieldValue = _ABSENT
    nationality_code: FieldValue = _ABSENT
    nationality_name: FieldValue = _ABSENT
    # "self_declared" when the document ITSELF prints a qualifier saying so
    # (confirmed on a real Ethiopian Digital ID: "Nationality (Self
    # Declared)") -- blank otherwise, never asserted or inferred. A caller
    # using nationality for KYC purposes needs this distinction
    # structurally: the issuer is explicitly marking this field as
    # holder-asserted, not independently verified, which is a materially
    # different fact from an ordinary unqualified nationality field.
    nationality_qualifier: str = ""
    personal_number: FieldValue = _ABSENT
    # How many independent OCR passes (different crop/preprocessing variants) landed on
    # this exact personal_number value, out of how many TD3 passes were attempted -- e.g.
    # "4/6". Cross-pass agreement is much stronger evidence than personal_number's own
    # check digit (a single digit over up to 14 characters -- see mrz_td3.py), so callers
    # that want a programmatic confidence signal for this field should use this rather
    # than parsing `warnings`. None when personal_number is blank or wasn't from the MRZ path.
    personal_number_ocr_agreement: Optional[str] = None
    place_of_birth: FieldValue = _ABSENT
    # Depth-generic administrative hierarchy (docs/GLOBAL_COVERAGE.md) --
    # coarse-to-fine, each level carrying the label AS PRINTED alongside
    # its value, so Ethiopia's Region/Zone/Woreda and a US license's
    # State/County/City share one structure without a schema change per
    # country. `address` (below) stays the flat, verbatim-as-printed
    # string every path already produces; `admin_levels` is populated
    # ADDITIONALLY, only where a document's own structure is discrete
    # enough to split (Ethiopia's region/zone/woreda today).
    admin_levels: list[dict[str, str]] = []   # [{"label": "Woreda", "value": "11"}, ...], coarse -> fine
    address: FieldValue = _ABSENT
    issuing_authority: FieldValue = _ABSENT
    phone_number: FieldValue = _ABSENT

    # Standard across every US/Canada AAMVA driver's license (not per-state), so unlike
    # region/zone/woreda-type country-specific fields these are worth modeling directly
    # rather than folding into `address`.
    license_class: FieldValue = _ABSENT
    # A raw class LETTER is not comparable across jurisdictions -- confirmed directly:
    # Virginia's "D" and California/Maryland's "C" are all the same standard non-
    # commercial operator class, so code that compares license_class strings across
    # states would report a privilege mismatch that doesn't exist. Populated ONLY from
    # a small, explicitly-confirmed mapping (see aamva.py's _CLASS_CODE_CANONICAL) --
    # never guessed -- so blank here means "not yet confirmed for this jurisdiction/
    # letter", not "no class". license_class itself always keeps the raw printed letter
    # regardless of whether this canonicalizes successfully.
    license_class_canonical: str = ""
    license_restrictions: FieldValue = _ABSENT
    license_endorsements: FieldValue = _ABSENT
    # Descriptive fields (docs/GLOBAL_COVERAGE.md's "universal component
    # set") -- genuinely universal in principle, but a real AAMVA
    # jurisdiction only sometimes prints them at all (confirmed: Maryland
    # prints neither; Washington prints eye colour but not hair; see
    # idx/ontology/aamva_jurisdictions.yaml's `field_profile`). `state`
    # distinguishes a confirmed-not-printed jurisdiction (`not_applicable`)
    # from one that just hasn't been observed yet, or where OCR simply
    # missed a field the jurisdiction DOES print (`absent_from_specimen`)
    # -- see pipeline.py's `_from_aamva` for how that distinction is made.
    eye_color: FieldValue = _ABSENT
    hair_color: FieldValue = _ABSENT
    # Array of records, not a flat string (docs/GLOBAL_COVERAGE.md) -- an
    # EU licence can hold category B valid to 2035 and category C valid to
    # 2028 on one card; `license_class`/`license_restrictions` above stay
    # for the (today, only) AAMVA case where a flat string has always been
    # sufficient, `categories` is populated additionally wherever
    # per-category validity actually needs representing. Not currently
    # populated by any builder in this codebase (no EU driving-licence
    # parser exists yet) -- present so the shape exists ahead of that.
    categories: list[dict[str, object]] = []   # [{"code", "code_canonical", "valid_from", "valid_to", "restrictions": [], "label_as_printed"}]
    # Tri-state, not bool -- None means "couldn't be determined" (e.g. no barcode and
    # the OCR-only front-text path found neither the compliance marking nor a "FEDERAL
    # LIMITS APPLY" banner), which is a materially different answer from "checked and
    # this card is not compliant" (False). Whether a card satisfies federal ID
    # requirements (TSA, federal facilities, I-9) turns on this distinction, so it's
    # never defaulted to either boolean value.
    real_id_compliant: Optional[bool] = None
    # True only when the card affirmatively PRINTS "NONE" (or the barcode encodes AAMVA's
    # own "NONE" placeholder) for that field -- distinct from license_restrictions/
    # license_endorsements simply being blank because the card prints nothing there at
    # all (confirmed both happen in practice: California's barcode encodes "NONE"
    # explicitly, Washington prints it, Maryland prints neither field's row with any
    # value). Blank does NOT imply "no restrictions" -- it means this was never actually
    # asserted one way or the other, so a caller checking only license_restrictions==""
    # would wrongly treat an unverified field as a verified-clean one.
    restrictions_asserted_none: bool = False
    endorsements_asserted_none: bool = False

    # The defining characteristic of a residence permit -- e.g. "A. Beperkt verblijf"
    # (Belgium's "limited residence" category). Near-universal across countries that
    # issue these (permits are always categorized by duration/purpose), unlike the
    # country-specific remarks/register-number fields that go in `extra` instead.
    permit_type: FieldValue = _ABSENT

    mrz_raw: list[str] = []
    mrz_checks: dict[str, bool] = {}
    # Generic (not MRZ-specific) home for non-checksum cross-validation signals from
    # barcode/QR-based document families -- e.g. an Ethiopian Digital ID's
    # `fan_qr_barcode_match` (the front barcode and back QR independently agreeing on the
    # same document number). Parallel to `mrz_checks`, just for documents with no MRZ.
    verification_checks: dict[str, object] = {}
    # Generic home for fields specific to one document family with no equivalent across
    # the others -- e.g. an AAMVA driver's license's height/weight/eye color/hair color/
    # document discriminator/inventory control number (the discriminator + ICN in
    # particular are exactly what a caller would use to detect a duplicate or reissued
    # card). Parallel to `verification_checks`, just for data instead of boolean checks --
    # same reasoning `ethiopia_id.py` already applies to region/zone/woreda not getting
    # dedicated top-level fields.
    extra: dict[str, object] = {}
    warnings: list[str] = []
    # CLAUDE.md's Precedence table: "Two channels disagreeing on a field
    # means both candidates are emitted and human_review_required is set."
    # The higher-tier channel's value is still what's KEPT on the field
    # itself (Precedence still resolves which value wins) -- this flag is
    # the queryable signal that a genuine cross-channel disagreement was
    # found and silently overridden, not just enriched, so a caller doesn't
    # have to parse `warnings` text to detect it. The specific mismatch
    # detail is in `warnings`/`verification_checks` either way.
    human_review_required: bool = False
    source_image: str = ""
