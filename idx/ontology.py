"""
Field ontology: canonical field concepts + their known label aliases.

Two data sources, merged at import:

1. `concepts/field_definitions.yaml` — field *profiles* (type, validators,
   extraction priority) plus aliases stored as they are printed, including
   bilingual lines (`ሙሉ ስም | Full Name`, `COGNOME / SURNAME`). This is the
   file that grows for worldwide coverage. A per-language YAML is not
   required to add a bilingual card.

2. `concepts/labels/<lang>.yaml` — extra translations keyed by language.
   Still loaded so existing files remain valid; new work should prefer (1).

See docs/FIELD_ONTOLOGY.md. Deliberately does NOT absorb aamva.py's
Convention 3 search-window quirks or ethiopia_id.py's QR/barcode parsers
— those stay family-specific (docs/ARCHITECTURE.md additive fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

_CONCEPTS_DIR = Path(__file__).parent / "concepts"
_LABEL_COMPONENT_SPLIT_RE = re.compile(r"[|/]+")
_NORMALIZE_STRIP_RE = re.compile(r"[\s_\-./:|]+")
_WHITESPACE_TOKEN_RE = re.compile(r"\S+")
_HAS_DIGIT_RE = re.compile(r"\d")
# Auto-expanded bilingual components that would silently corrupt other
# fields if registered as aliases (aamva.py's own comments: never add
# bare "class" / "end"; generic_ocr.py: never add bare "name").
_BANNED_EXPANDED = {
    "class", "end", "name", "of", "the", "and", "or", "to", "no", "id",
    "date", "place", "type", "card",
}


def expand_printed_alias(alias: str) -> list[str]:
    """A bilingual printed label is one observation, not two language
    files. `ሙሉ ስም | Full Name` and `COGNOME / SURNAME` expand to the
    full string plus each side so either OCR of the compound line or of
    a single language still matches.
    """
    text = alias.strip()
    if not text:
        return []
    out = [text]
    for part in _LABEL_COMPONENT_SPLIT_RE.split(text):
        part = part.strip(" .")
        if not part or part in out:
            continue
        folded = unicodedata.normalize("NFKC", part).casefold()
        ascii_letters = re.sub(r"[^a-z]", "", folded)
        if ascii_letters in _BANNED_EXPANDED:
            continue
        if ascii_letters and len(ascii_letters) < 3 and not any(ord(c) > 127 for c in part):
            continue
        out.append(part)
    return out


def _load_field_definitions() -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    import yaml

    path = _CONCEPTS_DIR / "field_definitions.yaml"
    if not path.exists():
        return {}, {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    profiles: dict[str, dict[str, Any]] = dict(raw.get("fields") or {})
    aliases: dict[str, list[str]] = {}
    for field, profile in profiles.items():
        collected: list[str] = []
        for alias in profile.get("aliases") or []:
            collected.extend(expand_printed_alias(str(alias)))
        aliases[field] = collected
    return aliases, profiles


def _load_language_label_files(known_fields: set[str]) -> dict[str, list[str]]:
    import yaml

    aliases: dict[str, list[str]] = {field: [] for field in known_fields}
    labels_dir = _CONCEPTS_DIR / "labels"
    if not labels_dir.is_dir():
        return aliases
    for lang_file in sorted(labels_dir.glob("*.yaml")):
        with open(lang_file) as f:
            per_lang = yaml.safe_load(f) or {}
        for field, words in per_lang.items():
            if field not in aliases:
                raise ValueError(f"{lang_file.name}: unknown field '{field}' not in fields.yaml / field_definitions.yaml")
            aliases[field].extend(str(w) for w in (words or []))
    return aliases


def _load_fields_list() -> list[str]:
    import yaml

    with open(_CONCEPTS_DIR / "fields.yaml") as f:
        return list(yaml.safe_load(f)["fields"])


def _dedupe(words: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(word)
    return out


def _load_ontology() -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    fields = _load_fields_list()
    from_profiles, profiles = _load_field_definitions()
    known = set(fields) | set(profiles)
    from_langs = _load_language_label_files(known)
    merged: dict[str, list[str]] = {field: [] for field in known}
    for field in known:
        merged[field] = _dedupe(from_profiles.get(field, []) + from_langs.get(field, []))
    return merged, profiles


_FIELD_ALIASES, _FIELD_PROFILES = _load_ontology()


@dataclass(frozen=True)
class FieldProfile:
    """Typed view of one field_definitions.yaml entry. Missing keys are
    honest defaults, never invented jurisdictional facts.
    """

    field: str
    expected_type: str = "text"
    expected_count: int = 1
    extraction_priority: int = 100
    allow_unlabeled: bool = False
    min_year: Optional[int] = None
    max_age: Optional[int] = None
    future_grace_days: Optional[int] = None
    enum_values: tuple[str, ...] = ()


def field_profile(field: str) -> FieldProfile:
    raw = _FIELD_PROFILES.get(field) or {}
    validators = raw.get("validators") or {}
    values = raw.get("values") or []
    return FieldProfile(
        field=field,
        expected_type=str(raw.get("expected_type") or "text"),
        expected_count=int(raw.get("expected_count") or 1),
        extraction_priority=int(raw.get("extraction_priority") or 100),
        allow_unlabeled=bool(raw.get("allow_unlabeled", False)),
        min_year=validators.get("min_year"),
        max_age=validators.get("max_age"),
        future_grace_days=validators.get("future_grace_days"),
        enum_values=tuple(str(v) for v in values),
    )


def aliases_for(field: str) -> list[str]:
    return list(_FIELD_ALIASES.get(field, []))


def normalize_label(text: str) -> str:
    """NFKC, casefold, strip `_ - . / : |` and whitespace — `First Name`,
    `first_name`, `FIRST-NAME`, `FirstName`, and `ሙሉስም|fullname` after
    delimiter strip all collapse.
    """
    return _NORMALIZE_STRIP_RE.sub("", unicodedata.normalize("NFKC", text).casefold())


def _label_components(text: str) -> list[str]:
    """CLAUDE.md compound-label rule, plus the pipe used on Ethiopian
    Digital IDs (`Zone/City Admin/Sub City`, `ሙሉ ስም | Full Name`).
    """
    parts = [text]
    parts.extend(_LABEL_COMPONENT_SPLIT_RE.split(text))
    return [p.strip() for p in parts if p.strip()]


def _exact_alias(component: str, field: str) -> bool:
    aliases = _FIELD_ALIASES.get(field)
    if not aliases:
        return False
    key = normalize_label(component)
    if key in _BANNED_EXPANDED:
        return False
    return key in {normalize_label(a) for a in aliases}


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _substantial_affix(shorter: str, longer: str) -> bool:
    """True when one normalized string is the other plus 2+ characters.
    'license' vs 'licenseno' is a truncated different label, not a typo.
    A single missing letter ('expir' vs 'expiry') is still a typo.
    """
    if longer.startswith(shorter) and len(longer) - len(shorter) >= 2:
        return True
    if longer.endswith(shorter) and len(longer) - len(shorter) >= 2:
        return True
    return False


def _fuzzy_alias_ratio(component: str, field: str) -> float:
    """0.0 when this is not a plausible OCR-typo of any alias. RapidOCR
    on a real Ethiopian driving licence read 'Full Name' as 'illsame' and
    'Expiry Date' as 'Expiy Date' — substitutions, not a different field.
    """
    needle = normalize_label(component)
    if len(needle) < 6:
        return 0.0
    best = 0.0
    for alias in _FIELD_ALIASES.get(field) or []:
        target = normalize_label(alias)
        if len(target) < 6:
            continue
        if abs(len(needle) - len(target)) > 1:
            continue
        shorter, longer = (needle, target) if len(needle) <= len(target) else (target, needle)
        if _substantial_affix(shorter, longer):
            continue
        dist = _levenshtein(needle, target)
        if dist == 0:
            return 1.0
        if dist > 3:
            continue
        ratio = 1.0 - dist / max(len(needle), len(target))
        if dist <= 2 and ratio >= 0.75:
            best = max(best, ratio)
        elif dist == 3 and ratio >= 0.62:
            best = max(best, ratio)
    return best


def _is_known_alias_token(component: str) -> bool:
    return any(_exact_alias(component, field) for field in _FIELD_ALIASES)


def match_label(text: str, field: str) -> bool:
    """True when ANY component of `text` (split on `/` and `|`) matches
    one of `field`'s known aliases after normalization.

    A multi-word component may also match when its first token is an
    alias for `field` AND every remaining token is itself a known label
    for some field (`SURNAME Forenames` on a Belgian permit). A value
    sitting on the same row (`Surname SMITH`) must NOT be swallowed as
    part of the label.

    OCR near-misses of a long alias are accepted only when this field
    scores strictly better than every other field (so 'Full Name' never
    also matches surname, and a header word 'License' never matches
    document_number via 'License No').
    """
    if not _FIELD_ALIASES.get(field):
        return False
    if _label_is_exact(text, field):
        return True
    ratio = _label_fuzzy_ratio(text, field)
    if ratio <= 0:
        return False
    for other in _FIELD_ALIASES:
        if other == field or not _FIELD_ALIASES.get(other):
            continue
        if _label_is_exact(text, other):
            return False
        if _label_fuzzy_ratio(text, other) > ratio:
            return False
    return True


def _label_is_exact(text: str, field: str) -> bool:
    for component in _label_components(text):
        if _exact_alias(component, field):
            return True
        words = component.split()
        if len(words) >= 2 and _exact_alias(words[0], field):
            if all(_is_known_alias_token(w) for w in words[1:]):
                return True
    return False


def _label_fuzzy_ratio(text: str, field: str) -> float:
    best = 0.0
    for component in _label_components(text):
        best = max(best, _fuzzy_alias_ratio(component, field))
        words = component.split()
        if len(words) >= 2:
            best = max(best, _fuzzy_alias_ratio(" ".join(words[:2]), field))
    return best


def find_label_spans(text: str, field: str) -> list[tuple[int, int]]:
    """Start/end offsets of every token-run in `text` that `match_label`
    accepts for `field`. Longer runs win (Issue Date before Date).
    """
    tokens = list(_WHITESPACE_TOKEN_RE.finditer(text))
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(tokens):
        matched_n = 0
        for n in range(min(4, len(tokens) - i), 0, -1):
            phrase_toks = tokens[i: i + n]
            # A label is words, not a date or ID sitting on the same row.
            if any(_HAS_DIGIT_RE.search(t.group()) for t in phrase_toks):
                continue
            phrase = text[phrase_toks[0].start(): phrase_toks[-1].end()]
            if match_label(phrase, field):
                matched_n = n
                spans.append((phrase_toks[0].start(), phrase_toks[-1].end()))
                break
        i += matched_n if matched_n else 1
    return spans


def label_alias_pattern(field: str) -> re.Pattern[str]:
    """Compiles the same case-insensitive-but-scoped alternation regex
    generic_ocr.py needs for a whole-text search. Word-boundary-wrapped,
    tolerant of variable whitespace in multi-word aliases. Never matches
    anything for a field with no known aliases.
    """
    aliases = _FIELD_ALIASES.get(field, [])
    if not aliases:
        return re.compile(r"(?!x)x")
    # A bilingual alias's own "/" is frequently printed as a LINE BREAK
    # instead -- confirmed against a real Italian Carta di Identita, whose
    # "CITTADINANZA / NATIONALITY" pair prints as two stacked lines with no
    # slash at all ("CITTADINANZA" bold, "NATIONALITY" italic underneath).
    # `\s*` (from the space replacement below) already crosses a newline;
    # making the slash itself optional too means the same alias entry
    # matches whichever layout a given card actually uses, without needing
    # a second "no-slash" alias for every bilingual pair in the ontology.
    escaped = sorted(
        (
            re.escape(alias).replace(r"\ ", r"\s*").replace("/", "/?")
            for alias in aliases
        ),
        key=len, reverse=True,
    )
    return re.compile(r"(?i:\b(?:" + "|".join(escaped) + r")\b)")


def _load_document_vocabulary() -> list[tuple[str, tuple[str, ...]]]:
    import yaml

    path = _CONCEPTS_DIR / "document_vocabulary.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    categories = raw.get("categories") or {}
    # Stable order: more specific families before "national_id" / "card".
    preferred = ("passport", "drivers_license", "residence_permit", "national_id")
    ordered: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for name in preferred:
        if name in categories:
            kws = tuple(str(k) for k in (categories[name].get("keywords") or []))
            ordered.append((name, kws))
            seen.add(name)
    for name, body in categories.items():
        if name in seen:
            continue
        kws = tuple(str(k) for k in ((body or {}).get("keywords") or []))
        ordered.append((name, kws))
    return ordered


_DOCUMENT_VOCABULARY = _load_document_vocabulary()


def document_vocabulary() -> list[tuple[str, tuple[str, ...]]]:
    return list(_DOCUMENT_VOCABULARY)
