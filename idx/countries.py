"""ICAO Doc 9303 / PRADO territory lookup.

ICAO Doc 9303 Part 3 encodes issuing state and nationality as ISO 3166-1
alpha-3 codes, plus a short list of exceptional codes (GBD/GBN/…, UN*,
XXA/XXB/XXC/XXX, EUE, UTO). PRADO uses the same three-letter issuing-state
codes. This module loads that table once and lets any one of

    country code  ↔  country name  ↔  nationality (demonym)

resolve the other two. Canonical labels are ALL CAPS (ETH / ETHIOPIA /
ETHIOPIAN). Lookup still accepts any casing.

That is canonicalisation of a single printed fact. It is not permission to
copy issuing country onto holder nationality or the reverse — a residence
permit issued by BEL stating nationality ETH is the documented counterexample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import re
import yaml

_TABLE_PATH = Path(__file__).parent / "ontology" / "icao_prado_entities.yaml"
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z"}

EntityKind = Literal["state", "nationality_status", "organization", "status", "test"]
TokenRole = Literal["code", "alpha2", "name", "nationality", "alias"]


@dataclass(frozen=True)
class Territory:
    code: str
    alpha2: str
    country_name: str
    nationality: str
    kind: str
    aliases: tuple[str, ...] = ()
    source: str = ""

    def display_name(self) -> str:
        """Issuing-state / country label; status codes fall back to the demonym."""
        return self.country_name or self.nationality


@dataclass(frozen=True)
class _TokenHit:
    territory: Territory
    role: TokenRole


def _fold(value: str) -> str:
    return " ".join((value or "").strip().casefold().replace(".", "").split())


def _canon_label(value: str) -> str:
    """VIZ/MRZ latin labels are emitted ALL CAPS: ETHIOPIA, ETHIOPIAN."""
    return (value or "").strip().upper()


def _load_entities() -> list[Territory]:
    raw = yaml.safe_load(_TABLE_PATH.read_text(encoding="utf-8")) or {}
    out: list[Territory] = []
    for row in raw.get("entities") or []:
        out.append(
            Territory(
                code=str(row.get("code") or "").upper(),
                alpha2=str(row.get("alpha2") or "").upper(),
                country_name=_canon_label(str(row.get("country_name") or "")),
                nationality=_canon_label(str(row.get("nationality") or "")),
                kind=str(row.get("kind") or "state"),
                aliases=tuple(str(a) for a in (row.get("aliases") or []) if a),
                source=str(row.get("source") or ""),
            )
        )
    return out


def _index_tokens(entities: list[Territory]) -> dict[str, list[_TokenHit]]:
    index: dict[str, list[_TokenHit]] = {}

    def add(token: str, territory: Territory, role: TokenRole) -> None:
        key = _fold(token)
        if not key:
            return
        hits = index.setdefault(key, [])
        if any(h.territory.code == territory.code and h.role == role for h in hits):
            return
        hits.append(_TokenHit(territory, role))

    for entity in entities:
        add(entity.code, entity, "code")
        add(entity.alpha2, entity, "alpha2")
        add(entity.country_name, entity, "name")
        add(entity.nationality, entity, "nationality")
        for alias in entity.aliases:
            add(alias, entity, "alias")
    return index


_ENTITIES: list[Territory] = _load_entities()
_BY_CODE: dict[str, Territory] = {e.code: e for e in _ENTITIES}
_INDEX: dict[str, list[_TokenHit]] = _index_tokens(_ENTITIES)


def pycountry_available() -> bool:
    try:
        import pycountry  # noqa: F401

        return True
    except ImportError:
        return False


def _hits(token: str) -> list[_TokenHit]:
    return list(_INDEX.get(_fold(token), ()))


def _unique_territory(hits: list[_TokenHit], *, prefer_state: bool) -> Territory | None:
    codes = {h.territory.code for h in hits}
    if len(codes) == 1:
        return hits[0].territory
    if not prefer_state or not hits:
        return None
    states = [h.territory for h in hits if h.territory.kind == "state"]
    state_codes = {s.code for s in states}
    if len(state_codes) == 1:
        return states[0]
    return None


def resolve(token: str) -> Territory | None:
    """Resolve a code, country name, demonym, or alias to one territory.

    Returns None when the token is unknown or ambiguous (e.g. 'Congolese'
    is COD and COG; 'Dominican' is DMA and DOM).
    """
    text = (token or "").strip()
    if not text:
        return None
    compact = text.upper().rstrip("<")
    if compact in _BY_CODE:
        return _BY_CODE[compact]
    hits = _hits(text)
    entity = _unique_territory(hits, prefer_state=True)
    if entity:
        return entity
    if any(ch.isdigit() for ch in compact) and 2 <= len(compact) <= 3:
        fixed = "".join(_DIGIT_TO_LETTER.get(ch, ch) for ch in compact)
        if fixed in _BY_CODE:
            return _BY_CODE[fixed]
    return None


def resolve_issuing_state(token: str) -> Territory | None:
    """Issuing-state slot: codes, country names, and name aliases only.

    A demonym-only token ('Ethiopian', 'Swedish') does not fill issuing
    country — that would collapse nationality into issuer.
    """
    text = (token or "").strip()
    if not text:
        return None
    compact = text.upper().rstrip("<")
    if compact in _BY_CODE:
        return _BY_CODE[compact]
    hits = [h for h in _hits(text) if h.role in ("code", "alpha2", "name", "alias")]
    entity = _unique_territory(hits, prefer_state=True)
    if entity:
        return entity
    if any(ch.isdigit() for ch in compact) and 2 <= len(compact) <= 3:
        fixed = "".join(_DIGIT_TO_LETTER.get(ch, ch) for ch in compact)
        if fixed in _BY_CODE:
            return _BY_CODE[fixed]
    return None


def resolve_nationality(token: str) -> Territory | None:
    """Nationality slot: codes, demonyms, and country names used as nationality."""
    return resolve(token)


def lookup_country(alpha3: str) -> str:
    """Return the country name for an ICAO/MRZ code, or '' if unknown."""
    entity = resolve_issuing_state(alpha3)
    if not entity:
        return ""
    return entity.display_name()


def lookup_nationality(alpha3: str) -> str:
    """Return the English demonym for an ICAO/MRZ nationality code."""
    entity = resolve_nationality(alpha3)
    if not entity:
        return ""
    return entity.nationality or entity.display_name()


def resolve_country_code(alpha3: str) -> tuple[str, str]:
    """Return (best_guess_code, country_name) for an MRZ issuing-state field."""
    raw = (alpha3 or "").strip().upper().rstrip("<")
    if not raw:
        return "", ""
    entity = resolve_issuing_state(raw)
    if entity:
        return entity.code, entity.display_name()
    return raw, ""


def resolve_nationality_code(alpha3: str) -> tuple[str, str]:
    """Return (best_guess_code, demonym) for an MRZ nationality field."""
    raw = (alpha3 or "").strip().upper().rstrip("<")
    if not raw:
        return "", ""
    entity = resolve_nationality(raw)
    if entity:
        return entity.code, entity.nationality or entity.display_name()
    return raw, ""


def complete_issuing(code: str = "", name: str = "") -> Territory | None:
    return resolve_issuing_state(code) or resolve_issuing_state(name)


def complete_nationality(code: str = "", name: str = "") -> Territory | None:
    return resolve_nationality(code) or resolve_nationality(name)


def _compact_letters(value: str) -> str:
    return re.sub(r"[^a-z]", "", (value or "").casefold())


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


def header_iso_state_codes(text: str, max_lines: int = 5) -> list[str]:
    """ISO alpha-3 state codes printed in the opening header lines.

    Residence-permit and ID headers often show the issuing state as a
    standalone three-letter code (``BEL``, ``SWE``, ``USA``) too short
    for the country-name matcher. Order is reading order. ``CAN`` next
    to a digit group is a Card Access Number, not Canada.
    """
    if not (text or "").strip():
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:max_lines]
    header = "\n".join(lines)
    out: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b([A-Z]{3})\b", header):
        code = match.group(1)
        if code == "CAN" and re.search(r"\bCAN\s+\d", text, re.I):
            continue
        # A person's 3-letter surname can coincidentally match a real ISO
        # alpha-3 code -- confirmed on a real Australian (ACT) driver
        # licence printing "AGO Binyam Wendemu" as an unlabeled name line,
        # where "AGO" (Angola's own code) sits within the header window.
        # A genuine issuing-state header code is printed ALONE on its own
        # line (see the Belgian-permit "BEL" case this function already
        # handles); one immediately followed by a Title-Case word on the
        # SAME line is a name line, not an issuer statement.
        line_end = header.find("\n", match.end())
        rest_of_line = header[match.end(): line_end if line_end >= 0 else len(header)]
        if re.match(r"[ \t]+[A-Z][a-z]", rest_of_line):
            continue
        entity = _BY_CODE.get(code)
        if not entity or entity.kind not in ("state", "test"):
            continue
        if entity.code in seen:
            continue
        seen.add(entity.code)
        out.append(entity.code)
    return out


_LINE_HAS_DATE_RE = re.compile(
    r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b|\b(?:19|20)\d{2}\b"
)
_DEMONYM_JURISDICTION_RE = re.compile(
    r"\b([A-Z][a-z]+)[ \t]+(?:[A-Z][a-z]+[ \t]+){0,2}"
    r"(?:Territory|State|Province|Region|District|Commonwealth)\b",
    re.IGNORECASE,
)


def _all_mentions_are_birth_context(haystack: str, pattern: re.Pattern[str]) -> bool:
    """True when EVERY occurrence of `pattern` sits on a line that also
    carries a date-shaped token -- the same "3.<DOB> <country>" row a
    real EU/UK driving licence prints for place of birth, not a genuine
    statement of which country issued the document. Confirmed against a
    real UK Provisional Driving Licence stating birth-country ERITREA on
    the same numbered line as the holder's date of birth: without this
    check, the only country name anywhere in the text became the ISSUER
    by default -- the same mistake this module's own Belgian-permit
    lesson (nationality is not the issuer) warns against, just inverted.

    Takes the already-compiled label pattern (the caller just used it to
    confirm a match exists at all) rather than compiling its own copy --
    `find_issuing_in_text` tries roughly a thousand candidate labels per
    call, the overwhelming majority of which never match, so avoiding a
    second/third compile of the same pattern in the rare matched branch
    is a real, free win, not premature optimization.
    """
    found_any = False
    for line in haystack.splitlines():
        if pattern.search(line):
            found_any = True
            if not _LINE_HAS_DATE_RE.search(line):
                return False
    return found_any


def _all_mentions_are_nationality_context(haystack: str, pattern: re.Pattern[str]) -> bool:
    """Same idea as `_all_mentions_are_birth_context`, but for a country
    name sitting on a line with its own "Nationality"/"Citizenship"-type
    label -- a holder-nationality STATEMENT, not a claim about who issued
    the document. Confirmed against a real UAE Resident Identity Card
    printing "Nationality Ethiopia": the only OTHER country name anywhere
    on the card is the genuine issuer (United Arab Emirates), but a blind
    whole-text country-name search made both candidates equally "exact"
    hits, so neither won -- the same Belgian-permit lesson this module
    already applies to demonyms, just for a printed country NAME instead.
    """
    from . import ontology

    found_any = False
    for line in haystack.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        found_any = True
        if ontology.find_label_spans(line, "nationality"):
            continue
        # The label is sometimes glued directly to the value with no
        # space at all (just a stray OCR period, or nothing) -- confirmed
        # against a real UAE Resident Identity Card, where "Nationality"
        # itself came back OCR-typo'd ("Natjonality") AND glued straight
        # onto "Ethiopia" with no space in between. `find_label_spans`
        # above tokenizes on whitespace only, so the glued whole line is
        # seen as one token and never gets a chance at a fuzzy match --
        # checking just the text before this line's own country match
        # (where the label would sit) catches it instead.
        prefix = line[:m.start()].strip(" .:|/")
        if prefix and ontology.match_label(prefix, "nationality"):
            continue
        return False
    return found_any


def find_issuing_in_text(text: str) -> Territory | None:
    """Best issuing-state mention in OCR'd header text.

    Exact country-name hits win. Long names may lose a letter
    (``NETHERLANO``) — accepted only when a single state remains.
    Demonym tokens (``Nederlandse``, ``Ethiopian``) are not the issuer.
    """
    if not (text or "").strip():
        return None
    haystack = text
    exact: list[Territory] = []
    for entity in _ENTITIES:
        if entity.kind not in ("state", "test"):
            continue
        labels = [entity.country_name]
        demonym = _fold(entity.nationality)
        for alias in entity.aliases:
            if _fold(alias) == demonym:
                continue
            labels.append(alias)
        for label in labels:
            compact_len = len(re.sub(r"\s+", "", label.strip()))
            # Country names like SWEDEN/SPAIN are 5–6 letters; aliases such
            # as "Kingdom of the Netherlands" stay long-only so short
            # English words do not match.
            min_len = 4 if _fold(label) == _fold(entity.country_name) else 8
            if compact_len < min_len:
                continue
            label_pattern = re.compile(r"\b" + re.escape(label) + r"\b", re.I)
            if not label_pattern.search(haystack):
                continue
            if _all_mentions_are_birth_context(haystack, label_pattern):
                continue
            if _all_mentions_are_nationality_context(haystack, label_pattern):
                continue
            exact.append(entity)
            break
    codes = {e.code for e in exact}
    if len(codes) == 1:
        return exact[0]
    if exact:
        return None

    header = header_iso_state_codes(text)
    if header:
        header_entity = _BY_CODE.get(header[0])
        if header_entity and header_entity.kind in ("state", "test"):
            return header_entity

    # A demonym used as an ADJECTIVE inside a subnational jurisdiction's
    # own proper name ("Australian Capital Territory", "British Columbia")
    # is a place name, not a personal nationality statement -- confirmed
    # against a real Australian driver licence that never separately
    # spells out "Australia" anywhere, only this territory name and the
    # issuing government's own caption. Narrow on purpose: only a demonym
    # immediately followed by a recognizable jurisdiction-type word counts,
    # so an ordinary "Ethiopian"/"Nederlandse" nationality statement
    # elsewhere in the text is never mistaken for this.
    jurisdiction_m = _DEMONYM_JURISDICTION_RE.search(text)
    if jurisdiction_m:
        jurisdiction_entity = resolve_nationality(jurisdiction_m.group(1))
        if jurisdiction_entity and jurisdiction_entity.kind in ("state", "test"):
            return jurisdiction_entity

    fuzzy: list[Territory] = []
    for word in re.findall(r"[A-Za-z]{8,24}", text):
        folded = _compact_letters(word)
        if resolve_nationality(word) and not resolve_issuing_state(word):
            continue
        hits: list[Territory] = []
        for entity in _ENTITIES:
            if entity.kind != "state":
                continue
            needle = _compact_letters(entity.country_name)
            if not (5 <= len(needle) <= 24):
                continue
            if abs(len(needle) - len(folded)) > 2:
                continue
            if _levenshtein(folded, needle) <= 2:
                hits.append(entity)
        hit_codes = {h.code for h in hits}
        if len(hit_codes) == 1:
            fuzzy.append(hits[0])
    fuzzy_codes = {e.code for e in fuzzy}
    if len(fuzzy_codes) == 1:
        return fuzzy[0]
    return None
