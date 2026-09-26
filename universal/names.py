"""
Person-name parts: first / middle / last, from what the document states.

A document states at most two name fields — the surname and the given names
(or one full name with a known cultural order). Each space-separated word of
the given names is its own name, so once the given names are known exactly
they split deterministically: the first word is the first name, the rest are
middle names. The surname is ONE printed field and is never split, whatever
its spaces ("GEBRE MEDHIN", "GARCÍA LÓPEZ", "VAN DER BERG").

Cultural rules that keep this correct worldwide:

  compound words  particles that belong to the next word stay with it —
                  ABDUL RAHMAN, ABD AL, ABU BAKR, UMM KULTHUM, DE LOS ANGELES
  hyphens         "JEAN-PIERRE" is one name
  CJK romanized   Chinese/Korean/Japanese given names are one name even when
                  romanized with a space ("MIN JUN"): never split into middle
  patronymic      Ethiopian/Eritrean/Somali names with no surname label are
                  given name + father's name + grandfather's name, which map
                  to first / middle / last in that printed order
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words that bind to the FOLLOWING word to form one name.
_BINDING_PREFIXES = {
    "ABD", "ABDUL", "ABDEL", "ABDUR", "ABDOL", "ABU", "ABOU", "UMM", "UM",
    "DE", "DEL", "DELLA", "DI", "DA", "DOS", "DAS", "DO", "VAN", "VON", "DER", "DEN", "LA", "LE", "LOS", "LAS",
    "MAC", "MC", "SAINT", "ST", "SANTA", "SAN",
}
# 'AL'/'EL' bind both ways inside Arabic compounds (ABD AL RAHMAN, NOUR EL DIN).
_INNER_ARTICLES = {"AL", "EL", "UL", "UD", "ED"}

PATRONYMIC_COUNTRIES = {"ETH", "ERI", "SOM"}
FAMILY_FIRST_COUNTRIES = {"CHN", "TWN", "HKG", "MAC", "KOR", "PRK", "JPN", "VNM", "HUN"}
UNSPLIT_GIVEN_COUNTRIES = {"CHN", "TWN", "HKG", "MAC", "KOR", "PRK", "JPN"}


@dataclass
class NameParts:
    first: str = ""
    middle: str = ""
    last: str = ""

    @property
    def given(self) -> str:
        return " ".join(p for p in (self.first, self.middle) if p)


def name_words(text: str) -> list[str]:
    """Space-separated names, with binding particles and Arabic inner articles
    merged into the name they belong to."""
    raw = [t for t in re.split(r"[\s,]+", (text or "").strip()) if t]
    words: list[str] = []
    i = 0
    while i < len(raw):
        word = raw[i]
        # ABD AL RAHMAN / NOUR EL DIN: article joins its neighbours into one name
        while i + 2 < len(raw) and raw[i + 1].upper() in _INNER_ARTICLES:
            word = f"{word} {raw[i + 1]} {raw[i + 2]}"
            i += 2
        # ABDUL RAHMAN / DE LOS ANGELES: prefix binds to the next word(s)
        while word.split()[-1].upper() in _BINDING_PREFIXES and i + 1 < len(raw):
            word = f"{word} {raw[i + 1]}"
            i += 1
        words.append(word)
        i += 1
    return words


def split_given(given: str, country: str = "") -> tuple[str, str]:
    """-> (first name, middle names)."""
    given = " ".join((given or "").split())
    if not given:
        return "", ""
    if country in UNSPLIT_GIVEN_COUNTRIES:
        return given, ""
    words = name_words(given)
    return words[0], " ".join(words[1:])


def parts_from_labels(surname: str, given: str, country: str = "") -> NameParts:
    first, middle = split_given(given, country)
    return NameParts(first=first, middle=middle, last=" ".join((surname or "").split()))


def parts_from_full_name(full: str, country: str) -> NameParts:
    """Only for conventions where order alone is decisive: patronymic names
    (given, father, grandfather) and family-name-first cultures."""
    words = name_words(full)
    if len(words) < 2:
        return NameParts()
    if country in PATRONYMIC_COUNTRIES:
        return NameParts(first=words[0], middle=" ".join(words[1:-1]), last=words[-1])
    if country in FAMILY_FIRST_COUNTRIES:
        first, middle = split_given(" ".join(words[1:]), country)
        return NameParts(first=first, middle=middle, last=words[0])
    return NameParts()


def compose_local_name(surname: str, given: str, country: str, script: str) -> str:
    """The native-script full name in the same natural order as the Latin one:
    given + surname, or family-name-first where the culture puts it first.
    Chinese / Japanese / Korean names are written without a space (王 + 芳 -> 王芳)."""
    surname, given = " ".join((surname or "").split()), " ".join((given or "").split())
    if not (surname and given):
        return surname or given
    family_first = country in FAMILY_FIRST_COUNTRIES or script == "CJK"
    first, second = (surname, given) if family_first else (given, surname)
    joiner = "" if script == "CJK" and " " not in surname + given else " "
    return f"{first}{joiner}{second}"


# Scripts each issuing country prints names in, besides Latin. A country not
# listed prints Latin only — a native-script name "read" from its documents is
# a model inventing a transliteration (seen: Amharic, Cyrillic and Bengali
# renderings of Latin names on US, Canadian and EU cards).
_SINGLE_SCRIPT = {
    "Ethiopic": ("ETH", "ERI"),
    "Arabic": ("SAU", "ARE", "EGY", "IRQ", "JOR", "SYR", "LBN", "YEM", "OMN", "QAT", "KWT", "BHR", "MAR", "DZA",
               "TUN", "LBY", "SDN", "IRN", "AFG", "PAK", "PSE", "MRT"),
    "CJK": ("CHN", "TWN", "HKG", "MAC", "JPN", "KOR", "PRK", "SGP"),
    "Cyrillic": ("RUS", "UKR", "BLR", "BGR", "SRB", "MKD", "KAZ", "KGZ", "TJK", "MNG", "MNE", "BIH", "UZB"),
    "Greek": ("GRC", "CYP"), "Hebrew": ("ISR",), "Thai": ("THA",), "Georgian": ("GEO",), "Armenian": ("ARM",),
    "Devanagari": ("NPL",), "Bengali": ("BGD",), "Sinhala": ("LKA",), "Burmese": ("MMR",), "Khmer": ("KHM",),
    "Lao": ("LAO",),
}
LOCAL_SCRIPTS: dict[str, set[str]] = {}
for _script, _countries in _SINGLE_SCRIPT.items():
    for _c in _countries:
        LOCAL_SCRIPTS.setdefault(_c, set()).add(_script)
LOCAL_SCRIPTS["IND"] = {"Devanagari", "Bengali", "Tamil", "Telugu", "Gujarati", "Gurmukhi", "Kannada", "Malayalam"}
LOCAL_SCRIPTS["LKA"] |= {"Tamil"}
LOCAL_SCRIPTS["SGP"] |= {"Tamil"}


def local_script_plausible(script: str, country: str) -> bool:
    """Unknown country: can't judge, so allow. Known country: only its scripts."""
    return not country or script in LOCAL_SCRIPTS.get(country, set())
