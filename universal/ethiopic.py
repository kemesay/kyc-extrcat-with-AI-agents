"""
Cross-script check: an Ethiopic (Ge'ez) name verified against the Latin
spelling of the same name on the same document.

Ge'ez is a syllabary — every character is one consonant row x one vowel
order — so each character romanizes deterministically (ሆ = "ho", ሮ = "ro",
ከ = "ke", ካ = "ka"). Vision models misread single Ge'ez characters
(ረና for ሀና, ካበደ for ከበደ) far more often than they misread the Latin
line printed beside it. When a word's romanization disagrees with its Latin
counterpart and exactly one character substitution makes them agree, that
substitution is applied. Ambiguous cases are left alone: የ (yä) and ዬ (yē)
both romanize "ye" in Ethiopian spelling practice, so neither is preferred.

Romanization follows Ethiopian naming practice (as on passports): the first
order ä is written e or a, the sixth order ə is usually silent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

# Regular 8-character rows of the Ethiopic block: base code point -> consonant romanizations.
_ROWS: dict[int, tuple[str, ...]] = {
    0x1200: ("h",), 0x1208: ("l",), 0x1210: ("h",), 0x1218: ("m",), 0x1220: ("s",), 0x1228: ("r",),
    0x1230: ("s",), 0x1238: ("sh",), 0x1240: ("q", "k"), 0x1250: ("q",), 0x1260: ("b",), 0x1268: ("v",),
    0x1270: ("t",), 0x1278: ("ch", "c"), 0x1280: ("h",), 0x1290: ("n",), 0x1298: ("gn", "ny"),
    0x12A0: ("",), 0x12A8: ("k",), 0x12B8: ("h", "kh"), 0x12C8: ("w",), 0x12D0: ("",), 0x12D8: ("z",),
    0x12E0: ("zh", "j"), 0x12E8: ("y",), 0x12F0: ("d",), 0x1300: ("j",), 0x1308: ("g",), 0x1320: ("t", "th"),
    0x1328: ("ch", "c"), 0x1330: ("p", "ph"), 0x1338: ("ts", "tz", "s"), 0x1340: ("ts", "tz", "s"),
    0x1348: ("f",), 0x1350: ("p",),
}
# Vowel orders 0-7 as written in Ethiopian Latin names.
_VOWELS: tuple[tuple[str, ...], ...] = (
    ("e", "a"), ("u",), ("i",), ("a",), ("e", "ie"), ("", "i", "e"), ("o",), ("wa", "ua"),
)
# Homophone rows, most common first: prefer ሀ over ሐ over ኀ, ሰ over ሠ, አ over ዐ, ጸ over ፀ.
_ROW_PREFERENCE = [0x1200, 0x1208, 0x1218, 0x1228, 0x1230, 0x1238, 0x1240, 0x1260, 0x1268, 0x1270,
                   0x1278, 0x1290, 0x1298, 0x12A0, 0x12A8, 0x12C8, 0x12D8, 0x12E0, 0x12E8, 0x12F0,
                   0x1300, 0x1308, 0x1320, 0x1328, 0x1330, 0x1338, 0x1348, 0x1350,
                   0x1210, 0x1280, 0x12B8, 0x1220, 0x12D0, 0x1340, 0x1250]


def _row_order(ch: str) -> Optional[tuple[int, int]]:
    cp = ord(ch)
    row = cp - (cp - 0x1200) % 8
    if row in _ROWS and 0x1200 <= cp <= 0x1357:
        return row, cp - row
    return None


def romanizations(ch: str) -> tuple[str, ...]:
    ro = _row_order(ch)
    if ro is None:
        return ()
    row, order = ro
    consonants = _ROWS[row]
    # Gemination is unmarked in Ge'ez but doubled in Latin names (ከበደ = KEBBEDE, ለማ = LEMMA).
    doubled = tuple(c[0] + c for c in consonants if c)
    return tuple(c + v for c in consonants + doubled for v in _VOWELS[order])


@lru_cache(maxsize=4096)
def _spells(word: str, latin: str) -> bool:
    """Can the Ge'ez word be romanized to exactly this Latin word?"""
    if not word:
        return latin == ""
    options = romanizations(word[0])
    if not options:
        return False
    return any(latin.startswith(o) and _spells(word[1:], latin[len(o):]) for o in options)


def _candidates(ch: str) -> list[str]:
    """Characters one step away: same consonant another vowel, or same vowel another consonant."""
    ro = _row_order(ch)
    if ro is None:
        return []
    row, order = ro
    same_row = [chr(row + o) for o in range(8) if o != order]
    same_order = [chr(r + order) for r in _ROW_PREFERENCE if r != row]
    return same_row + same_order


def correct_word(word: str, latin: str) -> Optional[str]:
    """A single-character fix that makes the Ge'ez word spell the Latin word,
    when there is exactly one (homophone rows collapsed to the preferred one)."""
    latin = latin.lower()
    if not word or _spells(word, latin):
        return None
    fixes: list[str] = []
    for i, ch in enumerate(word):
        for cand in _candidates(ch):
            fixed = word[:i] + cand + word[i + 1:]
            if _spells(fixed, latin):
                fixes.append(fixed)
                break                      # candidates are ordered by preference
    distinct = list(dict.fromkeys(fixes))
    return distinct[0] if len(distinct) == 1 else None


def is_ethiopic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(0x1200 <= ord(c) <= 0x139F for c in letters)


def correct_name(local: str, latin: str) -> tuple[str, list[tuple[str, str]]]:
    """-> (corrected local name, [(old word, new word), ...]). Word counts must
    match; the Latin name must be in the same order as the local one."""
    words, lat = local.split(), [w for w in latin.replace(",", " ").split()]
    if len(words) != len(lat) or not is_ethiopic(local):
        return local, []
    out, changes = [], []
    for w, l in zip(words, lat):
        fixed = correct_word(w, "".join(c for c in l.lower() if c.isalpha()))
        if fixed:
            changes.append((w, fixed))
        out.append(fixed or w)
    return " ".join(out), changes


def spelling_score(local: str, latin: str) -> int:
    """How well a Ge'ez name spells the Latin name: 2 per word that spells it
    exactly, 1 per word one character away, -1 if the word counts differ."""
    words, lat = local.split(), [w for w in latin.replace(",", " ").split()]
    if not is_ethiopic(local) or len(words) != len(lat):
        return -1
    score = 0
    for w, l in zip(words, lat):
        key = "".join(c for c in l.lower() if c.isalpha())
        score += 2 if _spells(w, key) else (1 if correct_word(w, key) else 0)
    return score


def word_score(word: str, latin_word: str) -> int:
    key = "".join(c for c in latin_word.lower() if c.isalpha())
    return 2 if _spells(word, key) else (1 if correct_word(word, key) else 0)


def combine_readings(readings: list[str], latin: str) -> str:
    """Word-by-word: for each position take the reading whose word best spells
    the Latin word (ties keep the first reading). Readings with the wrong word
    count don't vote."""
    lat = latin.replace(",", " ").split()
    usable = [r.split() for r in readings if is_ethiopic(r) and len(r.split()) == len(lat)]
    if not usable:
        return readings[0] if readings else ""
    out = []
    for i, lw in enumerate(lat):
        out.append(max((r[i] for r in usable), key=lambda w: word_score(w, lw)))
    return " ".join(out)
