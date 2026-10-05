"""
Which languages a document is printed in — from the printed text, with the
agents' opinion as one vote.

Every agent reports language_primary / language_secondary, the fast scanner
included (a small model, but a useful first impression). Those are votes.
The printed text supplies the evidence:

  labels     every recognised field label names its language ("Date Issued"
             English, "የተሰጠበት ቀን" Amharic, "Guyyaa Kenname" Oromo)
  markers    words that belong to one language: "Oromiyaa", "Naannoo",
             "Jamhuuriyadda", "République", "Republic of"
  script     text in a script used by one language, or by one language in
             the issuing country (Ge'ez: Amharic in ETH, Tigrinya in ERI;
             Hangul: Korean; Kana: Japanese; Arabic script: Persian in IRN,
             Urdu in PAK, Arabic elsewhere; ...)

The primary language is the best-evidenced language other than English when
the document is bilingual with English (English is the translation line on
most national documents); otherwise the best-evidenced one. All others are
secondary. Nothing is guessed without evidence: with no printed evidence the
agents' majority stands.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Optional

from . import labels

# Words that mark one language (beyond field labels). Normalised forms.
_MARKERS: dict[str, tuple[str, ...]] = {
    "om": ("oromiyaa", "naannoo", "bulchiinsa", "godina", "aanaa", "ganda", "baankii", "hojii", "gamtaa",
           "mootummaa", "biyyoolessaa", "federaalawaa", "dimokraatawaa", "rippabliikii", "itoophiyaa",
           "eenyummaa", "hojjetaa", "waajjira", "kan", "fi"),
    "en": ("republic", "of the", "federal", "democratic", "bank", "card", "identity", "ministry", "the"),
    "fr": ("republique", "carte", "nationale", "d identite", "ministere"),
    "es": ("republica", "documento", "nacional", "identidad"),
    "pt": ("republica", "cartao", "cidadao"),
    "de": ("bundesrepublik", "deutschland", "ausweis"),
    "so": ("jamhuuriyadda", "soomaaliya", "federaalka", "dowladda"),
    "sw": ("jamhuri ya", "muungano", "kitambulisho", "taifa"),
    "tr": ("turkiye", "cumhuriyeti", "kimlik"),
    "id": ("republik indonesia", "provinsi", "kabupaten"),
}

_SCRIPT_RANGES = (
    ("Ethiopic", 0x1200, 0x139F), ("Arabic", 0x0600, 0x06FF), ("Arabic", 0x0750, 0x077F),
    ("Hebrew", 0x0590, 0x05FF), ("Cyrillic", 0x0400, 0x04FF), ("Greek", 0x0370, 0x03FF),
    ("Devanagari", 0x0900, 0x097F), ("Bengali", 0x0980, 0x09FF), ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF), ("Tamil", 0x0B80, 0x0BFF), ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF), ("Malayalam", 0x0D00, 0x0D7F), ("Sinhala", 0x0D80, 0x0DFF),
    ("Thai", 0x0E00, 0x0E7F), ("Lao", 0x0E80, 0x0EFF), ("Myanmar", 0x1000, 0x109F), ("Khmer", 0x1780, 0x17FF),
    ("Georgian", 0x10A0, 0x10FF), ("Armenian", 0x0530, 0x058F), ("Hangul", 0xAC00, 0xD7AF),
    ("Kana", 0x3040, 0x30FF), ("Han", 0x4E00, 0x9FFF), ("Thaana", 0x0780, 0x07BF),
)

# Script -> language when only the script is known.
_BY_SCRIPT = {
    "Hebrew": "Hebrew", "Greek": "Greek", "Bengali": "Bengali", "Gurmukhi": "Punjabi", "Gujarati": "Gujarati",
    "Tamil": "Tamil", "Telugu": "Telugu", "Kannada": "Kannada", "Malayalam": "Malayalam", "Sinhala": "Sinhala",
    "Thai": "Thai", "Lao": "Lao", "Myanmar": "Burmese", "Khmer": "Khmer", "Georgian": "Georgian",
    "Armenian": "Armenian", "Hangul": "Korean", "Kana": "Japanese", "Thaana": "Dhivehi",
}
_BY_SCRIPT_COUNTRY = {
    ("Ethiopic", "ERI"): "Tigrinya", ("Ethiopic", ""): "Amharic",
    ("Arabic", "IRN"): "Persian", ("Arabic", "AFG"): "Dari", ("Arabic", "PAK"): "Urdu", ("Arabic", ""): "Arabic",
    ("Cyrillic", "UKR"): "Ukrainian", ("Cyrillic", "BGR"): "Bulgarian", ("Cyrillic", "SRB"): "Serbian",
    ("Cyrillic", "MKD"): "Macedonian", ("Cyrillic", "BLR"): "Belarusian", ("Cyrillic", "KAZ"): "Kazakh",
    ("Cyrillic", "KGZ"): "Kyrgyz", ("Cyrillic", "MNG"): "Mongolian", ("Cyrillic", "TJK"): "Tajik",
    ("Cyrillic", ""): "Russian",
    ("Devanagari", "NPL"): "Nepali", ("Devanagari", ""): "Hindi",
    ("Han", "JPN"): "Japanese", ("Han", "KOR"): "Korean", ("Han", ""): "Chinese",
}

# Agents write language names many ways.
_ALIASES = {
    "afaan oromo": "Oromo", "afaan oromoo": "Oromo", "oromiffa": "Oromo", "oromigna": "Oromo", "oromoo": "Oromo",
    "affan oromo": "Oromo", "amharic": "Amharic", "amhara": "Amharic", "tigrigna": "Tigrinya", "tigrinya": "Tigrinya",
    "farsi": "Persian", "persian": "Persian", "mandarin": "Chinese", "chinese": "Chinese", "filipino": "Filipino",
    "tagalog": "Filipino", "bahasa indonesia": "Indonesian", "bahasa melayu": "Malay", "kiswahili": "Swahili",
}


def canonical_language(name: str) -> str:
    text = re.sub(r"\(.*?\)|[^\w\s]", " ", name or "").strip().lower()
    text = " ".join(text.split())
    if not text or text in ("none", "unknown", "n a", "na"):
        return ""
    if text in _ALIASES:
        return _ALIASES[text]
    for key, value in _ALIASES.items():
        if key in text:
            return value
    names = {v.lower(): v for v in labels.LANGUAGE_NAMES.values()}
    return names.get(text, text.title())


def scripts_in(text: str) -> Counter:
    counts: Counter = Counter()
    for ch in text or "":
        cp = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                counts["Latin"] += 1
            continue
        for script, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[script] += 1
                break
        else:
            if ch.isalpha() and cp < 0x0250:
                counts["Latin"] += 1
    return counts


def detect(text: str, country: str = "") -> list[dict]:
    """[{language, weight, evidence[]}] best first, from the printed text alone."""
    weight: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)
    seen: set = set()
    for hit in labels.find_labels(text, fuzzy=False):
        name = labels.LANGUAGE_NAMES.get(hit.lang, hit.lang)
        if (name, hit.phrase) in seen:
            continue
        seen.add((name, hit.phrase))
        weight[name] += 2.0
        evidence[name].append(f"label '{hit.phrase}'")
    norm = labels.normalize(text)
    words = set(norm.split())
    for lang, markers in _MARKERS.items():
        name = labels.LANGUAGE_NAMES[lang]
        for m in markers:
            if (" " in m and re.search(rf"(?<!\w){re.escape(m)}(?!\w)", norm)) or m in words:
                if len(m) <= 3 and lang == "om":
                    continue   # 'kan', 'fi' only confirm Oromo already evidenced below
                weight[name] += 1.0
                evidence[name].append(f"word '{m}'")
    if weight.get("Oromo"):
        for m in ("kan", "fi"):
            if m in words:
                weight["Oromo"] += 0.5
    scripts = scripts_in(text)
    for script, count in scripts.items():
        if script == "Latin" or count < 3:
            continue
        name = (_BY_SCRIPT.get(script) or _BY_SCRIPT_COUNTRY.get((script, country))
                or _BY_SCRIPT_COUNTRY.get((script, "")))
        if name:
            weight[name] += 1.5
            evidence[name].append(f"{count} {script} characters")
    ranked = sorted(weight, key=lambda n: -weight[n])
    return [{"language": n, "weight": round(weight[n], 1), "evidence": evidence[n][:6]} for n in ranked]


def decide(detected: list[dict], agent_votes: list[str]) -> tuple[str, str, list[dict]]:
    """-> (primary, secondary, languages). Agents vote 1 each; printed evidence
    weighs by what it found."""
    score: dict[str, float] = defaultdict(float)
    info = {d["language"]: dict(d, votes=0) for d in detected}
    for d in detected:
        score[d["language"]] += d["weight"]
    for vote in agent_votes:
        for part in re.split(r"[,/;&]| and ", vote or ""):
            name = canonical_language(part)
            if not name:
                continue
            score[name] += 1.0
            info.setdefault(name, {"language": name, "weight": 0.0, "evidence": [], "votes": 0})["votes"] += 1
    if not score:
        return "", "", []
    evidenced = [n for n in score if info[n]["weight"] > 0]
    pool = evidenced or list(score)
    ranked = sorted(pool, key=lambda n: (-score[n], n))
    local = [n for n in ranked if n != "English" and info[n]["weight"] >= 2.0]
    primary = local[0] if "English" in ranked and local else ranked[0]
    others = [n for n in ranked if n != primary]
    languages = [dict(info[n], score=round(score[n], 1)) for n in [primary] + others]
    return primary, ", ".join(others), languages


def script_label(text: str) -> str:
    """'Latin', 'Ethiopic', ... or 'Mixed (Ethiopic, Latin)' for the printed text."""
    counts = {s: c for s, c in scripts_in(text).items() if c >= 3}
    if not counts:
        return ""
    names = sorted(counts, key=lambda s: -counts[s])
    names = ["CJK" if n in ("Han", "Kana", "Hangul") else n for n in names]
    names = list(dict.fromkeys(names))
    return names[0] if len(names) == 1 else f"Mixed ({', '.join(names)})"


def is_known_language(name: Optional[str]) -> bool:
    return bool(name) and canonical_language(name) != ""
