"""
Deterministic field normalization for agent output.

The prompts ask the agents to embed their own uncertainty in values
("value [LOW CONFIDENCE]", "value [CORRECTED]", "value?") and to put an
English transliteration in parentheses after non-Latin names. Callers of an
API need clean values, so this module strips those markers into per-field
flags and splits bilingual names into their local-script / Latin parts.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from idx.countries import resolve_country_code, resolve_nationality_code

FLAG_LOW_CONFIDENCE = "low_confidence"
FLAG_CORRECTED = "corrected"
FLAG_UNCERTAIN = "uncertain"

_MARKER_RE = re.compile(r"\s*\[\s*(LOW[\s_-]*CONFIDENCE|CORRECTED)\s*\]\s*", re.I)
_PLACEHOLDER_VALUES = {
    "n/a", "na", "none", "null", "nil", "-", "--", "—", "not visible", "not available",
    "not applicable", "not shown", "not present", "illegible", "unreadable", "unknown",
    "not found", "no data", "empty", "?", "xxx",
    # "none" as agents write it in the document's own language
    "无", "無", "없음", "なし", "لا يوجد", "لايوجد", "የለም", "нет", "keine", "aucun", "aucune", "ninguno", "nenhum",
}


_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_MD_EMPHASIS = re.compile(r"(\*\*|__|`)")


def _strip_markup(text: str) -> tuple[str, bool]:
    """Agents occasionally wrap values in markdown links or append URLs
    ('[ABEBE](https://i.imgur.com/...)') — never part of a printed value."""
    cleaned = _MD_EMPHASIS.sub("", _URL.sub("", _MD_LINK.sub(r"\1", text)))
    cleaned = " ".join(cleaned.split())
    return cleaned, cleaned != " ".join(text.split())


def strip_markers(value: str) -> tuple[str, set[str]]:
    flags: set[str] = set()
    text, had_markup = _strip_markup((value or "").strip())
    if had_markup:
        flags.add("markup_removed")
    for m in _MARKER_RE.finditer(text):
        flags.add(FLAG_CORRECTED if "CORRECT" in m.group(1).upper() else FLAG_LOW_CONFIDENCE)
    text = _MARKER_RE.sub(" ", text).strip()
    if text.endswith("?") and len(text) > 1:
        flags.add(FLAG_UNCERTAIN)
        text = text.rstrip("?").strip()
    if text.lower() in _PLACEHOLDER_VALUES:
        text = ""
    return text, flags


# ─────────────────────────────────────────────────────────────
# Script detection
# ─────────────────────────────────────────────────────────────

_SCRIPT_PREFIXES = (
    ("ETHIOPIC", "Ethiopic"), ("ARABIC", "Arabic"), ("HEBREW", "Hebrew"),
    ("CJK", "CJK"), ("HIRAGANA", "CJK"), ("KATAKANA", "CJK"), ("HANGUL", "CJK"),
    ("CYRILLIC", "Cyrillic"), ("DEVANAGARI", "Devanagari"), ("GREEK", "Greek"),
    ("THAI", "Thai"), ("GEORGIAN", "Georgian"), ("ARMENIAN", "Armenian"),
    ("BENGALI", "Bengali"), ("TAMIL", "Tamil"), ("TELUGU", "Telugu"),
    ("GUJARATI", "Gujarati"), ("GURMUKHI", "Gurmukhi"), ("KANNADA", "Kannada"),
    ("MALAYALAM", "Malayalam"), ("SINHALA", "Sinhala"), ("MYANMAR", "Burmese"),
    ("KHMER", "Khmer"), ("LAO", "Lao"), ("MONGOLIAN", "Mongolian"), ("TIBETAN", "Tibetan"),
    ("LATIN", "Latin"),
)


def detect_script(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text or "":
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for prefix, script in _SCRIPT_PREFIXES:
            if name.startswith(prefix):
                counts[script] = counts.get(script, 0) + 1
                break
        else:
            counts["Other"] = counts.get("Other", 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)


def is_latin(text: str) -> bool:
    return detect_script(text) in ("Latin", "")


# ─────────────────────────────────────────────────────────────
# Names
# ─────────────────────────────────────────────────────────────

_PAREN_NAME = re.compile(r"^(?P<a>[^()]+?)\s*\((?P<b>[^()]+)\)\s*$")


def split_bilingual_name(value: str) -> tuple[str, str]:
    """'መሳይ ታደሰ (MESAY TADESSE)' -> ('መሳይ ታደሰ', 'MESAY TADESSE').
    Returns (local_script_part, latin_part); either may be ''."""
    text = " ".join((value or "").split())
    if not text:
        return "", ""
    m = _PAREN_NAME.match(text)
    if m:
        a, b = m.group("a").strip(), m.group("b").strip()
        if is_latin(a) and not is_latin(b):
            return b, a
        if not is_latin(a) and is_latin(b):
            return a, b
        return "", text if is_latin(text) else ""
    return ("", text) if is_latin(text) else (text, "")


def clean_name(value: str) -> str:
    # No identity document prints square brackets in a name: '[a]HELFE',
    # '[CHECKED] ...' are model annotations / placeholder echoes.
    text = re.sub(r"\[[^\]]*\]|[\[\]]", " ", value or "")
    text = re.sub(r"[<>«»]+", " ", text)          # MRZ fillers (and OCR's '>' misread of them)
    text = " ".join(text.split())
    return text.strip(" ,;")


# ─────────────────────────────────────────────────────────────
# Sex / gender
# ─────────────────────────────────────────────────────────────

_MALE = {
    "m", "male", "man", "masculin", "masculine", "masculino", "maschile", "männlich", "mannlich",
    "мужской", "муж", "м", "чоловіча", "erkek", "laki-laki", "laki laki", "lelaki", "nam",
    "ذكر", "مذكر", "مرد", "男", "남", "남자", "ወንድ", "पुरुष", "ชาย", "זכר", "άρρεν", "ανδρας",
    "dhiira", "lab", "labood", "mume",                               # Oromo, Somali, Swahili
}
_FEMALE = {
    "f", "female", "woman", "féminin", "feminin", "feminine", "femenino", "feminino", "femminile",
    "weiblich", "женский", "жен", "ж", "жіноча", "kadın", "kadin", "perempuan", "wanita", "nữ", "nu",
    "أنثى", "انثى", "مؤنث", "زن", "女", "여", "여자", "ሴት", "महिला", "स्त्री", "หญิง", "נקבה", "θήλυ", "γυναικα",
    "dhalaa", "dubartii", "dheddig", "mke",                        # Oromo, Somali, Swahili
}
_NONBINARY = {"x", "<", "other", "diverse", "divers", "unspecified", "non-binary", "nonbinary"}


def normalize_sex(value: str) -> tuple[str, set[str]]:
    """-> ('Male'|'Female'|'X'|'Unknown'|'', flags). Bilingual 'M/M' or
    'F / ሴት' is read part by part; a contradiction is flagged, not guessed."""
    text = (value or "").strip()
    if not text:
        return "", set()
    parts = [p.strip().lower().strip(".") for p in re.split(r"[/|,]", text) if p.strip()]
    found = set()
    for part in parts or [text.lower()]:
        if part in _MALE:
            found.add("Male")
        elif part in _FEMALE:
            found.add("Female")
        elif part in _NONBINARY:
            found.add("X")
        elif part == "unknown":
            found.add("Unknown")
    if len(found) == 1:
        return found.pop(), set()
    if len(found) > 1:
        return "", {"conflicting_values"}
    return "", {"unrecognized_value"}


# ─────────────────────────────────────────────────────────────
# Document type
# ─────────────────────────────────────────────────────────────

_DOC_TYPE_RULES = (
    ("refugee_travel_document", ("refugee", "convention travel", "unhcr", "1951 convention")),
    # Work / student / civic cards, before the generic "id card" of national_id.
    ("work_permit", ("work permit", "employment pass", "permis de travail", "arbeitserlaubnis", "የሥራ ፈቃድ",
                     "የስራ ፈቃድ")),
    ("employee_id", ("employee", "staff id", "staff card", "work id", "work card", "service card", "company id",
                     "corporate id", "personnel", "carte professionnelle", "dienstausweis", "mitarbeiterausweis",
                     "tarjeta de empleado", "carnet de empleado", "crachá", "tessera aziendale", "工作证", "工作證",
                     "员工证", "員工證", "社員証", "職員証", "사원증", "بطاقة موظف", "بطاقة عمل", "ሰራተኛ", "ሠራተኛ",
                     "hojjetaa")),
    ("student_id", ("student", "étudiant", "etudiant", "estudiante", "studente", "studentenausweis", "学生证",
                    "學生證", "学生証", "학생증", "ተማሪ", "barataa")),
    ("voter_id", ("voter", "electoral", "électeur", "elector", "የመራጭ")),
    ("health_insurance_card", ("health insurance", "health card", "medicare", "carte vitale",
                               "gesundheitskarte", "krankenversicherung")),
    ("tax_id", ("taxpayer", "tax id", "tin card", "permanent account number")),
    ("residence_permit", ("residence", "resident permit", "residency", "iqama", "brp", "green card",
                          "permanent resident", "titre de séjour", "aufenthalt", "permesso di soggiorno",
                          "tarjeta de residencia", "إقامة", "اقامة")),
    ("drivers_license", ("driver", "driving", "licence to drive", "permis de conduire", "führerschein",
                         "fuhrerschein", "licencia de conducir", "carteira de habilitação", "patente",
                         "rijbewijs", "運転免許", "驾驶证", "운전면허", "رخصة قيادة", "መንጃ ፈቃድ")),
    ("visa", ("visa", "e-visa", "evisa", "entry permit", "تأشيرة")),
    ("military_id", ("military", "armed forces", "army", "stanag", "defence id", "defense id")),
    ("passport", ("passport", "passeport", "pasaporte", "reisepass", "passaporto", "паспорт",
                  "جواز", "护照", "護照", "여권", "旅券", "ፓስፖርት")),
    ("national_id", ("national id", "identity card", "id card", "identification", "citizen",
                     "carte d'identité", "carte nationale", "personalausweis", "documento nacional",
                     "cédula", "cedula", "dni", "aadhaar", "fayda", "nin", "huduma", "emirates id",
                     "identiteitskaart", "identitetskort", "henkilökortti", "carta d'identità",
                     "carta di identità", "carta di identita", "dowód osobisty", "občanský průkaz",
                     "cartão de cidadão", "documento de identidad", "kimlik kartı", "osobna iskaznica",
                     "személyazonosító", "identitātes karte", "asmens tapatybės", "isikutunnistus",
                     "resident identity", "居民身份证", "身份证", "身分證", "주민등록", "بطاقة", "هوية",
                     "ብሔራዊ", "መታወቂያ", "national")),
)


def normalize_document_type(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    for canonical, keywords in _DOC_TYPE_RULES:
        if any(_keyword_in(k, text) for k in keywords):
            return canonical
    return "other"


def _keyword_in(keyword: str, text: str) -> bool:
    # Short Latin keywords ("dni", "nin", "brp") must match as whole words.
    if len(keyword) <= 4 and keyword.isascii():
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
    return keyword in text


# ─────────────────────────────────────────────────────────────
# Countries
# ─────────────────────────────────────────────────────────────

# Country names as printed in their own language/script, which the ICAO
# table (English/Latin) doesn't carry. Keys are compared case-folded.
NATIVE_COUNTRY_NAMES = {
    "中国": "CHN", "中华人民共和国": "CHN", "中華人民共和國": "CHN", "中國": "CHN",
    "中華民國": "TWN", "中华民国": "TWN", "台灣": "TWN", "台湾": "TWN", "香港": "HKG", "澳門": "MAC", "澳门": "MAC",
    "日本": "JPN", "日本国": "JPN", "大韓民国": "KOR", "대한민국": "KOR", "한국": "KOR", "조선": "PRK",
    "ኢትዮጵያ": "ETH", "የኢትዮጵያ": "ETH", "ኢትዮጵያዊ": "ETH", "ኤርትራ": "ERI",
    "ኢትዮጲያ": "ETH", "ኢትዮጲያዊ": "ETH", "ኢትዮጵያዊት": "ETH",           # common spellings
    "itiyoophiyaa": "ETH", "itoophiyaa": "ETH", "itoophiyaa'i": "ETH",   # Afaan Oromo
    "itoobiya": "ETH", "itoobiyaan": "ETH",                            # Somali
    "مصر": "EGY", "جمهورية مصر العربية": "EGY", "السعودية": "SAU", "المملكة العربية السعودية": "SAU",
    "الإمارات": "ARE", "الامارات": "ARE", "الإمارات العربية المتحدة": "ARE", "قطر": "QAT", "الكويت": "KWT",
    "البحرين": "BHR", "عمان": "OMN", "الأردن": "JOR", "العراق": "IRQ", "سوريا": "SYR", "لبنان": "LBN",
    "اليمن": "YEM", "المغرب": "MAR", "الجزائر": "DZA", "تونس": "TUN", "ليبيا": "LBY", "السودان": "SDN",
    "فلسطين": "PSE", "ایران": "IRN", "إيران": "IRN", "افغانستان": "AFG", "پاکستان": "PAK",
    "россия": "RUS", "российская федерация": "RUS", "україна": "UKR", "беларусь": "BLR",
    "қазақстан": "KAZ", "казахстан": "KAZ", "кыргызстан": "KGZ", "ўзбекистон": "UZB", "узбекистан": "UZB",
    "тоҷикистон": "TJK", "монгол улс": "MNG", "србија": "SRB", "българия": "BGR", "македонија": "MKD",
    "भारत": "IND", "भारत गणराज्य": "IND", "नेपाल": "NPL", "বাংলাদেশ": "BGD", "ශ්‍රී ලංකාව": "LKA",
    "ประเทศไทย": "THA", "ไทย": "THA", "việt nam": "VNM", "မြန်မာ": "MMR", "កម្ពុជា": "KHM", "ລາວ": "LAO",
    "ישראל": "ISR", "ελλάδα": "GRC", "ελληνική δημοκρατία": "GRC", "საქართველო": "GEO", "հայաստան": "ARM",
    "türkiye": "TUR", "turkiye": "TUR", "españa": "ESP", "méxico": "MEX", "brasil": "BRA", "italia": "ITA",
    "nederland": "NLD", "polska": "POL", "österreich": "AUT", "schweiz": "CHE", "suisse": "CHE",
    "belgique": "BEL", "belgië": "BEL", "sverige": "SWE", "norge": "NOR", "danmark": "DNK", "suomi": "FIN",
    "česká republika": "CZE", "magyarország": "HUN", "românia": "ROU", "hrvatska": "HRV", "éire": "IRL",
    "soomaaliya": "SOM", "jamhuuriyadda federaalka soomaaliya": "SOM", "kenya": "KEN", "tanzania": "TZA",
    # nationality as printed in the document's own language
    "nederlandse": "NLD", "nederlands": "NLD", "deutsch": "DEU", "deutsche": "DEU", "svensk": "SWE",
    "norsk": "NOR", "dansk": "DNK", "suomalainen": "FIN", "italiana": "ITA", "italiano": "ITA",
    "française": "FRA", "francaise": "FRA", "français": "FRA", "francais": "FRA", "española": "ESP",
    "espanola": "ESP", "español": "ESP", "espanol": "ESP", "belge": "BEL", "belgische": "BEL",
    "portuguesa": "PRT", "portuguesa/o": "PRT", "polskie": "POL", "polska obywatelstwo": "POL",
    "österreich staatsbürgerschaft": "AUT", "schweizer": "CHE", "ελληνική": "GRC", "türk": "TUR",
    "ኢትዮጵያዊት": "ETH", "إثيوبيا": "ETH", "اثيوبيا": "ETH", "أثيوبيا": "ETH", "مصري": "EGY", "سعودي": "SAU",
    "إماراتي": "ARE", "ایرانی": "IRN", "中国籍": "CHN", "日本国籍": "JPN", "한국": "KOR",
}


def _native_country(text: str) -> str:
    folded = " ".join(text.casefold().split())
    if folded in NATIVE_COUNTRY_NAMES:
        return NATIVE_COUNTRY_NAMES[folded]
    for name, code in NATIVE_COUNTRY_NAMES.items():
        if len(name) >= 2 and name in folded and not name.isascii():
            return code
    return ""


# Models sometimes wrap a code or name: '[CA]', '(CAN)', '"Canada"'.
_WRAPPING = " \t[](){}<>\"'`“”‘’"


def _by_parts(value: str, resolve) -> tuple[str, str]:
    """'CA[CANADA]', 'ETH / Ethiopia': every part that resolves must name the
    same country, else nothing is guessed."""
    parts = [p for p in re.split(r"[\[\](){}/|,;]+", value or "") if p.strip(_WRAPPING)]
    if len(parts) < 2:
        return "", ""
    found = {r for r in (resolve(p) for p in parts) if r[0]}
    return found.pop() if len({code for code, _ in found}) == 1 else ("", "")


def resolve_country(value: str) -> tuple[str, str]:
    """Any code / name / demonym -> (ISO alpha-3, English name), or ('', '')
    when idx's ICAO/PRADO table can't place it."""
    text = (value or "").strip(_WRAPPING)
    if not text:
        return "", ""
    whole = _resolve_country_whole(text)
    return whole if whole[0] else _by_parts(text, _resolve_country_whole)


def _resolve_country_whole(text: str) -> tuple[str, str]:
    text = text.strip(_WRAPPING)
    if not text:
        return "", ""
    code, name = resolve_country_code(text)
    if name:
        return code, name
    code, demonym = resolve_nationality_code(text)
    if demonym and code:
        code2, name2 = resolve_country_code(code)
        return code, name2
    native = _native_country(text)
    if native:
        return native, resolve_country_code(native)[1]
    return "", ""


def resolve_nationality(value: str) -> tuple[str, str]:
    """-> (ISO alpha-3, demonym) or ('', '')."""
    text = (value or "").strip(_WRAPPING)
    if not text:
        return "", ""
    whole = _resolve_nationality_whole(text)
    return whole if whole[0] else _by_parts(text, _resolve_nationality_whole)


def _resolve_nationality_whole(text: str) -> tuple[str, str]:
    text = text.strip(_WRAPPING)
    if not text:
        return "", ""
    code, demonym = resolve_nationality_code(text)
    if demonym:
        return code, demonym
    code, name = resolve_country_code(text)
    if name:
        code2, demonym2 = resolve_nationality_code(code)
        return code, demonym2 or name
    native = _native_country(text)
    if native:
        return native, resolve_nationality_code(native)[1] or resolve_country_code(native)[1]
    return "", ""


# ─────────────────────────────────────────────────────────────
# Small fields
# ─────────────────────────────────────────────────────────────

_BLOOD = re.compile(r"^\s*(AB|A|B|O|0)\s*(\+|-|POS(?:ITIVE)?|NEG(?:ATIVE)?|RH\s*\+|RH\s*-)?\s*$", re.I)


def normalize_blood_type(value: str) -> str:
    m = _BLOOD.match(value or "")
    if not m:
        return (value or "").strip()
    group = "O" if m.group(1) == "0" else m.group(1).upper()
    rh = (m.group(2) or "").upper().replace(" ", "")
    sign = "+" if rh in ("+", "POS", "POSITIVE", "RH+") else "-" if rh in ("-", "NEG", "NEGATIVE", "RH-") else ""
    return group + sign


_ID_LABEL = re.compile(
    r"^(?:(?:FAN|FIN|NIN|UIN|UID|CIN|CNIC|NIC|NIK|PSN|TCKN|ID|DOC|CARD|NO|NR|NUM|NUMBER|N°|Nº|#)\.?\s*[:#.]?\s*)+",
    re.I)


def strip_identifier_label(value: str) -> str:
    """'FAN 1234 5678' / 'No.: A123' -> the number. Only strips when what
    remains still carries digits, so a number that merely starts with
    letters ('ID1234' style serials without a space are untouched)."""
    text = (value or "").strip()
    if _ID_LABEL.fullmatch(text):
        return ""            # the agent returned the label itself ("FAN", "No.")
    m = _ID_LABEL.match(text)
    if not m or not m.group().strip() or not re.search(r"\s|[:#.]$", m.group()):
        return text
    rest = text[m.end():].strip()
    return rest if sum(c.isdigit() for c in rest) >= 4 else text


def compact_identifier(value: str) -> str:
    return re.sub(r"[\s\-./]", "", (value or "")).upper()


def comparison_key(field: str, value: str) -> str:
    """Key under which two agents' values count as agreeing."""
    text = unicodedata.normalize("NFKC", value or "").casefold()
    if field in ("document_number", "personal_number") or field.startswith("mrz_line"):
        return compact_identifier(text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def first_nonempty(*values: Optional[str]) -> str:
    return next((v for v in values if v), "")
