"""
National identity number registry — checksum + embedded-data validation.

Many countries' national numbers are not random: they carry a check digit
and often encode the holder's birth date and sex. That turns a VLM's read of
the number into something verifiable, and lets it cross-check the VLM's
reads of date_of_birth and sex — two independent visual reads agreeing
through arithmetic is strong evidence; disagreeing is a review flag.

Each scheme is data-plus-a-function, keyed by ISO 3166-1 alpha-3. Adding a
country = adding one entry to SCHEMES. Schemes with no published checksum
(Ethiopian FAN/FIN, Nigerian NIN, ...) still validate structure, and are
reported with checksum_valid=None rather than pretending to verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────
# Check-digit algorithms
# ─────────────────────────────────────────────────────────────

def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(digits: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


# ─────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────

@dataclass
class IDNumberCheck:
    country: str
    scheme: str
    number: str
    format_valid: bool
    checksum_valid: Optional[bool] = None
    date_of_birth: str = ""        # ISO, when the number encodes it
    birth_year: str = ""           # when only the year is encoded
    sex: str = ""                  # "Male" / "Female", when encoded
    details: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """Checksum passed — the number itself can be trusted as read."""
        return self.format_valid and self.checksum_valid is True


def _date(y: int, m: int, d: int) -> str:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return ""


def _century_guess(yy: int) -> int:
    return 2000 if yy <= date.today().year % 100 else 1900


# ─────────────────────────────────────────────────────────────
# Schemes
# ─────────────────────────────────────────────────────────────

def _chn(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{17}[\dX]", n):
        return None
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    expected = "10X98765432"[sum(int(a) * w for a, w in zip(n[:17], weights)) % 11]
    dob = _date(int(n[6:10]), int(n[10:12]), int(n[12:14]))
    return IDNumberCheck("CHN", "resident_identity_card", n, bool(dob), expected == n[17],
                         date_of_birth=dob, sex="Male" if int(n[16]) % 2 else "Female",
                         details={"region_code": n[:6]})


def _ind_aadhaar(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"[2-9]\d{11}", n):
        return None
    return IDNumberCheck("IND", "aadhaar", n, True, verhoeff_valid(n),
                         notes=["Aadhaar numbers are masked in output (UIDAI guidance)"])


def _are(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"784\d{12}", n):
        return None
    return IDNumberCheck("ARE", "emirates_id", n, True, luhn_valid(n), birth_year=n[3:7])


def _zaf(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{13}", n):
        return None
    yy = int(n[:2])
    dob = _date(_century_guess(yy) + yy, int(n[2:4]), int(n[4:6]))
    citizenship = {"0": "citizen", "1": "permanent_resident", "2": "refugee"}.get(n[10], "")
    return IDNumberCheck("ZAF", "sa_identity_number", n, bool(dob) and bool(citizenship), luhn_valid(n),
                         date_of_birth=dob, sex="Male" if int(n[6:10]) >= 5000 else "Female",
                         details={"citizenship": citizenship})


_EGY_GOVERNORATES = {"01", "02", "03", "04", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                     "21", "22", "23", "24", "25", "26", "27", "28", "29", "31", "32", "33", "34",
                     "35", "88"}


def _egy(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"[23]\d{13}", n):
        return None
    century = 1900 if n[0] == "2" else 2000
    dob = _date(century + int(n[1:3]), int(n[3:5]), int(n[5:7]))
    return IDNumberCheck("EGY", "egyptian_national_id", n, bool(dob) and n[7:9] in _EGY_GOVERNORATES, None,
                         date_of_birth=dob, sex="Male" if int(n[12]) % 2 else "Female",
                         details={"governorate_code": n[7:9]},
                         notes=["check-digit algorithm not published — structure and embedded date verified only"])


def _tur(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"[1-9]\d{10}", n):
        return None
    d = [int(c) for c in n]
    ok = ((sum(d[0:9:2]) * 7 - sum(d[1:8:2])) % 10 == d[9]) and (sum(d[:10]) % 10 == d[10])
    return IDNumberCheck("TUR", "tc_kimlik_no", n, True, ok)


def _bra_cpf(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{11}", n) or len(set(n)) == 1:
        return None
    d = [int(c) for c in n]
    c1 = (sum(d[i] * (10 - i) for i in range(9)) * 10) % 11 % 10
    c2 = (sum(d[i] * (11 - i) for i in range(10)) * 10) % 11 % 10
    return IDNumberCheck("BRA", "cpf", n, True, c1 == d[9] and c2 == d[10])


def _sau(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"[12]\d{9}", n):
        return None
    holder = "citizen" if n[0] == "1" else "resident (iqama)"
    return IDNumberCheck("SAU", "saudi_id_or_iqama", n, True, luhn_valid(n), details={"holder_type": holder})


def _kor(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{13}", n):
        return None
    century = {"9": 1800, "0": 1800, "1": 1900, "2": 1900, "5": 1900, "6": 1900,
               "3": 2000, "4": 2000, "7": 2000, "8": 2000}[n[6]]
    dob = _date(century + int(n[:2]), int(n[2:4]), int(n[4:6]))
    weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    ok = (11 - sum(int(a) * w for a, w in zip(n[:12], weights)) % 11) % 10 == int(n[12])
    check = IDNumberCheck("KOR", "resident_registration_number", n, bool(dob), ok if ok else None,
                          date_of_birth=dob, sex="Male" if int(n[6]) % 2 else "Female",
                          details={"foreign_resident": n[6] in "5678"})
    if not ok:
        check.notes.append("checksum not satisfied — RRNs issued since Oct 2020 have no check digit")
    return check


def _esp(n: str) -> Optional[IDNumberCheck]:
    m = re.fullmatch(r"([XYZ]?)(\d{7,8})([A-Z])", n)
    if not m:
        return None
    prefix, digits, letter = m.groups()
    if prefix and len(digits) != 7 or not prefix and len(digits) != 8:
        return None
    number = int(str("XYZ".index(prefix)) + digits) if prefix else int(digits)
    scheme = "nie" if prefix else "dni"
    return IDNumberCheck("ESP", scheme, n, True, "TRWAGMYFPDXBNJZSQVHLCKE"[number % 23] == letter)


def _pak(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{13}", n):
        return None
    return IDNumberCheck("PAK", "cnic", n, True, None, sex="Male" if int(n[12]) % 2 else "Female",
                         notes=["CNIC has no public check digit — structure verified only"])


def _idn(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{16}", n):
        return None
    dd, mm, yy = int(n[6:8]), int(n[8:10]), int(n[10:12])
    female = dd > 40
    dob = _date(_century_guess(yy) + yy, mm, dd - 40 if female else dd)
    return IDNumberCheck("IDN", "nik", n, bool(dob), None, date_of_birth=dob,
                         sex="Female" if female else "Male", details={"region_code": n[:6]},
                         notes=["NIK has no check digit — structure and embedded date verified only"])


def _mys(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{12}", n):
        return None
    yy = int(n[:2])
    dob = _date(_century_guess(yy) + yy, int(n[2:4]), int(n[4:6]))
    return IDNumberCheck("MYS", "mykad", n, bool(dob), None, date_of_birth=dob,
                         sex="Male" if int(n[11]) % 2 else "Female", details={"birthplace_code": n[6:8]},
                         notes=["MyKad has no check digit — structure and embedded date verified only"])


def _kaz(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{12}", n) or n[6] not in "123456":
        return None
    century = {"1": 1800, "2": 1800, "3": 1900, "4": 1900, "5": 2000, "6": 2000}[n[6]]
    dob = _date(century + int(n[:2]), int(n[2:4]), int(n[4:6]))
    d = [int(c) for c in n]
    r = sum(d[i] * (i + 1) for i in range(11)) % 11
    if r == 10:
        r = sum(d[i] * ((i + 2) % 11 + 1) for i in range(11)) % 11
    return IDNumberCheck("KAZ", "iin", n, bool(dob), r != 10 and r == d[11], date_of_birth=dob,
                         sex="Male" if int(n[6]) % 2 else "Female")


def _eth_fan(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{16}", n):
        return None
    return IDNumberCheck("ETH", "fayda_fan", n, True, None,
                         notes=["FAN has no public check digit — structure verified only"])


def _eth_fin(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{12}", n):
        return None
    return IDNumberCheck("ETH", "fayda_fin", n, True, None,
                         notes=["FIN has no public check digit — structure verified only"])


def _nga(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{11}", n):
        return None
    return IDNumberCheck("NGA", "nin", n, True, None,
                         notes=["NIN has no public check digit — structure verified only"])


def _gbr_driving_licence(n: str) -> Optional[IDNumberCheck]:
    """DVLA licence number: SURNAME(5, 9-padded) + decade + month(+50 female)
    + day + year digit + initials(2, 9-padded) + check + 2 letters."""
    m = re.fullmatch(r"([A-Z9]{5})(\d)(\d{2})(\d{2})(\d)([A-Z9]{2})(\d)([A-Z]{2})(\d{2})?", n)
    if not m:
        return None
    surname, decade, month, day, year, initials = m.group(1), m.group(2), int(m.group(3)), m.group(4), m.group(5), m.group(6)
    female = month > 50
    yy = int(decade + year)
    dob = _date(_century_guess(yy) + yy, month - 50 if female else month, int(day))
    return IDNumberCheck("GBR", "dvla_driving_licence", n, bool(dob), None, date_of_birth=dob,
                         sex="Female" if female else "Male",
                         details={"surname_prefix": surname.rstrip("9"), "initials": initials.rstrip("9")},
                         notes=["check character not published — structure, embedded date, sex and name "
                                "letters verified"])


def _can_ontario_licence(n: str) -> Optional[IDNumberCheck]:
    """Ontario licence: surname initial + 14 digits, the last six YYMMDD
    with the month +50 for women."""
    m = re.fullmatch(r"([A-Z])(\d{14})", n)
    if not m:
        return None
    digits = m.group(2)
    yy, month, day = int(digits[-6:-4]), int(digits[-4:-2]), int(digits[-2:])
    female = month > 50
    dob = _date(_century_guess(yy) + yy, month - 50 if female else month, day)
    return IDNumberCheck("CAN", "ontario_driving_licence", n, bool(dob), None, date_of_birth=dob,
                         sex="Female" if female else "Male", details={"surname_prefix": m.group(1)},
                         notes=["no check digit — structure, embedded date, sex and surname initial verified"])


def _swe_personnummer(n: str) -> Optional[IDNumberCheck]:
    m = re.fullmatch(r"(\d{2})?(\d{6})(\d{4})", n)
    if not m:
        return None
    ten = m.group(2) + m.group(3)
    yy, month, day = int(ten[:2]), int(ten[2:4]), int(ten[4:6])
    century = int(m.group(1)) * 100 if m.group(1) else _century_guess(yy)
    dob = _date(century + yy if m.group(1) else century + yy, month, day)
    return IDNumberCheck("SWE", "personnummer", n, bool(dob), luhn_valid(ten), date_of_birth=dob,
                         sex="Male" if int(ten[8]) % 2 else "Female", details={"role": "personal"})


def _bel_national_register(n: str) -> Optional[IDNumberCheck]:
    if not re.fullmatch(r"\d{11}", n):
        return None
    body, check = n[:9], int(n[9:])
    before_2000 = 97 - int(body) % 97 == check
    after_2000 = 97 - int("2" + body) % 97 == check
    century = 2000 if after_2000 and not before_2000 else 1900
    dob = _date(century + int(n[:2]), int(n[2:4]), int(n[4:6]))
    return IDNumberCheck("BEL", "national_register_number", n, True, before_2000 or after_2000,
                         date_of_birth=dob, sex="Male" if int(n[6:9]) % 2 else "Female",
                         details={"role": "personal"})


SCHEMES: dict[str, list[Callable[[str], Optional[IDNumberCheck]]]] = {
    "CHN": [_chn], "IND": [_ind_aadhaar], "ARE": [_are], "ZAF": [_zaf], "EGY": [_egy],
    "TUR": [_tur], "BRA": [_bra_cpf], "SAU": [_sau], "KOR": [_kor], "ESP": [_esp],
    "PAK": [_pak], "IDN": [_idn], "MYS": [_mys], "KAZ": [_kaz], "ETH": [_eth_fan, _eth_fin],
    "NGA": [_nga], "GBR": [_gbr_driving_licence], "CAN": [_can_ontario_licence],
    "SWE": [_swe_personnummer], "BEL": [_bel_national_register],
}

# Schemes that ARE the card's own document number (vs. a secondary number
# such as Brazil's CPF or the Fayda FIN printed on the back).
PRIMARY_DOCUMENT_SCHEMES = {
    "resident_identity_card", "aadhaar", "emirates_id", "sa_identity_number", "egyptian_national_id",
    "tc_kimlik_no", "saudi_id_or_iqama", "resident_registration_number", "dni", "nie", "cnic", "nik",
    "mykad", "iin", "fayda_fan", "nin", "dvla_driving_licence", "ontario_driving_licence",
}

# Schemes whose false-positive rate on a random number is low enough to
# suggest the issuing country when the agents didn't identify it.
_DISTINCTIVE = (_chn, _are, _tur, _kaz)


def compact(value: str) -> str:
    return re.sub(r"[\s\-./]", "", value or "").upper()


def check_numbers(country: str, candidates: dict[str, str]) -> list[IDNumberCheck]:
    """Validates each candidate (field name -> printed value) against the
    issuing country's schemes. With no country, only distinctive schemes
    are tried and matches are returned as hints (details['inferred'])."""
    results: list[IDNumberCheck] = []
    seen: set[str] = set()
    schemes = SCHEMES.get(country, [])
    inferred = not schemes and not country
    if inferred:
        schemes = list(_DISTINCTIVE)
    for field_name, value in candidates.items():
        number = compact(value)
        if not number or number in seen:
            continue
        for scheme in schemes:
            check = scheme(number)
            if check is None:
                continue
            if inferred and not check.verified:
                continue
            check.details["field"] = field_name
            if inferred:
                check.details["inferred"] = True
            results.append(check)
            seen.add(number)
            break
    return results


def mask_aadhaar(value: str) -> str:
    digits = compact(value)
    if not re.fullmatch(r"\d{12}", digits):
        return value
    return f"XXXX XXXX {digits[-4:]}"


def supported_schemes() -> dict[str, list[str]]:
    return {country: [fn.__name__.lstrip("_") for fn in fns] for country, fns in SCHEMES.items()}
