import pytest

from universal.id_numbers import (
    check_numbers, luhn_valid, mask_aadhaar, verhoeff_check_digit, verhoeff_valid,
)
from universal.normalize import (
    normalize_document_type, normalize_sex, resolve_country, resolve_nationality,
    split_bilingual_name, strip_identifier_label, strip_markers,
)


def test_check_digit_primitives():
    assert luhn_valid("79927398713")
    assert verhoeff_valid("2363") and verhoeff_check_digit("236") == "3"


@pytest.mark.parametrize("country, number, dob, sex", [
    ("CHN", "11010519491231002X", "1949-12-31", "Female"),
    ("ZAF", "8001015009087", "1980-01-01", "Male"),
    ("KOR", "8801011234568", "1988-01-01", "Male"),
    ("TUR", "10000000146", "", ""),
    ("BRA", "529.982.247-25", "", ""),
    ("ESP", "12345678Z", "", ""),
    ("ESP", "X1234567L", "", ""),
])
def test_checksum_schemes(country, number, dob, sex):
    (check,) = check_numbers(country, {"document_number": number})
    assert check.verified
    assert (check.date_of_birth, check.sex) == (dob, sex)


@pytest.mark.parametrize("country, number", [("CHN", "110105194912310021"), ("TUR", "10000000147")])
def test_checksum_failures(country, number):
    (check,) = check_numbers(country, {"document_number": number})
    assert check.checksum_valid is False


@pytest.mark.parametrize("country, number, dob, sex", [
    ("EGY", "29001011234567", "1990-01-01", "Female"),
    ("IDN", "3174055001900001", "1990-01-10", "Female"),   # day + 40 marks female
    ("MYS", "900515-14-5677", "1990-05-15", "Male"),
])
def test_structural_schemes_decode_embedded_data(country, number, dob, sex):
    (check,) = check_numbers(country, {"document_number": number})
    assert check.format_valid and check.checksum_valid is None
    assert (check.date_of_birth, check.sex) == (dob, sex)


def test_aadhaar_verhoeff_and_mask():
    body = "23456789012"
    number = body + verhoeff_check_digit(body)
    (check,) = check_numbers("IND", {"document_number": number})
    assert check.verified
    assert mask_aadhaar(number) == f"XXXX XXXX {number[-4:]}"


def test_unknown_country_infers_only_distinctive_verified_schemes():
    (check,) = check_numbers("", {"document_number": "11010519491231002X"})
    assert check.country == "CHN" and check.details["inferred"]
    assert check_numbers("", {"document_number": "8001015009087"}) == []   # ZAF isn't distinctive


@pytest.mark.parametrize("raw, value, flags", [
    ("ABEBE [LOW CONFIDENCE]", "ABEBE", {"low_confidence"}),
    ("L898902C3 [CORRECTED]", "L898902C3", {"corrected"}),
    ("1990-01-20?", "1990-01-20", {"uncertain"}),
    ("N/A", "", set()),
    ("无", "", set()),
    ("[ABEBE KEBED](https://i.imgur.com/a/x)", "ABEBE KEBED", {"markup_removed"}),
])
def test_strip_markers(raw, value, flags):
    assert strip_markers(raw) == (value, flags)


@pytest.mark.parametrize("raw, expected", [
    ("M", "Male"), ("F", "Female"), ("ወንድ | Male".replace("|", "/"), "Male"), ("女", "Female"),
    ("أنثى", "Female"), ("Мужской", "Male"), ("X", "X"),
])
def test_sex(raw, expected):
    assert normalize_sex(raw)[0] == expected


def test_sex_conflict_is_flagged_not_guessed():
    assert normalize_sex("M/F") == ("", {"conflicting_values"})


def test_bilingual_name_split():
    assert split_bilingual_name("መሳይ ታደሰ (MESAY TADESSE)") == ("መሳይ ታደሰ", "MESAY TADESSE")
    assert split_bilingual_name("AHMED ALI (أحمد علي)") == ("أحمد علي", "AHMED ALI")


@pytest.mark.parametrize("raw, expected", [
    ("Passport", "passport"), ("Saudi Iqama", "residence_permit"), ("Driving Licence", "drivers_license"),
    ("Resident Identity Card", "national_id"), ("UNHCR Convention Travel Document", "refugee_travel_document"),
    ("Running shoe", "other"),
])
def test_document_type(raw, expected):
    assert normalize_document_type(raw) == expected


@pytest.mark.parametrize("raw, code", [
    ("Ethiopia", "ETH"), ("ethiopian", "ETH"), ("中国", "CHN"), ("ኢትዮጵያ", "ETH"),
    ("المملكة العربية السعودية", "SAU"), ("Россия", "RUS"), ("UAE", "ARE"),
])
def test_country_resolution(raw, code):
    assert resolve_country(raw)[0] == code
    assert resolve_nationality(raw)[0] == code


@pytest.mark.parametrize("raw, expected", [
    ("FAN 1234 5678 9012 3456", "1234 5678 9012 3456"), ("No.: A1234567", "A1234567"),
    ("L898902C3", "L898902C3"), ("EP1234567", "EP1234567"),
])
def test_identifier_label(raw, expected):
    assert strip_identifier_label(raw) == expected


@pytest.mark.parametrize("raw", ["[CA]", "(CAN)", '"Canada"', "“CANADA”", " 'CAN' "])
def test_wrapped_country_values_resolve(raw):
    from universal.normalize import resolve_country, resolve_nationality
    assert resolve_country(raw)[0] == "CAN" and resolve_nationality(raw)[0] == "CAN"


@pytest.mark.parametrize("raw", ["CA[CANADA]", "CAN / Canada", "Canada (CAN)", "[CA] CANADA"])
def test_country_annotations_resolve_when_every_part_agrees(raw):
    from universal.normalize import resolve_country, resolve_nationality
    assert resolve_country(raw)[0] == "CAN" and resolve_nationality(raw)[0] == "CAN"


def test_disagreeing_parts_are_not_guessed():
    from universal.normalize import resolve_country
    assert resolve_country("CAN / ETH") == ("", "")


@pytest.mark.parametrize("raw, want", [("[a]HELFE BEREKET", "HELFE BEREKET"),
                                       ("[CHECKED] DOE Jane", "DOE Jane"),
                                       ("DOE<<JANE", "DOE JANE"),
                                       ("ABEBE (አበበ)", "ABEBE (አበበ)")])       # parentheses are kept
def test_clean_name_drops_bracket_annotations(raw, want):
    from universal.normalize import clean_name
    assert clean_name(raw) == want
