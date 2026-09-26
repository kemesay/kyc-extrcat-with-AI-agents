"""Name parts — each word of the given names is its own name; surnames never split."""

import pytest

from universal.llm import clean_unicode
from universal.names import name_words, parts_from_full_name, parts_from_labels, split_given
from universal.prompts import EXTRACTION_FIELDS
from universal.reconcile import Context, FieldResult, settle_name_parts


@pytest.mark.parametrize("given, first, middle", [
    ("ABEBE KEBEDE", "ABEBE", "KEBEDE"), ("SARA T", "SARA", "T"), ("MARTA SOLOMON", "MARTA", "SOLOMON"),
    ("ABDUL RAHMAN KHALID", "ABDUL RAHMAN", "KHALID"), ("ABD AL RAHMAN", "ABD AL RAHMAN", ""),
    ("NOUR EL DIN HASSAN", "NOUR EL DIN", "HASSAN"), ("MARIA DE LOS ANGELES", "MARIA", "DE LOS ANGELES"),
    ("Jean-Pierre Marc", "Jean-Pierre", "Marc"), ("BEREKET", "BEREKET", ""),
])
def test_split_given(given, first, middle):
    assert split_given(given) == (first, middle)


def test_romanized_cjk_given_name_not_split():
    assert split_given("MIN JUN", "KOR") == ("MIN JUN", "")


def test_surname_kept_whole():
    p = parts_from_labels("GARCÍA LÓPEZ", "ANA SOFÍA")
    assert (p.first, p.middle, p.last) == ("ANA", "SOFÍA", "GARCÍA LÓPEZ")


def test_patronymic_order():
    p = parts_from_full_name("HANA TADESSE BEKELE", "ETH")
    assert (p.first, p.middle, p.last) == ("HANA", "TADESSE", "BEKELE")


def test_family_name_first_order():
    p = parts_from_full_name("KIM MIN JUN", "KOR")
    assert (p.first, p.middle, p.last) == ("MIN JUN", "", "KIM")


def test_no_order_guess_for_western_full_name_without_labels():
    assert parts_from_full_name("ANNA MARIA SMITH", "USA").last == ""


def _fields(**v):
    f = {n: FieldResult() for n in EXTRACTION_FIELDS}
    for k, val in v.items():
        f[k] = FieldResult(value=val, confidence=0.7, source="x")
    return f


def test_settle_splits_multiword_first_name():
    f = _fields(first_name="ABEBE KEBEDE", last_name="TESFAYE")
    assert settle_name_parts(f, Context(country="USA")) == "ABEBE KEBEDE"
    assert (f["first_name"].value, f["middle_name"].value) == ("ABEBE", "KEBEDE")


def test_settle_uses_patronymic_order_when_no_parts():
    f = _fields(full_name="HANA TADESSE BEKELE")
    settle_name_parts(f, Context(country="ETH"))
    assert (f["first_name"].value, f["middle_name"].value, f["last_name"].value) == ("HANA", "TADESSE", "BEKELE")


def test_lone_surrogate_cleaned():
    assert clean_unicode("CARTA DI IDENTIT\udcfa") == "CARTA DI IDENTIT�"
    assert clean_unicode("normal") == "normal"


def test_patronymic_first_word_never_a_surname():
    f = _fields(full_name="HANA TADESSE BEKELE", last_name="HANA", first_name="TADESSE BEKELE")
    settle_name_parts(f, Context(country="ETH"))
    assert (f["first_name"].value, f["middle_name"].value, f["last_name"].value) == ("HANA", "TADESSE", "BEKELE")


def test_labelled_ethiopian_passport_surname_kept():
    f = _fields(full_name="DAWIT ALEMU TESHOME", last_name="TESHOME", first_name="DAWIT ALEMU")
    settle_name_parts(f, Context(country="ETH"))
    assert (f["first_name"].value, f["middle_name"].value, f["last_name"].value) == ("DAWIT", "ALEMU", "TESHOME")


def test_full_name_ordered_first_middle_last():
    f = _fields(full_name="GEBRE ANNA MARIA", first_name="ANNA MARIA", last_name="GEBRE")
    settle_name_parts(f, Context(country="USA"))
    assert f["full_name"].value == "ANNA MARIA GEBRE" and f["full_name"].original == "GEBRE ANNA MARIA"


def test_family_first_country_keeps_family_first():
    f = _fields(full_name="WANG FANG", first_name="FANG", last_name="WANG")
    settle_name_parts(f, Context(country="CHN"))
    assert f["full_name"].value == "WANG FANG"


def test_full_name_with_names_not_in_parts_is_kept():
    f = _fields(full_name="ANNA MARIA GEBRE TESFA", first_name="ANNA", last_name="GEBRE")
    settle_name_parts(f, Context(country="USA"))
    assert f["full_name"].value == "ANNA MARIA GEBRE TESFA"


def test_label_word_is_not_a_name():
    from universal.reconcile import normalize_value
    assert normalize_value("full_name", "NAME", Context())[0] == ""
    assert normalize_value("full_name_latin", "NAME KEBBEDE", Context())[0] == "KEBBEDE"
    assert normalize_value("last_name", "SURNAME", Context())[0] == ""


def test_honorific_does_not_block_natural_order():
    f = _fields(full_name="SMITH MR JOHN PAUL", first_name="JOHN PAUL", last_name="SMITH")
    settle_name_parts(f, Context(country="GBR"))
    assert f["full_name"].value == "JOHN PAUL SMITH"


def test_full_name_missing_surname_rebuilt_from_parts():
    f = _fields(full_name="MARCO LUCA TORINO", first_name="MARCO LUCA", last_name="ROSSI")
    settle_name_parts(f, Context(country="ITA"))
    assert f["full_name"].value == "MARCO LUCA ROSSI"


@pytest.mark.parametrize("surname, given, country, script, expected", [
    ("ለማ", "መሳይ ከበደ", "ETH", "Ethiopic", "መሳይ ከበደ ለማ"),
    ("王", "芳", "CHN", "CJK", "王芳"),
    ("حسن", "خالد عمر", "ARE", "Arabic", "خالد عمر حسن"),
    ("Иванов", "Иван", "RUS", "Cyrillic", "Иван Иванов"),
])
def test_compose_local_name(surname, given, country, script, expected):
    from universal.names import compose_local_name
    assert compose_local_name(surname, given, country, script) == expected


def test_local_name_from_transcriber():
    from universal.printed import resolve_local_name
    f = _fields()
    report = resolve_local_name(f, {"transcriber": {"surname_local_script": "ለማ", "given_names_local_script": "መሳይ ከበደ"}},
                                Context(country="ETH"))
    assert f["full_name_local"].value == "መሳይ ከበደ ለማ" and report["surname_local"] == "ለማ"


def test_transliteration_in_local_field_ignored():
    from universal.printed import resolve_local_name
    f = _fields()
    assert resolve_local_name(f, {"transcriber": {"full_name_local_script": "MESAY KEBBEDE"}}, Context()) is None
    assert f["full_name_local"].value == ""


def test_one_local_slot_is_the_whole_name():
    from universal.printed import resolve_local_name
    f = _fields()
    resolve_local_name(f, {"transcriber": {"surname_local_script": "ሀና ታደሰ በቀለ"}}, Context(country="ETH"))
    assert f["full_name_local"].value == "ሀና ታደሰ በቀለ"


def test_same_local_name_in_both_slots_not_doubled():
    from universal.printed import resolve_local_name
    f = _fields()
    resolve_local_name(f, {"transcriber": {"surname_local_script": "አበበ ከበደ", "given_names_local_script": "አበበ ከበደ"}},
                       Context(country="ETH"))
    assert f["full_name_local"].value == "አበበ ከበደ"


def test_mixed_script_local_name_rejected():
    from universal.reconcile import normalize_value
    assert normalize_value("full_name_local", "pora ተጋባ ሕክታራ", Context())[0] == ""


def test_native_script_parts_never_rebuild_latin_full_name():
    f = _fields(full_name="ABEBE KEBEDE TESFAYE", first_name="አበበ", last_name="ቃብዎደ")
    settle_name_parts(f, Context(country="ETH"))
    assert f["full_name"].value == "ABEBE KEBEDE TESFAYE"
    assert (f["first_name"].value, f["middle_name"].value, f["last_name"].value) == ("ABEBE", "KEBEDE", "TESFAYE")


@pytest.mark.parametrize("local, latin, expected", [
    ("ረና ታደሰ በቀለ", "HANA TADESSE BEKELE", "ሀና ታደሰ በቀለ"),       # ረ (re) -> ሀ (ha)
    ("አበበ ካበደ ተስፋየ", "ABEBE KEBEDE TESFAYE", "አበበ ከበደ ተስፋየ"),     # ካ (ka) -> ከ (ke)
    ("መሳይ ከበደ ለማ", "MESAY KEBBEDE LEMMA", "መሳይ ከበደ ለማ"),         # gemination: already right
])
def test_geez_name_checked_against_latin_spelling(local, latin, expected):
    from universal.ethiopic import correct_name
    assert correct_name(local, latin)[0] == expected


def test_geez_ambiguous_vowel_left_alone():
    from universal.ethiopic import correct_word
    assert correct_word("ተስፋየ", "tesfaye") is None      # የ and ዬ both spell "ye"


def test_holder_nationality_decides_naming_order():
    f = _fields(full_name="DAWIT ALEMU TESHOME", nationality="ETHIOPIAN")
    settle_name_parts(f, Context(country="ARE"))
    assert (f["first_name"].value, f["middle_name"].value, f["last_name"].value) == ("DAWIT", "ALEMU", "TESHOME")


def test_garbled_geez_reading_loses_to_one_that_spells_the_latin_name():
    from universal.reconcile import cross_script_check
    f = _fields(full_name="ABEBE KEBEDE TESFAYE", full_name_local="አበበ ድሬዳት ግብርሳድያ የተፈዯይ አበበ")
    f["full_name_local"].alternatives = ["አበበ ካበደ ተስፋየ"]
    cross_script_check(f, Context(country="ETH"))
    assert f["full_name_local"].value == "አበበ ከበደ ተስፋየ"       # chosen, then ካ -> ከ


def test_geez_readings_combined_word_by_word():
    from universal.reconcile import cross_script_check
    f = _fields(full_name="ABEBE KEBEDE TESFAYE", full_name_local="አበበ ክዎደድ ተስፋየ")
    f["full_name_local"].alternatives = ["አበበ ከበዷ ተስሩፍ"]          # OCR: word 2 is one letter off
    cross_script_check(f, Context(country="ETH"))
    assert f["full_name_local"].value == "አበበ ከበደ ተስፋየ"


def test_foreign_document_surname_label_not_overruled_by_holder_nationality():
    f = _fields(full_name="GIRMA SAMUEL TADESSE", last_name="GIRMA", first_name="SAMUEL TADESSE", nationality="ETHIOPIAN")
    settle_name_parts(f, Context(country="BEL"))
    assert (f["first_name"].value, f["middle_name"].value, f["last_name"].value) == ("SAMUEL", "TADESSE", "GIRMA")


@pytest.mark.parametrize("local, country, kept", [
    ("ገብረስላሴ እንዳለ", "USA", False), ("ДЕРСО ХЕЛИНА", "USA", False), ("হাব্তামু কিদিস্‌", "CAN", False),
    ("መሳይ ከበደ ለማ", "ETH", True), ("خالد عمر", "ARE", True), ("王芳", "CHN", True), ("Иван Иванов", "RUS", True),
    ("அருண் குமார்", "IND", True), ("ስርአክ ደጀነ", "", True),
])
def test_native_script_name_only_where_the_issuer_prints_it(local, country, kept):
    from universal.reconcile import guard_local_script
    f = _fields(full_name_local=local)
    rejected = guard_local_script(f, Context(country=country))
    assert (f["full_name_local"].value == local) is kept and rejected is not kept


def test_geez_text_that_doesnt_spell_the_name_is_dropped():
    from universal.reconcile import cross_script_check
    f = _fields(full_name="DAWIT ALEMU TESHOME", full_name_local="አዲስ አበባ")        # the place of birth
    cross_script_check(f, Context(country="ETH"))
    assert f["full_name_local"].value == "" and "does_not_spell_latin_name" in f["full_name_local"].flags


def test_place_of_birth_never_part_of_labelled_given_names():
    from universal.printed import resolve_names
    f = _fields(place_of_birth="TORINO (ITA)")
    resolve_names(f, {"transcriber": {"surname_as_labelled": "ROSSI",
                                      "given_names_as_labelled": "MARCO LUCA TORINO (ITA)"}}, Context(country="ITA"))
    assert (f["first_name"].value, f["middle_name"].value, f["last_name"].value) == ("MARCO", "LUCA", "ROSSI")
