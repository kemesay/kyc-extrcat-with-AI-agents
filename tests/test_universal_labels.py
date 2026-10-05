"""Multilingual labels, label-anchored layout, dates decided by label, languages, document types."""

import pytest

from universal import labels, languages, layout, understanding
from universal.layout import Anchored, Layout, Line
from universal.printed import resolve_dates
from universal.prompts import EXTRACTION_FIELDS
from universal.reconcile import Context, FieldResult


def _fields(**v):
    f = {n: FieldResult() for n in EXTRACTION_FIELDS}
    for k, val in v.items():
        f[k] = FieldResult(value=val, confidence=0.7, source="x")
    return f


# ── the lexicon ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, field, lang", [
    ("Date Issued", "issue_date", "en"), ("Date of Issue", "issue_date", "en"), ("Issue Date", "issue_date", "en"),
    ("የተሰጠበት ቀን", "issue_date", "am"), ("Guyyaa Kenname", "issue_date", "om"),
    ("Date de délivrance", "issue_date", "fr"), ("Fecha de expedición", "issue_date", "es"),
    ("Ausstellungsdatum", "issue_date", "de"), ("签发日期", "issue_date", "zh"), ("تاريخ الإصدار", "issue_date", "ar"),
    ("Expiry Date", "expiry_date", "en"), ("Valid Until", "expiry_date", "en"), ("የሚያበቃበት ቀን", "expiry_date", "am"),
    ("Guyyaa Itti Dhumu", "expiry_date", "om"), ("Date d'expiration", "expiry_date", "fr"),
    ("Gültig bis", "expiry_date", "de"), ("Действителен до", "expiry_date", "ru"), ("有効期限", "expiry_date", "ja"),
    ("Date of Birth", "date_of_birth", "en"), ("DOB", "date_of_birth", "en"), ("የልደት ቀን", "date_of_birth", "am"),
    ("Guyyaa Dhalootaa", "date_of_birth", "om"), ("Date de naissance", "date_of_birth", "fr"),
    ("Doğum Tarihi", "date_of_birth", "tr"), ("出生日期", "date_of_birth", "zh"), ("생년월일", "date_of_birth", "ko"),
    ("Date of Employment", "other_date", "en"), ("የተቀጠረበት ቀን", "other_date", "am"),
    ("Maqaa", "given_names", "om"), ("ስም", "given_names", "am"), ("Maqaa Abbaa", "father_name", "om"),
    ("የአባት ስም", "father_name", "am"), ("Maqaa Akaakayyuu", "grandfather_name", "om"),
    ("Saala", "sex", "om"), ("ፆታ", "sex", "am"), ("Lammummaa", "nationality", "om"), ("ዜግነት", "nationality", "am"),
    ("Employee No.", "employee_number", "en"), ("Lakk. Hojjetaa", "employee_number", "om"),
    ("የሥራ መደብ", "position", "am"), ("Gahee Hojii", "position", "om"), ("Department", "department", "en"),
    ("Kutaa", "department", "om"), ("ቅርንጫፍ", "branch", "am"), ("Damee", "branch", "om"),
])
def test_label_in_any_language_names_its_field(text, field, lang):
    hit = labels.label_field(text)
    assert hit is not None and (hit.field, hit.lang) == (field, lang), hit


def test_ocr_slips_accents_and_homophones_still_match():
    assert labels.label_field("Expiy Date").field == "expiry_date"            # one letter lost
    assert labels.label_field("DATE D EMISSION").field == "issue_date"         # accents / apostrophe
    assert labels.label_field("የተሠጠበት ቀን").field == "issue_date"             # ሠ written for ሰ
    assert labels.label_field("DOB2/6/1995").field == "date_of_birth"         # value glued on


def test_longest_label_wins_and_short_labels_need_word_boundaries():
    assert [h.field for h in labels.find_labels("Maqaa Abbaa")] == ["father_name"]
    assert [h.field for h in labels.find_labels("Family Name")] == ["surname"]
    assert labels.label_field("Expression") is None                            # 'exp' inside a word
    assert [h.field for h in labels.find_labels("Date of Issue / Date of Expiry")] == ["issue_date", "expiry_date"]


def test_document_titles_in_several_languages():
    titles = labels.document_type_titles("EXAMPLE BANK  EMPLOYEE ID CARD  የሰራተኛ መታወቂያ  Kaardii Eenyummaa Hojjetaa")
    assert {t[0] for t in titles} == {"employee_id"} and {t[1] for t in titles} == {"en", "am", "om"}
    assert labels.document_type_titles("Ethiopian Driving License")[0][0] == "drivers_license"


# ── layout pairing ──────────────────────────────────────────────────────

def _work_card():
    """A trilingual work card: labels in a left column, values to the right."""
    L = lambda t, l, top, r, src="rapidocr": Line(t, l, top, r, top + 20, 0, src)
    return Layout([
        L("EXAMPLE COOPERATIVE BANK", 100, 10, 600),
        L("EMPLOYEE ID CARD", 200, 40, 500),
        L("ስም / Maqaa / Name", 20, 80, 260, "tesseract"), L("ABEBE KEBEDE TESFAYE", 300, 80, 600),
        L("Gahee Hojii / Position", 20, 110, 260), L("Senior Officer", 300, 110, 480),
        L("Employee No.", 20, 140, 200), L("EMP-0042", 300, 140, 400),
        L("15/03/2024", 300, 170, 420),                   # value printed BEFORE its label in OCR order
        L("Date Issued", 20, 170, 200),
        L("Expiry Date", 20, 200, 200), L("14/03/2027", 300, 200, 420),
        L("Blood Type", 20, 230, 160),
        L("Region: Example Region, Sub City: Example Woreda 04 House No NEW", 20, 255, 700),
    ])


def test_dates_are_paired_with_their_labels_by_position():
    got = {(a.field, a.value, a.how) for a in layout.anchored_dates(_work_card())}
    assert got == {("issue_date", "15/03/2024", "same_row"), ("expiry_date", "14/03/2027", "same_row")}


def test_amharic_label_from_tesseract_anchors_a_rapidocr_date():
    lay = Layout([Line("የተሰጠበት ቀን", 20, 100, 180, 120, 0, "tesseract"),
                  Line("06/07/2016", 200, 100, 320, 120, 0, "rapidocr"),
                  Line("Guyyaa Itti Dhumu", 20, 150, 200, 170, 0, "rapidocr"),
                  Line("05/07/2019", 20, 175, 140, 195, 0, "rapidocr")])          # value BELOW its label
    got = {(a.field, a.value, a.how, a.lang) for a in layout.anchored_dates(lay)}
    assert got == {("issue_date", "06/07/2016", "same_row", "am"), ("expiry_date", "05/07/2019", "above", "om")}


def test_values_are_anchored_and_validated():
    got = {a.field: a.value for a in layout.anchored_values(_work_card())}
    assert got["position"] == "Senior Officer" and got["employee_number"] == "EMP-0042"
    assert "blood_type" not in got           # the full-width address line below is not a blood group


def test_transcribed_label_list_is_mapped_by_the_lexicon():
    got = layout.transcribed_labelled_dates("Guyyaa Kenname: 15/03/2024 | የሚያበቃበት ቀን: 14 / 03 / 2027 | Tel: 0911")
    assert [(a.field, a.value, a.lang) for a in got] == [("issue_date", "15/03/2024", "om"),
                                                         ("expiry_date", "14/03/2027", "am")]


# ── dates: the label decides the field ──────────────────────────────────

def test_issue_date_filed_as_birth_date_is_moved_by_its_label():
    fields = _fields(date_of_birth="2024-03-15")                       # agents' guess
    outputs = {"transcriber": {"date_of_birth_as_printed": "15/03/2024",
                               "dates_with_labels": "Date Issued: 15/03/2024 | Expiry Date: 14/03/2027"}}
    ctx = Context(country="ETH")
    resolve_dates(fields, outputs, ctx)
    assert fields["issue_date"].value == "2024-03-15" and fields["expiry_date"].value == "2027-03-14"
    assert fields["date_of_birth"].value == "" and "labelled_as_issue_date" in fields["date_of_birth"].flags
    assert "printed_label" in fields["issue_date"].verified_by
    assert any("beside a issue date label" in w for w in ctx.warnings)


def test_layout_labels_correct_the_agents_without_any_transcription():
    fields = _fields(date_of_birth="2024-03-15", issue_date="2027-03-14")
    anchored = [Anchored("issue_date", "15/03/2024", "date issued", "en", "same_row", 0, "rapidocr"),
                Anchored("expiry_date", "14/03/2027", "expiry date", "en", "same_row", 0, "rapidocr")]
    resolve_dates(fields, {}, Context(country="ETH"), anchored=anchored)
    assert (fields["date_of_birth"].value, fields["issue_date"].value, fields["expiry_date"].value) == \
           ("", "2024-03-15", "2027-03-14")


def test_employment_date_is_never_taken_as_issue_date():
    fields = _fields(issue_date="2019-09-01")
    outputs = {"transcriber": {"dates_with_labels": "Date of Employment: 01/09/2019 | Date Issued: 10/01/2024"}}
    resolve_dates(fields, outputs, Context(country="ETH"))
    assert fields["issue_date"].value == "2024-01-10"


def test_labelled_date_with_no_conflict_is_confirmed_not_changed():
    fields = _fields(date_of_birth="1990-05-01")
    outputs = {"transcriber": {"date_of_birth_as_printed": "01/05/1990", "dates_with_labels": "DOB: 01/05/1990"}}
    resolve_dates(fields, outputs, Context(country="GBR"))
    assert fields["date_of_birth"].value == "1990-05-01" and "printed_label" in fields["date_of_birth"].verified_by


# ── languages ───────────────────────────────────────────────────────────

def test_trilingual_card_languages():
    text = ("ስም / Maqaa / Name\nየተሰጠበት ቀን / Guyyaa Kenname / Date Issued\n"
            "Baankii Hojii Gamtaa Oromiyaa\nየሚያበቃበት ቀን / Guyyaa Itti Dhumu / Expiry Date")
    detected = {d["language"] for d in languages.detect(text, "ETH")}
    assert {"Amharic", "Oromo", "English"} <= detected
    primary, secondary, langs = languages.decide(languages.detect(text, "ETH"), ["English", "Amharic"])
    assert primary in ("Amharic", "Oromo") and "English" in secondary
    assert all(l["evidence"] for l in langs if l["language"] in ("Amharic", "Oromo", "English"))


def test_agent_votes_decide_when_nothing_is_printed_evidence():
    primary, secondary, _ = languages.decide([], ["Afaan Oromoo", "Oromo", "English"])
    assert primary == "Oromo" and secondary == "English"


@pytest.mark.parametrize("name, want", [("Afaan Oromoo", "Oromo"), ("Oromiffa", "Oromo"), ("Amharic (አማርኛ)", "Amharic"),
                                        ("Farsi", "Persian"), ("english", "English"), ("none", "")])
def test_language_names_are_canonical(name, want):
    assert languages.canonical_language(name) == want


def test_script_only_evidence_uses_the_country():
    assert languages.detect("ሀና ታደሰ በቀለ", "ETH")[0]["language"] == "Amharic"
    assert languages.detect("ሀና ታደሰ በቀለ", "ERI")[0]["language"] == "Tigrinya"
    assert languages.detect("김민준", "")[0]["language"] == "Korean"


# ── document type ───────────────────────────────────────────────────────

def test_printed_title_and_work_fields_make_an_employee_id():
    fields = _fields(document_type="national_id")
    und = understanding.Understanding(layout=_work_card(), values=layout.anchored_values(_work_card()))
    ctx = Context(country="ETH")
    report = understanding.decide_document_type(fields, [_work_card().text()], und, ctx)
    assert fields["document_type"].value == "employee_id" and fields["document_type"].alternatives == ["national_id"]
    assert report["work_card_fields"] == ["employee_number", "position"]


def test_agents_type_with_printed_support_is_kept():
    fields = _fields(document_type="drivers_license")
    report = understanding.decide_document_type(fields, ["Ethiopian Driving License"], understanding.Understanding(),
                                                Context())
    assert fields["document_type"].value == "drivers_license" and "printed_title" in fields["document_type"].verified_by
    assert "decided" not in report


def test_employee_number_becomes_the_document_number_on_a_work_card():
    fields = _fields(document_type="employee_id", document_number="0042")
    und = understanding.Understanding(values=layout.anchored_values(_work_card()))
    extra = understanding.apply_values(fields, und, Context())
    assert fields["document_number"].value == "EMP-0042" and extra["position"] == "Senior Officer"


def test_birth_date_expected_only_on_state_identity_documents():
    assert understanding.birth_date_expected("passport") and understanding.birth_date_expected("national_id")
    assert not understanding.birth_date_expected("employee_id") and not understanding.birth_date_expected("student_id")


def _two_column_rows():
    """Labels on two lines (Amharic above Latin), value centred to the right —
    the next row's label is CLOSER below than the value is to the right."""
    L = lambda t, l, top, r, h=22: Line(t, l, top, r, top + h, 0, "rapidocr")
    return Layout([L("Gahee Hojii / Position", 400, 230, 610), L("Senior Customer Service Officer", 830, 214, 1290, 30),
                   L("ክፍል /", 400, 282, 460), L("Kutaa / Department", 400, 312, 590),
                   L("Credit Operations", 830, 296, 1080, 30),
                   L("Guyyaa Kenname / Date Issued", 400, 640, 700), L("15/03/2024", 830, 624, 980, 30),
                   L("Guyyaa Itti Dhumu / Expiry Date", 400, 722, 710), L("14/03/2027", 830, 706, 980, 30)])


def test_value_on_the_same_row_beats_the_line_below():
    got = {a.field: a.value for a in layout.anchored_values(_two_column_rows())}
    assert got["position"] == "Senior Customer Service Officer" and got["department"] == "Credit Operations"


def test_each_date_keeps_only_its_best_label():
    got = [(a.field, a.value) for a in layout.anchored_dates(_two_column_rows())]
    assert sorted(got) == [("expiry_date", "14/03/2027"), ("issue_date", "15/03/2024")]


def test_generic_title_never_overrides_the_agents_specific_type():
    fields = _fields(document_type="residence_permit")
    report = understanding.decide_document_type(fields, ["IDENTITY CARD  RESIDENCE PERMIT"],
                                                understanding.Understanding(), Context())
    assert fields["document_type"].value == "residence_permit" and "decided" not in report
    empty = _fields(document_type="other")
    understanding.decide_document_type(empty, ["NATIONAL IDENTITY CARD"], understanding.Understanding(), Context())
    assert empty["document_type"].value == "national_id"


def test_label_above_a_value_never_overrules_the_model():
    fields = _fields(date_of_birth="1990-05-01")
    outputs = {"transcriber": {"date_of_birth_as_printed": "01/05/1990"}}
    weak = [Anchored("issue_date", "01/05/1990", "date issued", "en", "above", 0, "rapidocr")]
    resolve_dates(fields, outputs, Context(country="GBR"), anchored=weak)
    assert fields["date_of_birth"].value == "1990-05-01" and fields["issue_date"].value == ""


def test_strong_sources_that_disagree_move_nothing():
    fields = _fields(date_of_birth="1990-05-01")
    outputs = {"transcriber": {"date_of_birth_as_printed": "01/05/1990", "dates_with_labels": "DOB: 01/05/1990"}}
    layout_says = [Anchored("issue_date", "01/05/1990", "date issued", "en", "same_row", 0, "rapidocr")]
    resolve_dates(fields, outputs, Context(country="GBR"), anchored=layout_says)
    assert fields["date_of_birth"].value == "1990-05-01"


# ── an Ethiopian bank staff card (stacked trilingual labels, E.C dates) ──

def _bank_card_back():
    L = lambda t, l, top, r, src="rapidocr": Line(t, l, top, r, top + 26, 1, src)
    return Layout([
        L("Guyyaa itti kenname", 40, 190, 240), L("25/10/2016E.C", 420, 221, 600),
        L("ናተሰጠበት ቀን 25/10/2016 ይ.ር", 40, 225, 380, "tesseract"), L("Date Issued", 40, 254, 160),
        L("Guyyaa Tajaajilaa itti Xummuramu", 40, 300, 370), L("24/10/2021E.C", 420, 331, 600),
        L("Expiry Date", 40, 364, 160),
        L("The bearer of this ID card is", 180, 576, 460), L("an employee of our Bank.", 190, 610, 440),
    ])


def test_stacked_trilingual_labels_and_ethiopian_calendar_marker():
    got = {a.field: a.value for a in layout.anchored_dates(_bank_card_back())}
    assert got == {"issue_date": "25/10/2016 E.C", "expiry_date": "24/10/2021 E.C"}   # marker kept across engines


def test_ec_dates_convert_and_the_naive_misreading_is_not_a_birth_date():
    fields = _fields(document_type="employee_id", date_of_birth="2016-10-25", issue_date="2016-10-25")
    ctx = Context(country="ETH")
    resolve_dates(fields, {}, ctx, anchored=layout.anchored_dates(_bank_card_back()))
    assert (fields["issue_date"].value, fields["expiry_date"].value) == ("2024-07-02", "2029-07-01")
    assert fields["date_of_birth"].value == ""
    assert "calendar_pair" in fields["issue_date"].verified_by


def test_the_cards_own_statement_makes_it_an_employee_id():
    fields = _fields(document_type="national_id")
    understanding.decide_document_type(fields, [_bank_card_back().text()], understanding.Understanding(), Context())
    assert fields["document_type"].value == "employee_id"


def test_oromo_expiry_label_spellings():
    for text in ("Guyyaa Tajaajillisaa itti Xummuramu", "Guyyaa itti Xumuramu", "Guyyaa Itti Dhumu"):
        assert labels.label_field(text).field == "expiry_date", text


def test_organization_from_the_card_header_not_the_website():
    lay = Layout([Line("BaankiiHojiiGamtaaOromiyaa", 250, 40, 600, 70, 0), Line("Bank of Example", 20, 100, 200, 130, 0),
                  Line("www.example-bank.test", 150, 950, 450, 980, 0), Line("an employee of our Bank.", 190, 610, 440, 636, 1)])
    assert understanding.organization_names(lay) == ["Baankii Hojii Gamtaa Oromiyaa", "Bank of Example"]


def test_issue_dates_original_calendar_never_becomes_a_birth_date():
    from universal.plausibility import check_dual_calendar
    fields = _fields(issue_date="2024-07-02", date_of_birth_orig="25/10/2016 E.C.")
    fields["date_of_birth"] = FieldResult(value="", flags=["labelled_as_issue_date"])
    check_dual_calendar(fields, Context(country="ETH"))
    assert fields["date_of_birth"].value == "" and fields["issue_date"].value == "2024-07-02"
    assert fields["date_of_birth_orig"].value == ""
    # also without the label flag: the conversion IS the issue date
    fields = _fields(issue_date="2024-07-02", date_of_birth_orig="25/10/2016 E.C.")
    check_dual_calendar(fields, Context(country="ETH"))
    assert fields["date_of_birth"].value == ""


# ── Addis Ababa resident ID: month-name dual dates, one "Full Name" label ──

@pytest.mark.parametrize("printed, iso, verified", [
    ("የካቲት 07, 1987/FEB 14, 1995", "1995-02-14", True),        # both halves complete, agree
    ("07,1987/FEB 14,1995", "1995-02-14", True),                  # Ge'ez month lost: exactly one month fits
    ("መጋቢት 10, 2018/Mar 192026", "2026-03-19", True),            # day and year printed glued
    ("10.2027/Mar = 192030", "2030-03-19", False),                # Ethiopian half misread: Gregorian half only
])
def test_month_name_dual_calendar_dates(printed, iso, verified):
    from universal.printed import interpret_dual_date
    got = interpret_dual_date(printed, "ETH")
    assert (got["iso"], got["verified"]) == (iso, verified)


def test_disagreeing_calendar_halves_go_to_review_not_a_guess():
    from universal.printed import interpret_dual_date
    fields = _fields()
    ctx = Context(country="ETH")
    resolve_dates(fields, {"transcriber": {"date_of_birth_as_printed": "ጥር 1, 2000/Mar 19 2026"}}, ctx)
    assert fields["date_of_birth"].value == "" and any("disagree" in r for r in ctx.review_reasons)
    assert interpret_dual_date("ar192026", "ETH") is None          # an unreadable month is never filled in


def test_gregorian_month_name_is_never_converted_as_ethiopian():
    from universal.plausibility import check_dual_calendar
    fields = _fields(date_of_birth="1995-02-14", date_of_birth_orig="07/FEB 1995")
    check_dual_calendar(fields, Context(country="ETH"))
    assert fields["date_of_birth"].value == "1995-02-14"


def test_patronymic_middle_name_from_the_printed_full_name():
    from universal.reconcile import settle_name_parts
    for last in ("NEGASH", "GOSA NEGASH"):
        fields = _fields(full_name="EDEN GOSA NEGASH", first_name="EDEN", last_name=last)
        settle_name_parts(fields, Context(country="ETH"))
        assert (fields["first_name"].value, fields["middle_name"].value, fields["last_name"].value) == \
               ("EDEN", "GOSA", "NEGASH")


def test_glued_numbers_are_split_by_the_printed_id_label():
    fields = _fields(document_type="national_id", document_number="ID/KI/W04/02848628 AA0003239441")
    und = understanding.Understanding(values=[
        Anchored("document_number", "ID/KI/W04/02848628", "id no", "en", "same_row", 0, "rapidocr"),
        Anchored("registration_number", "AA0003239441", "reg no", "en", "same_box", 0, "rapidocr")])
    extra = understanding.apply_values(fields, und, Context())
    assert fields["document_number"].value == "ID/KI/W04/02848628" and extra["registration_number"] == "AA0003239441"


def test_glued_ocr_labels_are_found():
    lay = Layout([Line("P+mn/IssueDt", 326, 919, 470, 945, 0), Line("P-A/DOBPh+07,1987/FEB14.1995", 329, 835, 750, 860, 0)])
    got = {a.field: a.value for a in layout.anchored_dates(lay)}
    assert got["date_of_birth"] == "Ph+07,1987/FEB14.1995"


# ── nothing the card doesn't print ──────────────────────────────────────

CARD_TEXT = ("Addis Ababa City Resident ID Card\nReg. No AA0003239441\nID No ID/KI/W04/02848628\n"
             "Full Name EDEN GOSA NEGASH\nDOB 07,1987/FEB 14, 1995\nBlood GroupAB+VE  Sex F\nIssue Dt\nExpiry Dt")


def test_unprinted_nationality_is_dropped_not_inferred():
    fields = _fields(nationality="ERITREAN")
    report = understanding.require_printed(fields, CARD_TEXT, understanding.Understanding(), Context(country="ETH"),
                                           ocr_substantial=True)
    assert fields["nationality"].value == "" and "not_printed" in fields["nationality"].flags
    assert report["nationality"].startswith("dropped")


@pytest.mark.parametrize("text", ["Nationality ETHIOPIAN", "Nationality ETH", "ዜግነት ኢትዮጵያዊ"])
def test_printed_nationality_is_kept(text):
    fields = _fields(nationality="ETHIOPIAN")
    understanding.require_printed(fields, text, understanding.Understanding(), Context(country="ETH"), True)
    assert fields["nationality"].value == "ETHIOPIAN"


def test_nothing_is_judged_when_ocr_read_too_little():
    fields = _fields(nationality="ERITREAN")
    understanding.require_printed(fields, "", understanding.Understanding(), Context(), ocr_substantial=False)
    assert fields["nationality"].value == "ERITREAN"


def test_invented_latin_name_is_replaced_by_the_printed_one():
    fields = _fields(full_name="EDEN GOSA NEGASH", full_name_latin="Eritrean woman's")
    fields["full_name"].verified_by = ["ocr"]
    understanding.require_printed(fields, CARD_TEXT, understanding.Understanding(), Context(country="ETH"), True)
    assert fields["full_name_latin"].value == "EDEN GOSA NEGASH"
    assert fields["full_name_latin"].alternatives[0] == "Eritrean woman's"


def test_printed_blood_group_is_restored_from_its_label():
    fields = _fields()
    fields["blood_type"] = FieldResult(value="", alternatives=["AB+VE"], flags=["dropped_possible_hallucination"])
    und = understanding.Understanding(values=[Anchored("blood_type", "AB +VE", "blood group", "en", "same_box", 0,
                                                        "rapidocr")])
    understanding.require_printed(fields, CARD_TEXT, und, Context(country="ETH"), True)
    assert fields["blood_type"].value and "printed_labels" in fields["blood_type"].verified_by


# ── Ethiopian driving licence: OCR-damaged dual dates, OCR-misread name ──

def test_calendar_pair_solver_on_ocr_damaged_dates():
    from universal.printed import solve_calendar_pair
    assert solve_calendar_pair(["1/21998", "12/31991"], []) == ("1998-11-21", "12/03/1991 EC")
    assert solve_calendar_pair(["7/292025", "22/12017", "729/2027"], []) == ("2025-07-29", "22/11/2017 EC")
    assert solve_calendar_pair(["n022/2019ExpiryDate"], []) is None          # nothing confirmed: no decision
    assert solve_calendar_pair(["12/31991"], ["1998-11-21"]) == ("1998-11-21", "12/03/1991 EC")   # agents confirm
    assert solve_calendar_pair(["12/31991"], ["1998-02-12"]) is None          # a wrong agent date confirms nothing


def test_solver_overrides_the_hybrid_transcription():
    fields = _fields(date_of_birth="1998-02-12", issue_date="2019-11-22")
    outputs = {"transcriber": {"issue_date_as_printed": "29/07/2017"}, "field_extractor": {"date_of_birth": "1998-02-12"}}
    ctx = Context(country="ETH")
    resolve_dates(fields, outputs, ctx, row_tokens={"date_of_birth": ["1/21998", "12/31991"],
                                                    "issue_date": ["7/292025", "22/12017"]})
    assert (fields["date_of_birth"].value, fields["issue_date"].value) == ("1998-11-21", "2025-07-29")
    assert fields["date_of_birth_orig"].value == "12/03/1991 EC"


def test_geez_spelling_overrules_an_ocr_misreading():
    fields = _fields(last_name="TEPERA", middle_name="MESELE", full_name_local="ያሬድ መሠለ ተፈራ")
    fields["last_name"].alternatives, fields["last_name"].flags = ["TEFERA"], ["ocr_spelling_corrected"]
    fields["middle_name"].alternatives, fields["middle_name"].flags = ["MEESELE"], ["ocr_spelling_corrected"]
    understanding.arbitrate_spelling_by_script(fields, ["ያሬድ መሠለ ተፈራ"], Context(country="ETH"))
    assert fields["last_name"].value == "TEFERA"          # ተፈራ spells TEFERA, not TEPERA
    assert fields["middle_name"].value == "MESELE"        # መሠለ supports the correction


def test_address_line_with_several_labels():
    for text, want in (("Region: Addis Ababa, SubCity: Nefas Silk-Lafto, Woreda: 06, HouseNo: NEW",
                        {"region": "Addis Ababa", "sub_city": "Nefas Silk-Lafto", "woreda": "06", "house_number": "NEW"}),
                       ("Region:AddisAbaba.SubCityNefas Silk-Lafto.Wored6HouNaM",
                        {"region": "Addis Ababa", "sub_city": "Nefas Silk-Lafto", "woreda": "6"})):
        got = {a.field: a.value for a in layout.anchored_values(Layout([Line(text, 92, 558, 700, 575, 0)]))}
        assert {k: got.get(k) for k in want} == want, got


def test_printed_region_beats_the_agents_sub_city():
    fields = _fields(region="Nefas Silk-Lafto", district="")
    und = understanding.Understanding(values=[
        Anchored("region", "Addis Ababa", "region", "en", "same_box", 0, "rapidocr"),
        Anchored("sub_city", "Nefas Silk-Lafto", "subcity", "en", "same_box", 0, "rapidocr"),
        Anchored("woreda", "06", "woreda", "en", "same_box", 0, "rapidocr")])
    understanding.apply_values(fields, und, Context(country="ETH"))
    assert (fields["region"].value, fields["district"].value, fields["sub_district"].value) == \
           ("Addis Ababa", "Nefas Silk-Lafto", "Woreda 06")


def test_misread_phone_number_is_not_a_personal_number():
    fields = _fields(personal_number="09331566565")                   # one extra digit misread
    understanding.require_printed(fields, "hAh/Te0933566565\nNationality Ethiopian", understanding.Understanding(),
                                  Context(country="ETH"), True)
    assert fields["personal_number"].value == "" and "is_phone_number" in fields["personal_number"].flags


def test_printed_nationality_beats_an_agents_guess():
    fields = _fields(nationality="DJIBOUTIAN")
    und = understanding.Understanding(values=[Anchored("nationality", "Ethiopian", "nationality", "en", "same_box", 0,
                                                        "rapidocr")])
    understanding.require_printed(fields, "Nationality Ethiopian", und, Context(country="ETH"), True)
    assert fields["nationality"].value == "ETHIOPIAN" and fields["nationality"].alternatives == ["DJIBOUTIAN"]


def test_ocr_near_reading_of_a_number_goes_to_the_reviewer_not_the_value():
    fields = _fields(document_number="051547")
    fields["document_number"].flags = ["not_found_in_ocr"]
    ctx = Context(country="ETH")
    understanding.require_printed(fields, "License No 951547\nDOB 11/21/1998", understanding.Understanding(), ctx, True)
    assert fields["document_number"].value == "051547" and fields["document_number"].alternatives[0] == "951547"
    assert any("OCR reads '951547'" in r for r in ctx.review_reasons)


# ── Shaggar City resident ID (Afaan Oromo + Amharic, EC dates, emergency contact) ──

SHAGGAR_FRONT = ["Waraqaa Eenyummaa Jiraattota", "Magaalaa Shaggar", "Maqaa Guutuu|",
                 "Kaasahuun Girmaa Wulatuu|hAu c mA", "Guyyaa Dhalootaa|P+@AE中）", "02/07/1988", "Saala|Pt", "Dhiira|",
                 "Itiyoophiyaa|A+.e", "Kutaa Magaaiaa/hsA htoy", "Sulultaama"]


def test_country_named_on_the_card_beats_the_agents_guess():
    fields = _fields(issuing_country="EGY")
    ctx = Context(country="EGY")
    report = understanding.decide_country(fields, ctx, "\n".join(SHAGGAR_FRONT))
    assert ctx.country == "ETH" and fields["issuing_country"].value == "ETH" and report["decided"] == "ETH"


def test_label_languages_give_the_country_when_none_is_named():
    fields = _fields(issuing_country="EGY")
    ctx = Context(country="EGY")
    understanding.decide_country(fields, ctx, "Maqaa Guutuu\nGuyyaa Dhalootaa 02/07/1988\nSaala Dhiira")
    assert ctx.country == "ETH"


def test_named_country_is_kept():
    fields = _fields(issuing_country="BEL")
    ctx = Context(country="BEL")
    understanding.decide_country(fields, ctx, "BELGIQUE BELGIE\nNationality: ETHIOPIAN")
    assert ctx.country == "BEL"                                    # the card names Belgium


def test_name_is_read_beside_its_label_and_a_title_is_never_a_name():
    lay = Layout([Line(t, 400, 50 + 30 * i, 900, 72 + 30 * i, 0) for i, t in enumerate(SHAGGAR_FRONT)])
    und = understanding.Understanding(layout=lay, values=layout.anchored_values(lay))
    fields = _fields(full_name="Waraqaa Eenyummaa Jiraattota", first_name="Waraqaa", last_name="Eenyummaa Jiraattota")
    understanding.apply_values(fields, und, Context(country="ETH"))
    assert fields["full_name"].value == "Kaasahuun Girmaa Wulatuu" and fields["first_name"].value == ""
    fields = _fields(full_name="Waraqaa Eenyummaa Jiraattota")
    ctx = Context(country="ETH")
    understanding.apply_values(fields, understanding.Understanding(), ctx)
    assert fields["full_name"].value == "" and "title_not_a_name" in fields["full_name"].flags


def test_printed_title_corrects_an_unsupported_agent_type():
    fields = _fields(document_type="drivers_license")
    understanding.decide_document_type(fields, ["Waraqaa Eenyummaa Jiraattota Magaalaa Shaggar"],
                                       understanding.Understanding(), Context())
    assert fields["document_type"].value == "national_id"


def test_two_ec_dates_in_one_field_are_both_placed_and_the_printed_issue_stands():
    fields = _fields(issue_date="2012-05-01", expiry_date="2012-06-01")
    outputs = {"transcriber": {"issue_date_as_printed": "30/01/2018 EC | 30/01/2021 EC"}}
    ctx = Context(country="ETH")
    resolve_dates(fields, outputs, ctx, ocr_text="30/01/2018EC\n30/01/2021EC")
    assert (fields["issue_date"].value, fields["expiry_date"].value) == ("2025-10-10", "2028-10-10")


def test_unmarked_birth_date_on_an_ec_card_is_ambiguous_not_guessed():
    fields = _fields(date_of_birth="1988-07-02")
    outputs = {"transcriber": {"date_of_birth_as_printed": "02/07/1988", "issue_date_as_printed": "30/01/2018 EC"}}
    ctx = Context(country="ETH")
    resolve_dates(fields, outputs, ctx, ocr_text="02/07/1988\n30/01/2018EC")
    assert fields["date_of_birth"].value == "" and "calendar_not_printed" in fields["date_of_birth"].flags
    assert {"1988-07-02", "1996-03-11"} <= set(fields["date_of_birth"].alternatives)
    assert any("no calendar mark" in r for r in ctx.review_reasons)


def test_unprinted_expiry_before_a_printed_issue_is_the_one_dropped():
    from universal.plausibility import check_plausibility
    fields = _fields(issue_date="2025-10-10", expiry_date="2012-06-01")
    fields["issue_date"].verified_by = ["printed_text"]
    fields["expiry_date"].flags = ["not_found_in_ocr"]
    check_plausibility(fields, Context(country="ETH"))
    assert fields["issue_date"].value == "2025-10-10" and fields["expiry_date"].value == ""


def test_emergency_contact_and_phone_are_not_the_holders():
    back = ["Lakkoofsa Bibilaa/ nah epc", "0925499938", "Maqaa Waamamaa Yeroo Rakkoo/", "ARAGGUU TASFAAYEE",
            "Teessoo Waamamaa Yeroo Rakkoo/", "SHAGGAR", "Lakk.bilbilaa Wamamaa Yeroo Rakkoo/", "0960632587",
            "Cosa Dhiigaa/", "O+"]
    lay = Layout([Line(t, 240, 350 + 34 * i, 600, 372 + 34 * i, 1) for i, t in enumerate(back)])
    got = {a.field: a.value for a in layout.anchored_values(lay)}
    assert got["phone"] == "0925499938" and got["emergency_contact_phone"] == "0960632587"
    assert got["emergency_contact_name"] == "ARAGGUU TASFAAYEE" and got["blood_type"] == "O+"
    assert "document_number" not in got


def test_a_misspelled_title_is_still_not_a_name():
    assert labels.looks_like_title("Waraqya Enyumu Jirattat") and labels.looks_like_title("EMPLOYEE ID CARD")
    assert not labels.looks_like_title("Kaasahuun Girmaa Wulatuu") and not labels.looks_like_title("EDEN GOSA NEGASH")


def test_the_other_language_half_of_a_label_is_not_a_value():
    lay = Layout([Line("Kutaa Magaaiaa/hsA htoy", 404, 628, 640, 650, 0), Line("Sulultaama", 404, 671, 520, 693, 0),
                  Line("Aanaa|@LR", 404, 717, 500, 739, 0), Line("Abba Gada| #", 406, 744, 540, 766, 0)])
    got = {a.field: a.value for a in layout.anchored_values(lay)}
    assert got["sub_city"] == "Sulultaama" and got["woreda"].startswith("Abba Gada")


def test_a_long_name_below_a_short_label_is_read():
    lay = Layout([Line("Maqaa Guutuu|", 402, 234, 560, 256, 0), Line("Kaasahuun Girmaa Wulatuu|hAu c mA", 400, 256, 930, 280, 0)])
    got = {a.field: a.value for a in layout.anchored_values(lay)}
    assert got["full_name"].startswith("Kaasahuun Girmaa Wulatuu")


def test_an_unprinted_original_calendar_date_is_not_used():
    from universal.plausibility import check_dual_calendar
    fields = _fields(date_of_birth="1988-07-02", date_of_birth_orig="01.05.1992")
    check_dual_calendar(fields, Context(country="ETH"), printed_text="Guyyaa Dhalootaa\n02/07/1988")
    assert fields["date_of_birth"].value == "1988-07-02" and "not_printed" in fields["date_of_birth_orig"].flags


def test_tilted_photo_overlapping_boxes_and_far_same_row_boxes():
    lay = Layout([Line("Maqaa Guutuu|", 402, 234, 657, 281, 0), Line("Kaasahuun Girmaa Wulatuu|hAu c mA", 400, 256, 930, 319, 0),
                  Line("Guyyaa Dhalootaa|P+@AE中）", 401, 307, 700, 340, 0),
                  Line("Aanaa|@LR", 404, 717, 500, 739, 0), Line("MuilattooAbouTaayiaa", 1040, 708, 1230, 730, 0),
                  Line("Abba Gada| #", 406, 744, 540, 766, 0)])
    got = {a.field: a.value for a in layout.anchored_values(lay)}
    assert got["full_name"].startswith("Kaasahuun Girmaa Wulatuu") and got["woreda"].startswith("Abba Gada")


def test_ambiguous_calendar_when_the_date_came_from_the_labelled_layout():
    fields = _fields(date_of_birth="1988-07-02")
    outputs = {"transcriber": {"issue_date_as_printed": "30/01/2018 EC"}}          # no DOB transcribed
    anch = [Anchored("date_of_birth", "02/07/1988", "guyyaa dhalootaa", "om", "above", 0, "rapidocr")]
    ctx = Context(country="ETH")
    resolve_dates(fields, outputs, ctx, ocr_text="02/07/1988\n30/01/2018EC", anchored=anch)
    assert fields["date_of_birth"].value == "" and "calendar_not_printed" in fields["date_of_birth"].flags


def test_a_plus_sign_alone_is_not_a_blood_group():
    from universal.reconcile import normalize_value
    assert normalize_value("blood_type", "+", Context())[0] == ""
    assert normalize_value("blood_type", "O+", Context())[0] == "O+"


def test_name_label_keeps_the_latin_name_below_and_the_geez_fragment_beside():
    lay = Layout([Line("Maqaa Guutuu|", 402, 234, 657, 281, 0),
                  Line("ግርማ ዉለቱ", 700, 240, 900, 270, 0, "tesseract"),
                  Line("Kaasahuun Girmaa Wulatuu|hAu c mA", 400, 256, 930, 319, 0)])
    names = [a.value for a in layout.anchored_values(lay) if a.field == "full_name"]
    assert any(n.startswith("Kaasahuun") for n in names) and "ግርማ ዉለቱ" in names


def test_calendar_mark_is_checked_on_the_birth_dates_own_reading():
    fields = _fields(date_of_birth="1988-07-02")
    outputs = {"transcriber": {"date_of_birth_as_printed": "02/07/1988 | 30/01/2018 EC | 30/01/2021 EC"}}
    ctx = Context(country="ETH")
    resolve_dates(fields, outputs, ctx, ocr_text="02/07/1988\n30/01/2018EC\n30/01/2021EC")
    assert fields["date_of_birth"].value == "" and "calendar_not_printed" in fields["date_of_birth"].flags
    assert (fields["issue_date"].value, fields["expiry_date"].value) == ("2025-10-10", "2028-10-10")
