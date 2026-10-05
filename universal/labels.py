"""
Printed field labels in the world's languages — how the code knows what a
value IS, instead of trusting a model's guess.

An identity or work card says what each value means: "Date Issued",
"የተሰጠበት ቀን", "Guyyaa Kenname", "Date de délivrance", "签发日期". This module
maps such a label, in any language or script, to a field. The layout reader
(universal.layout) pairs every printed value with the label next to it, and
the pipeline then decides with it: a date printed beside an issue-date label
is the issue date, whatever a model called it.

  normalisation   case, accents (Latin/Greek/Cyrillic), punctuation and the
                  Ethiopic homophone rows (ሠ/ሰ, ፀ/ጸ, ሐ/ኀ/ሀ, ዐ/አ) are folded, so
                  "Date d'Émission", "DATE D EMISSION" and "የተሠጠበት" all match
  specificity     the longest label wins ("Date of Birth" over "Date";
                  "የአባት ስም" over "ስም"); overlapping shorter matches are dropped
  OCR tolerance   in alphabetic scripts a word of 5+ letters may differ by one
                  character ("Expiy Date" → expiry_date)
  short labels    "DOB", "EXP", "Tel", "Nom" match only as whole words

Fields: the schema's own (date_of_birth, issue_date, expiry_date, sex,
nationality, ...) plus work-card fields (employee_number, position,
department, branch, organization-level labels) and `other_date` — dates
that must never be taken as birth/issue/expiry (date of employment,
registration, entry).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

# ─────────────────────────────────────────────────────────────
# The lexicon: field -> {language code: labels}
# ─────────────────────────────────────────────────────────────

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "pt": "Portuguese", "it": "Italian", "de": "German",
    "nl": "Dutch", "sv": "Swedish", "no": "Norwegian", "da": "Danish", "fi": "Finnish", "pl": "Polish",
    "cs": "Czech", "sk": "Slovak", "ro": "Romanian", "hu": "Hungarian", "sh": "Serbo-Croatian", "el": "Greek",
    "ru": "Russian", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic", "fa": "Persian", "ur": "Urdu",
    "he": "Hebrew", "hi": "Hindi", "bn": "Bengali", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay", "tl": "Filipino", "sw": "Swahili",
    "am": "Amharic", "ti": "Tigrinya", "om": "Oromo", "so": "Somali", "ha": "Hausa",
}

LEXICON: dict[str, dict[str, tuple[str, ...]]] = {
    "date_of_birth": {
        "en": ("date of birth", "birth date", "dob", "d o b", "born", "birthday", "date of birth dob"),
        "fr": ("date de naissance", "ne le", "nee le", "naissance"),
        "es": ("fecha de nacimiento", "fecha nacimiento", "f nacimiento", "fecha nac"),
        "pt": ("data de nascimento", "data nascimento", "data nasc"),
        "it": ("data di nascita", "nato il", "nata il"),
        "de": ("geburtsdatum", "geburtstag", "geboren am"),
        "nl": ("geboortedatum",),
        "sv": ("fodelsedatum", "fodelsedag"), "no": ("fodselsdato",), "da": ("fodselsdato",),
        "fi": ("syntymaaika",),
        "pl": ("data urodzenia",), "cs": ("datum narozeni",), "sk": ("datum narodenia",),
        "ro": ("data nasterii",), "hu": ("szuletesi ido", "szuletesi datum"),
        "sh": ("datum rodenja",),
        "el": ("ημερομηνια γεννησης",),
        "ru": ("дата рождения",), "uk": ("дата народження",),
        "tr": ("dogum tarihi",),
        "ar": ("تاريخ الميلاد", "تاريخ الولادة"), "fa": ("تاریخ تولد",), "ur": ("تاریخ پیدائش",),
        "he": ("תאריך לידה",),
        "hi": ("जन्म तिथि", "जन्मतिथि", "जन्म की तारीख"), "bn": ("জন্ম তারিখ",),
        "zh": ("出生日期", "出生"), "ja": ("生年月日",), "ko": ("생년월일",),
        "th": ("วันเกิด", "เกิดวันที่"),
        "vi": ("ngay sinh", "ngay thang nam sinh"),
        "id": ("tanggal lahir", "tgl lahir"), "ms": ("tarikh lahir",),
        "tl": ("petsa ng kapanganakan",),
        "sw": ("tarehe ya kuzaliwa",),
        # Ge'ez stems too: Tesseract often drops a label's first syllables ("ሰጠበት ቀ").
        "am": ("የልደት ቀን", "የትውልድ ቀን", "የተወለደበት ቀን", "ልደት ቀን", "ተወለደበት", "ልደት"),
        "ti": ("ዕለት ልደት", "ዕለተ ልደት", "ዝተወለደሉ ዕለት"),
        "om": ("guyyaa dhalootaa", "guyyaa dhaloota", "bara dhalootaa"),
        "so": ("taariikhda dhalashada",),
        "ha": ("ranar haihuwa",),
    },
    "issue_date": {
        "en": ("date of issue", "issue date", "date issued", "issued on", "issued", "doi", "iss", "issue dt",
               "date of iss", "iss date",
               "date of issuance", "issuance date", "valid from"),
        "fr": ("date de delivrance", "delivre le", "delivree le", "date d emission", "emis le", "fait le",
               "delivrance"),
        "es": ("fecha de expedicion", "fecha de emision", "fecha expedicion", "expedicion", "emision", "expedido"),
        "pt": ("data de emissao", "data de expedicao", "emissao", "emitido em"),
        "it": ("data di rilascio", "rilasciato il", "data di emissione", "rilascio"),
        "de": ("ausstellungsdatum", "ausgestellt am", "erteilt am", "ausstellung"),
        "nl": ("datum van afgifte", "afgiftedatum", "afgegeven op", "afgifte"),
        "sv": ("utfardandedatum", "utfardad", "utfardat"), "no": ("utstedelsesdato", "utstedt"),
        "da": ("udstedelsesdato", "udstedt"),
        "fi": ("myontamispaiva", "myonnetty"),
        "pl": ("data wydania",), "cs": ("datum vydani",), "sk": ("datum vydania",),
        "ro": ("data eliberarii", "eliberat"), "hu": ("kiallitas datuma", "kiallitas kelte", "kiallitva"),
        "sh": ("datum izdavanja", "izdano", "izdato"),
        "el": ("ημερομηνια εκδοσης",),
        "ru": ("дата выдачи",), "uk": ("дата видачі",),
        "tr": ("verilis tarihi", "duzenlenme tarihi"),
        "ar": ("تاريخ الإصدار", "تاريخ الاصدار", "تاريخ التحرير"), "fa": ("تاریخ صدور",),
        "ur": ("تاریخ اجراء", "تاریخ اجرا"),
        "he": ("תאריך הנפקה",),
        "hi": ("जारी करने की तिथि", "जारी करने की तारीख", "जारी"), "bn": ("প্রদানের তারিখ", "ইস্যুর তারিখ"),
        "zh": ("签发日期", "簽發日期", "发证日期", "發證日期"), "ja": ("交付年月日", "交付日", "発行日"),
        "ko": ("발급일자", "발급일", "발행일"),
        "th": ("วันออกบัตร", "วันที่ออก"),
        "vi": ("ngay cap", "cap ngay"),
        "id": ("tanggal dikeluarkan", "tanggal terbit", "dikeluarkan"), "ms": ("tarikh dikeluarkan",),
        "tl": ("petsa ng pagkakaloob", "petsa ng isyu"),
        "sw": ("tarehe ya kutolewa", "imetolewa"),
        "am": ("የተሰጠበት ቀን", "የተሰጠበት", "የወጣበት ቀን", "የተሰጠ", "ሰጠበት"),
        "ti": ("ዝተውሃበሉ ዕለት", "ዝተዋህበሉ ዕለት", "ዝተውሃበሉ"),
        "om": ("guyyaa itti kenname", "guyyaa kenname", "guyyaa kennamee", "itti kenname", "kan kenname"),
        "so": ("taariikhda la bixiyay", "la bixiyay"),
        "ha": ("ranar bayarwa",),
    },
    "expiry_date": {
        "en": ("expiry date", "date of expiry", "expiration date", "date of expiration", "expires on", "expires",
               "expiry dt", "exp dt", "exp date",
               "expiry", "exp", "valid until", "valid till", "valid thru", "valid through", "valid to"),
        "fr": ("date d expiration", "expire le", "valable jusqu au", "date de validite", "fin de validite",
               "expiration"),
        "es": ("fecha de caducidad", "fecha de vencimiento", "fecha de expiracion", "valido hasta", "caducidad",
               "vencimiento", "vence", "validez"),
        "pt": ("data de validade", "valido ate", "validade"),
        "it": ("data di scadenza", "valido fino al", "valida fino al", "scadenza"),
        "de": ("gultig bis", "ablaufdatum", "gultigkeit", "lauft ab"),
        "nl": ("geldig tot", "vervaldatum", "einde geldigheid"),
        "sv": ("giltig till", "giltig t o m", "sista giltighetsdag"), "no": ("gyldig til", "utlopsdato"),
        "da": ("gyldig til", "udlobsdato"),
        "fi": ("viimeinen voimassaolopaiva", "voimassa"),
        "pl": ("data waznosci", "wazny do", "termin waznosci"), "cs": ("platnost do", "plati do"),
        "sk": ("platny do", "platnost do"),
        "ro": ("data expirarii", "valabil pana la"), "hu": ("ervenyes", "lejarat"),
        "sh": ("vrijedi do", "vazi do", "datum isteka"),
        "el": ("ημερομηνια ληξης", "ισχυει εως", "ληξη"),
        "ru": ("действителен до", "срок действия", "дата окончания"), "uk": ("дійсний до", "термін дії"),
        "tr": ("son gecerlilik tarihi", "gecerlilik tarihi", "gecerlilik"),
        "ar": ("تاريخ الانتهاء", "تاريخ الإنتهاء", "صالح حتى", "صالحة حتى", "تاريخ الصلاحية", "ينتهي في"),
        "fa": ("تاریخ انقضا", "تاریخ اعتبار", "اعتبار تا"), "ur": ("تاریخ تنسیخ", "میعاد ختم"),
        "he": ("תאריך תפוגה", "בתוקף עד"),
        "hi": ("समाप्ति तिथि", "वैधता"), "bn": ("মেয়াদোত্তীর্ণের তারিখ", "মেয়াদ"),
        "zh": ("有效期限", "有效期至", "有效期", "截止日期"), "ja": ("有効期限",),
        "ko": ("만료일자", "만료일", "유효기간"),
        "th": ("วันบัตรหมดอายุ", "วันหมดอายุ", "หมดอายุ"),
        "vi": ("co gia tri den", "ngay het han", "gia tri den"),
        "id": ("berlaku hingga", "berlaku sampai", "masa berlaku"), "ms": ("tarikh luput", "sah sehingga"),
        "tl": ("balido hanggang",),
        "sw": ("tarehe ya kuisha", "tarehe ya mwisho"),
        "am": ("የሚያበቃበት ቀን", "የሚያበቃበት", "የሚያገለግልበት ጊዜ", "ያገለግላል እስከ", "የሚያልፍበት ቀን", "ማብቂያ",
               "ሚያበቃበት", "ያበቃበት"),
        "ti": ("ዘብቅዓሉ ዕለት", "ዘብቅዓሉ"),
        "om": ("guyyaa itti dhumu", "guyyaa dhumaa", "yeroon itti dhumu", "itti dhumu", "guyyaa xumuraa",
               "itti xumuramu", "itti xummuramu", "xumuramu", "xummuramu", "xumurama"),
        "so": ("taariikhda dhacitaanka",),
        "ha": ("ranar karewa",),
    },
    "other_date": {   # never a birth / issue / expiry date
        "en": ("date of employment", "employment date", "date employed", "date joined", "joining date",
               "date of joining", "hire date", "date of hire", "registration date", "date of registration",
               "date of entry", "entry date", "enrollment date", "date of enrollment"),
        "fr": ("date d embauche", "date d entree"), "es": ("fecha de ingreso", "fecha de contratacion"),
        "am": ("የተቀጠረበት ቀን", "የተቀጠሩበት ቀን", "የቅጥር ቀን", "የተመዘገበበት ቀን"),
        "om": ("guyyaa qacarame", "guyyaa itti qacarame", "guyyaa galmee"),
    },
    "sex": {
        "en": ("sex", "gender"), "fr": ("sexe",), "es": ("sexo",), "pt": ("sexo",), "it": ("sesso",),
        "de": ("geschlecht",), "nl": ("geslacht",), "sv": ("kon",), "no": ("kjonn",), "da": ("kon",),
        "fi": ("sukupuoli",), "pl": ("plec",), "cs": ("pohlavi",), "sk": ("pohlavie",), "hu": ("neme",),
        "sh": ("spol", "pol"), "el": ("φυλο",), "ru": ("пол",), "uk": ("стать",), "tr": ("cinsiyet", "cinsiyeti"),
        "ar": ("الجنس", "النوع"), "fa": ("جنسیت",), "he": ("מין",), "hi": ("लिंग",), "zh": ("性别", "性別"),
        "ja": ("性別",), "ko": ("성별",), "th": ("เพศ",), "vi": ("gioi tinh",), "id": ("jenis kelamin",),
        "ms": ("jantina",), "sw": ("jinsia",), "am": ("ፆታ", "ጾታ"), "ti": ("ጾታ",),
        "om": ("saalaa", "saala", "koorniyaa"), "so": ("jinsiga",),
    },
    "nationality": {
        "en": ("nationality", "citizenship"), "fr": ("nationalite",), "es": ("nacionalidad",),
        "pt": ("nacionalidade",), "it": ("cittadinanza", "nazionalita"), "de": ("staatsangehorigkeit",),
        "nl": ("nationaliteit",), "sv": ("medborgarskap", "nationalitet"), "pl": ("obywatelstwo",),
        "cs": ("statni obcanstvi",), "ro": ("cetatenie",), "hu": ("allampolgarsag",),
        "ru": ("гражданство",), "uk": ("громадянство",), "tr": ("uyrugu", "uyruk"), "ar": ("الجنسية",),
        "fa": ("تابعیت", "ملیت"), "he": ("אזרחות",), "zh": ("国籍", "國籍"), "ja": ("国籍",), "ko": ("국적",),
        "th": ("สัญชาติ",), "vi": ("quoc tich",), "id": ("kewarganegaraan",), "ms": ("warganegara",),
        "sw": ("uraia",), "am": ("ዜግነት",), "ti": ("ዜግነት",), "om": ("lammummaa",), "so": ("jinsiyadda",),
    },
    "place_of_birth": {
        "en": ("place of birth", "birthplace", "pob"), "fr": ("lieu de naissance",), "es": ("lugar de nacimiento",),
        "pt": ("local de nascimento", "naturalidade"), "it": ("luogo di nascita",), "de": ("geburtsort",),
        "nl": ("geboorteplaats",), "sv": ("fodelseort",), "pl": ("miejsce urodzenia",), "cs": ("misto narozeni",),
        "ro": ("locul nasterii",), "ru": ("место рождения",), "uk": ("місце народження",), "tr": ("dogum yeri",),
        "ar": ("مكان الميلاد", "محل الميلاد"), "fa": ("محل تولد",), "he": ("מקום לידה",), "zh": ("出生地",),
        "ko": ("출생지",), "vi": ("noi sinh", "que quan"), "id": ("tempat lahir",),
        "am": ("የትውልድ ቦታ", "የልደት ቦታ"), "ti": ("ቦታ ልደት",), "om": ("bakka dhalootaa", "iddoo dhalootaa"),
        "so": ("goobta dhalashada",), "sw": ("mahali pa kuzaliwa",),
    },
    "surname": {
        "en": ("surname", "last name", "family name"), "fr": ("nom de famille", "nom"), "es": ("apellidos", "apellido"),
        "pt": ("sobrenome", "apelido"), "it": ("cognome",), "de": ("nachname", "familienname"),
        "nl": ("achternaam",), "sv": ("efternamn",), "no": ("etternavn",), "da": ("efternavn",),
        "fi": ("sukunimi",), "pl": ("nazwisko",), "cs": ("prijmeni",), "sk": ("priezvisko",), "ro": ("nume",),
        "hu": ("csaladi nev", "vezeteknev"), "sh": ("prezime",), "el": ("επωνυμο",), "ru": ("фамилия",),
        "uk": ("прізвище",), "tr": ("soyadi",), "ar": ("اسم العائلة", "اللقب"), "fa": ("نام خانوادگی",),
        "he": ("שם משפחה",), "tl": ("apelyido",), "sw": ("jina la ukoo",), "so": ("magaca qoyska",),
    },
    "given_names": {
        "en": ("given names", "given name", "first name", "first names", "forenames", "forename"),
        "fr": ("prenoms", "prenom"), "es": ("nombres", "nombre"), "pt": ("prenome",), "it": ("nome",),
        "de": ("vornamen", "vorname"), "nl": ("voornamen",), "sv": ("fornamn",), "no": ("fornavn",),
        "da": ("fornavn",), "fi": ("etunimet",), "pl": ("imiona", "imie"), "cs": ("jmeno", "jmena"),
        "sk": ("meno",), "ro": ("prenume",), "hu": ("utonev", "keresztnev"), "sh": ("ime",), "el": ("ονομα",),
        "ru": ("имя",), "uk": ("імя", "ім я"), "tr": ("adi",), "ar": ("الاسم الأول",), "he": ("שם פרטי",),
        "tl": ("pangalan",), "am": ("ስም",), "ti": ("ስም",), "om": ("maqaa",),
    },
    "full_name": {
        "en": ("full name", "name", "names", "holder"), "fr": ("nom et prenoms",), "es": ("nombre completo",),
        "ar": ("الاسم الكامل", "الاسم"), "fa": ("نام و نام خانوادگی",), "zh": ("姓名",), "ja": ("氏名",),
        "ko": ("성명",), "vi": ("ho va ten", "ho ten"), "id": ("nama",), "ms": ("nama",), "hi": ("नाम",),
        "bn": ("নাম",), "th": ("ชื่อ",), "am": ("ሙሉ ስም",), "ti": ("ምሉእ ስም",), "om": ("maqaa guutuu",),
        "sw": ("jina kamili", "jina"), "so": ("magaca",),
    },
    "father_name": {
        "en": ("father s name", "father name", "fathers name", "son of", "s o", "d o"), "am": ("የአባት ስም",),
        "ti": ("ስም ኣቦ",), "om": ("maqaa abbaa",), "ar": ("اسم الأب",), "ur": ("ولدیت", "والد کا نام"),
        "hi": ("पिता का नाम",), "bn": ("পিতা",), "ru": ("отчество",), "uk": ("по батькові",),
        "so": ("magaca aabbaha",),
    },
    "grandfather_name": {
        "en": ("grandfather s name", "grandfather name"), "am": ("የአያት ስም",), "ti": ("ስም ኣባሓጎ",),
        "om": ("maqaa akaakayyuu",),
    },
    "mother_name": {
        "en": ("mother s name", "mother name", "mothers name"), "am": ("የእናት ስም",), "om": ("maqaa haadhaa",),
        "ar": ("اسم الأم",), "bn": ("মাতা",), "hi": ("माता का नाम",),
    },
    "document_number": {
        "en": ("document no", "document number", "card no", "card number", "id no", "id number", "licence no",
               "license no", "license number", "passport no", "passport number", "serial no"),
        "fr": ("numero du document", "numero de carte", "n du document", "n de carte"),
        "es": ("numero de documento", "no de documento"), "pt": ("numero do documento",),
        "it": ("numero documento", "n documento"), "de": ("ausweisnummer", "dokumentennummer", "kartennummer"),
        "nl": ("documentnummer",), "ru": ("номер документа",), "ar": ("رقم الهوية", "رقم البطاقة", "رقم الوثيقة"),
        "zh": ("公民身份号码", "证件号码", "證件號碼"), "am": ("የመታወቂያ ቁጥር", "መለያ ቁጥር", "የመንጃ ፈቃድ ቁ", "ቁጥር"),
        "om": ("lakk waraqaa eenyummaa", "lakkoofsa waraqaa eenyummaa", "lakkoofsa eenyummaa", "lakk waraqaa",
               "lakkoofsa", "lakk"),
    },
    "registration_number": {
        "en": ("reg no", "registration no", "registration number", "reg number"), "am": ("የምዝገባ ቁጥር", "የምዝገባ ቁ"),
        "om": ("lakkoofsa galmee",),
    },
    "emergency_contact_name": {
        "en": ("emergency contact name", "emergency contact", "in case of emergency", "next of kin"),
        "om": ("maqaa waamamaa yeroo rakkoo", "maqaa nama yeroo rakkoo"), "am": ("የአደጋ ጊዜ ተጠሪ ስም", "የአደጋ ጊዜ ተጠሪ"),
        "fr": ("personne a contacter",), "es": ("contacto de emergencia",),
    },
    "emergency_contact_address": {
        "en": ("emergency contact address",), "om": ("teessoo waamamaa yeroo rakkoo",),
        "am": ("የአደጋ ጊዜ ተጠሪ አድራሻ",),
    },
    "emergency_contact_phone": {
        "en": ("emergency contact phone", "emergency phone", "emergency contact no"),
        "om": ("lakk bilbilaa waamamaa yeroo rakkoo", "lakkoofsa bilbilaa waamamaa yeroo rakkoo"),
        "am": ("የአደጋ ጊዜ ተጠሪ ስልክ ቁጥር", "የአደጋ ጊዜ ተጠሪ ስልክ"),
    },
    "occupation": {
        "en": ("occupation", "profession"), "fr": ("profession",), "es": ("ocupacion", "profesion"),
        "om": ("hojii",), "am": ("ሥራ", "ስራ", "ሙያ"),
    },
    "employee_number": {
        "en": ("employee no", "employee id", "employee number", "staff no", "staff id", "staff number",
               "emp no", "emp id", "personnel no", "service no", "badge no", "id code"),
        "fr": ("matricule",), "es": ("numero de empleado", "legajo"), "pt": ("matricula",),
        "de": ("personalnummer",), "am": ("የሰራተኛ ቁጥር", "የሰራተኛ መለያ ቁጥር", "የሰራተኛ መለያ"),
        "om": ("lakk hojjetaa", "lakkoofsa hojjetaa"),
    },
    "position": {
        "en": ("job title", "position", "designation", "rank", "title", "role"),
        "fr": ("fonction", "poste", "profession"), "es": ("cargo", "puesto", "ocupacion"), "pt": ("cargo", "funcao"),
        "de": ("funktion", "beruf"), "ar": ("المسمى الوظيفي", "الوظيفة", "المهنة"),
        "am": ("የሥራ መደብ", "የስራ መደብ", "ማዕረግ"), "om": ("gahee hojii", "ramaddii hojii"),
    },
    "department": {
        "en": ("department", "dept", "division", "unit", "section", "directorate"),
        "fr": ("departement", "direction"), "es": ("departamento",), "de": ("abteilung",),
        "ar": ("القسم", "الإدارة"), "am": ("ክፍል", "መምሪያ", "ዘርፍ", "የስራ ክፍል"), "om": ("kutaa", "kutaa hojii"),
    },
    "branch": {
        "en": ("branch",), "fr": ("agence", "succursale"), "es": ("sucursal",), "am": ("ቅርንጫፍ",),
        "om": ("damee",),
    },
    "grade": {
        "en": ("grade", "class", "category", "level"), "am": ("ደረጃ",), "om": ("sadarkaa",),
    },
    "region": {
        "en": ("region", "state", "province"), "fr": ("region",), "am": ("ክልል",), "om": ("naannoo",),
    },
    "sub_city": {
        "en": ("subcity", "sub city", "zone"), "am": ("ክፍለ ከተማ", "ክ ከተማ", "ዞን"),
        "om": ("kutaa magaalaa", "godina", "magaalaa xiqqaa"),
    },
    "woreda": {
        "en": ("woreda", "wereda"), "am": ("ወረዳ",), "om": ("aanaa",),
    },
    "kebele": {
        "en": ("kebele",), "am": ("ቀበሌ",), "om": ("ganda",),
    },
    "house_number": {
        "en": ("house no", "house number", "houseno", "h no"), "am": ("የቤት ቁጥር", "የቤት ቁ"), "om": ("lakk manaa",),
    },
    "address": {
        "en": ("address", "residence"), "fr": ("adresse", "domicile"), "es": ("domicilio", "direccion"),
        "pt": ("endereco", "morada"), "it": ("indirizzo", "residenza"), "de": ("anschrift", "wohnort"),
        "ru": ("адрес", "место жительства"), "ar": ("العنوان",), "zh": ("住址", "地址"), "ja": ("住所",),
        "ko": ("주소",), "am": ("አድራሻ",), "om": ("teessoo", "iddoo jireenyaa"), "sw": ("anwani", "makazi"),
        "so": ("cinwaanka",),
    },
    "phone": {
        "en": ("telephone", "phone", "mobile", "tel"), "fr": ("telephone",), "ar": ("الهاتف", "جوال"),
        "am": ("ስልክ ቁጥር", "ስልክ"), "om": ("lakkoofsa bilbilaa", "lakk bilbilaa", "bilbilaa", "bilbila"),
    },
    "blood_type": {
        "en": ("blood type", "blood group"), "fr": ("groupe sanguin",), "es": ("grupo sanguineo", "tipo de sangre"),
        "ar": ("فصيلة الدم",), "am": ("የደም ዓይነት", "የደም አይነት", "ደም አይነት", "ደም ዓይነት"), "om": ("gosa dhiigaa",),
    },
    "height": {
        "en": ("height",), "fr": ("taille",), "es": ("estatura",), "de": ("grosse",), "am": ("ቁመት",),
        "om": ("dheerina",),
    },
    "issuing_authority": {
        "en": ("issued by", "issuing authority", "authority"), "fr": ("autorite", "delivre par"),
        "es": ("autoridad", "expedido por"), "de": ("ausstellende behorde", "behorde"),
        "am": ("የሰጠው ተቋም",), "om": ("qaama kenne",),
    },
    "issuer_name": {   # the issuing officer — a person, never the authority
        "en": ("issuer s name", "issuing officer"), "am": ("የሰጪ ስም", "የሰጪው ስም"), "om": ("maqaa kan kenne",),
    },
    "signature": {
        "en": ("holder s signature", "signature"), "fr": ("signature",), "es": ("firma",), "ar": ("التوقيع",),
        "am": ("ፊርማ",), "om": ("mallattoo",),
    },
}

# Document-type titles: what the card calls itself.
DOC_TYPE_TITLES: dict[str, dict[str, tuple[str, ...]]] = {
    "employee_id": {
        "en": ("employee id card", "employee identity card", "employee identification", "employee id",
               "staff id card", "staff identity card", "staff id", "work id", "service card", "company id",
               "corporate id", "employee card", "staff card",
               # the card's own statement: "The bearer of this ID card is an employee of our Bank."
               "is an employee of", "an employee of our", "is a staff member of", "is an employee"),
        "fr": ("carte professionnelle", "badge professionnel"), "de": ("dienstausweis", "mitarbeiterausweis"),
        "es": ("tarjeta de empleado", "carnet de empleado", "credencial de empleado"), "pt": ("cracha",),
        "it": ("tessera aziendale",), "zh": ("工作证", "工作證", "员工证", "員工證"), "ja": ("社員証", "職員証"),
        "ko": ("사원증",), "ar": ("بطاقة موظف", "بطاقة عمل", "بطاقة الموظف"),
        "am": ("የሰራተኛ መታወቂያ", "የሠራተኛ መታወቂያ", "የሰራተኞች መታወቂያ",
               "ባልደረባ መሆናቸውን", "ሰራተኛ መሆናቸውን", "የባንካችን ባልደረባ"),
        "om": ("waraqaa eenyummaa hojjetaa", "kaardii eenyummaa hojjetaa", "eenyummaa hojjetaa", "kaardii hojjetaa",
               "hojjetaa baankii", "hojjetaa dhaabbatichaa"),
    },
    "student_id": {
        "en": ("student id", "student card", "student identity card"), "fr": ("carte d etudiant",),
        "es": ("carnet de estudiante",), "de": ("studentenausweis", "schulerausweis"),
        "zh": ("学生证", "學生證"), "ja": ("学生証",), "ko": ("학생증",), "am": ("የተማሪ መታወቂያ",),
        "om": ("eenyummaa barataa",),
    },
    "voter_id": {
        "en": ("voter id", "voter card", "electoral card", "elector s photo identity card"),
        "fr": ("carte d electeur",), "am": ("የመራጭ ካርድ",),
    },
    "work_permit": {
        "en": ("work permit", "employment pass", "work visa"), "fr": ("permis de travail",),
        "de": ("arbeitserlaubnis",), "am": ("የሥራ ፈቃድ", "የስራ ፈቃድ"),
    },
    "health_insurance_card": {
        "en": ("health insurance card", "health card", "medicare"), "fr": ("carte vitale",),
        "de": ("krankenversicherungskarte", "gesundheitskarte"),
    },
    "tax_id": {
        "en": ("taxpayer identification", "tax id card", "permanent account number"),
    },
    "drivers_license": {
        "en": ("driving license", "driving licence", "driver license", "driver s license", "driver licence"),
        "fr": ("permis de conduire",), "de": ("fuhrerschein",), "es": ("licencia de conducir",),
        "am": ("የአሽከርካሪ ብቃት ማረጋገጫ ፈቃድ", "መንጃ ፈቃድ"), "om": ("hayyama konkolaachisummaa",),
    },
    "residence_permit": {
        "en": ("residence permit", "residence card", "permanent resident card", "resident permit"),
        "fr": ("titre de sejour", "carte de sejour"), "de": ("aufenthaltstitel", "aufenthaltserlaubnis"),
        "it": ("permesso di soggiorno",), "es": ("tarjeta de residencia",), "nl": ("verblijfsdocument",),
        "ar": ("إقامة", "تصريح إقامة"),
    },
    "passport": {
        "en": ("passport",), "fr": ("passeport",), "es": ("pasaporte",), "de": ("reisepass",),
        "am": ("ፓስፖርት",), "ar": ("جواز سفر", "جواز السفر"), "zh": ("护照", "護照"),
    },
    "national_id": {
        "en": ("national id", "national identity card", "identity card", "resident id", "residence id card"),
        "fr": ("carte nationale d identite", "carte d identite"), "de": ("personalausweis",),
        "es": ("documento nacional de identidad", "cedula de identidad"),
        "am": ("ብሔራዊ መታወቂያ", "የነዋሪነት መታወቂያ", "የነዋሪዎች መታወቂያ"),
        "om": ("waraqaa eenyummaa jiraattota", "waraqaa eenyummaa", "eenyummaa biyyoolessaa"),
        "zh": ("居民身份证",),
    },
}

# ─────────────────────────────────────────────────────────────
# Normalisation and matching
# ─────────────────────────────────────────────────────────────

# Ethiopic homophone rows folded to one (ሠ→ሰ, ፀ→ጸ, ሐ/ኀ→ሀ, ዐ→አ), same vowel order.
_ETHIOPIC_FOLD = {0x1220: 0x1230, 0x1340: 0x1338, 0x1210: 0x1200, 0x1280: 0x1200, 0x12D0: 0x12A0}
_FOLD_ACCENTS = re.compile(r"[̀-ͯ]")
_SEPARATORS = re.compile(r"[\s\.,:;/\\|()\[\]{}<>_\-–—'’`\"“”«»·•፡፣፤፥፦!?*#＃：，；（）]+")


def _fold_char(ch: str) -> str:
    cp = ord(ch)
    base = cp - (cp - 0x1200) % 8 if 0x1200 <= cp <= 0x137F else None
    if base in _ETHIOPIC_FOLD:
        return chr(_ETHIOPIC_FOLD[base] + (cp - base))
    return ch


def normalize(text: str) -> str:
    """Case-, accent- and punctuation-folded form used on both sides of a match."""
    text = unicodedata.normalize("NFKD", text or "")
    text = _FOLD_ACCENTS.sub("", text)                 # Latin/Greek/Cyrillic diacritics
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("ß", "ss").replace("ı", "i").replace("ø", "o").replace("æ", "ae").replace("đ", "d")
    text = "".join(_fold_char(c) for c in text)
    return " ".join(_SEPARATORS.sub(" ", text).split())


def _alphabetic(phrase: str) -> bool:
    """Scripts that separate words with spaces and case (Latin, Greek, Cyrillic)."""
    return all(c == " " or c.isascii() or 0x0370 <= ord(c) <= 0x052F for c in phrase)


@dataclass(frozen=True)
class LabelHit:
    field: str
    lang: str
    phrase: str          # normalised label
    start: int           # offsets in the normalised text
    end: int


@lru_cache(maxsize=1)
def _entries() -> list[tuple[str, str, str, "re.Pattern"]]:
    out = []
    seen = set()
    for field, by_lang in LEXICON.items():
        for lang, phrases in by_lang.items():
            for phrase in phrases:
                norm = normalize(phrase)
                if not norm or (field, lang, norm) in seen:
                    continue
                seen.add((field, lang, norm))
                if _alphabetic(norm):
                    pattern = re.compile(rf"(?<![^\W\d_]){re.escape(norm)}(?![^\W\d_])")
                else:
                    pattern = re.compile(re.escape(norm))
                out.append((field, lang, norm, pattern))
    out.sort(key=lambda e: -len(e[2]))
    return out


def _edit_leq1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _fuzzy_candidates(norm_text: str) -> list[LabelHit]:
    """Labels read with ONE OCR slip: a letter off in a long word ('expiy date',
    'nationallty'), or a single slip anywhere in an 8+-letter label ('cosa dhiigaa')."""
    words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", norm_text)]
    hits = []
    for field, lang, phrase, _ in _entries():
        if not _alphabetic(phrase):
            continue
        pw = phrase.split()
        if not any(len(w) >= 5 for w in pw):
            continue
        for i in range(len(words) - len(pw) + 1):
            window = words[i:i + len(pw)]
            pairs = list(zip((w for w, _, _ in window), pw))
            if all(w == p for w, p in pairs):
                continue                                   # an exact match, found by the exact pass
            per_word = all(w == p or (len(p) >= 5 and _edit_leq1(w, p)) for w, p in pairs)
            one_slip = (len(phrase) >= 8 and sum(w != p for w, p in pairs) == 1
                        and all(w == p or (len(p) >= 3 and _edit_leq1(w, p)) for w, p in pairs))
            if per_word or one_slip:
                hits.append(LabelHit(field, lang, phrase, window[0][1], window[-1][2]))
    return hits


def find_labels(text: str, fuzzy: bool = True) -> list[LabelHit]:
    """Every label in the text, no two overlapping; in text order. The LONGEST
    label wins, whether read exactly or with one OCR slip ('Lakkoofsa Bibilaa' is
    the phone label, not 'Lakkoofsa' + a stray word); on equal length, exact wins."""
    norm = normalize(text)
    if not norm:
        return []
    candidates: list[tuple[int, int, LabelHit]] = []
    for field, lang, phrase, pattern in _entries():
        if phrase not in norm:
            continue
        for m in pattern.finditer(norm):
            candidates.append((len(phrase), 1, LabelHit(field, lang, phrase, *m.span())))
    if fuzzy:
        candidates += [(len(h.phrase), 0, h) for h in _fuzzy_candidates(norm)]
    taken: list[tuple[int, int]] = []
    hits: list[LabelHit] = []
    # More WORDS wins (a one-slip 'lakkoofsa bilbilaa' over an exact 'lakkoofsa'); then exact over
    # fuzzy ('department' over a one-slip 'departement'); then length.
    for _, _, h in sorted(candidates, key=lambda c: (-len(c[2].phrase.split()), -c[1], -c[0], c[2].start)):
        if any(h.start < te and ts < h.end for ts, te in taken):
            continue
        taken.append((h.start, h.end))
        hits.append(h)
    return sorted(hits, key=lambda h: h.start)


def label_field(text: str, fields: Optional[Iterable[str]] = None) -> Optional[LabelHit]:
    """The field a piece of label text names (the last label in it, which is
    the one closest to a value that follows), optionally among `fields`."""
    wanted = set(fields) if fields else None
    hits = [h for h in find_labels(text) if wanted is None or h.field in wanted]
    return hits[-1] if hits else None


def is_label_only(text: str) -> bool:
    """True when the text is nothing but labels (so it is not a value)."""
    norm = normalize(text)
    if not norm:
        return True
    covered = sum(h.end - h.start for h in find_labels(text))
    return covered >= len(norm.replace(" ", "")) * 0.8


@lru_cache(maxsize=1)
def _title_entries() -> list[tuple[str, str, str, "re.Pattern"]]:
    out = []
    for doc_type, by_lang in DOC_TYPE_TITLES.items():
        for lang, phrases in by_lang.items():
            for phrase in phrases:
                norm = normalize(phrase)
                pattern = (re.compile(rf"(?<![^\W\d_]){re.escape(norm)}(?![^\W\d_])") if _alphabetic(norm)
                           else re.compile(re.escape(norm)))
                out.append((doc_type, lang, norm, pattern))
    out.sort(key=lambda e: -len(e[2]))
    return out


def looks_like_title(text: str) -> bool:
    """A card's title, exactly or with OCR / transcription slips ('Waraqya Enyumu
    Jirattat' for 'Waraqaa Eenyummaa Jiraattota'): most words close to a title's."""
    from difflib import SequenceMatcher
    words = normalize(text).split()
    if not words:
        return False
    if document_type_titles(text):
        return True
    for _, _, phrase, _ in _title_entries():
        pw = phrase.split()
        if len(pw) >= 2 and len(pw) == len(words):
            ratios = [SequenceMatcher(None, w, p).ratio() for w, p in zip(words, pw)]
            if min(ratios) >= 0.6 and sum(ratios) / len(ratios) >= 0.72:
                return True
    return False


def document_type_titles(text: str) -> list[tuple[str, str, str]]:
    """[(document type, language, title)] printed in the text, longest first."""
    norm = normalize(text)
    taken: list[tuple[int, int]] = []
    out = []
    for doc_type, lang, phrase, pattern in _title_entries():
        for m in pattern.finditer(norm):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            out.append((doc_type, lang, phrase))
    return out
