"""
Label-anchored reading of the card's layout: every printed value paired with
the label beside it, by position — so the code, not a model, decides that a
date is the issue date.

A card prints each value to the RIGHT of its label or directly BELOW it,
often with the same label in two or three languages ("የተሰጠበት ቀን /
Guyyaa Kenname / Date Issued"). OCR text order does not follow that — a
value can come out before its label — so pairing uses the boxes:

  1. same box       "Issue Date: 08/04/2025", "DOB2/6/1995"
  2. same row       the nearest label box to the LEFT whose vertical extent
                    overlaps the value's
  3. directly above the nearest label box above whose horizontal extent
                    overlaps the value's, within two line heights

Boxes come from RapidOCR (Latin / CJK, with positions — the same engine the
grounding uses) and, for documents whose labels are in another script
(Ge'ez, Arabic, ...), Tesseract word boxes grouped into lines. Each page
(front / back) is paired separately.

Dates are found as written (separators, month names, CJK 年月日, compact
06MAR1994); a separator-less run like "842025" is not a date here — the
transcriber and agents still read those.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from . import labels

logger = logging.getLogger(__name__)

DATE_FIELDS = ("date_of_birth", "issue_date", "expiry_date", "other_date")
TEXT_FIELDS = ("full_name", "occupation", "emergency_contact_name", "emergency_contact_address",
               "emergency_contact_phone", "region", "sub_city", "woreda", "kebele", "house_number",
               "registration_number", "employee_number", "position", "department", "branch", "grade", "phone", "blood_type",
               "father_name", "grandfather_name", "mother_name", "place_of_birth", "nationality", "sex",
               "issuing_authority", "document_number", "height")

_MONTHS = (r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|okt|nov|dec|ene|fev|fév|avr|mai|juin|juil|aou|aoû|"
           r"déc|dez|mär|mrz|gen|mag|giu|lug|ago|set|ott|dic|janv|févr|sept")
DATE_RE = re.compile(
    r"(?<!\d)(?:"
    r"\d{1,2}\s?[./\-]\s?\d{1,2}\s?[./\-]\s?\d{2,4}"                 # 08/04/2025, 8.4.25, 29 / 5 / 1987
    r"|\d{4}[./\-]\d{1,2}[./\-]\d{1,2}"                             # 2025-04-08
    r"|\d{1,2}\s?(?:" + _MONTHS + r")[a-zé]*\.?(?:\s?/\s?[a-z]{3,4}\.?)?\s?\d{2,4}"   # 29 OKT/OCT 1972, 06MAR1994
    r"|\d{4}\s?年\s?\d{1,2}\s?月\s?\d{1,2}\s?日"                     # 2020年1月1日
    r"|\d{1,2}\s\d{1,2}\s\d{4}"                                     # 29 5 1987 (spaced)
    r")(?!\d)"
    # A printed calendar marker belongs to the date: "25/10/2016 E.C" is Ethiopian.
    r"(?:\s?(?:E\.?\s?C\.?(?![A-Za-z])|ዓ\.?\s?ም\.?|A\.?H\.?(?![A-Za-z])|هـ))?", re.I)


@dataclass
class Line:
    text: str
    left: float
    top: float
    right: float
    bottom: float
    page: int = 0
    source: str = "rapidocr"

    @property
    def height(self) -> float:
        return max(1.0, self.bottom - self.top)

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class Anchored:
    field: str
    value: str
    label: str
    lang: str
    how: str                 # same_box | same_row | above
    page: int
    source: str

    def as_dict(self) -> dict:
        return {"field": self.field, "value": self.value, "label": self.label,
                "language": labels.LANGUAGE_NAMES.get(self.lang, self.lang), "how": self.how,
                "page": self.page, "source": self.source}


@dataclass
class Layout:
    lines: list[Line] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


# ── OCR with boxes ──────────────────────────────────────────────────────

def _load(path: str):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def rapid_lines(image_paths: list[str]) -> list[Line]:
    try:
        from idx.ocr_engine import _quad_to_ltwh, _run_rapidocr
    except Exception:
        return []
    out = []
    for page, path in enumerate(image_paths):
        try:
            result = _run_rapidocr(_load(path)) or []
        except Exception as e:
            logger.warning("layout OCR failed on %s: %s", path, e)
            continue
        for quad, text, _conf in result:
            if not (text or "").strip():
                continue
            l, t, w, h = _quad_to_ltwh(quad)
            out.append(Line(text.strip(), l, t, l + w, t + h, page, "rapidocr"))
    return out


def tesseract_lines(image_paths: list[str], langs: str) -> list[Line]:
    """Word boxes grouped into lines (block, paragraph, line)."""
    if not langs:
        return []
    try:
        import pytesseract
    except Exception:
        return []
    out = []
    for page, path in enumerate(image_paths):
        try:
            data = pytesseract.image_to_data(_load(path), lang=langs, output_type=pytesseract.Output.DICT,
                                             timeout=40)
        except Exception as e:
            logger.warning("layout tesseract failed on %s: %s", path, e)
            continue
        groups: dict[tuple, list[int]] = {}
        for i, word in enumerate(data["text"]):
            if word.strip() and float(data["conf"][i]) >= 0:
                groups.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(i)
        for idx in groups.values():
            words = [data["text"][i] for i in idx]
            l = min(data["left"][i] for i in idx)
            t = min(data["top"][i] for i in idx)
            r = max(data["left"][i] + data["width"][i] for i in idx)
            b = max(data["top"][i] + data["height"][i] for i in idx)
            out.append(Line(" ".join(words), l, t, r, b, page, "tesseract"))
    return out


def read_layout(image_paths: list[str], extra_langs: str = "") -> Layout:
    """RapidOCR lines, plus Tesseract lines in `extra_langs` (e.g. 'amh')."""
    lines = rapid_lines(image_paths)
    if extra_langs:
        lines += tesseract_lines(image_paths, extra_langs)
    return Layout(lines)


# ── pairing ─────────────────────────────────────────────────────────────

def _vertical_overlap(a: Line, b: Line) -> float:
    return max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top)) / min(a.height, b.height)


def _horizontal_overlap(a: Line, b: Line) -> bool:
    return min(a.right, b.right) > max(a.left, b.left)


def _unglued(text: str) -> str:
    """OCR glues words ('IssueDt', 'DOBPh+07'); labels are found in the unglued text
    (dates are read from the original, which ungluing would break apart)."""
    try:
        from idx.ocr_engine import unglue_ocr_text
        return unglue_ocr_text(text)
    except Exception:
        return text


def _month_name_date(text: str) -> Optional[str]:
    """A date written with a month name, possibly in two calendars
    ('+07,1987/FEB14.1995', 'መጋቢት 10, 2018/Mar 192026') — as printed."""
    from .printed import interpret_dual_date
    candidate = text.strip(" :：-+")
    return candidate if candidate and interpret_dual_date(candidate, "") else None


def _label_lines(lines: list[Line]) -> list[tuple[Line, labels.LabelHit]]:
    out = []
    for ln in lines:
        hit = labels.label_field(_unglued(ln.text))
        if hit is not None:
            out.append((ln, hit))
    return out


def _nearest_label(value: Line, candidates: list[tuple[Line, labels.LabelHit]],
                   wanted: tuple[str, ...]) -> Optional[tuple[labels.LabelHit, str, Line]]:
    best = None
    for ln, hit in candidates:
        # Both engines read the same full-resolution image, so an Amharic label from
        # Tesseract can anchor a date RapidOCR read (digits are RapidOCR's strength).
        if ln is value or ln.page != value.page or hit.field not in wanted:
            continue
        # Rank: a label on the value's own row always beats one above it (on a
        # two-column card the row gap is wider than the line gap); then distance.
        if ln.right <= value.left + value.height and _vertical_overlap(ln, value) >= 0.5:
            rank = (1, (value.left - ln.right) / value.height)
            how = "same_row"
        elif ln.right <= value.left + value.height and abs(ln.center_y - value.center_y) <= 1.6 * value.height:
            # A label STACK beside the value: "Guyyaa itti kenname / የተሰጠበት ቀን /
            # Date Issued" on three lines, the date level with the middle one.
            rank = (1.5, abs(ln.center_y - value.center_y) / value.height)
            how = "left_stack"
        elif ln.bottom <= value.top + value.height * 0.3 and _horizontal_overlap(ln, value) \
                and value.top - ln.bottom <= 2.0 * value.height:
            rank = (2, (value.top - ln.bottom) / value.height)
            how = "above"
        else:
            continue
        if best is None or rank < best[0]:
            best = (rank, hit, how, ln)
    return (best[1], best[2], best[3]) if best else None


_HOW_RANK = {"same_box": 0, "same_row": 1, "left_stack": 1.5, "above": 2, "below": 2, "transcribed_label": 3}


def _best_per_value(found: list[tuple[tuple, "Anchored"]]) -> list["Anchored"]:
    """One label per printed value (per page): the best-ranked across engines."""
    best: dict[tuple, tuple[tuple, Anchored]] = {}
    markers: dict[tuple, str] = {}
    for rank, a in found:
        key = (a.page, tuple(int(n) for n in re.findall(r"\d+", _MARKER.sub("", a.value))) or a.value)
        m = _MARKER.search(a.value)
        if m:
            markers.setdefault(key, m.group(1))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, a)
    out = []
    for key, (rank, a) in sorted(best.items(), key=lambda kv: kv[1][0]):
        # One engine may read the calendar marker the other lost ("E.C" as "ይ.ር").
        if key in markers and not _MARKER.search(a.value):
            a = Anchored(a.field, f"{a.value} {markers[key]}", a.label, a.lang, a.how, a.page, a.source)
        out.append(a)
    return out


def _stack_agrees(value: Line, label_lines: list, field: str) -> bool:
    """Every date label in the stack beside the value names the same field."""
    for ln, hit in label_lines:
        if hit.field in DATE_FIELDS and ln.page == value.page and ln.right <= value.left + value.height \
                and abs(ln.center_y - value.center_y) <= 1.6 * value.height and hit.field != field:
            return False
    return True


def anchored_dates(layout: Layout) -> list[Anchored]:
    lines = layout.lines
    label_lines = _label_lines(lines)
    found: list[tuple[tuple, Anchored]] = []
    for ln in lines:
        candidates = [(m.start(), _clean_date(m.group())) for m in DATE_RE.finditer(ln.text)]
        if not candidates:
            # A month-name / dual-calendar date: take the text after the line's label, if any.
            hit = labels.label_field(_unglued(ln.text), DATE_FIELDS)
            rest = ln.text
            if hit is not None:
                cut = _label_end_in_original(ln.text, hit)
                rest = ln.text[cut:] if cut is not None else ""
            value = _month_name_date(rest) if rest else None
            if value:
                candidates = [(len(ln.text) - len(rest), value)]
        for start, value in candidates:
            before = ln.text[:start]
            hit = labels.label_field(_unglued(before), DATE_FIELDS) if before.strip() else None
            if hit is not None:
                found.append(((0, 0.0), Anchored(hit.field, value, hit.phrase, hit.lang, "same_box", ln.page,
                                                 ln.source)))
                continue
            near = _nearest_label(ln, label_lines, DATE_FIELDS)
            if near and near[1] == "left_stack" and not _stack_agrees(ln, label_lines, near[0].field):
                near = None          # the stack beside it names two different dates: not decided here
            if near:
                hit, how, label_line = near
                gap = ((ln.left - label_line.right) if how == "same_row"
                       else abs(label_line.center_y - ln.center_y) if how == "left_stack"
                       else (ln.top - label_line.bottom))
                found.append(((_HOW_RANK[how], gap / ln.height),
                              Anchored(hit.field, value, hit.phrase, hit.lang, how, ln.page, ln.source)))
    return _best_per_value(found)


_BLOOD = re.compile(r"^\s*(AB|A|B|O|0)\s*[+\-−]?\s*(?:VE|IVE|POS(?:ITIVE)?|NEG(?:ATIVE)?)?\s*"
                    r"(?:RH\s*[+\-])?\s*$", re.I)


def _plausible(field: str, value: str) -> bool:
    """A value must look like its field: a blood group, a phone number, a
    number with digits — or it is not taken."""
    v = value.strip()
    if not v or len(v) > 80:
        return False
    digits = sum(c.isdigit() for c in v)
    if field == "blood_type":
        return bool(_BLOOD.match(v))
    if field in ("phone", "emergency_contact_phone"):
        return digits >= 7
    if field in ("full_name", "emergency_contact_name"):
        latin = v.split("|")[0]
        return len(re.findall(r"[^\W\d_]{2,}", latin)) >= 2 and digits == 0
    if field in ("employee_number", "document_number", "registration_number"):
        return digits >= 2 and len(v) <= 30
    if field == "height":
        return digits >= 2
    if field in ("woreda", "kebele", "house_number"):
        return 1 <= len(v) <= 30
    if field == "sex":
        from .normalize import normalize_sex
        return bool(normalize_sex(v)[0])
    return sum(c.isalpha() for c in v) >= 2 and digits <= len(v) // 2


def _normalized_with_map(text: str) -> tuple[str, list[int]]:
    """labels.normalize(text), plus the original index of every normalized char."""
    out, index = [], []
    for i, ch in enumerate(text):
        n = labels.normalize(ch)
        if not n:
            if out and out[-1] != " ":
                out.append(" ")
                index.append(i)
            continue
        for c in n:
            out.append(c)
            index.append(i)
    return "".join(out).strip(), index


def _segments(line: Line, fields: tuple[str, ...]) -> list[tuple[labels.LabelHit, str]]:
    """'Region: Addis Ababa. SubCity: Nefas Silk-Lafto. Woreda: 06' -> one value
    per label: the text between each label and the next (original spelling)."""
    text = _unglued(line.text)
    norm, index = _normalized_with_map(text)
    hits = [h for h in labels.find_labels(text) if h.field in fields or h.field in DATE_FIELDS]
    if len(hits) < 2:
        return []
    out = []
    for h, nxt in zip(hits, hits[1:] + [None]):
        start, end = h.end, (nxt.start if nxt else len(norm))
        if h.field not in fields or start >= end or start >= len(index):
            continue
        seg = text[index[start]:(index[end - 1] + 1 if end - 1 < len(index) else len(text))]
        value = re.sub(r"(?<=\w) - (?=\w)", "-", seg.strip(" :：.,-/|"))      # 'Silk - Lafto' -> 'Silk-Lafto'
        if h.field in ("woreda", "kebele") and re.match(r"\d", value):
            value = re.match(r"\d+", value).group()          # a level number, not the next (garbled) label
        if value and _plausible(h.field, value):
            out.append((h, value))
    return out


def anchored_values(layout: Layout, fields: tuple[str, ...] = TEXT_FIELDS) -> list[Anchored]:
    """Text values beside a label: the rest of the label's own box, else the
    nearest box to its right on the same row, else the box just below. A line
    carrying several labels gives one value per label."""
    lines = layout.lines
    out: list[Anchored] = []
    multi = set()
    for ln in lines:
        for hit, value in _segments(ln, fields):
            out.append(Anchored(hit.field, value, hit.phrase, hit.lang, "same_box", ln.page, ln.source))
            multi.add(id(ln))
    for ln, hit in _label_lines(lines):
        if id(ln) in multi:
            continue
        if hit.field not in fields:
            continue
        norm = labels.normalize(_unglued(ln.text))      # the text the label was found in
        rest = norm[hit.end:].strip()
        # 'Kutaa Magaalaa/ክፍለ ከተማ', 'Aanaa|ወረዳ': after '/' or '|' comes the label in its other
        # language, not a value — the value is below / to the right.
        unglued = _unglued(ln.text)
        _, index = _normalized_with_map(unglued)
        cut = index[hit.end - 1] + 1 if 0 < hit.end <= len(index) else None
        bilingual_label = cut is not None and re.match(r"\s*[/|]", unglued[cut:] or "")
        if rest and not bilingual_label and not labels.is_label_only(rest) and not DATE_RE.search(rest):
            # Take the original characters after the label (case and punctuation kept).
            value = _tail_after_label(ln.text, rest)
            if _plausible(hit.field, value):
                out.append(Anchored(hit.field, value, hit.phrase, hit.lang, "same_box", ln.page, ln.source))
                continue
        candidates: list = []
        for other in lines:
            if other is ln or other.page != ln.page:
                continue
            if labels.is_label_only(other.text) or DATE_RE.search(other.text):
                continue
            first = labels.find_labels(_unglued(other.text))
            if first and first[0].start == 0:
                continue                    # the next label line ('Guyyaa Dhalootaa|...'), not a value
            width_ok = ((other.right - other.left) <= 3 * (ln.right - ln.left)
                        or hit.field in ("full_name", "emergency_contact_name"))     # names run long
            left_aligned = abs(other.left - ln.left) <= 2 * ln.height
            if other.left >= ln.right - ln.height and _vertical_overlap(ln, other) >= 0.5:
                gap = (other.left - ln.right) / ln.height
                # Far along the row (a signature block across the card) ranks after a value below.
                dist, how = ((1, gap) if gap <= 8 else (3, gap)), "same_row"
            elif other.top >= ln.bottom - ln.height * 0.3 and _horizontal_overlap(ln, other) \
                    and other.top - ln.bottom <= 1.5 * ln.height and left_aligned and width_ok:
                # below: a value column under its label, not the next full-width line
                dist, how = (2, (other.top - ln.bottom) / ln.height), "below"
            elif other.center_y > ln.center_y + 0.4 * ln.height and other.top >= ln.top + 0.3 * ln.height \
                    and other.center_y - ln.center_y <= 2.0 * ln.height and left_aligned and width_ok:
                # A tilted photo: the boxes overlap, but the value sits lower, left-aligned.
                dist, how = (2, (other.center_y - ln.center_y) / ln.height), "below"
            else:
                continue
            if not _plausible(hit.field, other.text.strip(" :：-")):
                continue
            candidates.append((dist, other, how))
        if hit.field in ("full_name", "emergency_contact_name") and candidates:
            # A name label keeps its best Latin AND its best native-script reading:
            # a Ge'ez fragment on the label's row must not hide the Latin name below it.
            latin = [c for c in candidates if re.search(r"[A-Za-z]{2}", c[1].text)]
            native = [c for c in candidates if not re.search(r"[A-Za-z]{2}", c[1].text)]
            for group in (latin, native):
                if group:
                    d, other, how = min(group, key=lambda c: c[0])
                    out.append(Anchored(hit.field, _clean_value(hit.field, other.text), hit.phrase, hit.lang, how,
                                        ln.page, ln.source))
            continue
        best = min(candidates, key=lambda c: c[0]) if candidates else None
        if best:
            out.append(Anchored(hit.field, _clean_value(hit.field, best[1].text), hit.phrase, hit.lang, best[2],
                                ln.page, ln.source))
    return out


_MARKER = re.compile(r"\s*(E\.?\s?C\.?|ዓ\.?\s?ም\.?|A\.?H\.?|هـ)\s*$", re.I)


def _label_end_in_original(original: str, hit: labels.LabelHit) -> Optional[int]:
    """Where a label found in the unglued text ends in the ORIGINAL text."""
    words = hit.phrase.split()
    squashed = "".join(words)
    norm_chars = [(i, labels.normalize(c)) for i, c in enumerate(original)]
    acc, start = "", None
    for i, n in norm_chars:
        n = n.replace(" ", "")
        if not n:
            continue
        acc += n
        if squashed in acc:
            return i + 1
    return None


def _clean_value(field: str, raw: str) -> str:
    """'Abba Gada| #' -> 'Abba Gada': on a bilingual value line keep the readable
    half(s); names keep both halves ('Latin | Ge'ez') for the two name fields."""
    value = raw.strip(" :：-")
    if "|" not in value or field in ("full_name", "emergency_contact_name"):
        return value
    halves = [h.strip() for h in value.split("|")]
    readable = [h for h in halves if sum(c.isalpha() for c in h) >= 2]
    return " | ".join(readable) if len(readable) > 1 else (readable[0] if readable else value)


def _clean_date(raw: str) -> str:
    """'29 / 5 / 1987' -> '29/5/1987' (the date parsers take one separator);
    a glued calendar marker is set apart: '24/10/2021E.C' -> '24/10/2021 E.C'."""
    text = raw.strip()
    m = _MARKER.search(text)
    marker = m.group(1) if m else ""
    core = text[:m.start()] if m else text
    core = re.sub(r"\s*([./\-])\s*", r"\1", core.strip())
    return f"{core} {marker}" if marker else core


def _tail_after_label(original: str, normalized_rest: str) -> str:
    """The original-text suffix whose normalised form is the rest after the label."""
    target = normalized_rest.replace(" ", "")
    for i in range(len(original)):
        if labels.normalize(original[i:]).replace(" ", "") == target:
            return original[i:].strip(" :：-")
    return normalized_rest


def date_rows(layout: Layout) -> dict[str, list[str]]:
    """Every date-like token on each date label's ROW (any column): the
    Ethiopian and the Gregorian halves of a dual date usually sit left and right
    of the label. Each token goes to the nearest date label vertically."""
    label_lines = [(ln, hit) for ln, hit in _label_lines(layout.lines) if hit.field in DATE_FIELDS[:3]]
    rows: dict[str, list[str]] = {}
    for ln in layout.lines:
        if not re.search(r"(?:19|20)\d\d", ln.text) or sum(c.isdigit() for c in ln.text) < 5:
            continue
        near = [(abs(lab.center_y - ln.center_y), hit.field) for lab, hit in label_lines
                if lab.page == ln.page and abs(lab.center_y - ln.center_y) <= 1.2 * max(lab.height, ln.height)]
        if near:
            rows.setdefault(min(near)[1], []).append(ln.text)
    return rows


def labelled_dates_by_field(anchored: list[Anchored]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a in anchored:
        if a.value not in out.setdefault(a.field, []):
            out[a.field].append(a.value)
    return out


def transcribed_labelled_dates(pieces: str) -> list[Anchored]:
    """The transcriber's 'LABEL: date | LABEL: date' list, mapped by the
    lexicon (the model copied the label; the code decides what it means)."""
    out = []
    for piece in re.split(r"\s*\|\s*", pieces or ""):
        if ":" not in piece and "：" not in piece:
            continue
        label, value = re.split(r"[:：]", piece, maxsplit=1)
        m = DATE_RE.search(value)
        hit = labels.label_field(label, DATE_FIELDS)
        if m and hit:
            out.append(Anchored(hit.field, _clean_date(m.group()), hit.phrase, hit.lang, "transcribed_label", 0,
                                "transcriber"))
    return out
