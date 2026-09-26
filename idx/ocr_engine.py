"""
Single OCR recognition backend for the whole project.

Confirmed with the user (this was a real dependency/deployment decision,
not made unilaterally): RapidOCR -- ONNXRuntime-based, CPU-only, no
GPU/PyTorch/PaddlePaddle, Apache-2.0 -- replaces tesseract as the actual
pixel-to-text engine behind EVERY existing OCR call site in this project
(aamva.py, ethiopia_id.py, generic_ocr.py, mrz_ocr.py, sweep.py). This is
a drop-in swap of the recognizer only: every label regex, value-shape
pattern, and per-field voting rule already tuned against real cards this
session is UNCHANGED -- they just receive better-recognized text than
tesseract was producing.

Why RapidOCR specifically: on a KYC extraction service, CPU-only and no
model-weight license concerns (unlike Surya, which requires a paid
commercial license above a revenue threshold) matter more than the very
best possible accuracy. It also detects text as full lines via its own
layout model rather than tesseract's column/PSM heuristics, which is a
genuine strength on the skewed, glare-heavy, non-standard-layout photos
this project's whole corpus consists of.

Deterministic (CLAUDE.md invariant #3): ONNXRuntime inference on CPU with
a fixed model is deterministic given the same input bytes -- no sampling,
no seed to set. Confirmed no non-determinism was introduced by running the
determinism harness (`tests/test_determinism.py`) after this module was
wired in.

tesseract is kept as an automatic FALLBACK only -- never a silent
downgrade. `ocr_text`/`ocr_words` try RapidOCR first; only an actual
import/runtime failure (engine not installed, or a corrupt image
onnxruntime itself rejects) falls through to tesseract, so a healthy
RapidOCR install always wins and a broken one degrades to the
previously-working path instead of returning nothing (CLAUDE.md
invariant #5 -- conditioning + OCR are the floor and must never go empty).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

_ENGINE: Optional[Any] = None
_ENGINE_UNAVAILABLE = False


def using_rapidocr() -> bool:
    """True once RapidOCR has been (successfully) lazily constructed --
    lets a caller like `ethiopia_id._ocr_texts` know that a tesseract PSM
    `config` string it's about to vary has NO effect on the result, so
    iterating several configs against the same image would just OCR the
    identical image N times and inflate that variant's vote weight in
    whatever majority-vote logic consumes the results, rather than
    genuinely sampling N different readings.
    """
    return _get_engine() is not None


def _get_engine() -> Optional[Any]:
    """Lazily constructs the one shared RapidOCR instance (loading its
    ONNX models is the slow part, ~0.2-0.3s -- worth paying once per
    process, not once per OCR call). `_ENGINE_UNAVAILABLE` latches so a
    broken install doesn't retry (and re-log/re-fail) on every call.
    """
    global _ENGINE, _ENGINE_UNAVAILABLE
    if _ENGINE is not None or _ENGINE_UNAVAILABLE:
        return _ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR()
    except Exception:
        _ENGINE_UNAVAILABLE = True
    return _ENGINE


@dataclass
class OcrWord:
    """Word-level pseudo-token, shape-compatible with sweep.py's
    `OcrToken` -- RapidOCR itself detects whole LINES, not individual
    words, so `_line_to_words` below splits a line's box into per-word
    boxes proportional to character span. An approximation, not a real
    per-glyph measurement, but sufficient for sweep.py's own use (same-row
    grouping and left-to-right ordering; it never needs a pixel-exact
    word boundary). `confidence` is the parent line's RapidOCR score
    (0-1), copied onto each split word so scoring can use it.
    """

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 0.0


def _quad_to_ltwh(quad: list[list[float]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    left, top = min(xs), min(ys)
    return int(left), int(top), int(max(xs) - left), int(max(ys) - top)


_CHAR_CLASS_RUN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"[A-Z][a-z]+|[A-Z]+(?![a-z])|[a-z]+")


def _split_camel(part: str) -> list[str]:
    """LicenseNo → License, No; SexM → Sex, M. All-caps identifiers
    (DOB, FAN, HORA) stay whole.
    """
    if not part.isalpha() or part.isupper() or part.islower():
        return [part]
    bits = _CAMEL_RE.findall(part)
    return bits if len(bits) > 1 else [part]


def _looks_like_mixed_identifier(chunk: str) -> bool:
    """ICAO-style document numbers alternate letters and digits
    (``IW5J5R3K3``, ``AB1234567``). Those must stay one token.
    CamelCase label+value (``LicenseNo952592``) still splits.
    """
    if not re.fullmatch(r"[A-Za-z0-9]{8,16}", chunk):
        return False
    if not re.search(r"[A-Za-z]", chunk) or not re.search(r"\d", chunk):
        return False
    if re.search(r"[a-z]{3,}", chunk):
        return False
    return True


def _split_glued_chunk(chunk: str) -> list[str]:
    """Splits a space-free chunk further at letter/digit/other CLASS
    transitions -- RapidOCR's own space-dropping (see module docstring)
    means a whole printed row can arrive as ONE chunk with zero internal
    spaces at all, e.g. a real Ethiopian driving license's "License No
    952592" row recognized as "LicenseNo952592".

    ALL-CAPS letter runs glued to digits (``AB1234567``, ``EQ1775002``)
    are document numbers, not labels — those stay one token. Mixed-case
    glue (``LicenseNo952592``) is a label+value and is split, then
    CamelCased (``License``, ``No``, ``952592``).
    """
    if _looks_like_mixed_identifier(chunk):
        return [chunk]
    parts = [p for p in _CHAR_CLASS_RUN_RE.findall(chunk) if p.strip()]
    merged: list[str] = []
    i = 0
    while i < len(parts):
        if (
            i + 1 < len(parts)
            and parts[i].isalpha()
            and parts[i].isupper()
            and parts[i + 1].isdigit()
        ):
            merged.append(parts[i] + parts[i + 1])
            i += 2
            continue
        merged.extend(_split_camel(parts[i]))
        i += 1
    return merged


_OCR_WORD_GAP_HYPHEN_RE = re.compile(r"(?<=[A-Z]{2})-(?=[A-Z]{2})")


def ocr_space_for_printed_word_gap(value: str) -> str:
    """RapidOCR often puts a hyphen in a printed space between two
    uppercase words -- given names (`MELAT-ASRAT`), street tokens
    (`LASKY-CT`). After `unglue_ocr_text` that hyphen is its own token
    (`MELAT - ASRAT`). Identifiers that mix letters and digits (`MD-1027`)
    are unchanged: both sides must be letters.
    """
    value = _OCR_WORD_GAP_HYPHEN_RE.sub(" ", value)
    return re.sub(r"(?<=[A-Z]{2}) - (?=[A-Z]{2})", " ", value).strip()


def unglue_ocr_text(text: str) -> str:
    """Re-space glued OCR for whole-text regex search (generic_ocr).
    Newlines are kept so line-oriented address capture still works.
    """
    lines = []
    for line in text.splitlines():
        tokens: list[str] = []
        for chunk in line.split(" "):
            if chunk:
                tokens.extend(_split_glued_chunk(chunk))
        lines.append(" ".join(tokens))
    return "\n".join(lines)


def _line_to_words(quad: list[list[float]], text: str, confidence: float = 0.0) -> list[OcrWord]:
    left, top, width, height = _quad_to_ltwh(quad)
    words = [part for chunk in text.split(" ") if chunk for part in _split_glued_chunk(chunk)]
    total_chars = sum(len(w) for w in words) or 1
    out = []
    cursor = 0.0
    for w in words:
        w_width = width * (len(w) / total_chars)
        out.append(OcrWord(
            text=w, left=int(left + cursor), top=top, width=int(w_width), height=height,
            confidence=confidence,
        ))
        cursor += w_width
    return out


def _to_array(image: Any) -> Any:
    """Accepts a PIL Image (every `preprocess_variants` output), a raw
    numpy/cv2 array (the edge-sidebar crop callers pass directly, never
    round-tripped through PIL), or a bare file path string
    (`generic_ocr.extract_fields_via_sweep` passes the image path straight
    through to `sweep.tokens_from_image`) -- RapidOCR accepts a path
    natively, so a
    string is handed to it unchanged rather than treated as array data.
    A 2-D (grayscale, e.g. post-`enhance_contrast`) array is stacked into
    3 identical channels rather than rejected; RapidOCR needs 3 channels.
    """
    import numpy as np

    if isinstance(image, str):
        return image
    array = np.array(image.convert("RGB")) if hasattr(image, "convert") else image
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return array


def _run_rapidocr(image: Any) -> Optional[list[tuple[list[list[float]], str, float]]]:
    engine = _get_engine()
    if engine is None:
        return None
    result, _elapsed = engine(_to_array(image))
    return result or []


def ocr_words(pil_image: Any) -> list[OcrWord]:
    """Word-level tokens for `sweep.py`'s per-field positional search.
    Falls back to `pytesseract.image_to_data` (real per-word boxes) if
    RapidOCR is unavailable -- see module docstring.
    """
    lines = _run_rapidocr(pil_image)
    if lines is not None:
        words: list[OcrWord] = []
        for quad, text, conf in lines:
            words.extend(_line_to_words(quad, text.strip(), float(conf or 0.0)))
        return words

    import pytesseract

    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    out = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        out.append(OcrWord(text=text, left=data["left"][i], top=data["top"][i], width=data["width"][i], height=data["height"][i]))
    return out


def _rapidocr_joined_text(image: Any) -> Optional[str]:
    """RapidOCR's lines, joined in reading order (top-to-bottom, ties
    broken left-to-right) -- the shape every existing regex in this
    project already expects from `pytesseract.image_to_string`. None if
    RapidOCR is unavailable or found nothing at all.
    """
    try:
        lines = _run_rapidocr(image)
    except Exception:
        return None
    if lines is None:
        return None
    ordered = sorted(lines, key=lambda ln: (_quad_to_ltwh(ln[0])[1], _quad_to_ltwh(ln[0])[0]))
    text = "\n".join(t.strip() for _quad, t, _conf in ordered if t.strip())
    return text or None


def _tesseract_text(image: Any, tesseract_config: str) -> Optional[str]:
    try:
        import pytesseract

        text = str(pytesseract.image_to_string(image, config=tesseract_config))
    except Exception:
        return None
    return text or None


def ocr_tesseract_page(image: Any) -> Optional[str]:
    """Extra Tesseract votes against RapidOCR (PSM 6 block + PSM 4 column).

    RapidOCR remains the primary engine. Tesseract often keeps printed
    spaces and field-number/value pairing that RapidOCR glues or splits
    onto the next line. Callers should add this as one extra vote, not
    replace RapidOCR.
    """
    parts: list[str] = []
    for config in ("--psm 6", "--psm 4"):
        text = _tesseract_text(image, config)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else None


def ocr_text(pil_image: Any, *, tesseract_config: str = "") -> Optional[str]:
    """Whole-image text -- the shape every existing regex in this project
    already expects from `pytesseract.image_to_string`. Returns None on
    total failure (both engines), matching every caller's existing "skip
    this variant" convention -- never raises.

    `tesseract_config` (a tesseract PSM/config string, e.g. "--psm 6") is
    passed through to the fallback path ONLY -- RapidOCR has no equivalent
    knob; it detects line regions itself rather than needing a page-
    segmentation hint, which is a genuine strength on this project's mixed
    layouts (see module docstring), not a gap.

    Drop-in single-winner mode (confirmed with the user): RapidOCR if it
    produced anything, tesseract only as an automatic fallback -- never
    both on every call. An earlier "additive second channel" mode (both
    engines, always, reconciled through each caller's existing per-pass
    voting) was tried and deliberately reverted: running two engines on
    every single OCR call was more sustained cost than the project wants
    to carry, for a benefit tesseract-as-pure-fallback already captures
    on the rare case RapidOCR itself fails outright.
    """
    return _rapidocr_joined_text(pil_image) or _tesseract_text(pil_image, tesseract_config)
