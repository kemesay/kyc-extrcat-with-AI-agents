"""
Locate and OCR the Machine Readable Zone (MRZ) on a document photo, then
recover from common OCR mistakes using the ICAO check digits themselves —
the checksum tells us when a character was misread, which is what makes
error-correction possible even on low-quality images.
"""

from __future__ import annotations

from typing import Any, Optional

import concurrent.futures
import re
from dataclasses import dataclass

from PIL import Image

from .preprocess import preprocess_variants, crop_bottom_band
from .mrz_checksum import check_digit

MRZ_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
_TESS_CONFIG = (
    f"--oem 3 --psm 6 -c tessedit_char_whitelist={MRZ_CHARSET}"
)

# OCR frequently confuses these in the constrained MRZ font (OCR-B / monospace).
_CONFUSABLE = {
    "O": ["0"], "0": ["O"],
    "I": ["1"], "1": ["I", "L"], "L": ["1"],
    "S": ["5"], "5": ["S"],
    "Z": ["2"], "2": ["Z"],
    "B": ["8"], "8": ["B"],
    "G": ["6"], "6": ["G"],
    "D": ["0"],
    "<": ["K", "C"],
}


_MAX_SEGMENTS_PER_LINE = 3


def _ocr_line_candidates(pil_image: Image.Image) -> list[str]:
    """`tessedit_char_whitelist` above already constrains RECOGNITION to
    MRZ_CHARSET, so a stray non-charset, non-whitespace character
    surviving into `raw` would already be rare -- but tesseract still
    inserts whitespace at perceived word boundaries regardless of the
    whitelist, and a genuine MRZ row is printed as ONE continuous run
    with no internal gaps at all (ICAO pads with '<', never a space).
    More than an occasional single stray space is a sign this "line" is
    actually multiple UNRELATED text fragments merged together once
    whitespace gets stripped below -- e.g. fragments of a driver's
    license's guilloche/security-pattern noise happening to concatenate
    into a plausible-length blob, the same false-positive-MRZ risk
    `pipeline._mrz_gate_passes` already gates against downstream (a local
    AAMVA-barcode probe, built after a real Virginia license exposed this
    failure mode). Rejecting lines with too many space-separated segments
    here stops some of that for free, before that downstream gate is even
    reached -- a generous threshold (3) since a single stray gap on
    otherwise-genuine MRZ text is a common, harmless OCR quirk.
    """
    from .ocr_engine import ocr_text

    text = ocr_text(pil_image, tesseract_config=_TESS_CONFIG) or ""
    lines = []
    for raw in text.splitlines():
        upper = raw.upper()
        if len(upper.split()) > _MAX_SEGMENTS_PER_LINE:
            continue
        line = re.sub(r"[^A-Z0-9<]", "", upper)
        if len(line) >= 20:
            lines.append(line)
    return lines


_FORMAT_SPECS = {
    # format -> (number of MRZ lines, exact char length per line)
    "TD3": (2, 44),
    "TD2": (2, 36),
    "TD1": (3, 30),
}
_MIN_LINE_LEN = 15  # shortest OCR fragment worth considering as an MRZ line


def _normalize_line(line: str, target_len: int) -> str:
    """Right-pad with '<' (the correct filler char) or truncate to the exact
    MRZ line length. OCR very commonly drops trailing '<' runs — they're thin,
    low-contrast strokes — so padding rather than rejecting short lines is
    what actually recovers real-world MRZ reads, not just clean synthetic ones.
    """
    if len(line) < target_len:
        return line.ljust(target_len, "<")
    return line[:target_len]


@dataclass
class MrzCandidate:
    format: str          # "TD3" | "TD2" | "TD1"
    lines: list[str]


def _mrz_candidates_from_variants(variants: dict[str, Any], formats: tuple[str, ...]) -> list[MrzCandidate]:
    """The 4 band fractions x N preprocessing variants below are all
    INDEPENDENT `_ocr_line_candidates` calls -- each one a real
    `tesseract` OS subprocess (`pytesseract.run_tesseract` uses
    `subprocess.Popen`, not an in-process call), so they don't contend for
    Python's GIL the way pure-Python CPU work would. Run through a thread
    pool instead of a plain sequential loop for genuine wall-clock
    parallelism bounded by available CPU cores, not a marginal one --
    confirmed empirically: a real passport photo dropped from ~5.9s to a
    fraction of that on an 8-core machine. `ThreadPoolExecutor.map`
    returns results in submission order regardless of which finishes
    first, so "most-promising first" below is completely unaffected by
    the OS scheduling these subprocesses in whatever order it likes.
    """
    candidates: list[MrzCandidate] = []
    seen: set[tuple[Any, ...]] = set()

    crops = [
        crop_bottom_band(variant, band_fraction)
        for band_fraction in (0.30, 0.22, 0.15, 0.45)
        for variant in variants.values()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(crops))) as executor:
        # Materialized as a list WHILE still inside the `with` block --
        # `executor.map` returns a lazy iterator, and leaving it unconsumed
        # past the context exit would rely on `shutdown(wait=True)`'s
        # internal behavior rather than an explicit guarantee.
        all_lines = list(executor.map(_ocr_line_candidates, crops))

    for raw_lines in all_lines:
        lines = [line for line in raw_lines if len(line) >= _MIN_LINE_LEN]
        if len(lines) < 2:
            continue

        for fmt in formats:
            n_lines, target_len = _FORMAT_SPECS[fmt]
            if len(lines) < n_lines:
                continue
            group = lines[-n_lines:]
            normalized = tuple(_normalize_line(line, target_len) for line in group)
            key = (fmt, normalized)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(MrzCandidate(format=fmt, lines=list(normalized)))

    return candidates


def find_mrz_candidates(image_path: str, formats: tuple[str, ...] = ("TD3", "TD2", "TD1")) -> list[MrzCandidate]:
    """Scan multiple crops/preprocessing variants of the image, return every
    plausible MRZ line-group for each requested format, most-promising first.
    Callers try each candidate against the relevant parser until one passes
    checksums — this function doesn't try to judge correctness itself, only
    to gather enough raw-OCR variants that checksum validation has a shot.

    The MRZ is always expected at the BOTTOM of an upright document
    (`crop_bottom_band`, inside `_mrz_candidates_from_variants`), so a
    photo taken sideways or upside down needs the whole image rotated
    back to upright first, or that crop lands on the wrong part of the
    document entirely. Tries upright first and only pays for rotated
    attempts when that finds nothing -- same "only try harder when
    genuinely needed" pattern as `aamva._best_orientation_texts`, built
    for the identical real risk (confirmed on two real driver's licenses
    photographed sideways during that module's development). Re-rotates
    the SAME already-preprocessed variants rather than re-running
    `preprocess_variants` from scratch per rotation, since the CLAHE/
    denoise/threshold work itself doesn't depend on orientation.
    """
    variants = preprocess_variants(image_path)
    candidates = _mrz_candidates_from_variants(variants, formats)
    if candidates:
        return candidates

    for degrees in (90, 180, 270):
        rotated_variants = {name: img.rotate(degrees, expand=True) for name, img in variants.items()}
        candidates = _mrz_candidates_from_variants(rotated_variants, formats)
        if candidates:
            return candidates

    return []


def try_correct_field(field: str, expected_check: str, max_flips: int = 2) -> str | None:
    """If `field`'s checksum doesn't match `expected_check`, try flipping up
    to `max_flips` OCR-confusable characters to find a value that does.
    Returns the corrected field, or None if no small correction validates.
    This only searches characters that have a plausible confusion — bounded
    and fast even though it looks like brute force.
    """
    if not expected_check.isdigit():
        return None
    target = int(expected_check)
    if check_digit(field) == target:
        return field

    positions_with_alts = [
        (i, _CONFUSABLE[ch]) for i, ch in enumerate(field) if ch in _CONFUSABLE
    ]

    def _search(base: str, remaining_flips: int, start_idx: int) -> Optional[str]:
        if check_digit(base) == target:
            return base
        if remaining_flips == 0:
            return None
        for idx, (pos, alts) in enumerate(positions_with_alts):
            if idx < start_idx:
                continue
            for alt in alts:
                candidate = base[:pos] + alt + base[pos + 1:]
                result = _search(candidate, remaining_flips - 1, idx + 1)
                if result:
                    return result
        return None

    return _search(field, max_flips, 0)
