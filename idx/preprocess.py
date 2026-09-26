"""
Image preprocessing for phone-photographed / scanned ID documents.

Handles the common low-quality failure modes for this use case: skewed
phone photos, uneven lighting/glare on laminated cards, small MRZ text,
and compression noise. Produces several variants because different OCR
passes tend to win on different documents — no single pipeline is best
for every photo.

`rectify_card` is the "L0 rectification" step CLAUDE.md's field-first
design assumes: perspective-corrects a photographed card onto a canonical
ID-1 rectangle BEFORE anything else runs, so every downstream field
position is predictable. This is a genuinely different fix than `deskew`
below -- confirmed necessary on a real photo with actual keystone
distortion (camera not parallel to the card, so the near edge measurably
outmeasures the far edge): no single ROTATION angle can make a keystoned
quadrilateral rectangular, only a true 4-corner perspective transform can.
Deliberately conservative -- falls back to the original, unwarped image
whenever it isn't confident it found the card at all, since warping based
on a wrong guess would make extraction worse, not better.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Every OCR-heavy module in this package (mrz_ocr.py, aamva.py,
# ethiopia_id.py) imports this one, so setting this here -- once, at
# import time -- reliably applies before any of them spawns a tesseract
# subprocess, regardless of which module happens to be imported first.
#
# Confirmed empirically, and the opposite of what seemed obvious at
# first: tesseract's LSTM engine is itself internally multi-threaded
# (OpenMP) by default. Running several tesseract subprocess calls
# concurrently through a thread pool (see mrz_ocr.find_mrz_candidates,
# aamva._best_orientation_texts, ethiopia_id._ocr_texts) to get real
# wall-clock parallelism on genuinely independent OCR passes backfires
# badly without this: 8 concurrent calls, each ALSO trying to spawn its
# own internal OpenMP threads, oversubscribes the CPU and thrashes --
# measured 51s for a search that takes 5.9s run sequentially with no
# pool at all. Capping each individual tesseract invocation to one
# thread via OMP_THREAD_LIMIT removes that internal contention, and the
# SAME search with the pool AND this cap set measured 3.0s -- faster
# than the plain sequential baseline, not just faster than the
# oversubscribed disaster. `setdefault` so an operator who has
# deliberately set a different value in their own deployment
# environment is respected, not silently overridden.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")
    return img


def deskew(gray: np.ndarray) -> np.ndarray:
    """Detect and correct document skew using a Hough line transform."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return gray

    angles = []
    for line in lines[:40]:
        _, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return gray

    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


# ISO/IEC 7810 ID-1 card format (85.60mm x 53.98mm) -- the physical shape
# of virtually every driver's license/national ID this project reads.
_ID1_ASPECT_RATIO = 85.60 / 53.98


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 arbitrary corner points as top-left, top-right, bottom-
    right, bottom-left -- `cv2.getPerspectiveTransform` requires a
    consistent winding order, but `cv2.approxPolyDP`'s own point order
    depends on where its contour trace happened to start, not the card's
    actual orientation in the photo.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]        # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]        # bottom-right: largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # top-right: smallest y-x
    rect[3] = pts[np.argmax(diff)]     # bottom-left: largest y-x
    return rect


def _largest_card_like_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    """Finds the largest 4-point contour whose proportions are plausible
    for a photographed ID-1 card. Confirmed necessary on a real photo
    with genuine perspective distortion (camera not parallel to the card,
    so the near edge measurably outmeasures the far edge) -- no in-plane
    ROTATION angle can fix that; see `rectify_card`'s docstring.

    Deliberately conservative -- returns None (never guesses) unless a
    quadrilateral both dominates the frame (a real capture taken for this
    purpose fills most of it; this also keeps a small rectangular detail
    printed ON the card, e.g. a logo, from ever winning) but stops well
    short of the WHOLE frame (confirmed a real failure mode: a solid-
    background test image's outermost contour traces the image's own
    border, which trivially "dominates the frame" at exactly 100% -- a
    genuine card photo taken with intent almost always leaves some visible
    margin/background on at least one edge, so a quadrilateral that IS
    the frame is a sign detection found nothing card-shaped at all, not a
    confident detection) AND has a plausible card-like proportion in
    EITHER orientation. A caller that gets None falls back to the
    original, unrectified frame -- this must never warp an image based on
    a low-confidence guess.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Explicit 3x3 kernel (rather than letting cv2 default it from `None`)
    # -- behaviorally identical, but satisfies cv2's own type stub, which
    # doesn't accept `None` for this argument despite the C++ API doing so.
    dilate_kernel = np.ones((3, 3), dtype="uint8")
    edges = cv2.dilate(cv2.Canny(blurred, 50, 150), dilate_kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = gray.shape[0] * gray.shape[1]
    best: Optional[np.ndarray] = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (image_area * 0.2 <= area <= image_area * 0.92) or area <= best_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        ordered = _order_quad_points(approx.reshape(4, 2).astype("float32"))
        tl, tr, br, bl = ordered
        width = max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)))
        height = max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))
        if height == 0:
            continue
        ratio = width / height
        # Tolerant of both landscape and portrait capture, and of a real
        # margin around the true ~1.586 ID-1 ratio -- perspective
        # foreshortening itself skews the MEASURED ratio even on a
        # genuine card; this only needs to reject obviously-wrong shapes
        # (the whole desk surface, a rectangular shadow), not pin the
        # exact ratio.
        if not (1.2 <= ratio <= 2.0 or 0.5 <= ratio <= 0.83):
            continue
        best, best_area = ordered, area

    return best


def rectify_card(img: np.ndarray) -> np.ndarray:
    """Perspective-corrects a photographed ID card onto a canonical ID-1
    frame before any other preprocessing runs -- the "L0 rectification"
    step every per-field position downstream implicitly assumes. Confirmed
    necessary on a real photo where the card was measurably keystoned
    (near edge longer than the far edge, several degrees of skew) -- this
    is a DIFFERENT distortion than `deskew` above corrects (a single
    in-plane rotation angle): no rotation, at any angle, makes a
    keystoned quadrilateral rectangular again. Only a true 4-corner
    perspective transform does.

    Falls back to the ORIGINAL image, completely unchanged, whenever
    `_largest_card_like_quad` isn't confident it found the card -- a
    blank/synthetic test image, an already well-framed photo with no
    detectable quadrilateral, or a detection too small to trust. This is
    intentionally conservative: warping a photo based on a wrong guess
    would actively make extraction worse, not better, so doing nothing is
    the safe default whenever the detection itself isn't solid.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    quad = _largest_card_like_quad(gray)
    if quad is None:
        return img

    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(width / _ID1_ASPECT_RATIO)
    if width < 200 or height < 100:
        return img

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def upscale_if_small(img: np.ndarray, min_dim: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) >= min_dim:
        return img
    scale = min_dim / min(h, w)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def enhance_contrast(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel — recovers text under glare/uneven lighting
    on laminated ID cards without blowing out the rest of the image."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def preprocess_variants(image_path: str) -> dict[str, Image.Image]:
    """Return several preprocessed PIL variants of the same image, each
    tuned for a different failure mode. Callers OCR whichever variants
    they need — trying more variants costs time but recovers more text
    from bad photos."""
    img = load_image(image_path)
    # Perspective correction FIRST, before anything else -- a keystoned
    # photo's field positions are only predictable once the card itself
    # is mapped onto a canonical rectangle; `deskew` below only corrects
    # in-plane rotation, a different (and by itself insufficient) problem
    # for a photo taken at a genuine angle. See `rectify_card`'s own
    # docstring -- a no-op fallback whenever detection isn't confident.
    img = rectify_card(img)
    img = upscale_if_small(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)
    enhanced_bgr = enhance_contrast(img)
    enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    def _sharpen(arr: np.ndarray) -> Image.Image:
        pil = Image.fromarray(arr)
        pil = pil.filter(ImageFilter.SHARPEN)
        return ImageEnhance.Contrast(pil).enhance(1.4)

    return {
        "color_enhanced": Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)),
        "gray": Image.fromarray(gray),
        "gray_enhanced": Image.fromarray(enhanced_gray),
        "binary_sharpened": _sharpen(binary),
    }


def crop_bottom_band(pil_image: Image.Image, fraction: float) -> Image.Image:
    """Crop the bottom `fraction` of the image — where the MRZ sits on
    virtually every passport/ID layout."""
    w, h = pil_image.size
    top = int(h * (1 - fraction))
    return pil_image.crop((0, top, w, h))
