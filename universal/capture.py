"""
Capture-quality check — measured on the pixels, in milliseconds, before any
model runs.

Deterministic measurements at a fixed 1000 px scale (so a 12 MP photo and a
1 MP scan are judged alike):

  sharpness   variance of the Laplacian (focus / motion blur)
  exposure    mean brightness (too dark / washed out)
  contrast    brightness standard deviation (faded, foggy)
  glare       largest connected blown-out area (a reflection over text)
  resolution  pixels on the short side of the original

Thresholds for "poor" sit BELOW every capture the engine has read correctly
(the ids/ regression set: blurriest readable sharpness 38, darkest mean 81),
so a readable document is never turned away. Nothing here alters the image
the agents see — no denoising, no sharpening, no super-resolution, which
can invent strokes in small text. The grade goes into the result, and a
client can ask (reject_poor_capture=true) to have a poor capture refused at
upload, so the customer retakes the photo while still at the counter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageOps

# (poor below, fair below) — calibrated on the regression set, see module docstring.
SHARPNESS = (20.0, 60.0)
MIN_SHORT_SIDE = (300, 600)
DARK_MEAN, BRIGHT_MEAN = (45.0, 230.0), (60.0, 215.0)          # (poor, fair) bounds
CONTRAST = (12.0, 20.0)
GLARE_PERCENT = (20.0, 6.0)                                     # poor above, fair above


def assess(path: str) -> dict[str, Any]:
    import cv2
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
    width, height = im.size
    scale = 1000 / max(width, height)
    small = im.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
    g = np.asarray(small)
    sharpness = float(cv2.Laplacian(g, cv2.CV_64F).var())
    mean, contrast = float(g.mean()), float(g.std())
    blown = (g >= 250).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(blown)
    glare = 100.0 * max((stats[i, cv2.CC_STAT_AREA] for i in range(1, count)), default=0) / g.size
    short = min(width, height)

    poor, fair = [], []

    def grade(bad_poor: bool, bad_fair: bool, message: str) -> None:
        if bad_poor:
            poor.append(message)
        elif bad_fair:
            fair.append(message)

    grade(sharpness < SHARPNESS[0], sharpness < SHARPNESS[1], "image is blurred or out of focus")
    grade(short < MIN_SHORT_SIDE[0], short < MIN_SHORT_SIDE[1], f"resolution is low ({width}x{height})")
    grade(mean < DARK_MEAN[0], mean < BRIGHT_MEAN[0], "image is too dark")
    grade(mean > DARK_MEAN[1], mean > BRIGHT_MEAN[1], "image is overexposed")
    grade(contrast < CONTRAST[0], contrast < CONTRAST[1], "contrast is very low")
    grade(glare > GLARE_PERCENT[0], glare > GLARE_PERCENT[1], "glare covers part of the document")

    return {
        "grade": "poor" if poor else "fair" if fair else "good",
        "issues": poor + fair,
        "measurements": {"sharpness": round(sharpness, 1), "brightness": round(mean, 1),
                         "contrast": round(contrast, 1), "glare_percent": round(glare, 2),
                         "width": width, "height": height},
    }


def assess_all(paths: dict[str, str]) -> dict[str, Any]:
    """Per side, plus the worst grade overall."""
    sides = {side: assess(p) for side, p in paths.items()}
    order = {"good": 0, "fair": 1, "poor": 2}
    worst = max((s["grade"] for s in sides.values()), key=order.get, default="good")
    return {"grade": worst, "sides": sides}
