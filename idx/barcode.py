"""
Generic barcode/QR decode utility, shared by any document template that
carries a machine-readable barcode instead of (or alongside) an MRZ --
e.g. `ethiopia_id.py` (QR + Code128 FAN barcode) and `aamva.py` (PDF417).

Knows nothing about what any particular barcode's payload means -- that
parsing lives in each template module, same separation as
`mrz_checksum.py` (generic check-digit arithmetic) vs `mrz_td3.py`
(what the fields mean).

Decode strategy learned empirically against real phone photos: QR codes
decode reliably from the whole image with a modest upscale and no crop,
but small 1D barcodes (e.g. a FAN barcode a few mm tall on a card) need a
much larger upscale *and* a crop close to where the barcode actually sits
-- 1D symbologies have far less redundancy per pixel than a QR code and
are much more sensitive to surrounding clutter at low resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from .preprocess import enhance_contrast, load_image, upscale_if_small

# Sliding vertical bands through the bottom half of the frame (where a
# barcode strip on an ID card typically sits), each crossed with a couple
# of horizontal ranges. A crop that runs all the way to the bottom/side
# edge of the frame tends to drag in background clutter (shadows, table
# edge, a second ghost-photo watermark) that defeats 1D decoding even
# though the barcode itself is sharp -- so bands stop short of the edge
# rather than spanning edge-to-edge.
_BAND_CROPS = tuple(
    (y0, y0 + 0.25, x0, x1)
    for y0 in (0.40, 0.50, 0.60, 0.70)
    for x0, x1 in ((0.0, 1.0), (0.2, 1.0))
)


@dataclass
class BarcodeResult:
    format: str
    text: str


def _crop(img: Any, y0f: float, y1f: float, x0f: float, x1f: float) -> Any:
    h, w = img.shape[:2]
    return img[int(h * y0f):int(h * y1f), int(w * x0f):int(w * x1f)]


def decode_barcodes(image_path: str, formats: tuple[str, ...] = ()) -> list[BarcodeResult]:
    """Try to decode any barcode/QR code in the image, trying progressively
    more aggressive (and more expensive) image variants until one hits.
    `formats` restricts which symbologies to look for (e.g. ("QRCode",),
    ("Code128",)) -- narrowing this speeds up the search and avoids
    spurious matches. Empty tuple means "any format".
    """
    import zxingcpp

    zformats = (
        zxingcpp.BarcodeFormats([getattr(zxingcpp.BarcodeFormat, f) for f in formats]) if formats else None
    )

    img = load_image(image_path)

    # Pass 1: whole image at a few upscales -- this is what QR codes need,
    # cheap enough to always try first regardless of target format.
    for scale in (1, 2, 3):
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        hits = zxingcpp.read_barcodes(resized, formats=zformats) if zformats else zxingcpp.read_barcodes(resized)
        if hits:
            return [BarcodeResult(format=str(h.format), text=h.text) for h in hits]

    # Pass 2: small 1D barcodes need a bigger upscale and a tighter crop to
    # cut down background clutter -- bounded to a fixed grid of bands
    # rather than an unbounded sliding-window search.
    big = upscale_if_small(img, min_dim=2400)
    for y0, y1, x0, x1 in _BAND_CROPS:
        crop = _crop(big, y0, y1, x0, x1)
        enhanced = enhance_contrast(crop)
        hits = zxingcpp.read_barcodes(enhanced, formats=zformats) if zformats else zxingcpp.read_barcodes(enhanced)
        if hits:
            return [BarcodeResult(format=str(h.format), text=h.text) for h in hits]

    return []
