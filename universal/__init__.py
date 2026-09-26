"""
Universal ID extraction engine — any country, any script, any ID type.

Additive to the existing idx/ pipeline (nothing in idx/ is modified): the
prompts in universal_id_extraction_prompts.py drive a 5-agent Ollama team
(fast scanner -> field extractor -> accuracy verifier -> layout analyzer ->
supervisor), and every value they propose is then checked by deterministic
code wherever the document makes that possible:

  - MRZ lines the agents read are re-validated with idx's ICAO 9303 parsers
    (and re-read by local OCR when the agents' copy fails its check digits)
  - barcodes are decoded locally (AAMVA PDF417, Ethiopian Fayda QR, others)
  - national ID numbers are checked against their country's checksum and
    cross-checked against the birth date / sex many of them encode
  - non-Gregorian dates are converted by calendar arithmetic, not by the LLM
  - countries are resolved to ISO 3166-1 alpha-3 via idx's ICAO table

Entry point: universal.pipeline.extract_universal(). HTTP: universal.api.
"""

from .pipeline import extract_universal

__all__ = ["extract_universal"]
