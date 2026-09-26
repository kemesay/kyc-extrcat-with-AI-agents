"""
Test Client — ID Document Extraction API v5.0
=============================================
Multi-channel pipeline (barcode/QR -> MRZ -> Ollama AI-vision crosscheck ->
Ollama AI-vision universal fallback -> generic OCR floor). See idx/pipeline.py.

Usage:
  python test_client.py <image> [doc_type] [sync|async] [back_image]

Examples:
  python test_client.py passport.jpg passport sync
  python test_client.py ethiopian_id.jpg auto async
  python test_client.py dl_front.jpg drivers_license sync dl_back.jpg
"""

import sys
import time
import requests

BASE = "http://localhost:8000"


def print_banner():
    print("\n╔══════════════════════════════════════════════╗")
    print("║  ID Extraction API v5.0 — Test Client       ║")
    print("║  barcode/MRZ -> Ollama AI-vision -> OCR      ║")
    print("╚══════════════════════════════════════════════╝\n")


def print_result(data: dict):
    print("\n" + "=" * 58)
    print(f"📋 RESULT  |  pipeline: {data.get('pipeline', '?')}")
    print("=" * 58)

    ex = data.get("extracted_data", {})
    for k, v in ex.items():
        if v:
            print(f"  {k:<25} {v}")

    print(f"\n  {'Document type':<25} {data.get('document_type', '?')}")
    print(f"  {'Extraction method':<25} {data.get('extraction_method', '?')}")
    print(f"  {'Confidence':<25} {data.get('confidence', '?')}")
    print(f"  {'Coverage tier':<25} {data.get('coverage_tier', '?')}")
    print(f"  {'Human review required':<25} {data.get('human_review_required', '?')}")
    print(f"  {'Time':<25} {data.get('total_duration_seconds', '?')}s")

    for w in data.get("warnings") or []:
        print(f"  ⚠️  {w}")

    validation = data.get("validation") or {}
    flag = "⚠️  SUSPICIOUS" if validation.get("suspicious") else "✅ clean"
    print(f"  {'Sanity check':<25} {flag}")
    for reason in validation.get("reasons", []):
        print(f"      - {reason}")

    doc = data.get("document") or {}
    if doc.get("mrz_checks"):
        print(f"\n  MRZ checksums: {doc['mrz_checks']}")
    if doc.get("verification_checks"):
        print(f"  Verification checks: {doc['verification_checks']}")
    print()


def test_health():
    print("🔍 Checking API health...")
    r = requests.get(f"{BASE}/health")
    d = r.json()
    print(f"  Status: {d['status']}")
    print(f"  GPU:    {d.get('gpu', 'unknown')}")
    print(f"  OCR:    {d.get('local_ocr', 'unknown')}")
    for model, status in d.get("ollama_models", {}).items():
        print(f"  {status}  {model}")
    print()


def test_agents():
    print("🔗 Pipeline channels (precedence order):")
    r = requests.get(f"{BASE}/agents")
    d = r.json()
    for c in d["channels"]:
        print(f"  {c['tier']}. {c['name']:<16} {c['what']}")
    print()


def test_sync(image: str, doc_type: str, back_image: str | None):
    print(f"📄 Sync [{doc_type}]: {image}" + (f" + {back_image}" if back_image else ""))
    files = {"file": (image, open(image, "rb"), "image/jpeg")}
    if back_image:
        files["file_back"] = (back_image, open(back_image, "rb"), "image/jpeg")
    r = requests.post(f"{BASE}/extract/id/sync?doc_type={doc_type}", files=files)
    d = r.json()
    print_result(d.get("result") or d)


def test_async(image: str, doc_type: str, back_image: str | None):
    print(f"📄 Async [{doc_type}]: {image}" + (f" + {back_image}" if back_image else ""))
    files = {"file": (image, open(image, "rb"), "image/jpeg")}
    if back_image:
        files["file_back"] = (back_image, open(back_image, "rb"), "image/jpeg")
    r = requests.post(f"{BASE}/extract/id?doc_type={doc_type}", files=files)
    job = r.json()
    job_id = job["job_id"]
    print(f"  Job ID: {job_id}")
    print("  Polling", end="", flush=True)

    while True:
        time.sleep(3)
        r = requests.get(f"{BASE}/extract/status/{job_id}")
        d = r.json()
        print(".", end="", flush=True)
        if d["status"] == "completed":
            print(" Done!")
            print_result(d["result"])
            break
        elif d["status"] == "failed":
            print(f"\n  ❌ Failed: {d['message']}")
            break


if __name__ == "__main__":
    print_banner()
    test_health()
    test_agents()

    if len(sys.argv) < 2:
        print("Usage: python test_client.py <image> [doc_type] [sync|async] [back_image]")
        print()
        print("Examples:")
        print("  python test_client.py passport.jpg passport sync")
        print("  python test_client.py ethiopian_id.jpg auto async")
        print("  python test_client.py dl_front.jpg drivers_license sync dl_back.jpg")
        sys.exit(0)

    image      = sys.argv[1]
    doc_type   = sys.argv[2] if len(sys.argv) > 2 else "auto"
    style      = sys.argv[3] if len(sys.argv) > 3 else "sync"
    back_image = sys.argv[4] if len(sys.argv) > 4 else None

    if style == "async":
        test_async(image, doc_type, back_image)
    else:
        test_sync(image, doc_type, back_image)
