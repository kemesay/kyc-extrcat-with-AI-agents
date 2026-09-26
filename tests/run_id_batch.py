"""
Batch-run the universal engine over a folder of ID images.

    python tests/run_id_batch.py <image_dir> <output_dir> [--mode fast|full] [--passes auto|1|2|3]
                                 [--only name1,name2]

Writes one <image>.json per unique image (duplicates by content hash are
skipped) plus summary.tsv. Keep <output_dir> outside the repository when the
images are real documents — the results contain personal data.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from universal import extract_universal  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SUMMARY_FIELDS = ["full_name", "full_name_local", "first_name", "last_name", "date_of_birth", "gender",
                  "nationality", "document_number", "expiry_date", "issuing_country"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--mode", default="fast", choices=["fast", "full"])
    ap.add_argument("--passes", default="auto")
    ap.add_argument("--only", default="", help="comma-separated file names to run")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # A laptop that idle-sleeps mid-batch stalls a request for as long as it
    # sleeps (seen: 12 minutes on battery). Hold a no-idle-sleep assertion
    # for the life of this process on macOS.
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    seen: set[str] = set()
    rows = []
    for path in sorted(Path(args.image_dir).iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES or (only and path.name not in only):
            continue
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        if digest in seen:
            print(f"skip duplicate {path.name}")
            continue
        seen.add(digest)
        started = time.time()
        try:
            r = extract_universal(str(path), mode=args.mode, agent_passes=args.passes)
            (out / f"{path.name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
            d = r["extracted_data"]
            rows.append({"file": path.name, "type": r["document_type"], "country": r["issuing_country"],
                         "confidence": r["overall_confidence"], "review": r["human_review_required"],
                         "seconds": r["total_duration_seconds"], **{f: d.get(f, "") for f in SUMMARY_FIELDS},
                         "review_reasons": " | ".join(r["review_reasons"])})
        except Exception as e:
            traceback.print_exc()
            rows.append({"file": path.name, "type": "ERROR", "review_reasons": str(e),
                         "seconds": round(time.time() - started, 1)})
        with open(out / "summary.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "type", "country", "confidence", "review", "seconds",
                                              *SUMMARY_FIELDS, "review_reasons"], delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    print(f"done: {len(rows)} documents -> {out}")


if __name__ == "__main__":
    main()
