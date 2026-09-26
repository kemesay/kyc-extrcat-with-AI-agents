"""
Command-line use without the HTTP server:

    python -m universal front.jpg [back.jpg] [--mode fast|full] [--doc-type passport] [--passes auto|1|2|3]
                        [--agents] [--no-grounding] [--no-mrz-search] [--full-json]
"""

import argparse
import json

from .pipeline import MODES, extract_universal


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m universal", description="Universal ID extraction")
    parser.add_argument("front")
    parser.add_argument("back", nargs="?")
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--doc-type", default="auto")
    parser.add_argument("--agents", action="store_true", help="include each agent's raw JSON")
    parser.add_argument("--passes", default="auto", choices=["auto", "1", "2", "3"],
                        help="per-agent passes (auto = fewest that fit all relevant prompt knowledge)")
    parser.add_argument("--no-grounding", action="store_true", help="skip OCR grounding")
    parser.add_argument("--no-mrz-search", action="store_true", help="skip the local OCR MRZ re-read")
    parser.add_argument("--full-json", action="store_true", help="print the whole result, not the summary")
    args = parser.parse_args()

    result = extract_universal(
        args.front, args.back, mode=args.mode, doc_type_hint=args.doc_type,
        include_agent_outputs=args.agents, ocr_grounding=not args.no_grounding,
        local_mrz_search=not args.no_mrz_search, agent_passes=args.passes,
    )
    if not args.full_json:
        result = {k: result[k] for k in (
            "document_type", "issuing_country", "extracted_data", "overall_confidence",
            "completeness_percent", "document_currently_valid", "human_review_required",
            "review_reasons", "warnings", "total_duration_seconds")}
        result["extracted_data"] = {k: v for k, v in result["extracted_data"].items() if v}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
