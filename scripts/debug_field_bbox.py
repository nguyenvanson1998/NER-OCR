"""Diagnostic: in ra segment được map vào 1 field cụ thể + các segment cùng dòng.

Cách dùng:
    PYTHONPATH=. .venv/bin/python scripts/debug_field_bbox.py \
        "data/benchmark/3. QĐ KHLCNT A1.pdf" \
        --field signed_or_effective_date
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


def _load_app_modules():
    from app.main import run_fast_ocr_layout
    from app.services.extraction import extract_information
    from app.services.layout_matching import attach_field_boxes
    from app.services.ocr_cleaning import clean_ocr_layout_result

    return run_fast_ocr_layout, extract_information, attach_field_boxes, clean_ocr_layout_result


async def _run(pdf_path: Path, field_key: str, extraction_type: str) -> dict[str, Any]:
    run_fast_ocr_layout, extract_information, attach_field_boxes, clean_ocr_layout_result = (
        _load_app_modules()
    )

    raw_layout, provider = await run_fast_ocr_layout(pdf_path, request_id="debug")
    layout = clean_ocr_layout_result(raw_layout)
    text = str(layout.get("text") or "")

    # Force rule-only path so we can inspect the local extraction without LLM noise.
    os.environ.setdefault("LLM_PROVIDER", "none")
    os.environ.setdefault("LOCAL_EXTRACTION_ENABLED", "true")
    result = await extract_information(text, extraction_type=extraction_type)

    fields = result["data"]["fields"]
    attach_field_boxes(result["data"], layout.get("segments") or [])

    field = fields.get(field_key)
    if not field:
        return {"error": f"field {field_key!r} not in extraction output", "available": list(fields)}

    box = field.get("box") or {}
    matched_ids = set(box.get("source_segment_ids") or ([box.get("source_segment_id")] if box.get("source_segment_id") else []))

    # Build a small report: matched segment(s) + the 3 nearest neighbours per page.
    segments = layout.get("segments") or []
    matched_segments = [seg for seg in segments if str(seg.get("id")) in matched_ids]
    neighbours: list[dict[str, Any]] = []
    if matched_segments:
        first = matched_segments[0]
        page = first.get("page")
        y_center = (first["bbox"]["y"] + first["bbox"]["height"] / 2)
        same_page = [seg for seg in segments if seg.get("page") == page]
        same_page.sort(key=lambda seg: abs((seg["bbox"]["y"] + seg["bbox"]["height"] / 2) - y_center))
        neighbours = same_page[:6]

    return {
        "ocr_provider": provider,
        "field": field_key,
        "value": field.get("value"),
        "evidence": field.get("evidence"),
        "box": box,
        "matched_segments": [
            {"id": seg["id"], "text": seg.get("text"), "bbox": seg.get("bbox")}
            for seg in matched_segments
        ],
        "nearby_segments_same_page": [
            {
                "id": seg["id"],
                "text": seg.get("text"),
                "bbox": seg.get("bbox"),
                "is_matched": str(seg.get("id")) in matched_ids,
            }
            for seg in neighbours
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--field", default="signed_or_effective_date")
    parser.add_argument("--type", default="document", choices=["document", "contract"])
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        raise SystemExit(2)

    report = asyncio.run(_run(args.pdf, args.field, args.type))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
