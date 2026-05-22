import difflib
import re
import unicodedata
from typing import Any
from typing import Optional


def attach_field_boxes(extraction: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    fields = extraction.get("fields", {})
    if not isinstance(fields, dict):
        return extraction

    used_segment_ids: set[str] = set()
    for field_key, field in fields.items():
        if not isinstance(field, dict):
            continue

        match = find_best_segment(field, segments, used_segment_ids)
        if match:
            segment, score = match
            used_segment_ids.add(segment["id"])
            field["box"] = {
                "id": f"box-{field_key}",
                "field_key": field_key,
                "page": segment["page"],
                "bbox": segment["bbox"],
                "score": round(score, 4),
                "source_segment_id": segment["id"],
                "text": segment["text"],
            }
        else:
            field["box"] = None

    return extraction


def find_best_segment(
    field: dict[str, Any],
    segments: list[dict[str, Any]],
    used_segment_ids: set[str],
) -> Optional[tuple[dict[str, Any], float]]:
    candidates = [
        field.get("evidence"),
        field.get("value"),
        field.get("normalized_value"),
    ]
    candidates = [str(candidate) for candidate in candidates if candidate not in (None, "")]
    if not candidates:
        return None

    best_segment = None
    best_score = 0.0
    for segment in segments:
        segment_text = segment.get("text") or ""
        for candidate in candidates:
            score = match_score(candidate, segment_text)
            if segment["id"] in used_segment_ids:
                score *= 0.94
            if score > best_score:
                best_segment = segment
                best_score = score

    if best_segment and best_score >= 0.48:
        return best_segment, best_score
    return None


def match_score(needle: str, haystack: str) -> float:
    needle_norm = normalize_text(needle)
    haystack_norm = normalize_text(haystack)
    if not needle_norm or not haystack_norm:
        return 0.0

    if needle_norm in haystack_norm:
        coverage = min(1.0, len(needle_norm) / max(len(haystack_norm), 1))
        return 0.82 + 0.18 * coverage

    if haystack_norm in needle_norm and len(haystack_norm) >= 4:
        coverage = min(1.0, len(haystack_norm) / max(len(needle_norm), 1))
        return 0.72 + 0.18 * coverage

    needle_tokens = set(needle_norm.split())
    haystack_tokens = set(haystack_norm.split())
    overlap = len(needle_tokens & haystack_tokens) / max(len(needle_tokens), 1)
    sequence = difflib.SequenceMatcher(None, needle_norm, haystack_norm).ratio()
    return max(sequence * 0.78, overlap * 0.84)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    value = re.sub(r"[^a-z0-9%/.,:-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()
