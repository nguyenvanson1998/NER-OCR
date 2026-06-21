import difflib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import Optional


MAX_FUZZY_GROUPS = 50


@dataclass(frozen=True)
class SegmentGroupRecord:
    segments: list[dict[str, Any]]
    text: str
    normalized: str
    tokens: frozenset[str]
    trigrams: frozenset[str]


class SegmentMatchIndex:
    """Precomputed layout text index shared by every extracted field."""

    def __init__(self, segments: list[dict[str, Any]]):
        self.segments = segments
        self.records: list[SegmentGroupRecord] = []
        self.token_groups: dict[str, list[int]] = defaultdict(list)
        self.trigram_groups: dict[str, list[int]] = defaultdict(list)
        self.segment_groups: dict[str, list[int]] = defaultdict(list)

        for group in candidate_segment_groups(segments):
            text = "\n".join(str(segment.get("text") or "") for segment in group)
            normalized = normalize_text(text)
            record = SegmentGroupRecord(
                segments=group,
                text=text,
                normalized=normalized,
                tokens=frozenset(normalized.split()),
                trigrams=text_trigrams(normalized),
            )
            group_index = len(self.records)
            self.records.append(record)
            for token in record.tokens:
                self.token_groups[token].append(group_index)
            for trigram in record.trigrams:
                self.trigram_groups[trigram].append(group_index)
            for segment in group:
                self.segment_groups[str(segment.get("id"))].append(group_index)

        self.single_group_count = len(segments)

    def shortlist(self, candidate_norm: str, limit: int = MAX_FUZZY_GROUPS) -> list[int]:
        tokens = frozenset(candidate_norm.split())
        trigrams = text_trigrams(candidate_norm)
        group_count = len(self.records)
        if not group_count:
            return []

        exact: list[int] = []
        for index, record in enumerate(self.records):
            if candidate_norm in record.normalized or (
                record.normalized in candidate_norm and len(record.normalized) >= 4
            ):
                exact.append(index)

        rarity_limit = max(12, group_count // 8)
        rare_tokens = sorted(
            (token for token in tokens if token in self.token_groups),
            key=lambda token: len(self.token_groups[token]),
        )
        rare_tokens = [token for token in rare_tokens if len(self.token_groups[token]) <= rarity_limit]
        pool: set[int] = set(exact)
        for token in rare_tokens[:4]:
            pool.update(self.token_groups[token])

        # With only generic tokens, identify promising single lines first, then
        # expand only those lines into their nearby multi-line groups.
        if not rare_tokens:
            single_scores = [
                (cheap_similarity(tokens, trigrams, record), index)
                for index, record in enumerate(self.records[: self.single_group_count])
            ]
            single_scores.sort(key=lambda item: (-item[0], item[1]))
            for score, index in single_scores[:20]:
                if score <= 0:
                    continue
                pool.add(index)
                segment_id = str(self.records[index].segments[0].get("id"))
                pool.update(self.segment_groups.get(segment_id, ()))

        # Character trigrams recover OCR spelling/punctuation variations when
        # exact token lookup is sparse without reopening a full fuzzy scan.
        if len(pool) < limit and trigrams:
            trigram_votes: dict[int, int] = defaultdict(int)
            for trigram in trigrams:
                postings = self.trigram_groups.get(trigram, ())
                if len(postings) <= max(40, group_count // 3):
                    for index in postings:
                        trigram_votes[index] += 1
            voted = sorted(trigram_votes, key=lambda index: (-trigram_votes[index], index))
            pool.update(voted[: limit * 2])

        ranked = sorted(
            pool,
            key=lambda index: (-cheap_similarity(tokens, trigrams, self.records[index]), index),
        )
        exact_set = set(exact)
        kept_exact = [index for index in exact if index in pool]
        fuzzy = [index for index in ranked if index not in exact_set][:limit]
        return kept_exact + fuzzy


def attach_field_boxes(extraction: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    used_segment_ids: set[str] = set()
    match_index = SegmentMatchIndex(segments)
    attach_boxes_to_field_group(
        extraction.get("fields", {}), segments, used_segment_ids, "field", match_index=match_index
    )
    attach_boxes_to_field_group(
        extraction.get("work_detail_fields", {}),
        segments,
        used_segment_ids,
        "work-detail",
        match_index=match_index,
    )
    return extraction


def attach_boxes_to_field_group(
    fields: Any,
    segments: list[dict[str, Any]],
    used_segment_ids: set[str],
    box_prefix: str,
    match_index: Optional[SegmentMatchIndex] = None,
) -> None:
    if not isinstance(fields, dict):
        return

    for field_key, field in fields.items():
        if not isinstance(field, dict):
            continue

        match = find_best_segment_group(field, segments, used_segment_ids, match_index=match_index)
        if match:
            matched_segments, score = match
            used_segment_ids.update(segment["id"] for segment in matched_segments)
            first_segment = matched_segments[0]
            field["box"] = {
                "id": f"box-{box_prefix}-{field_key}",
                "field_key": field_key,
                "page": first_segment["page"],
                "bbox": union_bbox(matched_segments),
                "score": round(score, 4),
                "source_segment_id": first_segment["id"],
                "source_segment_ids": [segment["id"] for segment in matched_segments],
                "text": "\n".join(str(segment.get("text") or "") for segment in matched_segments),
            }
        else:
            field["box"] = None


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


def find_best_segment_group(
    field: dict[str, Any],
    segments: list[dict[str, Any]],
    used_segment_ids: set[str],
    match_index: Optional[SegmentMatchIndex] = None,
) -> Optional[tuple[list[dict[str, Any]], float]]:
    candidates = [field.get("evidence"), field.get("value"), field.get("normalized_value")]
    candidate_texts = [str(candidate) for candidate in candidates if candidate not in (None, "")]
    if not candidate_texts:
        return None

    index = match_index or SegmentMatchIndex(segments)
    candidate_data = [
        (candidate, normalize_text(candidate), frozenset(normalize_text(candidate).split()))
        for candidate in candidate_texts
    ]
    group_indices: list[int] = []
    seen_group_indices: set[int] = set()
    for _candidate, candidate_norm, _candidate_tokens in candidate_data:
        if not candidate_norm:
            continue
        for group_index in index.shortlist(candidate_norm):
            if group_index not in seen_group_indices:
                seen_group_indices.add(group_index)
                group_indices.append(group_index)

    best_group: Optional[list[dict[str, Any]]] = None
    best_score = 0.0
    for group_index in group_indices:
        record = index.records[group_index]
        group = record.segments
        for _candidate, candidate_norm, candidate_tokens in candidate_data:
            score = match_score_normalized(candidate_norm, record.normalized)
            used_count = sum(segment.get("id") in used_segment_ids for segment in group)
            if used_count:
                score *= 0.94 ** used_count
            if len(group) > 1:
                coverage = len(candidate_tokens & record.tokens) / max(len(candidate_tokens), 1)
                if coverage >= 0.8:
                    score = min(1.0, score + 0.04)
            if score > best_score:
                best_group = group
                best_score = score

    if best_group and best_score >= 0.48:
        return best_group, best_score
    return None


def candidate_segment_groups(segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = [[segment] for segment in segments]
    ordered = sorted(
        [segment for segment in segments if isinstance(segment.get("bbox"), dict)],
        key=lambda segment: (
            segment.get("page", 0),
            segment["bbox"].get("y", 0),
            segment["bbox"].get("x", 0),
        ),
    )
    for start_index, start in enumerate(ordered):
        group = [start]
        previous = start
        for candidate in ordered[start_index + 1 :]:
            if candidate.get("page") != start.get("page"):
                break
            if not segments_share_column(previous, candidate):
                continue
            group.append(candidate)
            groups.append(list(group))
            previous = candidate
            if len(group) >= 4:
                break
    return groups


def segments_share_column(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_box = first.get("bbox") or {}
    second_box = second.get("bbox") or {}
    first_bottom = float(first_box.get("y", 0)) + float(first_box.get("height", 0))
    vertical_gap = float(second_box.get("y", 0)) - first_bottom
    if vertical_gap < -0.01 or vertical_gap > 0.08:
        return False

    first_x = float(first_box.get("x", 0))
    second_x = float(second_box.get("x", 0))
    first_right = first_x + float(first_box.get("width", 0))
    second_right = second_x + float(second_box.get("width", 0))
    horizontal_overlap = min(first_right, second_right) - max(first_x, second_x)
    return abs(first_x - second_x) <= 0.18 and horizontal_overlap >= -0.02


def union_bbox(segments: list[dict[str, Any]]) -> dict[str, float]:
    boxes = [segment.get("bbox") or {} for segment in segments]
    left = min(float(box.get("x", 0)) for box in boxes)
    top = min(float(box.get("y", 0)) for box in boxes)
    right = max(float(box.get("x", 0)) + float(box.get("width", 0)) for box in boxes)
    bottom = max(float(box.get("y", 0)) + float(box.get("height", 0)) for box in boxes)
    return {
        "x": round(left, 6),
        "y": round(top, 6),
        "width": round(right - left, 6),
        "height": round(bottom - top, 6),
    }


def match_score(needle: str, haystack: str) -> float:
    needle_norm = normalize_text(needle)
    haystack_norm = normalize_text(haystack)
    return match_score_normalized(needle_norm, haystack_norm)


def match_score_normalized(needle_norm: str, haystack_norm: str) -> float:
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


def text_trigrams(value: str) -> frozenset[str]:
    compact = f"  {value}  "
    return frozenset(compact[index : index + 3] for index in range(max(0, len(compact) - 2)))


def cheap_similarity(
    candidate_tokens: frozenset[str],
    candidate_trigrams: frozenset[str],
    record: SegmentGroupRecord,
) -> float:
    token_overlap = len(candidate_tokens & record.tokens) / max(len(candidate_tokens), 1)
    trigram_overlap = len(candidate_trigrams & record.trigrams) / max(len(candidate_trigrams), 1)
    return token_overlap * 0.72 + trigram_overlap * 0.28


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    value = re.sub(r"[^a-z0-9%/.,:-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()
