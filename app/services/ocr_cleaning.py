import re
import unicodedata
from copy import deepcopy
from typing import Any
from typing import Optional


MAX_REPORT_ITEMS = 60


def clean_ocr_plain_text(text: str) -> str:
    return clean_ocr_plain_text_with_report(text)["text"]


def clean_ocr_plain_text_with_report(text: str) -> dict[str, Any]:
    normalized_text = normalize_ocr_chars(text)
    lines = normalized_text.splitlines()
    cleaned_lines, report = clean_ocr_lines(lines)
    return {
        "text": "\n".join(cleaned_lines),
        **report,
    }


def clean_ocr_chunks(chunks: list[tuple[str, dict]]) -> tuple[list[tuple[str, dict]], dict[str, Any]]:
    cleaned_chunks: list[tuple[str, dict]] = []
    summary = empty_report()

    for text, metadata in chunks:
        report = clean_ocr_plain_text_with_report(text)
        merge_report(summary, report)
        cleaned_text = report["text"].strip()
        if not cleaned_text:
            continue
        cleaned_metadata = deepcopy(metadata) if isinstance(metadata, dict) else {}
        if isinstance(cleaned_metadata.get("content"), str):
            cleaned_metadata["content"] = cleaned_text
        cleaned_chunks.append((cleaned_text, cleaned_metadata))

    return cleaned_chunks, compact_report(summary)


def clean_ocr_layout_result(layout: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(layout)
    raw_text = str(layout.get("text") or "")
    text_report = clean_ocr_plain_text_with_report(raw_text)
    cleaned["raw_text"] = raw_text
    cleaned["text"] = text_report["text"]

    chunks, chunk_report = clean_ocr_chunks(layout.get("chunks") or [])
    cleaned["chunks"] = chunks

    segments, segment_report = clean_layout_segments(layout.get("segments") or [])
    cleaned["segments"] = segments

    combined_report = empty_report()
    merge_report(combined_report, text_report)
    merge_report(combined_report, chunk_report)
    merge_report(combined_report, segment_report)
    cleaned["cleaning"] = compact_report(combined_report)
    return cleaned


def clean_layout_segments(segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    pages_with_incoming_stamp = {
        segment.get("page")
        for segment in segments
        if is_incoming_stamp_line(str(segment.get("text") or ""))
    }

    for segment in segments:
        text = str(segment.get("text") or "")
        reason = layout_segment_noise_reason(segment, pages_with_incoming_stamp)
        if reason:
            removed.append(
                {
                    "text": clean_inline_text(text),
                    "reason": reason,
                    "page": segment.get("page"),
                    "bbox": segment.get("bbox"),
                }
            )
            continue

        bbox = segment.get("bbox") if isinstance(segment.get("bbox"), dict) else {}
        cleaned_text = clean_segment_text(text, segment.get("page"), safe_float(bbox.get("y")), pages_with_incoming_stamp)
        if not cleaned_text:
            removed.append(
                {
                    "text": clean_inline_text(text),
                    "reason": "empty_after_text_cleaning",
                    "page": segment.get("page"),
                    "bbox": segment.get("bbox"),
                }
            )
            continue

        cleaned_segment = deepcopy(segment)
        cleaned_segment["text"] = clean_inline_text(cleaned_text)
        kept.append(cleaned_segment)

    return kept, {
        "removed_segments": removed[:MAX_REPORT_ITEMS],
        "removed_segment_count": len(removed),
    }


def clean_segment_text(text: str, page: Any, y: float, pages_with_incoming_stamp: set[Any]) -> str:
    if page in pages_with_incoming_stamp and y <= 0.55:
        rewritten = strip_incoming_stamp_date_prefix(normalize_ocr_chars(text))
        if rewritten is not None:
            return rewritten
    return clean_ocr_plain_text(text)


def clean_ocr_lines(lines: list[str]) -> tuple[list[str], dict[str, Any]]:
    cleaned_lines: list[str] = []
    report = empty_report()
    incoming_stamp_seen = False
    incoming_stamp_number_seen = False
    incoming_stamp_date_removed = False

    for index, raw_line in enumerate(lines):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue

        inline_line, inline_reason = strip_inline_incoming_stamp(line)
        if inline_reason:
            if inline_line:
                add_report_item(report, "rewritten_lines", line, inline_line, inline_reason)
            else:
                add_report_item(report, "removed_lines", line, None, inline_reason)
            line = inline_line
            incoming_stamp_seen = True
            incoming_stamp_number_seen = True
            if not line:
                continue

        if is_incoming_stamp_line(line):
            incoming_stamp_seen = True
            add_report_item(report, "removed_lines", line, None, "incoming_stamp_header")
            continue

        if incoming_stamp_seen and is_incoming_stamp_number_line(line):
            incoming_stamp_number_seen = True
            add_report_item(report, "removed_lines", line, None, "incoming_stamp_number")
            continue

        if incoming_stamp_seen and not incoming_stamp_date_removed:
            rewritten_date_line = strip_incoming_stamp_date_prefix(line)
            if rewritten_date_line is not None:
                incoming_stamp_date_removed = True
                reason = "incoming_stamp_date"
                if rewritten_date_line:
                    add_report_item(report, "rewritten_lines", line, rewritten_date_line, reason)
                    line = rewritten_date_line
                else:
                    add_report_item(report, "removed_lines", line, None, reason)
                    continue
            elif incoming_stamp_number_seen and is_incoming_stamp_date_line(line):
                incoming_stamp_date_removed = True
                add_report_item(report, "removed_lines", line, None, "incoming_stamp_date")
                continue

        noise_reason = whole_line_noise_reason(line, index)
        if noise_reason:
            add_report_item(report, "removed_lines", line, None, noise_reason)
            continue

        stripped_line = strip_stamp_suffixes(line)
        if stripped_line != line:
            add_report_item(report, "rewritten_lines", line, stripped_line, "stamp_suffix")
            line = stripped_line

        line = clean_inline_text(line)
        if line:
            cleaned_lines.append(line)

    report["removed_line_count"] = len(report["removed_lines"])
    report["rewritten_line_count"] = len(report["rewritten_lines"])
    return cleaned_lines, compact_report(report)


def layout_segment_noise_reason(segment: dict[str, Any], pages_with_incoming_stamp: set[Any]) -> Optional[str]:
    text = str(segment.get("text") or "")
    bbox = segment.get("bbox") if isinstance(segment.get("bbox"), dict) else {}
    page = segment.get("page")
    y = safe_float(bbox.get("y"))

    if is_incoming_stamp_line(text):
        return "incoming_stamp_header"
    if page in pages_with_incoming_stamp and y <= 0.45 and is_incoming_stamp_number_line(text):
        return "incoming_stamp_number"
    if page in pages_with_incoming_stamp and y <= 0.55 and is_incoming_stamp_date_line(text):
        return "incoming_stamp_date"

    normalized = normalize_for_noise(text)
    if is_tax_or_round_seal_line(text):
        return "round_seal_artifact"
    if normalized in {"q ba dinh", "tp ha noi"} and y >= 0.20:
        return "round_seal_artifact"
    if looks_like_short_seal_fragment(text) and y >= 0.18:
        return "round_seal_fragment"
    return None


def normalize_ocr_chars(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    replacements = {
        "\ufeff": "",
        "\u200b": "",
        "\u00a0": " ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = normalize_ocr_date_artifacts(text)
    return text


def normalize_ocr_date_artifacts(text: str) -> str:
    def replace_day(match: re.Match) -> str:
        prefix = match.group(1)
        day = int(match.group(2))
        return f"{prefix} {day:02d}"

    text = re.sub(
        r"\b(ngày|ngay)\s*[.:]?\s*[DĐOQ]\s*[.:]?\s*(\d{1,2})(?=\s+tháng|\s+thang)",
        replace_day,
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\b(tháng|thang)\s+(\d{1,2})\s*[.]{1,3}\s*(?=năm|nam)",
        lambda match: f"{match.group(1)} {int(match.group(2))} ",
        text,
        flags=re.IGNORECASE,
    )


def normalize_for_noise(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    value = re.sub(r"[^a-z0-9%/.,:()_-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def clean_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()


def is_incoming_stamp_line(line: str) -> bool:
    normalized = normalize_for_noise(line).strip(" .,:;-")
    if len(normalized) > 70:
        return False
    return normalized in {
        "cong van den",
        "van ban den",
        "ho so den",
        "phieu chuyen",
        "phieu den",
    } or normalized.startswith(("cong van den ", "van ban den ", "ho so den "))


def is_incoming_stamp_number_line(line: str) -> bool:
    normalized = normalize_for_noise(line)
    return bool(
        re.match(r"^(?:so|số)\s*[:.]?\s*\d{1,6}$", line, flags=re.IGNORECASE)
        or re.match(r"^so\s*[:.]?\s*\d{1,6}$", normalized)
        or re.match(r"^(?:so den|den so)\s*[:.]?\s*\d{1,6}$", normalized)
    )


def is_incoming_stamp_date_line(line: str) -> bool:
    return strip_incoming_stamp_date_prefix(line) == ""


def strip_inline_incoming_stamp(line: str) -> tuple[str, Optional[str]]:
    if not is_incoming_stamp_line(line):
        return line, None

    cleaned = re.sub(r"\bC[ÔO]NG\s+V[ĂA]N\s+Đ[ẾE]N\b", " ", line, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bV[ĂA]N\s+B[ẢA]N\s+Đ[ẾE]N\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bH[ỒO]\s+S[ƠO]\s+Đ[ẾE]N\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bS[ỐO]\s*[:.]?\s*\d{1,6}\b", " ", cleaned, flags=re.IGNORECASE)
    return clean_inline_text(cleaned), "incoming_stamp_inline"


def strip_incoming_stamp_date_prefix(line: str) -> Optional[str]:
    pattern = re.compile(
        r"^\s*(?:ngày|ngay)\s*[:.]?\s*(?:[A-ZĐ]\s*[.:]?\s*)?(\d{1,2})\s*"
        r"(?:tháng|thang|/|-)\s*(\d{1,2})\s*(?:năm|nam|/|-)\s*(\d{2,4})\s*"
        r"[.,;:]?\s*(.*)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(line)
    if not match:
        return None

    tail = clean_inline_text(match.group(4))
    if not tail:
        return ""

    normalized_tail = normalize_for_noise(tail)
    if normalized_tail.startswith(("ve ", "v/v", "vv ", "trich yeu", "quyet dinh", "to trinh", "cong van")):
        return tail
    return None


def whole_line_noise_reason(line: str, line_index: int) -> Optional[str]:
    normalized = normalize_for_noise(line).strip(" .,:;-")
    if not normalized:
        return "blank_or_symbol"
    if re.fullmatch(r"\d{1,3}", normalized) and line_index > 6:
        return "page_number"
    if is_tax_or_round_seal_line(line):
        return "round_seal_artifact"
    if normalized in {"q ba dinh", "tp ha noi"} and line_index > 8:
        return "round_seal_artifact"
    if looks_like_short_seal_fragment(line) and line_index > 8:
        return "round_seal_fragment"
    if looks_like_bank_round_seal_line(line) and line_index > 28:
        return "round_seal_fragment"
    return None


def is_tax_or_round_seal_line(line: str) -> bool:
    normalized = normalize_for_noise(line)
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    return (
        "cttnhh" in compact
        or "msdn" in compact
        or "sdn" in compact and bool(re.search(r"\d{6,}", compact))
        or bool(re.search(r"\b[wm.]?s\.?d\.?n\b\s*[:.]?\s*\d", line, flags=re.IGNORECASE))
    )


def looks_like_short_seal_fragment(line: str) -> bool:
    normalized = normalize_for_noise(line).strip(" .,:;-")
    if len(normalized) > 18:
        return False
    fragments = {
        "ngan",
        "ngan hang",
        "nong",
        "nong n",
        "nong nghiep",
        "phat trien",
        "nong thon",
        "viet",
        "viet nam",
        "ang",
        "ghiep",
        "trien",
        "thon",
        "nam",
    }
    if normalized in fragments:
        return True
    return False


def looks_like_bank_round_seal_line(line: str) -> bool:
    normalized = normalize_for_noise(line)
    if len(line) > 70:
        return False
    if not any(
        fragment in normalized
        for fragment in ("ngan hang", "nong nghiep", "phat trien", "nong thon", "viet nam", "ba dinh", "ha noi")
    ):
        return False
    return uppercase_ratio(line) >= 0.65 or bool(re.search(r"^[*=•\d\s.-]+", line))


def strip_stamp_suffixes(line: str) -> str:
    stripped = line
    suffix_patterns = [
        r"\s+(?:Q|[*.=•]+)\s*$",
        r"\s+Q\.\s*BA\s+Đ[ÌI]NH\b.*$",
        r"\s+TP\.\s*H[ÀA]\s+N[ỘO]I\b.*$",
        r"\s+(?:[WM.]?S\.?D\.?N\b\s*[:.]?\s*\d[\d\s.,-]*(?:C\.?T\.?T\.?N\.?H\.?H\.?)?.*)$",
        r"\s+(?:V[ÀA]\s+PH[ÁA]|N[ÔO]NG\s+N)\s*$",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in suffix_patterns:
            updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
            if updated != stripped:
                stripped = updated.strip(" .,:;-")
                changed = True
    return clean_inline_text(stripped)


def uppercase_ratio(line: str) -> float:
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return 0.0
    uppercase_letters = [char for char in letters if char.upper() == char]
    return len(uppercase_letters) / len(letters)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def empty_report() -> dict[str, Any]:
    return {
        "removed_lines": [],
        "rewritten_lines": [],
        "removed_segments": [],
        "removed_line_count": 0,
        "rewritten_line_count": 0,
        "removed_segment_count": 0,
    }


def add_report_item(
    report: dict[str, Any],
    bucket: str,
    original: str,
    cleaned: Optional[str],
    reason: str,
) -> None:
    item = {"text": clean_inline_text(original), "reason": reason}
    if cleaned is not None:
        item["cleaned"] = clean_inline_text(cleaned)
    report.setdefault(bucket, []).append(item)


def merge_report(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("removed_lines", "rewritten_lines", "removed_segments"):
        target.setdefault(key, []).extend(source.get(key) or [])
    for key in ("removed_line_count", "rewritten_line_count", "removed_segment_count"):
        target[key] = int(target.get(key) or 0) + int(source.get(key) or 0)


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "removed_line_count": int(report.get("removed_line_count") or len(report.get("removed_lines") or [])),
        "rewritten_line_count": int(report.get("rewritten_line_count") or len(report.get("rewritten_lines") or [])),
        "removed_segment_count": int(report.get("removed_segment_count") or len(report.get("removed_segments") or [])),
        "removed_lines": (report.get("removed_lines") or [])[:MAX_REPORT_ITEMS],
        "rewritten_lines": (report.get("rewritten_lines") or [])[:MAX_REPORT_ITEMS],
        "removed_segments": (report.get("removed_segments") or [])[:MAX_REPORT_ITEMS],
    }
