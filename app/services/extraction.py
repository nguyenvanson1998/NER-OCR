import asyncio
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any
from typing import Optional

import httpx

CONTRACT_FORMS = [
    "Hợp đồng trọn gói",
    "Hợp đồng theo đơn giá cố định",
    "Hợp đồng theo đơn giá điều chỉnh",
    "Hợp đồng theo thời gian",
    "Hợp đồng theo chi phí cộng phí",
    "Hợp đồng theo tỷ lệ phần trăm",
    "Hợp đồng hỗn hợp",
    "Hợp đồng EPC",
    "Hợp đồng chìa khóa trao tay",
]

CONTRACTOR_GROUPS = [
    "Nhà thầu chính",
    "Nhà thầu phụ",
    "Liên danh nhà thầu",
    "Nhà thầu độc lập",
    "Tổng thầu",
    "Nhà thầu tư vấn",
    "Nhà thầu thi công xây lắp",
    "Nhà thầu cung cấp hàng hóa/thiết bị",
]

DOCUMENT_FIELDS = {
    "document_number": "Số văn bản",
    "signed_or_effective_date": "Thời gian thực hiện / thời gian ký",
    "approved_value": "Giá trị duyệt",
    "submitted_value": "Giá trị trình",
    "issuer": "Đơn vị phát hành / đơn vị thầu",
    "notes": "Ghi chú thông tin lưu ý",
    "title": "Title của tờ trình (tên của tờ trình)",
}

CONTRACT_FIELDS = {
    "work_name": "Tên công việc",
    "contract_name": "Tên hợp đồng",
    "contract_number": "Số hợp đồng",
    "signed_date": "Ngày ký hợp đồng",
    "execution_duration_days": "Thời gian thực hiện (số ngày)",
    "contract_form": "Hình thức hợp đồng",
    "contract_vat_percent": "Giá trị VAT của HĐ (%)",
    "settlement_request_vat_percent": "Giá trị VAT của ĐN quyết toán (%)",
    "appraisal_vat_percent": "Giá trị VAT của thẩm định (%)",
    "estimated_value": "Giá trị dự toán",
    "contract_value": "Giá trị hợp đồng",
    "contractor_group": "Nhóm nhà thầu",
    "contractor_name": "Tên nhà thầu",
    "contractor_contract_amount": "Tiền hợp đồng",
    "performance_guarantee_value": "Giá trị bảo lãnh thực hiện HĐ",
    "performance_guarantee_end_date": "Ngày kết thúc bảo lãnh THHĐ",
    "advance_guarantee_value": "Giá trị bảo lãnh tạm ứng",
    "advance_guarantee_end_date": "Ngày kết thúc bảo lãnh tạm ứng",
}


WORK_DETAIL_CONFIDENCE_FIELDS = (
    "document_number",
    "signed_or_effective_date",
    "issuer",
    "title",
)

LLM_SYSTEM_INSTRUCTION = (
    "Bạn là hệ thống trích xuất thông tin từ OCR tiếng Việt cho hợp đồng, "
    "quyết định, biên bản họp, tờ trình, hóa đơn, công văn trong dự án xây dựng. "
    "Chỉ trả JSON hợp lệ, không markdown. Nếu thiếu thông tin, để value là null "
    "và vẫn trả generic_extraction. "
    "QUY TẮC NGÀY THÁNG (BẮT BUỘC): với mọi field ngày (signed_or_effective_date, "
    "signed_date, performance_guarantee_end_date, advance_guarantee_end_date): "
    "đặt value theo định dạng DD/MM/YYYY và normalized_value theo định dạng "
    "ISO YYYY-MM-DD. Tuyệt đối KHÔNG để value là nguyên văn 'ngày XX tháng YY năm ZZZZ'. "
    "Nếu chỉ có ngày + tháng mà thiếu năm, để cả hai field là null."
)

CONTRACT_EXTRACTION_GUIDANCE = """
Luật bóc tách hợp đồng:
- Chỉ trích xuất thông tin có căn cứ trong tài liệu, không tự suy diễn hoặc tự tạo dữ liệu.
- Ưu tiên trang bìa, phần căn cứ, thành phần ký hợp đồng, điều khoản giá hợp đồng, thời gian thực hiện, thanh toán và bảo lãnh.
- Nếu cùng một thông tin xuất hiện nhiều lần nhưng khác nhau, ưu tiên điều khoản chính trong hợp đồng, sau đó trang ký kết/thông tin hợp đồng, phụ lục, cuối cùng là văn bản dẫn chiếu.
- signed_date (ngày ký hợp đồng): BẮT BUỘC lấy NGÀY CÓ ĐỊA ĐIỂM Ở ĐẦU HOẶC CUỐI HỢP ĐỒNG. PHẢI tìm dạng "Hà Nội, ngày 16 tháng 12 năm 2021", "..., hôm nay ngày...". PHẢI có địa điểm + dấu phẩy + ngày tháng năm. TUYỆT ĐỐI KHÔNG lấy ngày từ văn bản pháp lý như "Căn cứ Luật...", "Nghị định...". Output BẮT BUỘC: value="DD/MM/YYYY", normalized_value="YYYY-MM-DD".
- Tất cả các field ngày (signed_date, performance_guarantee_end_date, advance_guarantee_end_date): value = "DD/MM/YYYY", normalized_value = "YYYY-MM-DD". Không trả nguyên văn "ngày XX tháng YY năm ZZZZ".
- Giá trị tiền chuẩn hóa về số VND, không kèm chữ đồng.
- Tỷ lệ phần trăm chuẩn hóa về số, ví dụ 10 thay vì 10%.
- Tên công việc là field mapping: hiểu bản chất công việc rồi đề xuất task/subtask/subsubtask phù hợp, không lấy nguyên văn máy móc nếu tiêu đề quá dài.
- Hình thức hợp đồng, nhóm nhà thầu, tên nhà thầu là danh mục/mapping động; đề xuất giá trị phù hợp nhất và confidence, không hard-code ngoài danh mục được truyền vào.
- Giá trị dự toán không bắt buộc; chỉ lấy nếu có cụm như giá gói thầu, dự toán được duyệt, tổng mức dự toán, giá trị dự toán.
- Giá trị hợp đồng lấy từ điều khoản Giá hợp đồng/Giá trị hợp đồng. Nếu chỉ có một nhà thầu, tiền hợp đồng của nhà thầu bằng giá trị hợp đồng.
- Không tự tính bảo lãnh từ tỷ lệ phần trăm nếu tài liệu không nêu số tiền/ngày cụ thể.
""".strip()


def empty_field(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "value": None,
        "normalized_value": None,
        "evidence": None,
        "confidence": 0.0,
        "source": None,
    }


def normalize_result(payload: dict[str, Any], text: str, payload_source: str = "llm") -> dict[str, Any]:
    text = clean_ocr_text(text)
    detected_type = payload.get("document_type")
    if detected_type not in {"contract", "document"}:
        detected_type = classify_document(text)
    detected_intent = payload.get("document_intent") or classify_document_intent(text)

    field_schema = CONTRACT_FIELDS if detected_type == "contract" else DOCUMENT_FIELDS
    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    normalized_fields = {key: empty_field(label) for key, label in field_schema.items()}

    heuristic_notes: list[str] = []
    if detected_type == "document":
        heuristic_fields, heuristic_notes = heuristic_document_fields(text)
        for key, field in heuristic_fields.items():
            if key in normalized_fields:
                normalized_fields[key] = {**empty_field(field_schema[key]), **field}
    elif detected_type == "contract":
        heuristic_fields, heuristic_notes = heuristic_contract_fields(text)
        for key, field in heuristic_fields.items():
            if key in normalized_fields:
                normalized_fields[key] = {**empty_field(field_schema[key]), **field}

    for key, label in field_schema.items():
        raw = fields.get(key)
        if isinstance(raw, dict):
            candidate = {
                **empty_field(label),
                **raw,
                "label": raw.get("label") or label,
                "source": raw.get("source") or payload_source,
            }
            candidate = normalize_candidate_field(key, candidate)
            if should_replace_field(normalized_fields[key], candidate):
                normalized_fields[key] = candidate
        elif raw not in (None, ""):
            candidate = {
                **empty_field(label),
                "value": raw,
                "normalized_value": raw,
                "confidence": 0.4,
                "source": payload_source,
            }
            candidate = normalize_candidate_field(key, candidate)
            if should_replace_field(normalized_fields[key], candidate):
                normalized_fields[key] = candidate

    generic = payload.get("generic_extraction")
    if not isinstance(generic, dict):
        generic = heuristic_generic_extraction(text)
    payload_entities = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    payload_notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
    heuristic_generic = heuristic_generic_extraction(text)
    non_empty_generic = {key: value for key, value in generic.items() if value not in (None, "", [])}
    non_empty_entities = {key: value for key, value in payload_entities.items() if value not in (None, "", [])}

    result = {
        "document_type": detected_type,
        "document_type_label": "Hợp đồng" if detected_type == "contract" else "Tài liệu",
        "screen": "contract" if detected_type == "contract" else "work_detail",
        "document_intent": detected_intent,
        "fields": normalized_fields,
        "generic_extraction": {
            **heuristic_generic,
            **non_empty_generic,
        },
        "entities": non_empty_entities,
        "notes": unique_values([*heuristic_notes, *[str(note) for note in payload_notes if note]]),
    }
    result["local_confidence"] = calculate_local_confidence(result)
    result["pipeline"] = "local"
    result["llm_fallback_used"] = False
    result["llm_entity_extraction_used"] = False
    result["needs_review"] = result["local_confidence"] < env_float("LOCAL_CONFIDENCE_THRESHOLD", 0.68)
    return result


def classify_document(text: str) -> str:
    intent = classify_document_intent(text)
    if intent == "contract":
        return "contract"
    if intent != "unknown":
        return "document"

    sample = text[:5000].lower()
    normalized_sample = normalize_for_rules(sample)
    if "hop dong" in normalized_sample:
        return "contract"
    return "document"


def classify_document_intent(text: str) -> str:
    text = clean_ocr_text(text)
    sample = text[:5000].lower()
    document_title_prefixes = ("to trinh", "quyet dinh", "cong van", "bien ban", "bao cao", "thong bao")
    for line in meaningful_lines(text, limit=30):
        normalized_line = normalize_for_rules(line)
        for prefix in document_title_prefixes:
            if normalized_line.startswith(prefix):
                return prefix.replace(" ", "_")

    contract_hits = [
        "hợp đồng",
        "bên giao thầu",
        "bên nhận thầu",
        "giá trị hợp đồng",
        "bảo lãnh thực hiện",
        "tạm ứng hợp đồng",
    ]
    if sum(hit in sample for hit in contract_hits) >= 2:
        return "contract"
    return "unknown"


def heuristic_generic_extraction(text: str) -> dict[str, Any]:
    date_pattern = r"\b(?:ngày\s*)?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}"
    money_pattern = r"\b\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?\s*(?:đồng|vnđ|vnd)?\b|\b\d+(?:,\d+)?\s*(?:tỷ|triệu)\s*(?:đồng)?\b"
    doc_no_pattern = r"(?:số|so)\s*[:.]?\s*([A-Z0-9ĐĐa-zÀ-ỹ/._-]{2,})"
    approver_pattern = r"(?:chủ tịch|giám đốc|phó giám đốc|trưởng phòng|kt\.|ký bởi|người ký|đại diện)\s*[:.]?\s*([A-ZÀ-Ỹ][^\n]{2,80})"

    return {
        "document_title_or_type": guess_title(text),
        "project_name_candidates": guess_project_name_candidates(text),
        "dates": unique_matches(date_pattern, text, limit=20),
        "monetary_amounts": unique_matches(money_pattern, text, limit=20),
        "document_numbers": unique_matches(doc_no_pattern, text, limit=10, flags=re.IGNORECASE),
        "approvers_or_positions": unique_matches(approver_pattern, text, limit=10, flags=re.IGNORECASE),
    }


def guess_title(text: str) -> Optional[str]:
    title = extract_document_title(text)
    if title:
        return title

    title_starts = (
        "hop dong",
        "cong van",
        "quyet dinh",
        "van ban",
        "to trinh",
        "bien ban",
        "bao cao",
        "thong bao",
        "de nghi",
    )
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        if len(line) > 160:
            continue
        normalized = normalize_for_rules(line)
        if not normalized.startswith(title_starts):
            continue
        if len(line.split()) <= 4:
            continue
        return clean_text(line)
    return None


def heuristic_document_fields(text: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    title = extract_document_title(text)
    notes = extract_notes(text)
    fields: dict[str, dict[str, Any]] = {}

    document_number = extract_document_number(text)
    if document_number:
        value, evidence = document_number
        fields["document_number"] = build_field(
            DOCUMENT_FIELDS["document_number"],
            value=value,
            normalized_value=value,
            evidence=evidence,
            confidence=0.86,
        )

    signed_or_effective_date = extract_signed_or_effective_date(text)
    if signed_or_effective_date:
        value, normalized_value, evidence, confidence = signed_or_effective_date
        fields["signed_or_effective_date"] = build_field(
            DOCUMENT_FIELDS["signed_or_effective_date"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    approved_value = extract_labeled_money(text, MONEY_LABELS_APPROVED)
    if approved_value:
        value, normalized_value, evidence, confidence = approved_value
        fields["approved_value"] = build_field(
            DOCUMENT_FIELDS["approved_value"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    submitted_value = extract_labeled_money(text, MONEY_LABELS_SUBMITTED)
    if submitted_value:
        value, normalized_value, evidence, confidence = submitted_value
        fields["submitted_value"] = build_field(
            DOCUMENT_FIELDS["submitted_value"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    issuer = extract_issuer(text)
    if issuer:
        value, evidence = issuer
        fields["issuer"] = build_field(
            DOCUMENT_FIELDS["issuer"],
            value=value,
            normalized_value=value,
            evidence=evidence,
            confidence=0.82,
        )

    if notes:
        note_value = "; ".join(notes[:5])
        fields["notes"] = build_field(
            DOCUMENT_FIELDS["notes"],
            value=note_value,
            normalized_value=notes[:5],
            evidence=notes[0],
            confidence=0.66,
        )

    if title:
        fields["title"] = build_field(
            DOCUMENT_FIELDS["title"],
            value=title,
            normalized_value=title,
            evidence=title,
            confidence=0.84,
        )

    return fields, notes


def heuristic_contract_fields(text: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    fields: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    contract_name = extract_contract_name(text)
    if contract_name:
        fields["contract_name"] = build_field(
            CONTRACT_FIELDS["contract_name"],
            value=contract_name,
            normalized_value=contract_name,
            evidence=contract_name,
            confidence=0.84,
        )

        work_name = infer_work_name_from_contract_title(contract_name)
        if work_name:
            fields["work_name"] = build_field(
                CONTRACT_FIELDS["work_name"],
                value=work_name,
                normalized_value=work_name,
                evidence=contract_name,
                confidence=0.68,
            )

    contract_number = extract_contract_number(text)
    if contract_number:
        value, evidence = contract_number
        fields["contract_number"] = build_field(
            CONTRACT_FIELDS["contract_number"],
            value=value,
            normalized_value=value,
            evidence=evidence,
            confidence=0.86,
        )

    signed_date = extract_contract_signed_date(text)
    if signed_date:
        value, normalized_value, evidence, confidence = signed_date
        fields["signed_date"] = build_field(
            CONTRACT_FIELDS["signed_date"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    duration = extract_duration_days(text)
    if duration:
        value, normalized_value, evidence, confidence = duration
        fields["execution_duration_days"] = build_field(
            CONTRACT_FIELDS["execution_duration_days"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    contract_form = extract_contract_form(text)
    if contract_form:
        value, evidence, confidence = contract_form
        fields["contract_form"] = build_field(
            CONTRACT_FIELDS["contract_form"],
            value=value,
            normalized_value=value,
            evidence=evidence,
            confidence=confidence,
        )

    estimated_value = extract_labeled_money(text, CONTRACT_ESTIMATED_VALUE_LABELS)
    if estimated_value:
        value, normalized_value, evidence, confidence = estimated_value
        fields["estimated_value"] = build_field(
            CONTRACT_FIELDS["estimated_value"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    contract_value = extract_labeled_money(text, CONTRACT_VALUE_LABELS)
    if contract_value:
        value, normalized_value, evidence, confidence = contract_value
        fields["contract_value"] = build_field(
            CONTRACT_FIELDS["contract_value"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    contractor_name = extract_contractor_name(text)
    if contractor_name:
        value, evidence, confidence = contractor_name
        fields["contractor_name"] = build_field(
            CONTRACT_FIELDS["contractor_name"],
            value=value,
            normalized_value=value,
            evidence=evidence,
            confidence=confidence,
        )

    contractor_group = infer_contractor_group(text, contract_name or "")
    if contractor_group:
        value, evidence, confidence = contractor_group
        fields["contractor_group"] = build_field(
            CONTRACT_FIELDS["contractor_group"],
            value=value,
            normalized_value=value,
            evidence=evidence,
            confidence=confidence,
        )

    if fields.get("contractor_name", {}).get("value") and fields.get("contract_value", {}).get("normalized_value"):
        contract_value_field = fields["contract_value"]
        fields["contractor_contract_amount"] = build_field(
            CONTRACT_FIELDS["contractor_contract_amount"],
            value=contract_value_field["value"],
            normalized_value=contract_value_field["normalized_value"],
            evidence=contract_value_field["evidence"],
            confidence=0.72,
        )

    contract_vat = extract_labeled_percent(text, CONTRACT_VAT_LABELS)
    if contract_vat:
        value, normalized_value, evidence, confidence = contract_vat
        fields["contract_vat_percent"] = build_field(
            CONTRACT_FIELDS["contract_vat_percent"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    settlement_vat = extract_labeled_percent(text, SETTLEMENT_VAT_LABELS)
    if settlement_vat:
        value, normalized_value, evidence, confidence = settlement_vat
        fields["settlement_request_vat_percent"] = build_field(
            CONTRACT_FIELDS["settlement_request_vat_percent"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    appraisal_vat = extract_labeled_percent(text, APPRAISAL_VAT_LABELS)
    if appraisal_vat:
        value, normalized_value, evidence, confidence = appraisal_vat
        fields["appraisal_vat_percent"] = build_field(
            CONTRACT_FIELDS["appraisal_vat_percent"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    performance_guarantee = extract_labeled_money(text, PERFORMANCE_GUARANTEE_LABELS)
    if performance_guarantee:
        value, normalized_value, evidence, confidence = performance_guarantee
        fields["performance_guarantee_value"] = build_field(
            CONTRACT_FIELDS["performance_guarantee_value"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    performance_end_date = extract_labeled_date(text, PERFORMANCE_GUARANTEE_END_LABELS)
    if performance_end_date:
        value, normalized_value, evidence, confidence = performance_end_date
        fields["performance_guarantee_end_date"] = build_field(
            CONTRACT_FIELDS["performance_guarantee_end_date"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    advance_guarantee = extract_labeled_money(text, ADVANCE_GUARANTEE_LABELS)
    if advance_guarantee:
        value, normalized_value, evidence, confidence = advance_guarantee
        fields["advance_guarantee_value"] = build_field(
            CONTRACT_FIELDS["advance_guarantee_value"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    advance_end_date = extract_labeled_date(text, ADVANCE_GUARANTEE_END_LABELS)
    if advance_end_date:
        value, normalized_value, evidence, confidence = advance_end_date
        fields["advance_guarantee_end_date"] = build_field(
            CONTRACT_FIELDS["advance_guarantee_end_date"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    return fields, notes


def build_field(
    label: str,
    value: Any,
    normalized_value: Any,
    evidence: Optional[str],
    confidence: float,
    source: str = "rule",
) -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "normalized_value": normalized_value,
        "evidence": evidence,
        "confidence": round(confidence, 4),
        "source": source,
    }


def normalize_candidate_field(key: str, field: dict[str, Any]) -> dict[str, Any]:
    value = field.get("value")
    normalized_value = field.get("normalized_value")
    source_value = normalized_value if normalized_value not in (None, "") else value

    money_fields = {
        "approved_value",
        "submitted_value",
        "estimated_value",
        "contract_value",
        "contractor_contract_amount",
        "performance_guarantee_value",
        "advance_guarantee_value",
    }
    percent_fields = {
        "contract_vat_percent",
        "settlement_request_vat_percent",
        "appraisal_vat_percent",
    }

    if key in money_fields and isinstance(source_value, str):
        parsed_money = normalize_money(source_value)
        if parsed_money is not None:
            field["normalized_value"] = parsed_money

    if key in percent_fields and isinstance(source_value, str):
        percent = find_first_percent(source_value)
        if percent:
            field["normalized_value"] = percent[1]

    if key == "execution_duration_days" and isinstance(source_value, str):
        duration = re.search(r"(\d{1,4})\s*ngày", source_value, flags=re.IGNORECASE)
        if duration:
            field["normalized_value"] = int(duration.group(1))

    if key in {"signed_or_effective_date", "signed_date", "performance_guarantee_end_date", "advance_guarantee_end_date"}:
        # Prefer the raw value (likely Vietnamese phrasing) over a pre-normalized ISO string,
        # because extract_first_date matches dd-mm-yy and would mis-parse "2021-12-16" as 21/12/2016.
        date_source = value if isinstance(value, str) and value.strip() else source_value
        if isinstance(date_source, str):
            iso_match = re.match(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$", date_source)
            if iso_match:
                year, month, day = iso_match.groups()
                normalized_date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            else:
                date = extract_first_date(date_source)
                normalized_date = normalize_date(date or date_source)
            if normalized_date:
                field["normalized_value"] = normalized_date
                display = format_display_date(normalized_date)
                if display:
                    field["value"] = display

    return field


def should_replace_field(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    # If candidate has no value, don't replace
    if candidate.get("value") in (None, ""):
        return False

    # If existing has no value, always replace with candidate
    if existing.get("value") in (None, ""):
        return True

    existing_confidence = safe_float(existing.get("confidence"))
    candidate_confidence = safe_float(candidate.get("confidence"))
    existing_source = existing.get("source", "rule")
    candidate_source = candidate.get("source", "rule")

    # Priority 1: LLM source should win over local/rule when confidence is close
    if candidate_source == "llm" and existing_source == "rule":
        # LLM wins if confidence is within 0.15 or higher
        if candidate_confidence >= existing_confidence - 0.15:
            return True
        return False

    # Priority 2: If both same source, require significant confidence gain
    if candidate_confidence >= existing_confidence + 0.08:
        return True

    # Priority 3: Prefer deterministic parsers for layout-sensitive fields when confidence is close
    if candidate.get("label") == DOCUMENT_FIELDS.get("issuer") and existing_confidence >= 0.7:
        return False

    return False


def extract_document_number(text: str) -> Optional[tuple[str, str]]:
    for line in meaningful_lines(text, limit=60):
        normalized_line = normalize_for_rules(line)
        if not re.match(r"^(so|số)\b", normalized_line):
            continue
        if any(skip in normalized_line for skip in ("tai khoan", "dien thoai", "cmnd", "cccd")):
            continue

        match = re.search(r"(?:số|so)\s*[:.]?\s*([^\n\r]{2,80})", line, flags=re.IGNORECASE)
        if not match:
            continue
        value = clean_text(match.group(1))
        value = re.split(r"\s+(?:ngày|ngay)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
        value = value.strip(" .,:;-")
        if re.search(r"\d", value):
            return value, line
    return None


def extract_signed_or_effective_date(text: str) -> Optional[tuple[str, Any, str, float]]:
    # Priority 1: Date at document header (e.g., "Hà Nội, ngày 16 tháng 12 năm 2021")
    # This pattern is most reliable for official documents
    header_text = text[:1500]  # Search in first 1500 chars where document header usually is
    header_pattern = r"(?:[\w\s.]+,\s*)?ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})"
    header_match = re.search(header_pattern, header_text, flags=re.IGNORECASE)
    if header_match:
        day, month, year = header_match.groups()
        normalized = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        display = f"{int(day):02d}/{int(month):02d}/{int(year):04d}"
        evidence = clean_text(header_match.group(0))
        return display, normalized, evidence, 0.94

    # Priority 2: Explicitly labeled dates (ngày ký, ngày ban hành, etc.)
    # But exclude dates from legal references (Luật, Nghị định, Thông tư, etc.)
    labeled_patterns = [
        r"(?:ngày\s+ký|ngày\s+hiệu\s+lực|ngày\s+ban\s+hành|thời\s+gian\s+ký)\s*[:\-]?\s*([^\n]{4,80})",
        r"(?:thời\s+gian\s+thực\s+hiện(?:\s+hợp\s+đồng)?|tiến\s+độ\s+thực\s+hiện)\s*[:\-]?\s*([^\n]{4,120})",
    ]
    for pattern in labeled_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = clean_text(match.group(1))

        # Skip if this looks like a legal reference (contains Luật, Nghị định, etc.)
        context = text[max(0, match.start() - 100):match.end() + 50].lower()
        if any(keyword in context for keyword in ["luật", "nghị định", "nghị quyết", "thông tư", "quyết định số", "căn cứ"]):
            continue

        date = extract_first_date(raw)
        if date:
            normalized = normalize_date(date)
            display = format_display_date(normalized) if normalized else None
            return display or date, normalized or date, clean_text(match.group(0)), 0.86
        return raw, raw, clean_text(match.group(0)), 0.72

    # Priority 3: Date in meaningful lines (fallback)
    for line in meaningful_lines(text, limit=80):
        if "ngày" not in line.lower() and not re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", line):
            continue

        # Skip lines that look like legal references
        if any(keyword in line.lower() for keyword in ["luật", "nghị định", "nghị quyết", "thông tư", "căn cứ"]):
            continue

        date = extract_first_date(line)
        if date:
            normalized = normalize_date(date)
            display = format_display_date(normalized) if normalized else None
            return display or date, normalized or date, line, 0.74
    return None


MONEY_PATTERN = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[,.]\d+)?)\s*(?:đồng|vnđ|vnd|tỷ|triệu)?",
    flags=re.IGNORECASE,
)

MONEY_LABELS_APPROVED = [
    "giá gói thầu",
    "giá trị gói thầu",
    "giá dự toán gói thầu",
    "giá gói thầu được duyệt",
    "giá trị gói thầu được duyệt",
    "giá trị dự toán gói thầu",
    "giá trị phê duyệt",
    "giá trị được duyệt",
    "giá trị duyệt",
    "giá trị sau thẩm định",
    "giá trị thẩm định",
    "dự toán được duyệt",
    "dự toán phê duyệt",
    "tổng mức đầu tư được phê duyệt",
    "tổng mức đầu tư phê duyệt",
]

MONEY_LABELS_SUBMITTED = [
    "giá trị trình",
    "giá trị đề nghị",
    "giá trị xin phê duyệt",
    "giá trình duyệt",
    "giá trị trình duyệt",
    "giá trị dự toán trình duyệt",
    "giá dự toán trình duyệt",
    "dự toán gói thầu trình duyệt",
    "dự toán trình",
    "dự toán đề nghị",
    "tổng mức đầu tư trình",
    "tổng mức đầu tư đề nghị",
    "kinh phí đề nghị",
    "giá trị đề nghị phê duyệt",
]

CONTRACT_ESTIMATED_VALUE_LABELS = [
    "giá gói thầu",
    "giá trị gói thầu",
    "giá dự toán gói thầu",
    "dự toán được duyệt",
    "tổng mức dự toán",
    "giá trị dự toán",
    "tổng dự toán",
]

CONTRACT_VALUE_LABELS = [
    "giá trị hợp đồng",
    "giá hợp đồng",
    "tổng giá trị hợp đồng",
    "giá trị ký kết",
    "giá trị hđ",
]

CONTRACT_VAT_LABELS = [
    "thuế vat",
    "thuế gtgt",
    "vat",
    "gtgt",
]

SETTLEMENT_VAT_LABELS = [
    "vat đề nghị quyết toán",
    "gtgt đề nghị quyết toán",
    "thuế vat quyết toán",
    "thuế gtgt quyết toán",
]

APPRAISAL_VAT_LABELS = [
    "vat thẩm định",
    "gtgt thẩm định",
    "thuế vat thẩm định",
    "thuế gtgt thẩm định",
]

PERFORMANCE_GUARANTEE_LABELS = [
    "bảo lãnh thực hiện hợp đồng",
    "bảo đảm thực hiện hợp đồng",
    "đảm bảo thực hiện hợp đồng",
]

PERFORMANCE_GUARANTEE_END_LABELS = [
    "bảo lãnh thực hiện hợp đồng",
    "bảo đảm thực hiện hợp đồng",
    "đảm bảo thực hiện hợp đồng",
]

ADVANCE_GUARANTEE_LABELS = [
    "bảo lãnh tiền tạm ứng",
    "bảo lãnh tạm ứng",
    "bảo đảm tiền tạm ứng",
    "bảo đảm tạm ứng",
]

ADVANCE_GUARANTEE_END_LABELS = [
    "bảo lãnh tiền tạm ứng",
    "bảo lãnh tạm ứng",
    "bảo đảm tiền tạm ứng",
    "bảo đảm tạm ứng",
]


def extract_contract_name(text: str) -> Optional[str]:
    lines = meaningful_lines(text, limit=120)
    for index, line in enumerate(lines):
        normalized = normalize_for_rules(line)
        if "hop dong" not in normalized:
            continue
        if normalized.startswith(("can cu", "dieu ", "gia ", "ben ")):
            continue

        title_lines = [line]
        for next_line in lines[index + 1 : index + 7]:
            normalized_next = normalize_for_rules(next_line)
            if re.match(r"^(so|số)\b", normalized_next):
                break
            if normalized_next.startswith(("hom nay", "can cu", "ben a", "ben b", "chu dau tu", "nha thau", "dieu ")):
                break
            if len(next_line) <= 180 and is_probable_contract_title_continuation(next_line):
                title_lines.append(next_line)
                continue
            if len(title_lines) > 1:
                break

        title = clean_text(" ".join(title_lines))
        if len(title) >= 8:
            return title[:500]
    return None


def is_probable_contract_title_continuation(line: str) -> bool:
    normalized = normalize_for_rules(line)
    if any(keyword in normalized for keyword in ("tu van", "thiet ke", "thi cong", "du toan", "xay dung", "mua sam", "goi thau", "cong trinh", "du an")):
        return True
    letters = [char for char in line if char.isalpha()]
    uppercase_letters = [char for char in letters if char.upper() == char]
    return bool(letters) and len(line) <= 130 and len(uppercase_letters) / max(len(letters), 1) >= 0.6


def infer_work_name_from_contract_title(contract_name: str) -> Optional[str]:
    value = re.sub(r"\b(hợp\s*đồng|hop\s*dong)\b", "", contract_name, flags=re.IGNORECASE)
    value = re.split(r"\b(?:số|so)\b\s*[:.]?", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = clean_text(value.strip(" -:;,."))
    return value[:260] if len(value) >= 6 else None


def extract_contract_number(text: str) -> Optional[tuple[str, str]]:
    patterns = [
        r"(?:số\s+hợp\s+đồng|số\s+hđ|số|so)\s*[:.]?\s*([A-Z0-9Đa-zÀ-ỹ/._-]{2,80})",
    ]
    for line in meaningful_lines(text, limit=90):
        normalized = normalize_for_rules(line)
        if "hop dong" not in normalized and not re.match(r"^(so|số)\b", normalized):
            continue
        if any(skip in normalized for skip in ("tai khoan", "dien thoai", "ma so thue")):
            continue
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if not match:
                continue
            value = clean_text(match.group(1))
            value = re.split(r"\s+(?:ngày|ngay)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
            value = value.strip(" .,:;-")
            if re.search(r"\d", value) and ("hd" in normalize_for_rules(value) or "hđ" in value.lower() or "/" in value):
                return value, line
    return None


def extract_contract_signed_date(text: str) -> Optional[tuple[str, Any, str, float]]:
    patterns = [
        r"(hôm\s+nay\s*,?\s*ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
        r"(?:ký\s+ngày|ngày\s+ký|ngày\s+lập\s+hợp\s+đồng)\s*[:\-]?\s*([^\n]{4,100})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = clean_text(match.group(1))
        date = extract_first_date(raw)
        if date:
            normalized = normalize_date(date)
            display = format_display_date(normalized) if normalized else None
            return display or date, normalized or date, clean_text(match.group(0)), 0.88

    date = extract_signed_or_effective_date(text)
    if date:
        value, normalized_value, evidence, _confidence = date
        return value, normalized_value, evidence, 0.72
    return None


def extract_duration_days(text: str) -> Optional[tuple[str, Any, str, float]]:
    patterns = [
        r"(?:thời\s+gian(?:\s+và\s+tiến\s+độ)?\s+thực\s+hiện(?:\s+hợp\s+đồng)?|tiến\s+độ\s+thực\s+hiện)[^\n.]{0,160}?(\d{1,4})\s*ngày",
        r"(\d{1,4})\s*ngày\s+kể\s+từ\s+ngày\s+hợp\s+đồng",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = f"{match.group(1)} ngày"
            return value, int(match.group(1)), first_meaningful_line(match.group(0)), 0.84
    return None


def extract_contract_form(text: str) -> Optional[tuple[str, str, float]]:
    normalized_text = normalize_for_rules(text)
    form_aliases = [
        ("Hợp đồng trọn gói", ("hop dong tron goi", "tron goi")),
        ("Hợp đồng theo đơn giá cố định", ("don gia co dinh",)),
        ("Hợp đồng theo đơn giá điều chỉnh", ("don gia dieu chinh",)),
        ("Hợp đồng theo thời gian", ("theo thoi gian",)),
        ("Hợp đồng theo tỷ lệ phần trăm", ("theo ty le", "ty le phan tram")),
    ]
    for form, aliases in form_aliases:
        for alias in aliases:
            idx = normalized_text.find(alias)
            if idx == -1:
                continue
            raw_idx = approximate_raw_index(text, normalized_text, idx)
            evidence = first_meaningful_line(text[max(0, raw_idx - 120) : raw_idx + 180])
            return form, evidence, 0.82
    return None


def extract_contractor_name(text: str) -> Optional[tuple[str, str, float]]:
    patterns = [
        r"(?:bên\s+b|nhà\s+thầu|bên\s+nhận\s+thầu|đại\s+diện\s+bên\s+b)\s*[:\-]?\s*([A-ZÀ-Ỹ][^\n]{5,180})",
        r"(?:tên\s+nhà\s+thầu|tên\s+bên\s+b)\s*[:\-]?\s*([A-ZÀ-Ỹ][^\n]{5,180})",
    ]
    for line in meaningful_lines(text, limit=240):
        normalized = normalize_for_rules(line)
        if not any(keyword in normalized for keyword in ("ben b", "nha thau", "ben nhan thau")):
            continue
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if not match:
                continue
            value = clean_contractor_candidate(match.group(1))
            if value:
                return value, line, 0.82

    lines = meaningful_lines(text, limit=260)
    for index, line in enumerate(lines):
        normalized = normalize_for_rules(line)
        if normalized.startswith(("ben b", "nha thau", "ben nhan thau")):
            for next_line in lines[index + 1 : index + 5]:
                if is_probable_issuer_line(next_line):
                    return clean_text(next_line), next_line, 0.72
    return None


def clean_contractor_candidate(value: str) -> str:
    value = clean_text(value)
    value = re.split(
        r"\s+(?:mã\s+số\s+thuế|mst|địa\s+chỉ|đại\s+diện|chức\s+vụ|tài\s+khoản|số\s+tài\s+khoản)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = value.strip(" .,:;-")
    return value[:220] if len(value) >= 5 else ""


def infer_contractor_group(text: str, contract_name: str) -> Optional[tuple[str, str, float]]:
    source = f"{contract_name}\n{text[:6000]}"
    normalized = normalize_for_rules(source)
    group_aliases = [
        ("Nhà thầu tư vấn", ("tu van", "thiet ke", "giam sat", "lap du toan", "tham tra")),
        ("Nhà thầu thi công xây lắp", ("thi cong", "xay lap", "xay dung")),
        ("Nhà thầu cung cấp hàng hóa/thiết bị", ("mua sam hang hoa", "cung cap thiet bi", "mua sam thiet bi")),
        ("Nhà thầu phi tư vấn", ("phi tu van", "bao tri", "bao duong", "kiem dinh")),
    ]
    for group, aliases in group_aliases:
        if any(alias in normalized for alias in aliases):
            return group, contract_name or first_meaningful_line(text), 0.58
    return None


def extract_labeled_money(text: str, labels: list[str]) -> Optional[tuple[str, Any, str, float]]:
    normalized_text = normalize_for_rules(text)
    best: Optional[tuple[str, Any, str, float]] = None
    best_index = len(text) + 1

    for label in labels:
        normalized_label = normalize_for_rules(label)
        start = 0
        while True:
            idx = normalized_text.find(normalized_label, start)
            if idx == -1:
                break
            raw_idx = approximate_raw_index(text, normalized_text, idx)
            window = text[raw_idx : raw_idx + 320]
            money = find_first_money(window)
            if money:
                value, normalized_value = money
                evidence = first_meaningful_line(window)
                candidate = (value, normalized_value, evidence, 0.84)
                if idx < best_index:
                    best = candidate
                    best_index = idx
            start = idx + len(normalized_label)

    if best:
        return best

    for line in meaningful_lines(text, limit=200):
        normalized_line = normalize_for_rules(line)
        if any(normalize_for_rules(label) in normalized_line for label in labels):
            money = find_first_money(line)
            if money:
                value, normalized_value = money
                return value, normalized_value, line, 0.78
    return None


PERCENT_PATTERN = re.compile(r"(?<!\d)(\d{1,2}(?:[,.]\d+)?)\s*%", flags=re.IGNORECASE)


def extract_labeled_percent(text: str, labels: list[str]) -> Optional[tuple[str, Any, str, float]]:
    normalized_text = normalize_for_rules(text)
    for label in labels:
        normalized_label = normalize_for_rules(label)
        start = 0
        while True:
            idx = normalized_text.find(normalized_label, start)
            if idx == -1:
                break
            raw_idx = approximate_raw_index(text, normalized_text, idx)
            window = text[max(0, raw_idx - 100) : raw_idx + 280]
            percent = find_first_percent(window)
            if percent:
                value, normalized_value = percent
                return value, normalized_value, first_meaningful_line(window), 0.82
            start = idx + len(normalized_label)
    return None


def find_first_percent(text: str) -> Optional[tuple[str, Any]]:
    for match in PERCENT_PATTERN.finditer(text):
        raw = clean_text(match.group(0))
        normalized_number = match.group(1).replace(",", ".")
        try:
            value = float(normalized_number)
        except ValueError:
            continue
        if value.is_integer():
            return raw, int(value)
        return raw, value
    return None


def extract_labeled_date(text: str, labels: list[str]) -> Optional[tuple[str, Any, str, float]]:
    normalized_text = normalize_for_rules(text)
    for label in labels:
        normalized_label = normalize_for_rules(label)
        start = 0
        while True:
            idx = normalized_text.find(normalized_label, start)
            if idx == -1:
                break
            raw_idx = approximate_raw_index(text, normalized_text, idx)
            window = text[raw_idx : raw_idx + 420]
            date = extract_first_date(window)
            if date:
                normalized = normalize_date(date)
                display = format_display_date(normalized) if normalized else None
                return display or date, normalized or date, first_meaningful_line(window), 0.76
            start = idx + len(normalized_label)
    return None


def extract_issuer(text: str) -> Optional[tuple[str, str]]:
    lines = meaningful_lines(text, limit=35)
    if not lines:
        return None

    stop_index = len(lines)
    for index, line in enumerate(lines):
        normalized = normalize_for_rules(line)
        if normalized.startswith(("cong hoa", "doc lap")) or "to trinh" in normalized or "quyet dinh" in normalized:
            stop_index = min(stop_index, index)
        if re.match(r"^(so|số)\b", normalized):
            stop_index = min(stop_index, index)
            break

    candidates: list[str] = []
    for line in lines[: max(stop_index, 1)]:
        normalized = normalize_for_rules(line)
        if not line or len(line) < 3:
            continue
        if normalized.startswith(("cong hoa", "doc lap", "so ", "so:", "ngay ")):
            continue
        if is_probable_issuer_line(line):
            candidates.append(line)
        if len(candidates) >= 4:
            break

    if not candidates:
        for line in lines[:12]:
            if is_probable_issuer_line(line):
                candidates.append(line)
                if len(candidates) >= 3:
                    break

    if not candidates:
        return None

    issuer_lines = candidates[:4]
    issuer = clean_text(" ".join(issuer_lines))
    return issuer, " / ".join(issuer_lines)


def extract_document_title(text: str) -> Optional[str]:
    lines = meaningful_lines(text, limit=90)
    if not lines:
        return None

    title_starts = (
        "hop dong",
        "cong van",
        "quyet dinh",
        "van ban",
        "to trinh",
        "bien ban",
        "bao cao",
        "thong bao",
        "de nghi",
    )
    stop_prefixes = (
        "kinh gui",
        "can cu",
        "noi nhan",
        "nguoi ky",
        "dai dien",
        "chu dau tu",
        "ben moi thau",
    )

    for index, line in enumerate(lines):
        normalized = normalize_for_rules(line)
        if not normalized.startswith(title_starts):
            continue

        title_lines = [line]
        for next_line in lines[index + 1 : index + 8]:
            normalized_next = normalize_for_rules(next_line)
            if normalized_next.startswith(stop_prefixes):
                break
            if re.match(r"^(so|số)\b", normalized_next):
                break
            if normalized_next.startswith(("cong hoa", "doc lap", "ngay ")):
                break
            if len(next_line) > 180:
                break
            if is_probable_title_continuation(next_line):
                title_lines.append(next_line)
                continue
            if len(title_lines) > 1:
                break

        title = clean_text(" ".join(title_lines))
        if title and len(title.split()) > 4:
            return title[:500]

    subject = extract_subject_line(lines)
    if subject and len(subject.split()) > 4:
        return subject

    return None


def extract_subject_line(lines: list[str]) -> Optional[str]:
    for index, line in enumerate(lines[:80]):
        normalized = normalize_for_rules(line)
        if normalized.startswith(("v/v", "ve viec", "ve viec", "trich yeu")):
            subject_lines = [line]
            for next_line in lines[index + 1 : index + 5]:
                normalized_next = normalize_for_rules(next_line)
                if normalized_next.startswith(("kinh gui", "can cu", "ngay ", "noi nhan")):
                    break
                if len(next_line) <= 180:
                    subject_lines.append(next_line)
            return clean_text(" ".join(subject_lines))[:500]
    return None


def is_probable_title_continuation(line: str) -> bool:
    normalized = normalize_for_rules(line)
    if normalized.startswith(("ve viec", "v/v", "phe duyet", "tham dinh", "dieu chinh", "bo sung")):
        return True
    if any(keyword in normalized for keyword in ("du an", "cong trinh", "goi thau", "hang muc", "chu truong dau tu")):
        return True
    return False


def extract_notes(text: str) -> list[str]:
    notes: list[str] = []
    note_keywords = (
        "ghi chu",
        "luu y",
        "khong bao gom",
        "chua bao gom",
        "can luu y",
    )
    for line in meaningful_lines(text, limit=260):
        normalized = normalize_for_rules(line)
        if normalized.startswith(("ghi chu", "luu y", "can luu y", "kem theo")):
            notes.append(line)
        elif any(keyword in normalized for keyword in note_keywords):
            notes.append(line)
        if len(notes) >= 8:
            break
    return unique_values(notes)


def guess_project_name_candidates(text: str) -> list[str]:
    title = extract_document_title(text) or ""
    candidates: list[str] = []
    source = "\n".join([title, *meaningful_lines(text, limit=160)])
    patterns = [
        r"(?:tên\s+dự\s+án|dự\s+án|công\s+trình)\s*[:\-]\s*([^\n]{8,220})",
        r"(?:thuộc\s+dự\s+án)\s*[:\-]?\s*([^\n]{8,220})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            value = clean_project_candidate(match.group(1))
            if value:
                candidates.append(value)

    if title:
        title_tail = re.split(r"\b(?:dự án|công trình)\b\s*[:\-]?", title, flags=re.IGNORECASE)
        if len(title_tail) > 1:
            value = clean_project_candidate(title_tail[-1])
            if value:
                candidates.append(value)

    return unique_values(candidates)[:8]


def clean_project_candidate(value: str) -> str:
    value = clean_text(value)
    value = re.split(
        r"\s+(?:địa điểm|chủ đầu tư|nguồn vốn|tổng mức|hình thức|gói thầu|hạng mục|kính gửi|căn cứ)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = value.strip(" .,:;-")
    if len(value) < 8:
        return ""
    return value[:220]


def is_probable_issuer_line(line: str) -> bool:
    normalized = normalize_for_rules(line)
    if any(
        keyword in normalized
        for keyword in (
            "uy ban",
            "ban quan ly",
            "cong ty",
            "agribank",
            "ngan hang",
            "chi nhanh",
            "phong",
            "so ",
            "tong cong ty",
            "tap doan",
            "trung tam",
        )
    ):
        return True

    letters = [char for char in line if char.isalpha()]
    uppercase_letters = [char for char in letters if char.upper() == char]
    return bool(letters) and len(line) <= 90 and len(uppercase_letters) / max(len(letters), 1) >= 0.65


def find_first_money(text: str) -> Optional[tuple[str, Any]]:
    for match in MONEY_PATTERN.finditer(text):
        value = clean_text(match.group(0))
        if not re.search(r"\d", value):
            continue
        if looks_like_non_money_number(text, match, value):
            continue
        normalized = normalize_money(value)
        if normalized is not None:
            return value, normalized
    return None


def looks_like_non_money_number(text: str, match: re.Match, value: str) -> bool:
    tail = text[match.end() : match.end() + 2]
    if "%" in tail:
        return True

    normalized = normalize_for_rules(value)
    has_money_unit = any(unit in normalized for unit in ("dong", "vnd", "vnđ", "ty", "trieu"))
    has_thousand_separator = bool(re.search(r"\d{1,3}(?:[.\s]\d{3})+", value))
    digit_count = len(re.sub(r"\D", "", value))
    if has_money_unit or has_thousand_separator or digit_count >= 7:
        return False

    return True


def normalize_money(value: str) -> Optional[int]:
    normalized = normalize_for_rules(value)
    multiplier = 1
    if "ty" in normalized:
        multiplier = 1_000_000_000
    elif "trieu" in normalized:
        multiplier = 1_000_000

    number = re.sub(r"(dong|vnd|vnd|vnđ|ty|trieu)", "", normalized, flags=re.IGNORECASE).strip()
    number = number.replace(" ", "")
    if multiplier == 1:
        digits = re.sub(r"[^\d]", "", number)
        if not digits:
            return None
        return int(digits)

    number = number.replace(",", ".")
    number = re.sub(r"[^0-9.]", "", number)
    if not number:
        return None
    try:
        return int(float(number) * multiplier)
    except ValueError:
        return None


DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", flags=re.IGNORECASE),
]


def extract_first_date(value: str) -> Optional[str]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(value)
        if match:
            return clean_text(match.group(0))
    return None


def normalize_date(value: str) -> Optional[str]:
    slash_match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", value)
    if slash_match:
        day, month, year = slash_match.groups()
        if len(year) == 2:
            year = f"20{year}" if int(year) < 50 else f"19{year}"
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    words_match = re.search(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        value,
        flags=re.IGNORECASE,
    )
    if words_match:
        day, month, year = words_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return None


def format_display_date(iso_date: str) -> Optional[str]:
    """Convert YYYY-MM-DD to DD/MM/YYYY for human-readable display."""
    if not isinstance(iso_date, str):
        return None
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", iso_date.strip())
    if not match:
        return None
    year, month, day = match.groups()
    return f"{day}/{month}/{year}"


def meaningful_lines(text: str, limit: Optional[int] = None) -> list[str]:
    lines = [clean_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return lines[:limit] if limit else lines


def first_meaningful_line(text: str) -> str:
    lines = meaningful_lines(text, limit=1)
    return lines[0] if lines else clean_text(text)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()


def normalize_for_rules(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    value = re.sub(r"[^a-z0-9%/.,:()_-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def unique_values(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        value = clean_text(value)
        key = normalize_for_rules(value)
        if value and key not in {normalize_for_rules(item) for item in seen}:
            seen.append(value)
    return seen


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def approximate_raw_index(raw_text: str, normalized_text: str, normalized_index: int) -> int:
    if not raw_text or not normalized_text:
        return 0
    ratio = len(raw_text) / max(len(normalized_text), 1)
    return max(0, min(len(raw_text) - 1, int(normalized_index * ratio)))


def clean_ocr_text(text: str) -> str:
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

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def non_boilerplate_lines(text: str, limit: int = 80) -> list[str]:
    boilerplate_prefixes = (
        "cong hoa xa hoi chu nghia",
        "doc lap",
        "socialist republic",
    )
    lines: list[str] = []
    for line in meaningful_lines(text, limit=limit):
        normalized = normalize_for_rules(line)
        if normalized.startswith(boilerplate_prefixes):
            continue
        lines.append(line)
    return lines


def calculate_local_confidence(data: dict[str, Any]) -> float:
    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    if data.get("screen") == "work_detail":
        scores = []
        for key in WORK_DETAIL_CONFIDENCE_FIELDS:
            field = fields.get(key) if isinstance(fields.get(key), dict) else {}
            scores.append(safe_float(field.get("confidence")) if field.get("value") not in (None, "") else 0.0)
        return round(sum(scores) / max(len(scores), 1), 4)

    present_scores = [
        safe_float(field.get("confidence"))
        for field in fields.values()
        if isinstance(field, dict) and field.get("value") not in (None, "")
    ]
    if present_scores:
        return round(sum(present_scores) / len(present_scores), 4)
    return 0.0


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def unique_matches(pattern: str, text: str, limit: int, flags: int = 0) -> list[str]:
    seen: list[str] = []
    for match in re.findall(pattern, text, flags=flags):
        value = match if isinstance(match, str) else " ".join([part for part in match if part])
        value = re.sub(r"\s+", " ", value).strip()
        if value and value not in seen:
            seen.append(value)
        if len(seen) >= limit:
            break
    return seen


async def extract_information(text: str) -> dict[str, Any]:
    text = clean_ocr_text(text)
    llm_config = get_llm_config()
    local_enabled = env_bool("LOCAL_EXTRACTION_ENABLED", True)
    entity_extraction_enabled = env_bool("LLM_ENTITY_EXTRACTION_ENABLED", True)

    # Priority 1: Try LLM first if available
    if llm_config:
        # Optionally run local parser for entity extraction prompt or validation
        local_data = None
        if local_enabled and entity_extraction_enabled:
            local_data = normalize_result({}, text, payload_source="rule")

        # Build prompt (with or without local context)
        if entity_extraction_enabled and local_data:
            prompt = build_entity_extraction_prompt(text, local_data)
            llm_mode = "entity_extraction"
        else:
            prompt = build_prompt(text)
            llm_mode = "direct"

        try:
            parsed = await call_llm_extraction(llm_config, prompt)
            data = normalize_result(parsed, text, payload_source="llm")
            data["pipeline"] = "llm_first"
            data["llm_extraction_mode"] = llm_mode
            if local_data:
                data["local_confidence"] = local_data["local_confidence"]
            data["needs_review"] = calculate_local_confidence(data) < env_float("LOCAL_CONFIDENCE_THRESHOLD", 0.68)
            return {
                "provider": f"{llm_config['provider']}_{llm_mode}",
                "model": llm_config["model"],
                "data": data,
            }
        except Exception as exc:
            # LLM failed, fallback to local if enabled
            if local_enabled:
                if local_data is None:
                    local_data = normalize_result({}, text, payload_source="rule")
                local_data["pipeline"] = "local_fallback_from_llm"
                local_data["llm_error"] = compact_error(exc)
                local_data["needs_review"] = True
                return {
                    "provider": "local",
                    "model": None,
                    "data": local_data,
                }
            else:
                # No local fallback, return error
                data = normalize_result({}, text, payload_source="rule")
                data["pipeline"] = "llm_failed_no_fallback"
                data["llm_error"] = compact_error(exc)
                data["needs_review"] = True
                return {
                    "provider": "error",
                    "model": None,
                    "data": data,
                }

    # Priority 2: No LLM available, use local parser
    if local_enabled:
        data = normalize_result({}, text, payload_source="rule")
        data["pipeline"] = "local_only"
        return {
            "provider": "local",
            "model": None,
            "data": data,
        }

    # No LLM and no local enabled - return empty
    data = normalize_result({}, text, payload_source="rule")
    data["pipeline"] = "no_extraction_available"
    data["needs_review"] = True
    return {
        "provider": "none",
        "model": None,
        "data": data,
    }


def get_llm_config() -> Optional[dict[str, Any]]:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in {"gemini", "google", "vertex", "gemini_vertex"}:
        return None

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return {
            "provider": "gemini",
            "auth_mode": "api_key",
            "api_key": api_key,
            "model": model,
        }

    project_id = (
        os.getenv("GEMINI_VERTEX_PROJECT_ID")
        or os.getenv("VERTEX_AI_PROJECT_ID")
        or os.getenv("GOOGLE_AI_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    location = os.getenv("GEMINI_VERTEX_LOCATION") or os.getenv("VERTEX_AI_LOCATION") or "us-central1"
    has_google_credentials = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CLOUD_PROJECT"))
    if project_id and has_google_credentials:
        return {
            "provider": "gemini_vertex",
            "auth_mode": "vertex",
            "project_id": project_id,
            "location": location,
            "model": model,
        }

    return None


async def call_llm_extraction(config: dict[str, Any], prompt: str) -> dict[str, Any]:
    if config.get("auth_mode") == "api_key":
        return await call_gemini_api_key_extraction(config, prompt)
    if config.get("auth_mode") == "vertex":
        return await call_vertex_gemini_extraction(config, prompt)
    raise ValueError(f"Unsupported LLM auth mode: {config.get('auth_mode')}")


async def call_gemini_api_key_extraction(config: dict[str, Any], prompt: str) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/{gemini_api_model_path(config['model'])}:generateContent"
    async with httpx.AsyncClient(timeout=env_float("GEMINI_TIMEOUT_SECONDS", 90.0)) as client:
        response = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": config["api_key"],
            },
            json=build_gemini_payload(prompt),
        )
        response.raise_for_status()
    return parse_gemini_response(response.json())


async def call_vertex_gemini_extraction(config: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Call Vertex AI Gemini using new Google GenAI SDK for enterprise models like gemini-3.5-flash"""
    try:
        from google import genai
        from google.genai import types

        # Use enterprise SDK for newer models
        def _call_genai():
            client = genai.Client(
                vertexai=True,
                project=config["project_id"],
                location=config["location"]
            )

            generation_config_kwargs = {
                "system_instruction": LLM_SYSTEM_INSTRUCTION,
                "temperature": 0,
                "response_mime_type": "application/json",
            }
            # Disable internal "thinking" on Gemini 2.5 to reduce latency.
            thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
            try:
                generation_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
            except AttributeError:
                pass

            response = client.models.generate_content(
                model=config["model"],
                contents=[prompt],
                config=types.GenerateContentConfig(**generation_config_kwargs),
            )
            return response.text

        response_text = await asyncio.to_thread(_call_genai)
        return json.loads(response_text)

    except ImportError:
        # Fallback to old REST API if google-genai not installed
        access_token = await asyncio.to_thread(get_google_access_token)
        model_path = vertex_model_path(config["project_id"], config["location"], config["model"])
        url = f"https://{config['location']}-aiplatform.googleapis.com/v1/{model_path}:generateContent"
        async with httpx.AsyncClient(timeout=env_float("GEMINI_TIMEOUT_SECONDS", 90.0)) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=build_gemini_payload(prompt),
            )
            response.raise_for_status()
        return parse_gemini_response(response.json())


def build_gemini_payload(prompt: str) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "temperature": 0,
        "responseMimeType": "application/json",
    }
    # Disable "thinking" on Gemini 2.5 family for faster latency.
    thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
    generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    return {
        "systemInstruction": {"parts": [{"text": LLM_SYSTEM_INSTRUCTION}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": generation_config,
    }


def get_google_access_token() -> str:
    normalize_google_credentials_path()
    import google.auth
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


def normalize_google_credentials_path() -> None:
    raw_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not raw_path:
        return
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return

    repo_path = Path(__file__).resolve().parents[2] / raw_path
    if repo_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(repo_path)


def vertex_model_path(project_id: str, location: str, model: str) -> str:
    if model.startswith("projects/"):
        return model
    if model.startswith("publishers/"):
        return f"projects/{project_id}/locations/{location}/{model}"
    if model.startswith("models/"):
        return f"projects/{project_id}/locations/{location}/publishers/google/{model}"
    return f"projects/{project_id}/locations/{location}/publishers/google/models/{model}"


def gemini_api_model_path(model: str) -> str:
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def parse_gemini_response(payload: dict[str, Any]) -> dict[str, Any]:
    content_parts: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts", []) if isinstance(content, dict) else []
        for part in parts:
            text = part.get("text") if isinstance(part, dict) else None
            if text:
                content_parts.append(text)

    if not content_parts:
        return {}
    return parse_json_content("\n".join(content_parts))


def parse_json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {}

    return parsed if isinstance(parsed, dict) else {}


def compact_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:300] if exc.response is not None else ""
        return f"{exc.response.status_code} {exc.response.reason_phrase}: {body}".strip()
    message = str(exc)
    return f"{type(exc).__name__}: {message[:300]}"


def should_use_llm_fallback(data: dict[str, Any]) -> bool:
    if data.get("screen") == "contract":
        return True

    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    title = fields.get("title") if isinstance(fields.get("title"), dict) else {}
    issuer = fields.get("issuer") if isinstance(fields.get("issuer"), dict) else {}
    if title.get("value") in (None, "") or issuer.get("value") in (None, ""):
        return True

    return safe_float(data.get("local_confidence")) < env_float("LOCAL_CONFIDENCE_THRESHOLD", 0.68)


def build_entity_extraction_prompt(text: str, local_data: dict[str, Any]) -> str:
    fields = local_data.get("fields") if isinstance(local_data.get("fields"), dict) else {}
    compact_fields = {
        key: {
            "value": field.get("value"),
            "normalized_value": field.get("normalized_value"),
            "evidence": field.get("evidence"),
            "confidence": field.get("confidence"),
            "source": field.get("source"),
        }
        for key, field in fields.items()
        if isinstance(field, dict)
    }
    if local_data.get("screen") == "contract":
        return build_contract_entity_extraction_prompt(text, local_data, compact_fields)

    trimmed_text = select_entity_extraction_text(text)
    return f"""
Bạn hãy đọc OCR tiếng Việt và trích xuất intent/entities nghiệp vụ càng đầy đủ càng tốt.
Local rule bên dưới chỉ là gợi ý, được phép sửa nếu OCR chứng minh rõ hơn. Không suy đoán khi không có chứng cứ.
Chỉ trả JSON hợp lệ, không markdown.

Local rule result:
{json.dumps(compact_fields, ensure_ascii=False, indent=2)}

Quy tắc field quan trọng:
- document_number: số văn bản/số tờ trình/số quyết định ở đầu tài liệu.
- signed_or_effective_date: BẮT BUỘC lấy NGÀY CÓ ĐỊA ĐIỂM Ở ĐẦU VÀN BẢN (thường trong 500 ký tự đầu tiên).
  * PHẢI tìm dạng: "Hà Nội, ngày 16 tháng 12 năm 2021", "TP. Hồ Chí Minh, ngày 25 tháng 3 năm 2022", "Đà Nẵng, ngày..."
  * PHẢI có địa điểm (tên thành phố/tỉnh) + dấu phẩy + "ngày XX tháng YY năm ZZZZ"
  * TUYỆT ĐỐI KHÔNG lấy ngày từ: "Căn cứ Luật...", "Nghị định số...", "Thông tư số...", "Quyết định số... ngày..."
  * Chỉ khi KHÔNG tìm thấy ngày có địa điểm ở đầu văn bản, mới lấy "ngày ký:", "ngày ban hành:", "thời gian ký:"
  * Output BẮT BUỘC: value = "DD/MM/YYYY" (ví dụ "16/12/2021"), normalized_value = "YYYY-MM-DD" (ví dụ "2021-12-16"). Không trả nguyên văn "ngày 16 tháng 12 năm 2021".
- approved_value: giá trị duyệt/phê duyệt/được duyệt/sau thẩm định. Với tờ trình, "giá gói thầu", "giá trị gói thầu", "giá dự toán gói thầu" thường chính là giá trị duyệt.
- submitted_value: giá trị trình/đề nghị/xin phê duyệt/trình duyệt/trước thẩm định.
- issuer: đơn vị phát hành/đơn vị thầu ở góc trái đầu tài liệu, không lấy "Kính gửi".
- title: BẮT BUỘC trả về. Là tên/tựa đầy đủ của tài liệu, gồm loại văn bản (HỢP ĐỒNG / CÔNG VĂN / QUYẾT ĐỊNH / VĂN BẢN / TỜ TRÌNH / BIÊN BẢN / BÁO CÁO / THÔNG BÁO / ĐỀ NGHỊ) và dòng "Về việc..." nếu có. Title phải bắt đầu bằng một trong các loại văn bản trên và dài hơn 4 từ. Nếu OCR không có dòng nào thoả, để value=null.

Ngoài fields, hãy trả nhiều entity để matcher local dùng ở bước sau:
- project_name_candidates: tên dự án/công trình đầy đủ và biến thể ngắn.
- task_title_candidates: các cụm tên công việc con có thể match task.
- work_item_candidates: hạng mục, nội dung công việc, gói thầu, hồ sơ.
- procurement_package_candidates: tên/số gói thầu.
- task_keywords: cụm nghiệp vụ như phê duyệt chủ trương, thẩm định thiết kế, phê duyệt kế hoạch lựa chọn nhà thầu, chỉ định thầu, nghiệm thu, quyết toán.
- monetary_entities: tất cả dòng tiền quan trọng kèm role nếu nhận ra: approved_value, submitted_value, package_price, estimate, contract_value, unknown.

JSON output schema:
{{
  "document_type": "contract|document",
  "document_intent": "to_trinh|quyet_dinh|bien_ban|cong_van|contract|bao_cao|unknown",
  "fields": {{
    "document_number": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "signed_or_effective_date": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "approved_value": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "submitted_value": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "issuer": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "notes": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "title": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}}
  }},
  "generic_extraction": {{
    "document_title_or_type": null,
    "project_name_candidates": [],
    "task_title_candidates": [],
    "work_item_candidates": [],
    "procurement_package_candidates": [],
    "task_keywords": [],
    "dates": [],
    "monetary_amounts": [],
    "document_numbers": [],
    "approvers_or_positions": []
  }},
  "entities": {{
    "projects": [],
    "tasks": [],
    "work_items": [],
    "procurement_packages": [],
    "business_actions": [],
    "monetary_entities": [],
    "organizations": [],
    "people": [],
    "locations": []
  }},
  "notes": []
}}

OCR text đã clean:
\"\"\"{trimmed_text}\"\"\"
""".strip()


def build_contract_entity_extraction_prompt(text: str, local_data: dict[str, Any], compact_fields: dict[str, Any]) -> str:
    trimmed_text = select_entity_extraction_text(text)
    return f"""
Bạn hãy đọc OCR hợp đồng tiếng Việt và trích xuất đầy đủ các field để fill form thêm hợp đồng.
Local rule bên dưới chỉ là gợi ý/guardrail, được phép sửa nếu OCR chứng minh rõ hơn. Không suy đoán khi không có chứng cứ.
Chỉ trả JSON hợp lệ, không markdown.

{CONTRACT_EXTRACTION_GUIDANCE}

Danh mục hình thức hợp đồng tham chiếu:
{json.dumps(CONTRACT_FORMS, ensure_ascii=False)}

Danh mục nhóm nhà thầu tham chiếu:
{json.dumps(CONTRACTOR_GROUPS, ensure_ascii=False)}

Local rule result:
{json.dumps(compact_fields, ensure_ascii=False, indent=2)}

JSON output schema:
{{
  "document_type": "contract",
  "document_intent": "contract",
  "fields": {{
    "work_name": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "contract_name": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "contract_number": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "signed_date": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "execution_duration_days": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "contract_form": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "contract_vat_percent": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "settlement_request_vat_percent": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "appraisal_vat_percent": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "estimated_value": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "contract_value": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "contractor_group": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "contractor_name": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "contractor_contract_amount": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "performance_guarantee_value": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "performance_guarantee_end_date": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "advance_guarantee_value": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "advance_guarantee_end_date": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}}
  }},
  "generic_extraction": {{
    "document_title_or_type": null,
    "task_title_candidates": [],
    "work_item_candidates": [],
    "contract_parties": [],
    "monetary_amounts": [],
    "dates": [],
    "document_numbers": [],
    "guarantee_terms": []
  }},
  "entities": {{
    "tasks": [],
    "work_items": [],
    "contract_parties": [],
    "contractors": [],
    "guarantees": [],
    "monetary_entities": [],
    "organizations": [],
    "people": [],
    "dates": []
  }},
  "notes": []
}}

OCR text đã clean:
\"\"\"{trimmed_text}\"\"\"
""".strip()


def select_entity_extraction_text(text: str) -> str:
    max_chars = int(os.getenv("LLM_ENTITY_MAX_CHARS", os.getenv("LLM_MAX_OCR_CHARS", "30000")))
    lines = non_boilerplate_lines(text, limit=900)
    if not lines:
        return text[:max_chars]

    selected = lines[: min(120, len(lines))]
    keywords = (
        "gia tri",
        "gia goi thau",
        "du toan",
        "tong muc",
        "goi thau",
        "du an",
        "cong trinh",
        "hang muc",
        "phe duyet",
        "thanh dinh",
        "tham dinh",
        "lua chon nha thau",
        "chi dinh thau",
        "nghiem thu",
        "quyet toan",
        "ghi chu",
        "luu y",
    )
    for line in lines[120:]:
        normalized = normalize_for_rules(line)
        if any(keyword in normalized for keyword in keywords):
            selected.append(line)
        if len("\n".join(selected)) >= max_chars:
            break

    return "\n".join(unique_values(selected))[:max_chars]


def build_fallback_prompt(text: str, local_data: dict[str, Any]) -> str:
    fields = local_data.get("fields") if isinstance(local_data.get("fields"), dict) else {}
    compact_fields = {
        key: {
            "value": field.get("value"),
            "normalized_value": field.get("normalized_value"),
            "evidence": field.get("evidence"),
            "confidence": field.get("confidence"),
            "source": field.get("source"),
        }
        for key, field in fields.items()
        if isinstance(field, dict)
    }
    if local_data.get("screen") == "contract":
        return build_contract_entity_extraction_prompt(text, local_data, compact_fields)

    snippets = select_fallback_snippets(text, local_data)
    return f"""
Kết quả rule-based dưới đây có thể thiếu field. Hãy chỉ sửa/bổ sung các field còn thiếu hoặc rõ ràng sai.
Không suy đoán nếu snippet không đủ chứng cứ. Chỉ trả JSON hợp lệ.

Loại hiện tại: {local_data.get("document_type")} / intent: {local_data.get("document_intent")}
Local confidence: {local_data.get("local_confidence")}

Field hiện tại:
{json.dumps(compact_fields, ensure_ascii=False, indent=2)}

Chỉ được trả theo schema:
{{
  "document_type": "contract|document",
  "document_intent": "to_trinh|quyet_dinh|bien_ban|cong_van|contract|bao_cao|unknown",
  "fields": {{
    "document_number": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "signed_or_effective_date": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "approved_value": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "submitted_value": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "issuer": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "notes": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}},
    "title": {{"value": null, "normalized_value": null, "evidence": null, "confidence": 0.0}}
  }},
  "generic_extraction": {{
    "project_name_candidates": [],
    "task_title_candidates": [],
    "work_item_candidates": [],
    "procurement_package_candidates": [],
    "task_keywords": []
  }},
  "entities": {{}},
  "notes": []
}}

Lưu ý: approved_value có thể xuất hiện dưới nhãn "giá gói thầu", "giá trị gói thầu" hoặc "giá dự toán gói thầu".
Lưu ý title: BẮT BUỘC trả về (nếu OCR có dòng phù hợp). Title phải bắt đầu bằng một trong các loại văn bản HỢP ĐỒNG / CÔNG VĂN / QUYẾT ĐỊNH / VĂN BẢN / TỜ TRÌNH / BIÊN BẢN / BÁO CÁO / THÔNG BÁO / ĐỀ NGHỊ, dài hơn 4 từ, và nên kèm dòng "Về việc..." nếu có.

OCR snippets chọn lọc, không phải toàn bộ tài liệu:
\"\"\"{snippets}\"\"\"
""".strip()


def select_fallback_snippets(text: str, local_data: dict[str, Any]) -> str:
    lines = meaningful_lines(text)
    selected = lines[: min(45, len(lines))]
    keywords = (
        "giá trị",
        "gia tri",
        "dự toán",
        "du toan",
        "tổng mức",
        "tong muc",
        "ghi chú",
        "ghi chu",
        "lưu ý",
        "luu y",
        "thời gian",
        "thoi gian",
        "ngày ký",
        "ngay ky",
    )
    for line in lines[45:260]:
        normalized = normalize_for_rules(line)
        if any(keyword in normalized for keyword in keywords):
            selected.append(line)
        if len(selected) >= 90:
            break

    for field in (local_data.get("fields") or {}).values():
        if isinstance(field, dict) and field.get("evidence"):
            selected.append(str(field["evidence"]))

    return "\n".join(unique_values(selected))[: int(os.getenv("LLM_FALLBACK_MAX_CHARS", "9000"))]


def build_prompt(text: str) -> str:
    trimmed_text = text[: int(os.getenv("LLM_MAX_OCR_CHARS", "60000"))]
    return f"""
Hãy tự phân loại tài liệu thành đúng một loại:
- contract: Hợp đồng
- document: Tài liệu khác như quyết định, biên bản họp, tờ trình, hóa đơn, công văn.

Nếu là document, trích các field keys:
{json.dumps(DOCUMENT_FIELDS, ensure_ascii=False, indent=2)}
Trong đó:
- title BẮT BUỘC trả về (nếu OCR có): là tên/tựa đầy đủ của tài liệu, bắt đầu bằng một trong các loại văn bản HỢP ĐỒNG / CÔNG VĂN / QUYẾT ĐỊNH / VĂN BẢN / TỜ TRÌNH / BIÊN BẢN / BÁO CÁO / THÔNG BÁO / ĐỀ NGHỊ, dài hơn 4 từ, và thường gồm cả dòng "Về việc..." kèm theo. Không lấy các dòng quảng cáo, header.
- signed_or_effective_date: BẮT BUỘC lấy NGÀY CÓ ĐỊA ĐIỂM Ở ĐẦU VÀN BẢN (thường trong 500 ký tự đầu). PHẢI tìm dạng "Hà Nội, ngày 16 tháng 12 năm 2021", "TP.HCM, ngày...". PHẢI có địa điểm + dấu phẩy + ngày tháng năm. TUYỆT ĐỐI KHÔNG lấy ngày từ "Căn cứ Luật...", "Nghị định...", "Thông tư...". Chỉ khi KHÔNG có ngày ở đầu mới lấy "ngày ký:", "ngày ban hành:". Output BẮT BUỘC: value="DD/MM/YYYY", normalized_value="YYYY-MM-DD".
- issuer phải lấy theo phần đầu góc trái tài liệu, không lấy cơ quan nhận ở phần "Kính gửi".
- approved_value là giá trị đã/được phê duyệt, giá trị duyệt hoặc sau thẩm định. Với tờ trình, "giá gói thầu", "giá trị gói thầu", "giá dự toán gói thầu" thường chính là giá trị duyệt.
- submitted_value là giá trị trình, đề nghị phê duyệt, trình duyệt hoặc trước thẩm định.

Nếu là contract, trích các field keys:
{json.dumps(CONTRACT_FIELDS, ensure_ascii=False, indent=2)}

{CONTRACT_EXTRACTION_GUIDANCE}

Mỗi field phải có dạng:
{{"label": "...", "value": "... hoặc null", "normalized_value": "... hoặc null", "evidence": "câu OCR ngắn chứng minh", "confidence": 0.0-1.0, "source": "llm"}}

Luôn trả thêm generic_extraction gồm:
- document_title_or_type
- project_name_candidates
- task_title_candidates
- work_item_candidates
- procurement_package_candidates
- task_keywords
- dates
- monetary_amounts
- document_numbers
- approvers_or_positions

Luôn trả thêm entities gồm nhiều tín hiệu match nhất có thể:
- projects
- tasks
- work_items
- procurement_packages
- business_actions
- monetary_entities với role: approved_value, submitted_value, package_price, estimate, contract_value, unknown
- organizations, people, locations

Các hình thức hợp đồng tham chiếu: {", ".join(CONTRACT_FORMS)}
Các nhóm nhà thầu tham chiếu: {", ".join(CONTRACTOR_GROUPS)}

JSON output schema:
{{
  "document_type": "contract|document",
  "fields": {{}},
  "generic_extraction": {{}},
  "entities": {{}},
  "notes": []
}}

OCR text:
\"\"\"{trimmed_text}\"\"\"
""".strip()
