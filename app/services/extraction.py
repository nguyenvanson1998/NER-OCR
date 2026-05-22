import json
import os
import re
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


def empty_field(label: str) -> dict[str, Any]:
    return {"label": label, "value": None, "normalized_value": None, "evidence": None, "confidence": 0.0}


def normalize_result(payload: dict[str, Any], text: str) -> dict[str, Any]:
    detected_type = payload.get("document_type")
    if detected_type not in {"contract", "document"}:
        detected_type = classify_document(text)

    field_schema = CONTRACT_FIELDS if detected_type == "contract" else DOCUMENT_FIELDS
    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    normalized_fields = {key: empty_field(label) for key, label in field_schema.items()}
    for key, label in field_schema.items():
        raw = fields.get(key)
        if isinstance(raw, dict):
            normalized_fields[key] = {**empty_field(label), **raw, "label": raw.get("label") or label}
        elif raw not in (None, ""):
            normalized_fields[key] = {**empty_field(label), "value": raw, "normalized_value": raw, "confidence": 0.4}

    generic = payload.get("generic_extraction")
    if not isinstance(generic, dict):
        generic = heuristic_generic_extraction(text)

    return {
        "document_type": detected_type,
        "document_type_label": "Hợp đồng" if detected_type == "contract" else "Tài liệu",
        "screen": "contract" if detected_type == "contract" else "work_detail",
        "fields": normalized_fields,
        "generic_extraction": {
            **heuristic_generic_extraction(text),
            **generic,
        },
        "notes": payload.get("notes") or [],
    }


def classify_document(text: str) -> str:
    sample = text[:5000].lower()
    contract_hits = [
        "hợp đồng",
        "bên giao thầu",
        "bên nhận thầu",
        "giá trị hợp đồng",
        "bảo lãnh thực hiện",
        "tạm ứng hợp đồng",
    ]
    return "contract" if sum(hit in sample for hit in contract_hits) >= 2 else "document"


def heuristic_generic_extraction(text: str) -> dict[str, Any]:
    date_pattern = r"\b(?:ngày\s*)?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}"
    money_pattern = r"\b\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?\s*(?:đồng|vnđ|vnd)?\b|\b\d+(?:,\d+)?\s*(?:tỷ|triệu)\s*(?:đồng)?\b"
    doc_no_pattern = r"(?:số|so)\s*[:.]?\s*([A-Z0-9ĐĐa-zÀ-ỹ/._-]{2,})"
    approver_pattern = r"(?:chủ tịch|giám đốc|phó giám đốc|trưởng phòng|kt\.|ký bởi|người ký|đại diện)\s*[:.]?\s*([A-ZÀ-Ỹ][^\n]{2,80})"

    return {
        "document_title_or_type": guess_title(text),
        "dates": unique_matches(date_pattern, text, limit=20),
        "monetary_amounts": unique_matches(money_pattern, text, limit=20),
        "document_numbers": unique_matches(doc_no_pattern, text, limit=10, flags=re.IGNORECASE),
        "approvers_or_positions": unique_matches(approver_pattern, text, limit=10, flags=re.IGNORECASE),
    }


def guess_title(text: str) -> Optional[str]:
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        if 6 <= len(line) <= 160 and not line.lower().startswith(("cộng hòa", "độc lập", "số:")):
            return line
    return None


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
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return {
            "provider": "heuristic",
            "model": None,
            "data": normalize_result({}, text),
        }

    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là hệ thống trích xuất thông tin từ OCR tiếng Việt cho hợp đồng, "
                    "quyết định, biên bản họp, tờ trình, hóa đơn, công văn trong dự án xây dựng. "
                    "Chỉ trả JSON hợp lệ, không markdown. Nếu thiếu thông tin, để value là null "
                    "và vẫn trả generic_extraction."
                ),
            },
            {"role": "user", "content": build_prompt(text)},
        ],
    }

    async with httpx.AsyncClient(timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}

    return {
        "provider": "openai",
        "model": model,
        "data": normalize_result(parsed, text),
    }


def build_prompt(text: str) -> str:
    trimmed_text = text[: int(os.getenv("LLM_MAX_OCR_CHARS", "60000"))]
    return f"""
Hãy tự phân loại tài liệu thành đúng một loại:
- contract: Hợp đồng
- document: Tài liệu khác như quyết định, biên bản họp, tờ trình, hóa đơn, công văn.

Nếu là document, trích các field keys:
{json.dumps(DOCUMENT_FIELDS, ensure_ascii=False, indent=2)}

Nếu là contract, trích các field keys:
{json.dumps(CONTRACT_FIELDS, ensure_ascii=False, indent=2)}

Mỗi field phải có dạng:
{{"label": "...", "value": "... hoặc null", "normalized_value": "... hoặc null", "evidence": "câu OCR ngắn chứng minh", "confidence": 0.0-1.0}}

Luôn trả thêm generic_extraction gồm:
- document_title_or_type
- dates
- monetary_amounts
- document_numbers
- approvers_or_positions

Các hình thức hợp đồng tham chiếu: {", ".join(CONTRACT_FORMS)}
Các nhóm nhà thầu tham chiếu: {", ".join(CONTRACTOR_GROUPS)}

JSON output schema:
{{
  "document_type": "contract|document",
  "fields": {{}},
  "generic_extraction": {{}},
  "notes": []
}}

OCR text:
\"\"\"{trimmed_text}\"\"\"
""".strip()
