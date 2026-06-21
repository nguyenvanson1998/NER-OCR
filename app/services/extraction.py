import asyncio
import difflib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any
from typing import Optional

import httpx

from app.services.ocr_cleaning import clean_ocr_plain_text
from app.services.timing import record_timing_event
from app.services.timing import timed_stage

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


EXTRACTION_TYPES = {"document", "contract"}
DEFAULT_EXTRACTION_TYPE = "document"

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
    "Bỏ qua các dấu/overlay không thuộc nội dung văn bản như CÔNG VĂN ĐẾN, "
    "Số đến, Ngày đến, con dấu tròn, mã số doanh nghiệp trên con dấu và các mảnh chữ bị chèn từ dấu. "
    "QUY TẮC NGÀY THÁNG (BẮT BUỘC): với mọi field ngày (signed_or_effective_date, "
    "signed_date, performance_guarantee_end_date, advance_guarantee_end_date): "
    "đặt value theo định dạng DD/MM/YYYY và normalized_value theo định dạng "
    "ISO YYYY-MM-DD. Tuyệt đối KHÔNG để value là nguyên văn 'ngày XX tháng YY năm ZZZZ'. "
    "Nếu chỉ có ngày + tháng mà thiếu năm, để cả hai field là null. "
    "VỊ TRÍ NGÀY BAN HÀNH (signed_or_effective_date): LUÔN là dòng địa danh + ngày "
    "(ví dụ 'Hà Nội, ngày 16 tháng 12 năm 2021', 'TP. Hồ Chí Minh, ngày ...') nằm "
    "NGAY DƯỚI tiêu ngữ 'Độc lập - Tự do - Hạnh phúc' ở góc phải/đầu văn bản. "
    "TUYỆT ĐỐI KHÔNG lấy ngày trong phần 'Căn cứ ...', 'Luật số ... ngày ...', "
    "'Nghị định ... ngày ...', 'Thông tư ... ngày ...', 'Quyết định số ... ngày ...', "
    "'Công văn số ... ngày ...' hoặc bất kỳ trích dẫn pháp lý nào — đó là ngày của "
    "văn bản được tham chiếu, KHÔNG phải ngày ban hành văn bản hiện tại. "
    "Nếu không tìm thấy dòng 'địa danh, ngày ... tháng ... năm ...' dưới tiêu ngữ, "
    "để signed_or_effective_date = null thay vì lấy đại một ngày khác. "
    "Ngoại lệ Giấy chứng nhận đăng ký doanh nghiệp/kinh doanh: document_number luôn null, "
    "không lấy Mã số doanh nghiệp; ngày văn bản là ngày của 'Đăng ký thay đổi lần thứ' lớn nhất. "
    "Với Báo cáo/Kế hoạch/Hồ sơ/Đề cương/Thuyết minh/Quy trình nội bộ, để ngày văn bản null. "
    "QUY TẮC CƠ QUAN BAN HÀNH: Biên bản/Báo cáo/Hợp đồng/Kế hoạch/Quy trình/Hồ sơ thiết kế "
    "nội bộ để issuer=null; không bao giờ lấy CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM. Với "
    "Chứng chỉ/Giấy chứng nhận, chỉ lấy cơ quan nhà nước (Bộ/Cục/Sở/UBND/Tổng cục/Chi cục) "
    "ở đầu tài liệu, không lấy tiêu đề hay 'Nơi cấp'. Với văn bản hành chính, lấy đủ mọi dòng "
    "của khối cơ quan gần 'Số:', không dựa vào in đậm/in thường. "
    "QUY TẮC GIÁ TRỊ: trước hết xác định loại giấy tờ. Tờ trình/Văn bản đề nghị/"
    "Hồ sơ trình (Tờ trình, TTr, đề nghị, trình thẩm định, trình phê duyệt, xin chấp thuận) "
    "thì mọi tổng mức đầu tư/dự toán/kinh phí/giá trị đề nghị/giá tiền là submitted_value, "
    "không phải approved_value dù tiêu đề có chữ phê duyệt. Quyết định/Văn bản phê duyệt/"
    "Văn bản chấp thuận (QĐ, phê duyệt, chấp thuận, đồng ý, cho phép) thì map số tiền kết quả "
    "đã duyệt/chấp thuận vào approved_value. Hợp đồng thì giá trị hợp đồng có thể map "
    "approved_value, không map submitted_value. Biên lai/Phiếu thu/Chứng từ phí và văn bản "
    "góp ý/tham gia ý kiến/cung cấp thông tin/giấy mời/phân công thì để trống giá trị "
    "trình/duyệt nếu không có ngữ cảnh duyệt thật rõ."
)

LLM_DELTA_SYSTEM_INSTRUCTION = (
    "Trích xuất có căn cứ từ OCR tiếng Việt. Chỉ trả JSON theo response schema; "
    "không suy diễn, thiếu thì null. Evidence tối đa 200 ký tự. Bỏ qua dấu đến, "
    "con dấu và nhiễu OCR. Ngày hiển thị DD/MM/YYYY, normalized_value YYYY-MM-DD; "
    "không lấy ngày trong phần Căn cứ hoặc văn bản được dẫn chiếu."
)

_GENAI_CLIENTS: dict[tuple[str, str], Any] = {}

COMMON_EXTRACTION_RULES = (
    "Chỉ trả JSON hợp lệ, không markdown. Chỉ lấy thông tin có chứng cứ; thiếu thì để null. "
    "Bỏ qua stamp/overlay OCR như CÔNG VĂN ĐẾN, Số đến, Ngày đến, con dấu, mã số DN và mảnh chữ từ dấu."
)

DOCUMENT_FIELD_RULES = """
Rules document:
- document_number: lấy "Số:" đầu tài liệu; Giấy ĐKDN/ĐKKD để null, KHÔNG lấy "Mã số doanh nghiệp".
- signed_or_effective_date: lấy địa danh+ngày dưới "Độc lập - Tự do - Hạnh phúc", value=DD/MM/YYYY, normalized=YYYY-MM-DD; KHÔNG lấy ngày trong Căn cứ/Luật/Nghị định/Thông tư/Quyết định/Công văn. Giấy ĐKDN lấy ngày "Đăng ký thay đổi lần thứ" lớn nhất. Báo cáo/Kế hoạch/Hồ sơ/Đề cương/Thuyết minh/Quy trình nội bộ để null.
- approved_value: Quyết định/phê duyệt/chấp thuận hoặc Hợp đồng; nếu bảng có "Tổng cộng" lấy số tiền tổng, không lấy dòng chi phí thành phần.
- submitted_value: Tờ trình/đề nghị/hồ sơ trình; map tổng giá trị/giá gói thầu/tổng mức/dự toán/kinh phí/giá trị đề nghị. Biên lai/Phiếu thu/góp ý/giấy mời/phân công để null.
- issuer: văn bản hành chính lấy đủ khối nhiều dòng gần "Số:". Biên bản/Báo cáo/Hợp đồng/Kế hoạch/Quy trình/Hồ sơ nội bộ để null. Chứng chỉ/Giấy chứng nhận chỉ lấy cơ quan nhà nước đầu tài liệu, không lấy tiêu đề, "Nơi cấp" hay "Cộng hòa...".
- title: loại văn bản + dòng Về việc/V/v/trích yếu nếu có; bắt đầu bằng HỢP ĐỒNG/CÔNG VĂN/QUYẾT ĐỊNH/VĂN BẢN/TỜ TRÌNH/BIÊN BẢN/BÁO CÁO/THÔNG BÁO/ĐỀ NGHỊ.
""".strip()

CONTRACT_EXTRACTION_GUIDANCE = """
Luật bóc tách hợp đồng:
- không tự suy diễn; ưu tiên điều khoản chính, trang ký/thông tin hợp đồng, rồi phụ lục/văn bản dẫn chiếu.
- signed_date: ưu tiên dòng "<Địa danh>, ngày DD tháng MM năm YYYY" dưới tiêu ngữ "Độc lập - Tự do - Hạnh phúc", hoặc "hôm nay, ngày ... tháng ... năm ..." ở đầu/cuối hợp đồng; value=DD/MM/YYYY, normalized_value=YYYY-MM-DD. KHÔNG lấy ngày trong Căn cứ/Luật/Nghị định/Thông tư/Quyết định số/Công văn số.
- Tiền chuẩn hóa về số VND; phần trăm về số. contract_value từ Giá/Giá trị hợp đồng; estimated_value khi có giá gói thầu/dự toán/tổng mức; không tự tính bảo lãnh.
- Tên công việc là field mapping: hiểu bản chất để đề xuất task/subtask/subsubtask, không lấy nguyên văn nếu quá dài.
- contract_form và contractor_group chọn từ danh mục tham chiếu nếu phù hợp.
""".strip()

CONTRACT_FIELD_RULES = CONTRACT_EXTRACTION_GUIDANCE

FIELD_OBJECT_HINT = 'Field object={"value":string|null,"normalized_value":any|null,"evidence":string|null,"confidence":0-1}'
DOCUMENT_SCHEMA_HINT = (
    'Schema={"document_type":"document","document_intent":string,'
    '"fields":{document_number,signed_or_effective_date,approved_value,submitted_value,issuer,notes,title},'
    '"generic_extraction":{document_title_or_type,project_name_candidates,task_title_candidates,work_item_candidates,procurement_package_candidates,task_keywords,dates,monetary_amounts,document_numbers,approvers_or_positions},'
    '"entities":{projects,tasks,work_items,procurement_packages,business_actions,monetary_entities,organizations,people,locations},"notes":[]}'
)
CONTRACT_SCHEMA_HINT = (
    'Schema={"document_type":"contract","document_intent":"contract",'
    '"fields":{work_name,contract_name,contract_number,signed_date,execution_duration_days,contract_form,contract_vat_percent,settlement_request_vat_percent,appraisal_vat_percent,estimated_value,contract_value,contractor_group,contractor_name,contractor_contract_amount,performance_guarantee_value,performance_guarantee_end_date,advance_guarantee_value,advance_guarantee_end_date},'
    '"generic_extraction":{document_title_or_type,task_title_candidates,work_item_candidates,contract_parties,monetary_amounts,dates,document_numbers,guarantee_terms},'
    '"entities":{tasks,work_items,contract_parties,contractors,guarantees,monetary_entities,organizations,people,dates},"notes":[]}'
)


def empty_field(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "value": None,
        "normalized_value": None,
        "evidence": None,
        "confidence": 0.0,
        "source": None,
    }


def normalize_extraction_type(value: Optional[str] = None) -> str:
    extraction_type = (value or DEFAULT_EXTRACTION_TYPE).strip().lower()
    if extraction_type not in EXTRACTION_TYPES:
        raise ValueError("extraction_type must be either 'document' or 'contract'")
    return extraction_type


def normalize_result(
    payload: dict[str, Any],
    text: str,
    payload_source: str = "llm",
    extraction_type: Optional[str] = DEFAULT_EXTRACTION_TYPE,
) -> dict[str, Any]:
    text = clean_ocr_text(text)
    detected_type = normalize_extraction_type(extraction_type)
    rule_intent = classify_document_intent(text)
    detected_intent = choose_document_intent(rule_intent, payload.get("document_intent"))
    if detected_type == "contract":
        detected_intent = "contract"

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
            if should_replace_field(normalized_fields[key], candidate, key):
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
            if should_replace_field(normalized_fields[key], candidate, key):
                normalized_fields[key] = candidate

    if detected_type == "document":
        apply_document_context_guardrails(normalized_fields, text, heuristic_fields)
        apply_document_intent_value_rules(normalized_fields, detected_intent)

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


PROPOSAL_INTENTS = {"to_trinh", "van_ban_de_nghi", "ho_so_trinh"}
APPROVAL_INTENTS = {"quyet_dinh", "van_ban_phe_duyet", "van_ban_chap_thuan"}
CONTRACT_INTENTS = {"contract"}
RECEIPT_INTENTS = {"bien_lai", "phieu_thu", "chung_tu_phi"}
NO_VALUE_INTENTS = {
    "gop_y",
    "tham_gia_y_kien",
    "cung_cap_thong_tin",
    "giay_moi",
    "phan_cong",
    "bien_ban",
    "bao_cao",
    "thong_bao",
    "cong_van",
    "unknown",
}


def choose_document_intent(rule_intent: str, payload_intent: Any) -> str:
    if rule_intent and rule_intent != "unknown":
        return rule_intent
    if isinstance(payload_intent, str) and payload_intent.strip():
        return normalize_for_rules(payload_intent).replace(" ", "_")
    return rule_intent or "unknown"


def classify_document_intent(text: str) -> str:
    text = clean_ocr_text(text)
    lines = meaningful_lines(text, limit=90)
    normalized_lines = [normalize_for_rules(line) for line in lines]
    context = normalize_for_rules(document_classification_context(lines))
    sample = normalize_for_rules(text[:5000])

    if has_strong_to_trinh_signal(normalized_lines, context):
        return "to_trinh"
    if has_receipt_signal(normalized_lines, context):
        if any(line.startswith("bien lai") for line in normalized_lines):
            return "bien_lai"
        if any(line.startswith("phieu thu") for line in normalized_lines):
            return "phieu_thu"
        return "chung_tu_phi"
    if has_contract_signal(normalized_lines, context, sample):
        return "contract"
    if has_strong_decision_signal(normalized_lines, context):
        return "quyet_dinh"
    informational_intent = classify_informational_intent(normalized_lines, context)
    if informational_intent:
        return informational_intent
    if has_proposal_context(context):
        if "ho so trinh" in context:
            return "ho_so_trinh"
        return "van_ban_de_nghi"
    if has_approval_context(context):
        if "phe duyet" in context:
            return "van_ban_phe_duyet"
        return "van_ban_chap_thuan"

    document_title_prefixes = ("cong van", "bien ban", "bao cao", "thong bao")
    for normalized_line in normalized_lines[:30]:
        for prefix in document_title_prefixes:
            if normalized_line.startswith(prefix):
                return prefix.replace(" ", "_")
    return "unknown"


def document_classification_context(lines: list[str]) -> str:
    selected: list[str] = []
    for line in lines[:60]:
        normalized = normalize_for_rules(line)
        if normalized.startswith(("can cu", "xet de nghi", "theo de nghi")):
            break
        selected.append(line)
    return "\n".join(selected or lines[:30])


def has_strong_to_trinh_signal(normalized_lines: list[str], context: str) -> bool:
    if any(line.startswith("to trinh") for line in normalized_lines[:40]):
        return True
    return bool(re.search(r"(?:^|[\s/-])ttr(?:[\s/-]|$)", context))


def has_receipt_signal(normalized_lines: list[str], context: str) -> bool:
    if any(line.startswith(("bien lai", "phieu thu", "chung tu phi")) for line in normalized_lines[:40]):
        return True
    return "so tien" in context and any(keyword in context for keyword in ("le phi", "phi tham dinh", "phi xay dung"))


def has_contract_signal(normalized_lines: list[str], context: str, sample: str) -> bool:
    if any(line.startswith("hop dong") for line in normalized_lines[:40]):
        return True
    contract_hits = (
        "hop dong",
        "ben a",
        "ben b",
        "ben giao thau",
        "ben nhan thau",
        "gia tri hop dong",
        "gia hop dong",
        "ky ket",
    )
    return sum(hit in sample for hit in contract_hits) >= 3 or (
        "hop dong" in context and any(label in sample for label in ("gia tri hop dong", "gia hop dong"))
    )


def has_strong_decision_signal(normalized_lines: list[str], context: str) -> bool:
    if any(line.startswith("quyet dinh") for line in normalized_lines[:40]):
        return True
    return bool(re.search(r"(?:^|[\s/-])qd(?:[\s/-]|$)", context))


def has_proposal_context(context: str) -> bool:
    proposal_phrases = (
        "van ban de nghi",
        "ho so trinh",
        "de nghi",
        "trinh tham dinh",
        "trinh phe duyet",
        "trinh chap thuan",
        "xin chap thuan",
        "xin phe duyet",
        "kinh de nghi",
    )
    return any(phrase in context for phrase in proposal_phrases)


def has_approval_context(context: str) -> bool:
    approval_phrases = (
        "van ban phe duyet",
        "van ban chap thuan",
        "phe duyet",
        "chap thuan",
        "dong y",
        "cho phep",
    )
    return any(phrase in context for phrase in approval_phrases)


def classify_informational_intent(normalized_lines: list[str], context: str) -> Optional[str]:
    if "tham gia y kien" in context:
        return "tham_gia_y_kien"
    if "cung cap thong tin" in context:
        return "cung_cap_thong_tin"
    if "gop y" in context:
        return "gop_y"
    if any(line.startswith("giay moi") for line in normalized_lines[:40]) or "moi hop" in context:
        return "giay_moi"
    if "phan cong" in context:
        return "phan_cong"
    return None


def document_value_role(document_intent: str) -> str:
    if document_intent in PROPOSAL_INTENTS:
        return "submitted"
    if document_intent in APPROVAL_INTENTS:
        return "approved"
    if document_intent in CONTRACT_INTENTS:
        return "contract_approved"
    if document_intent in RECEIPT_INTENTS:
        return "receipt"
    if document_intent in NO_VALUE_INTENTS:
        return "none"
    return "none"


def heuristic_generic_extraction(text: str) -> dict[str, Any]:
    date_pattern = r"\b(?:ngày\s*)?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|(?:ngày|ngay)\s*[,:.]?\s*\d{1,2}\s*[,;:.]?\s*(?:tháng|thang)\s+\d{1,2}\s*[,;:.]?\s*(?:năm|nam)\s+\d{4}"
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
    document_intent = classify_document_intent(text)
    value_role = document_value_role(document_intent)
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

    approved_value = None
    if value_role == "approved":
        approved_value = extract_labeled_money(text, MONEY_LABELS_DECISION_TOTAL)
        if not approved_value:
            approved_value = extract_labeled_money(
                text,
                MONEY_LABELS_APPROVED,
                exclude_context_keywords=APPROVED_VALUE_EXCLUDE_CONTEXT_KEYWORDS,
            )
        if not approved_value:
            approved_value = extract_labeled_money(text, MONEY_LABELS_GENERAL_AMOUNT)
    elif value_role == "contract_approved":
        approved_value = extract_labeled_money(text, CONTRACT_VALUE_LABELS)
    elif value_role == "receipt" and env_bool("MAP_RECEIPT_AMOUNT_TO_APPROVED", False):
        approved_value = extract_labeled_money(text, MONEY_LABELS_GENERAL_AMOUNT)
    if approved_value:
        value, normalized_value, evidence, confidence = approved_value
        fields["approved_value"] = build_field(
            DOCUMENT_FIELDS["approved_value"],
            value=value,
            normalized_value=normalized_value,
            evidence=evidence,
            confidence=confidence,
        )

    submitted_value = None
    if value_role == "submitted":
        submitted_value = extract_labeled_money(text, MONEY_LABELS_SUBMITTED)
        if not submitted_value:
            submitted_value = extract_labeled_money(text, MONEY_LABELS_GENERAL_AMOUNT)
        if not submitted_value:
            submitted_value = extract_labeled_money(text, MONEY_LABELS_APPROVED)
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

    if key in {"document_number", "contract_number"}:
        normalized_document_number = normalize_document_number(source_value)
        if normalized_document_number:
            field["value"] = normalized_document_number
            field["normalized_value"] = normalized_document_number

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


def should_replace_field(existing: dict[str, Any], candidate: dict[str, Any], key: Optional[str] = None) -> bool:
    # If candidate has no value, don't replace
    if candidate.get("value") in (None, ""):
        return False

    if key in {"signed_or_effective_date", "signed_date"} and looks_like_reference_date_field(candidate):
        return False

    # If existing has no value, always replace with candidate
    if existing.get("value") in (None, ""):
        return True

    existing_confidence = safe_float(existing.get("confidence"))
    candidate_confidence = safe_float(candidate.get("confidence"))
    existing_source = existing.get("source", "rule")
    candidate_source = candidate.get("source", "rule")

    if (
        key in {"signed_or_effective_date", "signed_date"}
        and existing_source == "rule"
        and candidate_source == "llm"
        and existing_confidence >= 0.9
    ):
        return False

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


def looks_like_reference_date_field(field: dict[str, Any]) -> bool:
    text = " ".join(
        str(field.get(key) or "")
        for key in ("value", "normalized_value", "evidence")
    )
    normalized = normalize_for_rules(text)
    return any(keyword in normalized for keyword in DATE_REFERENCE_CONTEXT_KEYWORDS)


def apply_document_context_guardrails(
    fields: dict[str, dict[str, Any]],
    text: str,
    rule_fields: dict[str, dict[str, Any]],
) -> None:
    """Keep layout-sensitive fields deterministic after merging an LLM response."""
    if is_business_registration_document(text):
        fields["document_number"] = empty_field(DOCUMENT_FIELDS["document_number"])
    elif rule_fields.get("document_number"):
        fields["document_number"] = dict(rule_fields["document_number"])
    elif not is_valid_document_number_field(fields.get("document_number"), text):
        fields["document_number"] = empty_field(DOCUMENT_FIELDS["document_number"])

    for key in ("signed_or_effective_date", "issuer"):
        trusted = rule_fields.get(key)
        fields[key] = dict(trusted) if trusted else empty_field(DOCUMENT_FIELDS[key])

    approved = rule_fields.get("approved_value")
    if approved and "tong cong" in normalize_for_rules(str(approved.get("evidence") or "")):
        fields["approved_value"] = dict(approved)


def is_valid_document_number_field(field: Any, text: str) -> bool:
    if not isinstance(field, dict) or field.get("value") in (None, ""):
        return False
    candidate_text = " ".join(str(field.get(key) or "") for key in ("value", "evidence"))
    normalized = normalize_for_rules(candidate_text)
    if any(label in normalized for label in ("ma so doanh nghiep", "ma so thue", "so tai khoan")):
        return False

    header = "\n".join(meaningful_lines(text, limit=60))
    normalized_value = normalize_document_number(field.get("normalized_value") or field.get("value"))
    if not normalized_value:
        return False
    for line in meaningful_lines(header):
        if not is_document_number_line(line):
            continue
        if normalized_value in (normalize_document_number(line) or ""):
            return True
    return False


def apply_document_intent_value_rules(fields: dict[str, dict[str, Any]], document_intent: str) -> None:
    value_role = document_value_role(document_intent)
    if value_role == "submitted":
        approved = fields.get("approved_value") or empty_field(DOCUMENT_FIELDS["approved_value"])
        submitted = fields.get("submitted_value") or empty_field(DOCUMENT_FIELDS["submitted_value"])
        if approved.get("value") not in (None, "") and submitted.get("value") in (None, ""):
            fields["submitted_value"] = {
                **submitted,
                **approved,
                "label": DOCUMENT_FIELDS["submitted_value"],
                "confidence": min(safe_float(approved.get("confidence")) or 0.78, 0.84),
            }
        fields["approved_value"] = empty_field(DOCUMENT_FIELDS["approved_value"])
        return

    if value_role in {"approved", "contract_approved"}:
        approved = fields.get("approved_value") or empty_field(DOCUMENT_FIELDS["approved_value"])
        submitted = fields.get("submitted_value") or empty_field(DOCUMENT_FIELDS["submitted_value"])
        if submitted.get("value") not in (None, "") and approved.get("value") in (None, ""):
            fields["approved_value"] = {
                **approved,
                **submitted,
                "label": DOCUMENT_FIELDS["approved_value"],
                "confidence": min(safe_float(submitted.get("confidence")) or 0.78, 0.84),
            }
            fields["submitted_value"] = empty_field(DOCUMENT_FIELDS["submitted_value"])
        elif (
            submitted.get("value") not in (None, "")
            and approved.get("normalized_value") not in (None, "")
            and approved.get("normalized_value") == submitted.get("normalized_value")
        ):
            fields["submitted_value"] = empty_field(DOCUMENT_FIELDS["submitted_value"])

    if value_role in {"none", "receipt"}:
        if value_role != "receipt" or not env_bool("MAP_RECEIPT_AMOUNT_TO_APPROVED", False):
            fields["approved_value"] = empty_field(DOCUMENT_FIELDS["approved_value"])
        fields["submitted_value"] = empty_field(DOCUMENT_FIELDS["submitted_value"])


def extract_document_number(text: str) -> Optional[tuple[str, str]]:
    for line in meaningful_lines(text, limit=60):
        normalized_line = normalize_for_rules(line)
        if not is_document_number_line(line):
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
            return normalize_document_number(value) or value, line
    return None


def is_document_number_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:số|so)\s*[:.]?\s*[A-Z0-9Đ]", line, flags=re.IGNORECASE))


def normalize_document_number(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None

    raw = clean_text(str(value))
    # Normalize common OCR artefacts in numbering (fullwidth/ellipsis/middle dot)
    # so "981．．．/TTr-...", "981…/TTr-..." behave like "981.../TTr-...".
    raw = raw.translate(str.maketrans({"．": ".", "·": ".", "•": ".", "…": "..."}))
    raw = re.sub(r"^(?:số|so)\s*[:.]?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.split(r"\s+(?:ngày|ngay)\s+", raw, maxsplit=1, flags=re.IGNORECASE)[0]
    raw = raw.strip(" .,:;-")
    if not raw or not re.search(r"\d", raw):
        return None

    raw = re.sub(r"\s*/\s*", "/", raw)
    raw = re.sub(r"\s*-\s*", "-", raw)
    raw = re.sub(r"\s+", "", raw)
    raw = re.sub(r"\.+(?=/)", "", raw)

    if "/" in raw:
        prefix, suffix = raw.split("/", 1)
        normalized_prefix = normalize_document_number_prefix(prefix)
        suffix = suffix.strip(" .,:;-")
        if normalized_prefix and suffix:
            return f"{normalized_prefix}/{suffix}"
        if normalized_prefix:
            return normalized_prefix

    return raw.strip(" .,:;-")


def normalize_document_number_prefix(prefix: str) -> str:
    prefix = prefix.strip(" .,:;-")
    if not prefix:
        return ""
    # Normalize OCR artefacts: fullwidth dot "．", ellipsis "…", middle dot "·",
    # non-breaking / zero-width whitespace — so "981．．．", "981…", "9 8 1" all
    # collapse to "981" before the digits-only check below.
    prefix = prefix.translate(str.maketrans({
        "．": ".",
        "·": ".",
        "•": ".",
        "…": ".",
        " ": " ",
        "​": "",
        " ": " ",
    }))
    if re.fullmatch(r"[\d.\s]+", prefix):
        digits = re.sub(r"\D", "", prefix)
        return digits or prefix
    return re.sub(r"\s+", "", prefix)


def apply_filename_document_number_hint(extraction_data: dict[str, Any], filename: Optional[str]) -> dict[str, Any]:
    filename_prefix = extract_filename_document_number_prefix(filename)
    if not filename_prefix:
        return extraction_data

    fields = extraction_data.get("fields") if isinstance(extraction_data.get("fields"), dict) else {}
    field = fields.get("document_number") if isinstance(fields.get("document_number"), dict) else None
    if not field:
        return extraction_data

    current = normalize_document_number(field.get("normalized_value") or field.get("value"))
    if not current or "/" not in current:
        return extraction_data

    current_prefix, suffix = current.split("/", 1)
    if should_use_filename_document_number_prefix(current_prefix, filename_prefix):
        corrected = f"{filename_prefix}/{suffix}"
        field["value"] = corrected
        field["normalized_value"] = corrected
        field["source"] = field.get("source") or "rule"
        notes = extraction_data.setdefault("notes", [])
        if isinstance(notes, list):
            notes.append(f"Số văn bản được hiệu chỉnh theo tên file: {corrected}")

    return extraction_data


def extract_filename_document_number_prefix(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    stem = Path(filename).stem
    match = re.match(r"^\s*(\d{2,8})(?=[.\s_-])", stem)
    return match.group(1) if match else None


def should_use_filename_document_number_prefix(current_prefix: str, filename_prefix: str) -> bool:
    current_digits = re.sub(r"\D", "", current_prefix)
    filename_digits = re.sub(r"\D", "", filename_prefix)
    if not current_digits or not filename_digits or current_digits == filename_digits:
        return False
    if abs(len(filename_digits) - len(current_digits)) > 2:
        return False
    similarity = difflib.SequenceMatcher(None, current_digits, filename_digits).ratio()
    return similarity >= 0.75


DATE_REFERENCE_CONTEXT_KEYWORDS = (
    "luat",
    "luat so",
    "nghi dinh",
    "nghi quyet",
    "thong tu",
    "quyet dinh so",
    "cong van so",
    "can cu",
)


DATE_NOT_APPLICABLE_TITLE_PREFIXES = (
    "bao cao",
    "ke hoach",
    "ho so",
    "de cuong",
    "thuyet minh",
    "quy trinh",
    "he thong qlcl",
    "he thong quan ly chat luong",
)

ISSUER_NOT_APPLICABLE_TITLE_PREFIXES = (
    "bien ban",
    "bao cao",
    "hop dong",
    "ke hoach",
    "quy trinh",
    "ho so",
    "de cuong",
    "thuyet minh",
    "he thong qlcl",
    "he thong quan ly chat luong",
)


def is_business_registration_document(text: str) -> bool:
    sample = normalize_for_rules("\n".join(meaningful_lines(text, limit=80)))
    certificate_phrases = (
        "giay chung nhan dang ky doanh nghiep",
        "giay chung nhan dang ky kinh doanh",
        "giay xac nhan dang ky kinh doanh",
        "giay phep dang ky kinh doanh",
        "giay phep dkkd",
    )
    return any(phrase in sample for phrase in certificate_phrases) or (
        "ma so doanh nghiep" in sample and ("dang ky" in sample or "dkkd" in sample)
    )


def first_document_title_kind(text: str) -> Optional[str]:
    prefixes = (
        *ISSUER_NOT_APPLICABLE_TITLE_PREFIXES,
        "chung chi",
        "giay chung nhan",
        "giay xac nhan",
        "to trinh",
        "quyet dinh",
        "thong bao",
        "cong van",
        "giay moi",
        "van ban",
    )
    for line in meaningful_lines(text, limit=90):
        normalized = normalize_for_rules(line)
        normalized = re.sub(r"^\d+[.)\s-]+", "", normalized)
        for prefix in prefixes:
            if normalized.startswith(prefix):
                return prefix
    return None


def is_date_not_applicable_document(text: str) -> bool:
    kind = first_document_title_kind(text)
    return bool(kind and kind.startswith(DATE_NOT_APPLICABLE_TITLE_PREFIXES))


def is_issuer_not_applicable_document(text: str) -> bool:
    kind = first_document_title_kind(text)
    return bool(kind and kind.startswith(ISSUER_NOT_APPLICABLE_TITLE_PREFIXES))


def extract_business_registration_date(text: str) -> Optional[tuple[str, Any, str, float]]:
    change_pattern = re.compile(
        r"((?:đăng|dang)\s+(?:ký|ky)\s+thay\s+(?:đổi|doi)\s+lần\s+(?:thứ|thu)\s*[:\-]?\s*(\d{1,3})"
        r"[^\n]*(?:\n(?!\s*(?:đăng|dang)\s+(?:ký|ky)\s+thay\s+(?:đổi|doi))[^\n]*){0,2})",
        flags=re.IGNORECASE,
    )
    candidates: list[tuple[int, int, str, str]] = []
    for match in change_pattern.finditer(text):
        block = clean_text(match.group(1))
        date = extract_first_date(block)
        if date:
            candidates.append((int(match.group(2)), match.start(), date, block))

    if candidates:
        _change_number, _position, date, evidence = max(candidates, key=lambda item: (item[0], item[1]))
        normalized = normalize_date(date)
        display = format_display_date(normalized) if normalized else None
        if normalized:
            return display or date, normalized, evidence, 0.96

    initial_pattern = re.search(
        r"((?:đăng|dang)\s+(?:ký|ky)\s+lần\s+(?:đầu|dau)[^\n]{0,160})",
        text,
        flags=re.IGNORECASE,
    )
    if initial_pattern:
        evidence = clean_text(initial_pattern.group(1))
        date = extract_first_date(evidence)
        normalized = normalize_date(date) if date else None
        display = format_display_date(normalized) if normalized else None
        if normalized:
            return display or date, normalized, evidence, 0.88
    return None


def extract_signed_or_effective_date(text: str) -> Optional[tuple[str, Any, str, float]]:
    if is_business_registration_document(text):
        return extract_business_registration_date(text)
    if is_date_not_applicable_document(text):
        return None

    # Priority 1: Date under the national motto ("Độc lập - Tự do - Hạnh phúc")
    # in the form "<địa danh>, ngày DD tháng MM năm YYYY". The location prefix
    # with a comma is REQUIRED to avoid matching legal references like
    # "Luật số 43/2013/QH13 ngày 26 tháng 11 năm 2013".
    # Vietnamese place-name prefix: Title-cased words (e.g. "Hà Nội", "TP. Hồ Chí Minh"),
    # 1-6 tokens, ending with a comma immediately before "ngày".
    location_date_pattern = re.compile(
        r"([A-ZĐÀ-Ỹ][\wÀ-ỹ.]*(?:[^\S\r\n]+[A-ZĐÀ-Ỹ\d][\wÀ-ỹ.]*){0,5})[^\S\r\n]*,[^\S\r\n]*"
        r"(?:ngày|ngay)\s*[,:.]?\s*(\d{1,2})\s*[,;:.]?\s*"
        r"(?:tháng|thang)\s+(\d{1,2})\s*[,;:.]?\s*(?:năm|nam)\s+(\d{4})",
        flags=re.IGNORECASE,
    )

    # Prefer scanning right after the national motto when present.
    motto_match = re.search(
        r"độc\s*lập\s*[-–—]\s*tự\s*do\s*[-–—]\s*hạnh\s*phúc",
        text,
        flags=re.IGNORECASE,
    )
    search_start = motto_match.end() if motto_match else 0
    header_text = text[search_start : search_start + 1500]

    for header_match in location_date_pattern.finditer(header_text):
        context_start = max(0, header_match.start() - 120)
        context = normalize_for_rules(header_text[context_start : header_match.end()])
        if any(keyword in context for keyword in DATE_REFERENCE_CONTEXT_KEYWORDS):
            continue
        _location, day, month, year = header_match.groups()
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
        context = normalize_for_rules(text[max(0, match.start() - 100):match.end() + 50])
        if any(keyword in context for keyword in DATE_REFERENCE_CONTEXT_KEYWORDS):
            continue

        date = extract_first_date(raw)
        if date:
            normalized = normalize_date(date)
            display = format_display_date(normalized) if normalized else None
            return display or date, normalized or date, clean_text(match.group(0)), 0.86
        continue

    # Priority 3: Date in meaningful lines (fallback)
    for line in meaningful_lines(text, limit=80):
        if "ngày" not in line.lower() and not re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", line):
            continue

        # Skip lines that look like legal references
        normalized_line = normalize_for_rules(line)
        if any(keyword in normalized_line for keyword in DATE_REFERENCE_CONTEXT_KEYWORDS):
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
    "tổng mức đầu tư",
    "tổng mức đầu tư dự kiến",
    "tổng mức đầu tư xây dựng",
    "tổng mức dự toán",
    "tổng mức được duyệt",
    "tổng mức duyệt",
    "tổng kinh phí được duyệt",
    "tổng kinh phí",
    "kinh phí thực hiện",
    "nguồn vốn đầu tư",
]

MONEY_LABELS_GENERAL_AMOUNT = [
    "tổng giá trị",
    "tổng giá tiền",
    "giá tiền",
    "tổng số tiền",
    "số tiền",
]

MONEY_LABELS_DECISION_TOTAL = [
    "tổng giá trị phần công việc",
    "tổng giá trị các gói thầu",
    "tổng giá các gói thầu",
    "tổng cộng",
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

APPROVED_VALUE_EXCLUDE_CONTEXT_KEYWORDS = [
    "de nghi",
    "xin phe duyet",
    "trinh duyet",
    "trinh phe duyet",
]

TOTAL_INVESTMENT_LABELS = {
    "tong muc dau tu",
    "tong muc dau tu du kien",
    "tong muc dau tu xay dung",
    "tong muc du toan",
    "tong muc duoc duyet",
    "tong muc duyet",
    "tong kinh phi duoc duyet",
    "tong kinh phi",
    "kinh phi thuc hien",
    "nguon von dau tu",
    "tong gia tri",
    "tong gia tri phan cong viec",
    "tong gia tri cac goi thau",
    "tong gia cac goi thau",
    "tong gia tien",
    "tong so tien",
    "tong cong",
}

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


def extract_labeled_money(
    text: str,
    labels: list[str],
    exclude_context_keywords: Optional[list[str]] = None,
) -> Optional[tuple[str, Any, str, float]]:
    normalized_text = normalize_for_rules(text)
    best: Optional[tuple[str, Any, str, float]] = None
    best_rank: Optional[tuple[int, int, int]] = None

    for label in labels:
        normalized_label = normalize_for_rules(label)
        prefer_largest_money = normalized_label in TOTAL_INVESTMENT_LABELS
        start = 0
        while True:
            idx = normalized_text.find(normalized_label, start)
            if idx == -1:
                break
            raw_idx = approximate_raw_index(text, normalized_text, idx)
            window = text[raw_idx : raw_idx + 320]
            if should_skip_money_context(window, exclude_context_keywords):
                start = idx + len(normalized_label)
                continue
            money = find_largest_money_with_offset(window) if prefer_largest_money else find_first_money(window)
            if money:
                if prefer_largest_money:
                    value, normalized_value, offset = money
                    evidence_window = window[max(0, offset - 220) : offset + len(value)]
                    evidence_lines = meaningful_lines(evidence_window)
                    evidence = clean_text(" ".join(evidence_lines[-4:])) if evidence_lines else first_meaningful_line(evidence_window)
                else:
                    value, normalized_value = money
                    evidence = first_meaningful_line(window)
                candidate = (value, normalized_value, evidence, 0.84)
                rank = money_candidate_rank(idx, normalized_value, prefer_largest_money)
                if best_rank is None or rank > best_rank:
                    best = candidate
                    best_rank = rank
            start = idx + len(normalized_label)

    if best:
        return best

    for line in meaningful_lines(text, limit=200):
        normalized_line = normalize_for_rules(line)
        if any(normalize_for_rules(label) in normalized_line for label in labels):
            if should_skip_money_context(line, exclude_context_keywords):
                continue
            money = find_first_money(line)
            if money:
                value, normalized_value = money
                return value, normalized_value, line, 0.78
    return None


def money_candidate_rank(idx: int, normalized_value: Any, prefer_largest_money: bool) -> tuple[int, int, int]:
    if prefer_largest_money and isinstance(normalized_value, int):
        return 1, normalized_value, -idx
    return 0, -idx, 0


def should_skip_money_context(text: str, exclude_context_keywords: Optional[list[str]]) -> bool:
    if not exclude_context_keywords:
        return False
    money_match = MONEY_PATTERN.search(text[:180])
    context = text[: money_match.start()] if money_match else text[:140]
    normalized = normalize_for_rules(context)
    return any(keyword in normalized for keyword in exclude_context_keywords)


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
    if is_issuer_not_applicable_document(text):
        return None

    lines = meaningful_lines(text, limit=45)
    if not lines:
        return None

    if is_certificate_document(text):
        issuer_lines = extract_certificate_issuer_lines(lines)
    else:
        issuer_lines = extract_administrative_issuer_lines(lines)
    if not issuer_lines:
        return None

    issuer = clean_text(" ".join(issuer_lines))
    return issuer, "\n".join(issuer_lines)


def is_certificate_document(text: str) -> bool:
    kind = first_document_title_kind(text)
    return kind in {"chung chi", "giay chung nhan", "giay xac nhan"}


def extract_certificate_issuer_lines(lines: list[str]) -> list[str]:
    title_index = len(lines)
    for index, line in enumerate(lines):
        normalized = normalize_for_rules(line)
        if normalized.startswith(("chung chi", "giay chung nhan", "giay xac nhan")):
            title_index = index
            break

    candidates: list[str] = []
    for line in lines[:title_index]:
        if issuer_line_is_excluded(line):
            continue
        if is_government_authority_line(line):
            candidates.append(line)
    return candidates[:5]


def extract_administrative_issuer_lines(lines: list[str]) -> list[str]:
    number_index: Optional[int] = None
    for index, line in enumerate(lines):
        if is_document_number_line(line):
            number_index = index
            break

    if number_index is not None:
        header_lines = lines[max(0, number_index - 7) : number_index]
    else:
        stop_index = len(lines)
        for index, line in enumerate(lines):
            normalized = normalize_for_rules(line)
            if normalized.startswith(("cong hoa", "doc lap")) or first_document_title_kind(line):
                stop_index = index
                break
        header_lines = lines[:stop_index]

    eligible = [line for line in header_lines if not issuer_line_is_excluded(line)]
    anchor_indexes = [index for index, line in enumerate(eligible) if is_probable_issuer_line(line)]
    if not anchor_indexes:
        return []

    start = anchor_indexes[0]
    end = anchor_indexes[-1]
    issuer_lines = eligible[start : end + 1]
    return issuer_lines[-6:]


def issuer_line_is_excluded(line: str) -> bool:
    normalized = normalize_for_rules(line)
    if not normalized or len(normalized) < 3:
        return True
    if is_document_number_line(line):
        return True
    excluded_prefixes = (
        "cong hoa xa hoi chu nghia",
        "doc lap",
        "socialist republic",
        "ngay ",
        "kinh gui",
        "noi cap",
        "noi nhan",
        "to trinh",
        "quyet dinh",
        "thong bao",
        "cong van",
        "giay moi",
        "van ban",
        "bien ban",
        "bao cao",
        "hop dong",
        "ke hoach",
        "quy trinh",
        "ho so",
        "de cuong",
        "thuyet minh",
        "chung chi",
        "giay chung nhan",
        "giay xac nhan",
    )
    return normalized.startswith(excluded_prefixes)


def is_government_authority_line(line: str) -> bool:
    normalized = normalize_for_rules(line)
    return bool(
        re.search(r"(?:^|\s)(?:bo|cuc|so|ubnd|tong cuc|chi cuc)(?:\s|$)", normalized)
        or "uy ban nhan dan" in normalized
        or "phong dang ky kinh doanh" in normalized
    )


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
        if looks_like_unit_hint(text, match, value):
            continue
        normalized = normalize_money(value)
        if normalized is not None:
            return value, normalized
    return None


def find_largest_money(text: str) -> Optional[tuple[str, Any]]:
    money = find_largest_money_with_offset(text)
    if not money:
        return None
    value, normalized, _offset = money
    return value, normalized


def find_largest_money_with_offset(text: str) -> Optional[tuple[str, Any, int]]:
    best: Optional[tuple[str, Any]] = None
    best_value = -1
    best_offset = 0
    for match in MONEY_PATTERN.finditer(text):
        value = clean_text(match.group(0))
        if not re.search(r"\d", value):
            continue
        if looks_like_non_money_number(text, match, value):
            continue
        if looks_like_unit_hint(text, match, value):
            continue
        normalized = normalize_money(value)
        if isinstance(normalized, int) and normalized > best_value:
            best = (value, normalized)
            best_value = normalized
            best_offset = match.start()
    if not best:
        return None
    value, normalized = best
    return value, normalized, best_offset


def looks_like_non_money_number(text: str, match: re.Match, value: str) -> bool:
    tail = text[match.end() : match.end() + 2]
    if "%" in tail:
        return True

    normalized = normalize_for_rules(value)
    has_money_unit = any(unit in normalized for unit in ("dong", "vnd", "vnđ", "ty", "trieu"))
    has_thousand_separator = bool(re.search(r"\d{1,3}(?:[.\s]\d{3})+", value))
    digit_count = len(re.sub(r"\D", "", value))
    if has_money_unit or digit_count >= 7:
        return False
    if has_thousand_separator and digit_count < 7:
        return True

    return True


# Column headers in Vietnamese tables often annotate a money column with the
# unit it uses, e.g. "Giá gói thầu (1.000 đồng)" or "Giá trị (triệu đồng)".
# Those parenthesised hints must NOT be treated as the actual approved amount.
_UNIT_HINT_VALUES = {1000, 1_000_000, 1_000_000_000}


def looks_like_unit_hint(text: str, match: re.Match, value: str) -> bool:
    """Return True when the matched money is just a column-unit annotation."""
    start, end = match.start(), match.end()
    before = text[max(0, start - 40) : start]
    after = text[end : end + 8]
    # Only consider candidates directly inside parentheses.
    if ")" not in after:
        return False
    last_open = before.rfind("(")
    last_close = before.rfind(")")
    if last_open <= last_close:
        return False
    # Anything other than whitespace / common label words between '(' and the
    # number means it's likely a real amount (e.g. "(Bằng chữ: 193 tỷ đồng)").
    in_paren = before[last_open + 1 :]
    if re.search(r"[A-Za-zÀ-ỹ]", in_paren):
        return False
    normalized = normalize_money(value)
    return isinstance(normalized, int) and normalized in _UNIT_HINT_VALUES


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
    re.compile(
        r"(?:ngày|ngay)\s*[,:.]?\s*\d{1,2}\s*[,;:.]?\s*"
        r"(?:tháng|thang)\s+\d{1,2}\s*[,;:.]?\s*(?:năm|nam)\s+\d{4}",
        flags=re.IGNORECASE,
    ),
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
        r"(?:ngày|ngay)\s*[,:.]?\s*(\d{1,2})\s*[,;:.]?\s*"
        r"(?:tháng|thang)\s+(\d{1,2})\s*[,;:.]?\s*(?:năm|nam)\s+(\d{4})",
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
    return clean_ocr_plain_text(text)


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


async def extract_information(
    text: str,
    extraction_type: Optional[str] = DEFAULT_EXTRACTION_TYPE,
) -> dict[str, Any]:
    with timed_stage("extraction_clean_text"):
        text = clean_ocr_text(text)
    extraction_type = normalize_extraction_type(extraction_type)
    llm_config = get_llm_config()
    local_enabled = env_bool("LOCAL_EXTRACTION_ENABLED", True)
    entity_extraction_enabled = env_bool("LLM_ENTITY_EXTRACTION_ENABLED", True)
    execution_mode = llm_execution_mode()

    local_data = None
    if local_enabled:
        with timed_stage("rule_parse", extraction_type=extraction_type):
            local_data = normalize_result({}, text, payload_source="rule", extraction_type=extraction_type)

    should_call_llm = False
    decision_reason = "llm_config_unavailable"
    if llm_config and execution_mode == "always":
        should_call_llm = True
        decision_reason = "mode_always"
    elif llm_config and execution_mode == "adaptive":
        if local_data is None:
            should_call_llm = True
            decision_reason = "local_parser_disabled"
        else:
            should_call_llm, decision_reason = adaptive_llm_decision(local_data, text, extraction_type)
    elif execution_mode == "off":
        decision_reason = "mode_off"

    record_timing_event(
        "llm_decision",
        mode=execution_mode,
        call_llm=should_call_llm,
        reason=decision_reason,
    )

    if should_call_llm and llm_config:
        with timed_stage("llm_prompt_build"):
            if entity_extraction_enabled and local_data:
                requested_fields = requested_llm_fields(local_data, extraction_type)
                prompt = build_entity_extraction_prompt(
                    text, local_data, requested_fields=requested_fields
                )
                llm_mode = "entity_extraction"
            else:
                requested_fields = list(
                    (CONTRACT_FIELDS if extraction_type == "contract" else DOCUMENT_FIELDS).keys()
                )
                prompt = build_prompt(text, extraction_type=extraction_type)
                llm_mode = "direct"
            llm_config = {
                **llm_config,
                "response_schema": build_llm_response_schema(
                    extraction_type, requested_fields
                ),
                "max_output_tokens": (
                    int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS_CONTRACT", "3072"))
                    if extraction_type == "contract"
                    else int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS_DOCUMENT", "1536"))
                ),
                "system_instruction": (
                    LLM_DELTA_SYSTEM_INSTRUCTION
                    if os.getenv("LLM_PROMPT_MODE", "delta").lower() == "delta"
                    else LLM_SYSTEM_INSTRUCTION
                ),
            }

        try:
            with timed_stage(
                "llm_call",
                provider=llm_config.get("provider"),
                model=llm_config.get("model"),
                prompt_chars=len(prompt),
            ):
                parsed = await call_llm_extraction(llm_config, prompt)
            with timed_stage("llm_result_normalize"):
                data = normalize_result(parsed, text, payload_source="llm", extraction_type=extraction_type)
            data["pipeline"] = "local_with_llm_fallback" if llm_mode == "entity_extraction" else "llm"
            data["llm_extraction_mode"] = llm_mode
            data["llm_fallback_used"] = True
            data["llm_entity_extraction_used"] = llm_mode == "entity_extraction"
            if local_data:
                data["local_confidence"] = local_data["local_confidence"]
            data["needs_review"] = calculate_local_confidence(data) < env_float("LOCAL_CONFIDENCE_THRESHOLD", 0.68)
            return {
                "provider": f"{llm_config['provider']}_{llm_mode}",
                "model": llm_config["model"],
                "data": data,
            }
        except Exception as exc:
            if local_data is not None:
                local_data["pipeline"] = "local"
                local_data["llm_fallback_error"] = compact_error(exc)
                local_data["needs_review"] = True
                return {"provider": "local", "model": None, "data": local_data}

            data = normalize_result({}, text, payload_source="rule", extraction_type=extraction_type)
            data["pipeline"] = "llm_failed_no_fallback"
            data["llm_error"] = compact_error(exc)
            data["needs_review"] = True
            return {"provider": "error", "model": None, "data": data}

    if local_data is not None:
        local_data["pipeline"] = "local"
        return {
            "provider": "local",
            "model": None,
            "data": local_data,
        }

    data = normalize_result({}, text, payload_source="rule", extraction_type=extraction_type)
    data["pipeline"] = "no_extraction_available"
    data["needs_review"] = True
    return {
        "provider": "none",
        "model": None,
        "data": data,
    }


def llm_execution_mode() -> str:
    legacy_fallback = os.getenv("LLM_FALLBACK_ENABLED")
    if legacy_fallback is not None and not env_bool("LLM_FALLBACK_ENABLED", True):
        return "off"

    explicit = os.getenv("LLM_EXECUTION_MODE")
    if explicit:
        mode = explicit.strip().lower()
        if mode in {"always", "adaptive", "off"}:
            return mode

    return "adaptive"


def adaptive_llm_decision(
    local_data: dict[str, Any],
    text: str,
    extraction_type: str,
) -> tuple[bool, str]:
    fields = local_data.get("fields") if isinstance(local_data.get("fields"), dict) else {}
    threshold = env_float("LLM_ADAPTIVE_CONFIDENCE_THRESHOLD", 0.72)

    if extraction_type == "contract":
        core_fields = ("contract_name", "contract_number", "signed_date", "contract_value", "contractor_name")
        present = [
            fields[key]
            for key in core_fields
            if isinstance(fields.get(key), dict) and fields[key].get("value") not in (None, "")
        ]
        if len(present) < 4:
            return True, f"contract_core_fields_{len(present)}_of_5"
        average_confidence = sum(safe_float(field.get("confidence")) for field in present) / len(present)
        if average_confidence < threshold:
            return True, "contract_core_confidence_low"
        return False, "contract_core_fields_sufficient"

    kind = first_document_title_kind(text)
    if not kind:
        title_field = fields.get("title") if isinstance(fields.get("title"), dict) else {}
        extracted_title = str(title_field.get("value") or "").strip()
        if extracted_title:
            kind = first_document_title_kind(extracted_title)
    internal_document = bool(kind and kind.startswith(ISSUER_NOT_APPLICABLE_TITLE_PREFIXES))
    business_registration = is_business_registration_document(text)
    expected_fields: list[str] = []

    if not kind:
        expected_fields.append("title")
    if not internal_document and not business_registration:
        expected_fields.extend(("document_number", "signed_or_effective_date", "issuer"))
    elif business_registration and not internal_document:
        expected_fields.extend(("signed_or_effective_date", "issuer"))

    missing = [
        key
        for key in expected_fields
        if not isinstance(fields.get(key), dict) or fields[key].get("value") in (None, "")
    ]
    if missing:
        return True, f"missing_{'_'.join(missing)}"

    present = [fields[key] for key in expected_fields if isinstance(fields.get(key), dict)]
    if present:
        average_confidence = sum(safe_float(field.get("confidence")) for field in present) / len(present)
        if average_confidence < threshold:
            return True, "document_required_confidence_low"

    value_role = document_value_role(str(local_data.get("document_intent") or "unknown"))
    value_key = "submitted_value" if value_role == "submitted" else "approved_value" if value_role in {"approved", "contract_approved"} else None
    if value_key and document_contains_money_signal(text):
        field = fields.get(value_key) if isinstance(fields.get(value_key), dict) else {}
        if field.get("value") in (None, ""):
            return True, f"missing_{value_key}"

    return False, "local_fields_sufficient"


def document_contains_money_signal(text: str) -> bool:
    normalized = normalize_for_rules(text[:20000])
    labels = (
        "gia tri",
        "gia goi thau",
        "tong muc dau tu",
        "du toan",
        "tong kinh phi",
        "tong cong",
    )
    return any(label in normalized for label in labels) and bool(MONEY_PATTERN.search(text[:20000]))


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
    location = os.getenv("GEMINI_VERTEX_LOCATION") or os.getenv("VERTEX_AI_LOCATION") or "global"
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
            json=build_gemini_payload(prompt, config),
        )
        response.raise_for_status()
    return parse_gemini_response(response.json())


async def call_vertex_gemini_extraction(config: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Call Vertex Gemini through a reusable async GenAI client."""
    try:
        from google import genai
        from google.genai import types

        key = (str(config["project_id"]), str(config["location"]))
        client = _GENAI_CLIENTS.get(key)
        if client is None:
            client = genai.Client(
                vertexai=True,
                project=config["project_id"],
                location=config["location"],
            )
            _GENAI_CLIENTS[key] = client

        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=config["model"],
                contents=[prompt],
                config=types.GenerateContentConfig(
                    **build_vertex_generation_config_kwargs(types, config)
                ),
            ),
            timeout=env_float("GEMINI_TIMEOUT_SECONDS", 90.0),
        )
        response_text = response if isinstance(response, str) else response.text
        return parse_json_content(str(response_text or ""))

    except ImportError:
        # Fallback to old REST API if google-genai not installed
        access_token = await asyncio.to_thread(get_google_access_token)
        model_path = vertex_model_path(config["project_id"], config["location"], config["model"])
        hostname = (
            "aiplatform.googleapis.com"
            if config["location"] == "global"
            else f"{config['location']}-aiplatform.googleapis.com"
        )
        url = f"https://{hostname}/v1/{model_path}:generateContent"
        async with httpx.AsyncClient(timeout=env_float("GEMINI_TIMEOUT_SECONDS", 90.0)) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=build_gemini_payload(prompt, config),
            )
            response.raise_for_status()
        return parse_gemini_response(response.json())


def build_vertex_generation_config_kwargs(
    types: Any, config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    config = config or {}
    generation_config_kwargs = {
        "system_instruction": config.get("system_instruction") or LLM_DELTA_SYSTEM_INSTRUCTION,
        "temperature": 0,
        "response_mime_type": "application/json",
        "max_output_tokens": int(config.get("max_output_tokens") or 1536),
    }
    if isinstance(config.get("response_schema"), dict):
        generation_config_kwargs["response_schema"] = config["response_schema"]
    # Disable internal "thinking" on Gemini 2.5 family to reduce latency.
    thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
    try:
        generation_config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )
    except AttributeError:
        pass
    return generation_config_kwargs


def build_gemini_payload(
    prompt: str, config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    config = config or {}
    generation_config: dict[str, Any] = {
        "temperature": 0,
        "responseMimeType": "application/json",
        "maxOutputTokens": int(config.get("max_output_tokens") or 1536),
    }
    # Disable "thinking" on Gemini 2.5 family for faster latency.
    thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
    generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    if isinstance(config.get("response_schema"), dict):
        generation_config["responseSchema"] = config["response_schema"]

    return {
        "systemInstruction": {
            "parts": [
                {
                    "text": config.get("system_instruction")
                    or LLM_DELTA_SYSTEM_INSTRUCTION
                }
            ]
        },
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


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_prompt_fields(fields: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, field in fields.items():
        if not isinstance(field, dict):
            continue
        value = field.get("value")
        normalized_value = field.get("normalized_value")
        evidence = field.get("evidence")
        confidence = safe_float(field.get("confidence"))
        if value in (None, "") and normalized_value in (None, "") and evidence in (None, "") and confidence <= 0:
            continue
        compact[key] = {
            "v": _prompt_value(value, 500),
            "n": _prompt_value(normalized_value, 300),
            "e": _prompt_value(evidence, 200),
            "c": field.get("confidence"),
        }
    return compact


def _prompt_value(value: Any, limit: int) -> Any:
    return value[:limit] if isinstance(value, str) else value


def requested_llm_fields(local_data: dict[str, Any], extraction_type: str) -> list[str]:
    schema = CONTRACT_FIELDS if extraction_type == "contract" else DOCUMENT_FIELDS
    fields = local_data.get("fields") if isinstance(local_data.get("fields"), dict) else {}
    threshold = env_float("LLM_ADAPTIVE_CONFIDENCE_THRESHOLD", 0.72)
    requested = []
    for key in schema:
        field = fields.get(key) if isinstance(fields.get(key), dict) else {}
        if field.get("value") in (None, "") or safe_float(field.get("confidence")) < threshold:
            requested.append(key)
    return requested or list(schema)


def build_llm_response_schema(
    extraction_type: str, requested_fields: list[str]
) -> dict[str, Any]:
    field_object = {
        "type": "OBJECT",
        "properties": {
            "value": {"type": "STRING", "nullable": True},
            "normalized_value": {"type": "STRING", "nullable": True},
            "evidence": {"type": "STRING", "nullable": True},
            "confidence": {"type": "NUMBER"},
        },
    }
    generic_names = (
        (
            "document_title_or_type",
            "project_name_candidates",
            "task_title_candidates",
            "work_item_candidates",
            "procurement_package_candidates",
            "task_keywords",
            "dates",
            "monetary_amounts",
            "document_numbers",
        )
        if extraction_type == "document"
        else (
            "document_title_or_type",
            "task_title_candidates",
            "work_item_candidates",
            "contract_parties",
            "monetary_amounts",
            "dates",
            "document_numbers",
            "guarantee_terms",
        )
    )
    generic_properties = {
        name: {"type": "ARRAY", "items": {"type": "STRING"}}
        for name in generic_names
    }
    generic_properties["document_title_or_type"] = {
        "type": "STRING",
        "nullable": True,
    }
    return {
        "type": "OBJECT",
        "properties": {
            "document_type": {"type": "STRING"},
            "document_intent": {"type": "STRING"},
            "fields": {
                "type": "OBJECT",
                "properties": {key: field_object for key in requested_fields},
            },
            "generic_extraction": {
                "type": "OBJECT",
                "properties": generic_properties,
            },
            "notes": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
    }


def build_entity_extraction_prompt(
    text: str,
    local_data: dict[str, Any],
    requested_fields: Optional[list[str]] = None,
) -> str:
    fields = local_data.get("fields") if isinstance(local_data.get("fields"), dict) else {}
    compact_fields = compact_prompt_fields(fields)
    if local_data.get("screen") == "contract":
        requested_fields = requested_fields or requested_llm_fields(local_data, "contract")
        return build_contract_entity_extraction_prompt(
            text, local_data, compact_fields, requested_fields
        )

    requested_fields = requested_fields or requested_llm_fields(local_data, "document")
    trimmed_text = select_entity_extraction_text(text, "document")
    return f"""
Trích xuất document từ OCR tiếng Việt. {COMMON_EXTRACTION_RULES}
Fields cần bổ sung/kiểm tra: {compact_json(requested_fields)}
Local fields gợi ý: {compact_json(compact_fields)}
{DOCUMENT_FIELD_RULES}
{FIELD_OBJECT_HINT}
Entity hints: project_name_candidates, task_title_candidates, work_item_candidates, procurement_package_candidates, task_keywords, monetary_entities.
OCR:
\"\"\"{trimmed_text}\"\"\"
""".strip()


def build_contract_entity_extraction_prompt(
    text: str,
    local_data: dict[str, Any],
    compact_fields: dict[str, Any],
    requested_fields: Optional[list[str]] = None,
) -> str:
    requested_fields = requested_fields or requested_llm_fields(local_data, "contract")
    trimmed_text = select_entity_extraction_text(text, "contract")
    return f"""
Trích xuất hợp đồng từ OCR tiếng Việt. {COMMON_EXTRACTION_RULES}
{CONTRACT_FIELD_RULES}
contract_forms={compact_json(CONTRACT_FORMS)}
contractor_groups={compact_json(CONTRACTOR_GROUPS)}
Fields cần bổ sung/kiểm tra: {compact_json(requested_fields)}
Local fields gợi ý: {compact_json(compact_fields)}
{FIELD_OBJECT_HINT}
OCR:
\"\"\"{trimmed_text}\"\"\"
""".strip()


def select_entity_extraction_text(text: str, extraction_type: str = "document") -> str:
    default_limit = 10000 if extraction_type == "contract" else 5000
    specific_limit = int(
        os.getenv(
            "LLM_CONTRACT_MAX_OCR_CHARS"
            if extraction_type == "contract"
            else "LLM_DOCUMENT_MAX_OCR_CHARS",
            str(default_limit),
        )
    )
    configured_limit = os.getenv("LLM_ENTITY_MAX_CHARS")
    if configured_limit:
        # Legacy deployments often set this to 30k. Delta mode must still honor
        # the smaller per-document latency budget.
        max_chars = (
            min(int(configured_limit), specific_limit)
            if os.getenv("LLM_PROMPT_MODE", "delta").lower() == "delta"
            else int(configured_limit)
        )
    else:
        max_chars = specific_limit
    lines = non_boilerplate_lines(text, limit=900)
    if not lines:
        return text[:max_chars]

    head_line_count = min(70, len(lines))
    selected = lines[:head_line_count]
    keywords = (
        "gia tri",
        "gia goi thau",
        "gia hop dong",
        "tong gia tri",
        "du toan",
        "tong muc",
        "tong kinh phi",
        "goi thau",
        "du an",
        "cong trinh",
        "hang muc",
        "hop dong",
        "nha thau",
        "vat",
        "bao lanh",
        "bao dam",
        "tam ung",
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
    for line in lines[head_line_count:]:
        normalized = normalize_for_rules(line)
        if any(keyword in normalized for keyword in keywords):
            selected.append(line)
        if len("\n".join(selected)) >= max_chars:
            break

    return "\n".join(unique_values(selected))[:max_chars]


def build_fallback_prompt(text: str, local_data: dict[str, Any]) -> str:
    fields = local_data.get("fields") if isinstance(local_data.get("fields"), dict) else {}
    compact_fields = compact_prompt_fields(fields)
    if local_data.get("screen") == "contract":
        return build_contract_entity_extraction_prompt(text, local_data, compact_fields)

    snippets = select_fallback_snippets(text, local_data)
    return f"""
Sửa/bổ sung field còn thiếu/sai từ snippets. {COMMON_EXTRACTION_RULES}
Loại hiện tại={local_data.get("document_type")}; intent={local_data.get("document_intent")}; confidence={local_data.get("local_confidence")}
Local fields={compact_json(compact_fields)}
{DOCUMENT_FIELD_RULES}
{FIELD_OBJECT_HINT}
{DOCUMENT_SCHEMA_HINT}
OCR snippets:
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

    return "\n".join(unique_values(selected))[: int(os.getenv("LLM_FALLBACK_MAX_CHARS", "7000"))]


def build_prompt(text: str, extraction_type: Optional[str] = DEFAULT_EXTRACTION_TYPE) -> str:
    extraction_type = normalize_extraction_type(extraction_type)
    trimmed_text = text[: int(os.getenv("LLM_MAX_OCR_CHARS", "60000"))]
    if extraction_type == "contract":
        return f"""
Trích xuất hợp đồng từ OCR tiếng Việt. {COMMON_EXTRACTION_RULES}
{CONTRACT_FIELD_RULES}
contract_forms={compact_json(CONTRACT_FORMS)}
contractor_groups={compact_json(CONTRACTOR_GROUPS)}
{FIELD_OBJECT_HINT}
{CONTRACT_SCHEMA_HINT}
OCR text:
\"\"\"{trimmed_text}\"\"\"
""".strip()

    return f"""
Trích xuất document từ OCR tiếng Việt. {COMMON_EXTRACTION_RULES}
{DOCUMENT_FIELD_RULES}
{FIELD_OBJECT_HINT}
{DOCUMENT_SCHEMA_HINT}
OCR text:
\"\"\"{trimmed_text}\"\"\"
""".strip()
