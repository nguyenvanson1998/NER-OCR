import asyncio

from app.services import extraction


SAMPLE_TEXT = """
CÔNG TY CP ĐẦU TƯ PHÁT TRIỂN SƠN LA
CHI NHÁNH TỈNH SƠN LA
Số: 12/TTr-SL
Sơn La, ngày 15 tháng 05 năm 2026

TỜ TRÌNH
Về việc phê duyệt chủ trương đầu tư dự án: Trụ sở Công ty CP Đầu tư Phát triển Sơn La chi nhánh tỉnh Sơn La

Giá trị trình: 24.500.000.000 đồng
Giá trị phê duyệt: 24.000.000.000 đồng
Ghi chú: Hồ sơ kèm theo bản vẽ và dự toán chi tiết.
"""


def test_local_extraction_gets_work_detail_fields(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(SAMPLE_TEXT))
    data = result["data"]
    fields = data["fields"]

    assert result["provider"] == "local"
    assert data["pipeline"] == "local"
    assert data["llm_fallback_used"] is False
    assert data["document_intent"] == "to_trinh"
    assert fields["document_number"]["value"] == "12/TTr-SL"
    assert fields["signed_or_effective_date"]["normalized_value"] == "2026-05-15"
    assert fields["approved_value"]["value"] is None
    assert fields["submitted_value"]["normalized_value"] == 24500000000
    assert fields["issuer"]["source"] == "rule"
    assert "CÔNG TY CP ĐẦU TƯ PHÁT TRIỂN SƠN LA" in fields["issuer"]["value"]
    assert fields["title"]["source"] == "rule"
    assert data["local_confidence"] >= 0.8
    assert data["needs_review"] is False


def test_gemini_entity_extraction_uses_compact_prompt(monkeypatch):
    long_tail = "\n".join(f"SECRET_TAIL_{index}" for index in range(300))
    text = "Không có header rõ ràng\n" + long_tail
    captured = {}

    async def fake_call(config: dict, prompt: str):
        captured["config"] = config
        captured["prompt"] = prompt
        return {
            "document_type": "document",
            "fields": {
                "title": {
                    "value": "TỜ TRÌNH Về việc bổ sung hồ sơ",
                    "normalized_value": "TỜ TRÌNH Về việc bổ sung hồ sơ",
                    "evidence": "TỜ TRÌNH",
                    "confidence": 0.8,
                }
            },
        }

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    monkeypatch.setattr(extraction, "call_llm_extraction", fake_call)

    result = asyncio.run(extraction.extract_information(text))

    assert result["provider"] == "gemini_entity_extraction"
    assert captured["config"]["auth_mode"] == "api_key"
    assert result["data"]["pipeline"] == "local_with_llm_fallback"
    assert result["data"]["llm_fallback_used"] is True
    assert result["data"]["llm_entity_extraction_used"] is True
    assert result["data"]["llm_extraction_mode"] == "entity_extraction"
    assert "SECRET_TAIL_299" not in captured["prompt"]
    assert len(captured["prompt"]) < len(text) + 1200


def test_document_prompt_keeps_critical_guardrails_compact(monkeypatch):
    text = """
NGÂN HÀNG NÔNG NGHIỆP
Số: 5.7.7.1. /NHNo-QLĐT
Hà Nội, ngày 29 tháng 6 năm 2021
CÔNG VĂN
Về chấp thuận phương án thiết kế kiến trúc công trình Trụ sở Agribank chi nhánh huyện Trấn Yên
Căn cứ Quyết định số 873/QĐ-HĐTV-QLĐT ngày 31/12/2020 của Hội đồng thành viên Agribank;
TỔNG MỨC ĐẦU TƯ
26.000.000.000
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    local_data = extraction.normalize_result({}, text, payload_source="rule")
    prompt = extraction.build_entity_extraction_prompt(text, local_data)

    assert "CÔNG VĂN ĐẾN" in prompt
    assert "Số đến" in prompt
    assert "Ngày đến" in prompt
    assert "Căn cứ" in prompt and "KHÔNG lấy ngày" in prompt
    assert "TỔNG MỨC ĐẦU TƯ" in prompt
    assert "26000000000" in prompt
    assert "document_number" in prompt
    assert "signed_or_effective_date" in prompt
    assert "approved_value" in prompt
    assert "task_title_candidates" in prompt
    assert '"v":' in prompt
    assert '"value":string|null' in prompt
    assert len(prompt) < 3600


def test_fallback_disabled_does_not_call_llm(monkeypatch):
    async def fail_call(*args, **kwargs):
        raise AssertionError("LLM should not be called when fallback is disabled")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
    monkeypatch.setattr(extraction, "call_llm_extraction", fail_call)

    result = asyncio.run(extraction.extract_information("Không có header rõ ràng"))

    assert result["provider"] == "local"
    assert result["data"]["pipeline"] == "local"
    assert result["data"]["needs_review"] is True


def test_gemini_response_json_parsing():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '```json\n{"document_type":"document","fields":{"title":{"value":"TỜ TRÌNH"}}}\n```'
                        }
                    ]
                }
            }
        ]
    }

    parsed = extraction.parse_gemini_response(payload)

    assert parsed["document_type"] == "document"
    assert parsed["fields"]["title"]["value"] == "TỜ TRÌNH"


def test_llm_generation_temperature_is_zero(monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "0")

    payload = extraction.build_gemini_payload("OCR text")
    assert payload["generationConfig"]["temperature"] == 0

    class FakeTypes:
        class ThinkingConfig:
            def __init__(self, thinking_budget):
                self.thinking_budget = thinking_budget

    vertex_config = extraction.build_vertex_generation_config_kwargs(FakeTypes)
    assert vertex_config["temperature"] == 0
    assert vertex_config["response_mime_type"] == "application/json"


def test_gemini_fallback_error_returns_local(monkeypatch):
    async def fail_call(*args, **kwargs):
        raise RuntimeError("credential denied")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    monkeypatch.setattr(extraction, "call_llm_extraction", fail_call)

    result = asyncio.run(extraction.extract_information("Không có header rõ ràng"))

    assert result["provider"] == "local"
    assert result["data"]["pipeline"] == "local"
    assert result["data"]["needs_review"] is True
    assert "credential denied" in result["data"]["llm_fallback_error"]


def test_package_price_in_to_trinh_maps_to_submitted_value(monkeypatch):
    text = """
BAN QUẢN LÝ DỰ ÁN ĐẦU TƯ XÂY DỰNG
Số: 03/TTr-BQLDA
Hà Nội, ngày 02 tháng 01 năm 2026
TỜ TRÌNH
Về việc phê duyệt kế hoạch lựa chọn nhà thầu gói thầu thi công xây dựng dự án: Trụ sở Agribank chi nhánh A
Giá gói thầu: 12.345.000.000 đồng
Giá trị trình duyệt: 12.500.000.000 đồng
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert fields["approved_value"]["value"] is None
    assert fields["submitted_value"]["normalized_value"] == 12500000000


def test_to_trinh_header_date_with_commas_ignores_referenced_decision_date(monkeypatch):
    text = """
NGÂN HÀNG NÔNG NGHIỆP
VÀ PHÁT TRIỂN NÔNG THÔN VIỆT NAM
Số: 981/TTr-QLXD-QLKT
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
Hanoi, Ngày 21, tháng 10, năm 2022
TỜ TRÌNH
Về việc thẩm định Báo cáo nghiên cứu khả thi công trình Trụ sở Agribank chi nhánh huyện Trấn Yên
Quyết định số 873/QĐ-HĐTV-QLĐT ngày 31/12/2020 của Hội đồng thành viên về thành lập Ban quản lý dự án;
Tổng giá trị: 12.345.000.000 đồng
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert result["data"]["document_intent"] == "to_trinh"
    assert fields["signed_or_effective_date"]["value"] == "21/10/2022"
    assert fields["signed_or_effective_date"]["normalized_value"] == "2022-10-21"
    assert "31/12/2020" not in fields["signed_or_effective_date"]["evidence"]
    assert fields["approved_value"]["value"] is None
    assert fields["submitted_value"]["normalized_value"] == 12345000000


def test_de_nghi_document_maps_money_to_submitted_value(monkeypatch):
    text = """
NGÂN HÀNG NÔNG NGHIỆP
Số: 15/CV-QLXD
Hà Nội, ngày 03 tháng 02 năm 2026
CÔNG VĂN
Về việc đề nghị chấp thuận tổng mức đầu tư công trình Trụ sở Agribank chi nhánh A
Tổng mức đầu tư: 18.000.000.000 đồng
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert result["data"]["document_intent"] == "van_ban_de_nghi"
    assert fields["approved_value"]["value"] is None
    assert fields["submitted_value"]["normalized_value"] == 18000000000


def test_decision_approval_maps_money_to_approved_value(monkeypatch):
    text = """
NGÂN HÀNG NÔNG NGHIỆP
Số: 22/QĐ-NHNo-QLĐT
Hà Nội, ngày 04 tháng 02 năm 2026
QUYẾT ĐỊNH
Về việc phê duyệt Báo cáo kinh tế kỹ thuật công trình Trụ sở Agribank chi nhánh A
Tổng mức đầu tư: 18.000.000.000 đồng
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert result["data"]["document_intent"] == "quyet_dinh"
    assert fields["approved_value"]["normalized_value"] == 18000000000
    assert fields["submitted_value"]["value"] is None


def test_comment_letter_does_not_map_money_without_clear_approval_context(monkeypatch):
    text = """
SỞ XÂY DỰNG
Số: 09/SXD-QLXD
Hà Nội, ngày 05 tháng 02 năm 2026
CÔNG VĂN
Về việc tham gia ý kiến về hồ sơ dự án Trụ sở Agribank chi nhánh A
Tổng mức đầu tư: 18.000.000.000 đồng
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert result["data"]["document_intent"] == "tham_gia_y_kien"
    assert fields["approved_value"]["value"] is None
    assert fields["submitted_value"]["value"] is None


def test_receipt_does_not_map_fee_amount_to_submitted_or_approved(monkeypatch):
    text = """
SỞ XÂY DỰNG
Số: 02/PT-SXD
Hà Nội, ngày 06 tháng 02 năm 2026
PHIẾU THU
Về việc thu phí thẩm định hồ sơ
Số tiền: 3.000.000 đồng
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert result["data"]["document_intent"] == "phieu_thu"
    assert fields["approved_value"]["value"] is None
    assert fields["submitted_value"]["value"] is None


def test_contract_document_mode_maps_contract_value_to_approved_value(monkeypatch):
    text = """
HỢP ĐỒNG THI CÔNG XÂY DỰNG
Số: 01/2026/HĐ-XD
Hà Nội, ngày 07 tháng 02 năm 2026
Bên A: Ban quản lý dự án đầu tư xây dựng
Bên B: CÔNG TY TNHH XÂY DỰNG ABC
Điều 5. Giá trị hợp đồng: 1.200.000.000 đồng.
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert result["data"]["document_type"] == "document"
    assert result["data"]["document_intent"] == "contract"
    assert fields["approved_value"]["normalized_value"] == 1200000000
    assert fields["submitted_value"]["value"] is None


def test_document_number_and_total_investment_are_normalized(monkeypatch):
    text = """
NGÂN HÀNG NÔNG NGHIỆP
Số: 5.7.7.1. /NHNo-QLĐT
Hà Nội, ngày 04 tháng 06 năm 2026
CÔNG VĂN
Về việc chấp thuận phương án kiến trúc công trình Trụ sở Agribank chi nhánh Trấn Yên, Yên Bái
Tổng mức đầu tư: 26 tỷ đồng.
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert fields["document_number"]["value"] == "5771/NHNo-QLĐT"
    assert fields["document_number"]["normalized_value"] == "5771/NHNo-QLĐT"
    assert fields["approved_value"]["value"] == "26 tỷ"
    assert fields["approved_value"]["normalized_value"] == 26000000000
    assert "Tổng mức đầu tư" in fields["approved_value"]["evidence"]


def test_filename_hint_can_recover_missing_document_number_digit():
    data = {
        "fields": {
            "document_number": {
                "value": "571/NHNo-QLĐT",
                "normalized_value": "571/NHNo-QLĐT",
                "source": "rule",
            }
        },
        "notes": [],
    }

    extraction.apply_filename_document_number_hint(
        data,
        "5771.NHNo-QLDT Ve chap thuan PAKT cong trinh Tru so.pdf",
    )

    assert data["fields"]["document_number"]["value"] == "5771/NHNo-QLĐT"
    assert data["fields"]["document_number"]["normalized_value"] == "5771/NHNo-QLĐT"
    assert "5771/NHNo-QLĐT" in data["notes"][0]


def test_header_date_with_ocr_month_dots_is_normalized(monkeypatch):
    text = """
NGÂN HÀNG NÔNG NGHIỆP
Số: 5771/NHNo-QLĐT
Hà Nội, ngày 29 tháng 6.. năm 2021
Về chấp thuận phương án thiết kế kiến trúc công trình
Căn cứ Quyết định số 873/QĐ-HĐTV-QLĐT ngày 31/12/2020 của Hội đồng thành viên Agribank;
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert fields["signed_or_effective_date"]["value"] == "29/06/2021"
    assert fields["signed_or_effective_date"]["normalized_value"] == "2021-06-29"


def test_submitted_total_investment_is_not_used_as_approved_value(monkeypatch):
    text = """
Số: 01/TTr-BQLDA
Hà Nội, ngày 04 tháng 06 năm 2026
TỜ TRÌNH
Về việc trình phê duyệt chủ trương đầu tư
Tổng mức đầu tư đề nghị: 26 tỷ đồng.
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert fields["approved_value"]["value"] is None
    assert fields["submitted_value"]["normalized_value"] == 26000000000


def test_total_investment_table_ignores_area_and_unit_price_numbers(monkeypatch):
    text = """
Số: 5771/NHNo-QLĐT
Hà Nội, ngày 29 tháng 6 năm 2021
CÔNG VĂN
Về chấp thuận phương án thiết kế kiến trúc công trình Trụ sở Agribank chi nhánh huyện Trấn Yên
d. Phương án thiết kế, dự kiến tổng mức đầu tư theo các bảng thống kê sau:
Bảng 3. Tổng mức đầu tư xây dựng (dự kiến):
Tầng 1,2,3
m2
1.261
8.500.000 10.718.500.000
IV Chi phí dự phòng (Gdp)
2.067.165.000
TỔNG MỨC ĐẦU TƯ
Gxd+Gtb+Gk+Gdp
26.000.000.000
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert fields["approved_value"]["value"] == "26.000.000.000"
    assert fields["approved_value"]["normalized_value"] == 26000000000
    assert "TỔNG MỨC ĐẦU TƯ" in fields["approved_value"]["evidence"]


def test_noisy_incoming_stamp_does_not_override_decision_fields(monkeypatch):
    text = """
NGÂN HÀNG NÔNG NGHIỆP
VÀ PHÁT TRIỂN NÔNG THÔN VIỆT NAM
Số: 784/QĐ-NHNo-QLĐT
CÔNG VĂN ĐẾN
Số 193
CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
Hà Nội, ngày. D.4 tháng 5 năm 2018.
QUYẾT ĐỊNH
Ngày 7 tháng 5 năm 2018 Về thành lập Ban điều hành dự án đầu tư xây dựng
Công trình: Trụ sở Agribank chi nhánh huyện Đan Phượng, Hà Tây.
TỔNG GIÁM ĐỐC
NGÂN HÀNG NÔNG NGHIỆP VÀ PHÁT TRIỂN NÔNG THÔN VIỆT NAM
Điều 3. Quyết định này có hiệu lực từ ngày ký cho đến khi kết thúc dự án.
"""

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(text))
    fields = result["data"]["fields"]

    assert result["data"]["document_intent"] == "quyet_dinh"
    assert fields["document_number"]["value"] == "784/QĐ-NHNo-QLĐT"
    assert fields["signed_or_effective_date"]["value"] == "04/05/2018"
    assert fields["signed_or_effective_date"]["normalized_value"] == "2018-05-04"
    assert fields["title"]["value"].startswith("QUYẾT ĐỊNH Về thành lập Ban điều hành")
    assert "Số 193" not in fields["document_number"]["evidence"]


def test_gemini_entities_are_kept_for_matching(monkeypatch):
    captured = {}

    async def fake_call(config: dict, prompt: str):
        captured["prompt"] = prompt
        return {
            "document_type": "document",
            "document_intent": "to_trinh",
            "fields": {
                "submitted_value": {
                    "value": "Giá gói thầu: 12.345.000.000 đồng",
                    "normalized_value": "12.345.000.000 đồng",
                    "evidence": "Giá gói thầu: 12.345.000.000 đồng",
                    "confidence": 0.92,
                }
            },
            "generic_extraction": {
                "task_title_candidates": ["phê duyệt kế hoạch lựa chọn nhà thầu"],
                "procurement_package_candidates": ["gói thầu thi công xây dựng"],
                "task_keywords": ["kế hoạch lựa chọn nhà thầu"],
            },
            "entities": {
                "business_actions": [{"name": "phê duyệt kế hoạch lựa chọn nhà thầu"}],
                "procurement_packages": [{"name": "gói thầu thi công xây dựng"}],
            },
        }

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    monkeypatch.setattr(extraction, "call_llm_extraction", fake_call)

    result = asyncio.run(extraction.extract_information(SAMPLE_TEXT))
    data = result["data"]

    assert result["provider"] == "gemini_entity_extraction"
    assert "task_title_candidates" in captured["prompt"]
    assert data["fields"]["approved_value"]["value"] is None
    assert data["fields"]["submitted_value"]["normalized_value"] == 12345000000
    assert data["generic_extraction"]["task_title_candidates"] == ["phê duyệt kế hoạch lựa chọn nhà thầu"]
    assert data["entities"]["procurement_packages"][0]["name"] == "gói thầu thi công xây dựng"
