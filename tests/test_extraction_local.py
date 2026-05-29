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
    assert fields["approved_value"]["normalized_value"] == 24000000000
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
    assert len(captured["prompt"]) < len(text) + 2000


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


def test_package_price_maps_to_approved_value(monkeypatch):
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

    assert fields["approved_value"]["normalized_value"] == 12345000000
    assert fields["submitted_value"]["normalized_value"] == 12500000000


def test_gemini_entities_are_kept_for_matching(monkeypatch):
    captured = {}

    async def fake_call(config: dict, prompt: str):
        captured["prompt"] = prompt
        return {
            "document_type": "document",
            "document_intent": "to_trinh",
            "fields": {
                "approved_value": {
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
    assert data["fields"]["approved_value"]["normalized_value"] == 12345000000
    assert data["generic_extraction"]["task_title_candidates"] == ["phê duyệt kế hoạch lựa chọn nhà thầu"]
    assert data["entities"]["procurement_packages"][0]["name"] == "gói thầu thi công xây dựng"
