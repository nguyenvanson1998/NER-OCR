import asyncio

from app.services import extraction


CONTRACT_TEXT = """
HỢP ĐỒNG TƯ VẤN THIẾT KẾ BẢN VẼ THI CÔNG, LẬP DỰ TOÁN CÔNG TRÌNH
Số: 01/2018/HĐ-TVVTK
Hôm nay, ngày 05 tháng 04 năm 2018, các bên gồm:
Bên A: Ban quản lý dự án đầu tư xây dựng
Bên B: CÔNG TY TNHH TƯ VẤN XÂY DỰNG ABC
Điều 3. Thời gian và tiến độ thực hiện hợp đồng: 45 ngày kể từ ngày hợp đồng có hiệu lực.
Điều 4. Loại hợp đồng: Hợp đồng trọn gói.
Điều 5. Giá hợp đồng: 1.200.000.000 đồng, đã bao gồm thuế VAT 10%.
Giá gói thầu: 1.300.000.000 đồng.
Bảo lãnh thực hiện hợp đồng: 120.000.000 đồng, có hiệu lực đến ngày 30/06/2018.
Bảo lãnh tiền tạm ứng: 200.000.000 đồng, hiệu lực đến ngày 31/07/2018.
"""


def test_contract_rules_extract_core_fields(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(CONTRACT_TEXT, extraction_type="contract"))
    data = result["data"]
    fields = data["fields"]

    assert data["document_type"] == "contract"
    assert fields["contract_name"]["value"].startswith("HỢP ĐỒNG TƯ VẤN")
    assert fields["work_name"]["value"].startswith("TƯ VẤN THIẾT KẾ")
    assert fields["contract_number"]["value"] == "01/2018/HĐ-TVVTK"
    assert fields["signed_date"]["normalized_value"] == "2018-04-05"
    assert fields["execution_duration_days"]["normalized_value"] == 45
    assert fields["contract_form"]["value"] == "Hợp đồng trọn gói"
    assert fields["contract_value"]["normalized_value"] == 1200000000
    assert fields["estimated_value"]["normalized_value"] == 1300000000
    assert fields["contract_vat_percent"]["normalized_value"] == 10
    assert fields["contractor_name"]["value"] == "CÔNG TY TNHH TƯ VẤN XÂY DỰNG ABC"
    assert fields["contractor_contract_amount"]["normalized_value"] == 1200000000
    assert fields["performance_guarantee_value"]["normalized_value"] == 120000000
    assert fields["performance_guarantee_end_date"]["normalized_value"] == "2018-06-30"
    assert fields["advance_guarantee_value"]["normalized_value"] == 200000000
    assert fields["advance_guarantee_end_date"]["normalized_value"] == "2018-07-31"

    assert "work_detail_fields" not in data


def test_default_extraction_type_is_document(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    result = asyncio.run(extraction.extract_information(CONTRACT_TEXT))
    data = result["data"]

    assert data["document_type"] == "document"
    assert data["screen"] == "work_detail"
    assert "contract_number" not in data["fields"]
    assert "document_number" in data["fields"]


def test_contract_prompt_uses_pdf_rules():
    local_data = extraction.normalize_result({}, CONTRACT_TEXT, payload_source="rule", extraction_type="contract")
    prompt = extraction.build_entity_extraction_prompt(CONTRACT_TEXT, local_data)

    assert "Luật bóc tách hợp đồng" in prompt
    assert "không tự suy diễn" in prompt
    assert "contract_value" in prompt
    assert "performance_guarantee_value" in prompt
    assert "Tên công việc là field mapping" in prompt
    assert "work_detail_fields" not in prompt
