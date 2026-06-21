import asyncio

from app.services import extraction
from app.services import layout_matching


def run_local(monkeypatch, text: str):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")
    return asyncio.run(extraction.extract_information(text))["data"]


def test_business_registration_ignores_enterprise_id_and_uses_latest_change_date(monkeypatch):
    text = """
SỞ KẾ HOẠCH VÀ ĐẦU TƯ THÀNH PHỐ HÀ NỘI
PHÒNG ĐĂNG KÝ KINH DOANH
GIẤY CHỨNG NHẬN ĐĂNG KÝ DOANH NGHIỆP
CÔNG TY CỔ PHẦN
Mã số doanh nghiệp: 0101234567
Đăng ký lần đầu: ngày 10 tháng 02 năm 2012
Đăng ký thay đổi lần thứ: 6, ngày 15 tháng 03 năm 2019
Đăng ký thay đổi lần thứ: 7, ngày 21 tháng 08 năm 2023
"""

    fields = run_local(monkeypatch, text)["fields"]

    assert fields["document_number"]["value"] is None
    assert fields["signed_or_effective_date"]["value"] == "21/08/2023"
    assert fields["signed_or_effective_date"]["normalized_value"] == "2023-08-21"
    assert "thay đổi lần thứ: 7" in fields["signed_or_effective_date"]["evidence"]


def test_llm_cannot_restore_business_id_as_document_number():
    text = """
GIẤY CHỨNG NHẬN ĐĂNG KÝ DOANH NGHIỆP
Mã số doanh nghiệp: 0101234567
Đăng ký thay đổi lần thứ: 7, ngày 21 tháng 08 năm 2023
"""
    payload = {
        "fields": {
            "document_number": {
                "value": "0101234567",
                "evidence": "Mã số doanh nghiệp: 0101234567",
                "confidence": 0.99,
            },
            "signed_or_effective_date": {
                "value": "10/02/2012",
                "evidence": "Đăng ký lần đầu: ngày 10 tháng 02 năm 2012",
                "confidence": 0.99,
            },
        }
    }

    fields = extraction.normalize_result(payload, text, payload_source="llm")["fields"]

    assert fields["document_number"]["value"] is None
    assert fields["signed_or_effective_date"]["value"] == "21/08/2023"
    assert fields["signed_or_effective_date"]["source"] == "rule"


def test_abbreviated_business_license_is_still_detected():
    text = """
GIẤY PHÉP ĐKKD - NĂNG LỰC HOẠT ĐỘNG XÂY DỰNG
Mã số doanh nghiệp: 0101234567
"""

    assert extraction.is_business_registration_document(text) is True
    fields = extraction.normalize_result(
        {"fields": {"document_number": {"value": "0101234567", "confidence": 0.99}}},
        text,
        payload_source="llm",
    )["fields"]
    assert fields["document_number"]["value"] is None


def test_internal_report_has_no_document_date_or_issuer_even_when_llm_invents_them():
    text = """
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
BÁO CÁO KHẢO SÁT ĐỊA CHẤT
Công trình: Trụ sở Agribank chi nhánh Trấn Yên
Căn cứ Quyết định số 12/QĐ-ABC ngày 04/05/2020
"""
    payload = {
        "fields": {
            "signed_or_effective_date": {
                "value": "04/05/2020",
                "evidence": "Căn cứ Quyết định số 12/QĐ-ABC ngày 04/05/2020",
                "confidence": 0.98,
            },
            "issuer": {
                "value": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
                "evidence": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
                "confidence": 0.98,
            },
        }
    }

    fields = extraction.normalize_result(payload, text, payload_source="llm")["fields"]

    assert fields["signed_or_effective_date"]["value"] is None
    assert fields["issuer"]["value"] is None


def test_certificate_issuer_is_government_authority_and_not_place_of_issue(monkeypatch):
    text = """
BỘ XÂY DỰNG
CỤC QUẢN LÝ HOẠT ĐỘNG XÂY DỰNG
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
CHỨNG CHỈ NĂNG LỰC HOẠT ĐỘNG XÂY DỰNG
CÔNG TY CỔ PHẦN CONTECH
Nơi cấp: Thành phố Hà Nội
"""

    fields = run_local(monkeypatch, text)["fields"]

    assert fields["issuer"]["value"] == "BỘ XÂY DỰNG CỤC QUẢN LÝ HOẠT ĐỘNG XÂY DỰNG"
    assert "Nơi cấp" not in fields["issuer"]["value"]
    assert "CHỨNG CHỈ" not in fields["issuer"]["value"]


def test_administrative_issuer_keeps_all_header_lines_and_beats_partial_llm_value():
    text = """
NGÂN HÀNG NÔNG NGHIỆP
và Phát triển Nông thôn Việt Nam
CHI NHÁNH TỈNH YÊN BÁI
Số: 676/NHNo.YB-TH
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
Yên Bái, ngày 15 tháng 05 năm 2025
CÔNG VĂN
V/v tham gia ý kiến phương án thiết kế cơ sở
"""
    payload = {
        "fields": {
            "issuer": {
                "value": "NGÂN HÀNG NÔNG NGHIỆP",
                "evidence": "NGÂN HÀNG NÔNG NGHIỆP",
                "confidence": 0.99,
            }
        }
    }

    field = extraction.normalize_result(payload, text, payload_source="llm")["fields"]["issuer"]

    assert field["source"] == "rule"
    assert field["value"] == (
        "NGÂN HÀNG NÔNG NGHIỆP và Phát triển Nông thôn Việt Nam "
        "CHI NHÁNH TỈNH YÊN BÁI"
    )
    assert field["evidence"].count("\n") == 2


def test_decision_total_row_wins_over_component_costs(monkeypatch):
    text = """
BAN QUẢN LÝ DỰ ÁN
Số: 22/QĐ-BQLDA
Hà Nội, ngày 04 tháng 02 năm 2026
QUYẾT ĐỊNH
Về việc phê duyệt dự toán công trình
Chi phí xây dựng: 8.500.000.000 đồng
Chi phí thiết bị: 1.500.000.000 đồng
TỔNG CỘNG
10.750.000.000 đồng
"""

    fields = run_local(monkeypatch, text)["fields"]

    assert fields["approved_value"]["normalized_value"] == 10_750_000_000
    assert "TỔNG CỘNG" in fields["approved_value"]["evidence"]


def test_llm_component_cost_cannot_override_decision_total():
    text = """
Số: 22/QĐ-BQLDA
Hà Nội, ngày 04 tháng 02 năm 2026
QUYẾT ĐỊNH
Về việc phê duyệt dự toán công trình
Chi phí xây dựng: 8.500.000.000 đồng
TỔNG CỘNG
10.750.000.000 đồng
"""
    payload = {
        "fields": {
            "approved_value": {
                "value": "8.500.000.000 đồng",
                "normalized_value": 8_500_000_000,
                "evidence": "Chi phí xây dựng: 8.500.000.000 đồng",
                "confidence": 0.99,
            }
        }
    }

    field = extraction.normalize_result(payload, text, payload_source="llm")["fields"]["approved_value"]

    assert field["normalized_value"] == 10_750_000_000
    assert field["source"] == "rule"


def test_layout_matching_combines_multiline_issuer_blocks():
    extraction_data = {
        "fields": {
            "issuer": {
                "value": "NGÂN HÀNG NÔNG NGHIỆP VÀ PHÁT TRIỂN NÔNG THÔN VIỆT NAM",
                "evidence": "NGÂN HÀNG NÔNG NGHIỆP\nVÀ PHÁT TRIỂN NÔNG THÔN VIỆT NAM",
            }
        }
    }
    segments = [
        {
            "id": "issuer-1",
            "page": 1,
            "text": "NGÂN HÀNG NÔNG NGHIỆP",
            "bbox": {"x": 0.08, "y": 0.08, "width": 0.28, "height": 0.025},
        },
        {
            "id": "motto-1",
            "page": 1,
            "text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
            "bbox": {"x": 0.55, "y": 0.08, "width": 0.38, "height": 0.025},
        },
        {
            "id": "issuer-2",
            "page": 1,
            "text": "VÀ PHÁT TRIỂN NÔNG THÔN VIỆT NAM",
            "bbox": {"x": 0.07, "y": 0.115, "width": 0.32, "height": 0.025},
        },
    ]

    layout_matching.attach_field_boxes(extraction_data, segments)
    box = extraction_data["fields"]["issuer"]["box"]

    assert box["source_segment_ids"] == ["issuer-1", "issuer-2"]
    assert box["bbox"] == {"x": 0.07, "y": 0.08, "width": 0.32, "height": 0.06}
    assert "CỘNG HÒA" not in box["text"]
