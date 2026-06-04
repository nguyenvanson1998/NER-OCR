from app.services.ocr_cleaning import clean_ocr_layout_result, clean_ocr_plain_text_with_report


NOISY_DECISION_TEXT = """
NGÂN HÀNG NÔNG NGHIỆP
VÀ PHÁT TRIỂN NÔNG THÔN VIỆT NAM
Số: 784/QĐ-NHNo-QLĐT
CÔNG VĂN ĐẾN
Số 193
CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
Hà Nội, ngày 04 tháng 5 năm 2018.
QUYẾT ĐỊNH
Ngày 7 tháng 5 năm 2018 Về thành lập Ban điều hành dự án đầu tư xây dựng
Công trình: Trụ sở Agribank chi nhánh huyện Đan Phượng, Hà Tây.
WS.D.N: 01006861
hướng dẫn một số điều của Nghị định số 59/2015/NĐ-CP của Chính phủ về hình Q. BA ĐÌNH
7. Thực hiện các nhiệm vụ khác do Chủ đầu tư giao. Q
1
"""


def test_clean_ocr_plain_text_removes_incoming_stamp_and_seal_noise():
    report = clean_ocr_plain_text_with_report(NOISY_DECISION_TEXT)
    text = report["text"]

    assert "CÔNG VĂN ĐẾN" not in text
    assert "Số 193" not in text
    assert "Ngày 7 tháng 5 năm 2018" not in text
    assert "WS.D.N" not in text
    assert "Q. BA ĐÌNH" not in text
    assert "giao. Q" not in text
    assert "\n1\n" not in f"\n{text}\n"

    assert "Số: 784/QĐ-NHNo-QLĐT" in text
    assert "Hà Nội, ngày 04 tháng 5 năm 2018." in text
    assert "QUYẾT ĐỊNH" in text
    assert "Về thành lập Ban điều hành dự án đầu tư xây dựng" in text
    assert report["removed_line_count"] >= 4
    assert report["rewritten_line_count"] >= 2


def test_clean_ocr_layout_result_filters_stamp_segments_but_keeps_document_number():
    layout = {
        "text": NOISY_DECISION_TEXT,
        "chunks": [(NOISY_DECISION_TEXT, {"content": NOISY_DECISION_TEXT, "pageSpan": {"pageStart": 1, "pageEnd": 1}})],
        "pages": [{"page": 1, "width": 1000.0, "height": 1000.0, "unit": "px"}],
        "segments": [
            {
                "id": "p1-s1",
                "page": 1,
                "type": "line",
                "text": "Số: 784/QĐ-NHNo-QLĐT",
                "bbox": {"x": 0.08, "y": 0.12, "width": 0.25, "height": 0.02},
            },
            {
                "id": "p1-s2",
                "page": 1,
                "type": "line",
                "text": "CÔNG VĂN ĐẾN",
                "bbox": {"x": 0.40, "y": 0.11, "width": 0.15, "height": 0.02},
            },
            {
                "id": "p1-s3",
                "page": 1,
                "type": "line",
                "text": "Số 193",
                "bbox": {"x": 0.43, "y": 0.14, "width": 0.08, "height": 0.02},
            },
            {
                "id": "p1-s4",
                "page": 1,
                "type": "line",
                "text": "WS.D.N: 01006861",
                "bbox": {"x": 0.70, "y": 0.42, "width": 0.18, "height": 0.02},
            },
            {
                "id": "p1-s5",
                "page": 1,
                "type": "line",
                "text": "Ngày 7 tháng 5 năm 2018 Về thành lập Ban điều hành dự án đầu tư xây dựng",
                "bbox": {"x": 0.26, "y": 0.24, "width": 0.45, "height": 0.03},
            },
        ],
    }

    cleaned = clean_ocr_layout_result(layout)

    assert cleaned["text"] != layout["text"]
    assert [segment["id"] for segment in cleaned["segments"]] == ["p1-s1", "p1-s5"]
    assert cleaned["segments"][1]["text"] == "Về thành lập Ban điều hành dự án đầu tư xây dựng"
    assert cleaned["chunks"][0][0] == cleaned["text"]
    assert cleaned["chunks"][0][1]["content"] == cleaned["text"]
    assert cleaned["cleaning"]["removed_segment_count"] == 3
