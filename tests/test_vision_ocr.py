import asyncio
import io

import pytest
from pypdf import PdfWriter

from app.services import vision_ocr


def _word(text: str, x: float, break_type: str = "SPACE") -> dict:
    symbols = []
    for index, character in enumerate(text):
        symbol = {"text": character, "confidence": 0.95}
        if index == len(text) - 1:
            symbol["property"] = {"detectedBreak": {"type": break_type}}
        symbols.append(symbol)
    return {
        "confidence": 0.95,
        "symbols": symbols,
        "boundingBox": {
            "normalizedVertices": [
                {"x": x, "y": 0.1},
                {"x": x + 0.2, "y": 0.1},
                {"x": x + 0.2, "y": 0.15},
                {"x": x, "y": 0.15},
            ]
        },
    }


def _vision_page(text: str, local_page: int) -> dict:
    return {
        "responses": [
            {
                "responses": [
                    {
                        "fullTextAnnotation": {
                            "text": text,
                            "pages": [
                                {
                                    "width": 1000,
                                    "height": 1400,
                                    "blocks": [
                                        {
                                            "paragraphs": [
                                                {
                                                    "words": [
                                                        _word(f"Trang{local_page}", 0.1),
                                                        _word("nội-dung-đủ-dài", 0.35, "LINE_BREAK"),
                                                    ]
                                                }
                                            ]
                                        }
                                    ],
                                }
                            ],
                        }
                    }
                ]
            }
        ]
    }


def test_split_pdf_uses_five_page_parts():
    writer = PdfWriter()
    for _ in range(10):
        writer.add_blank_page(width=100, height=100)
    output = io.BytesIO()
    writer.write(output)

    parts = vision_ocr.split_pdf_bytes(output.getvalue(), max_pages=5)

    assert [(offset, count) for _, offset, count in parts] == [(0, 5), (5, 5)]


def test_vision_conversion_groups_lines_and_offsets_pages():
    result = vision_ocr.parse_vision_response(
        _vision_page("Trang 1 nội dung đủ dài để OCR", 1),
        page_offset=5,
        expected_pages=1,
    )

    assert result["pages"][0]["page"] == 6
    assert result["chunks"][0][1]["pageSpan"] == {"pageStart": 6, "pageEnd": 6}
    assert result["segments"][0]["text"] == "Trang1 nội-dung-đủ-dài"
    assert result["segments"][0]["bbox"]["x"] == 0.1
    assert result["_word_confidences"] == [0.95, 0.95]


def test_parallel_parts_merge_in_page_order(monkeypatch):
    monkeypatch.setattr(
        vision_ocr,
        "split_pdf_bytes",
        lambda *_args, **_kwargs: [(b"first", 0, 5), (b"second", 5, 5)],
    )

    async def fake_annotate(content, offset, expected):
        if offset == 0:
            await asyncio.sleep(0.02)
        chunks = [
            (
                f"Nội dung trang {offset + index + 1} đủ dài để đạt quality gate",
                {
                    "pageSpan": {
                        "pageStart": offset + index + 1,
                        "pageEnd": offset + index + 1,
                    }
                },
            )
            for index in range(expected)
        ]
        return {
            "chunks": chunks,
            "pages": [{"page": offset + index + 1} for index in range(expected)],
            "segments": [
                {"id": f"p{offset + index + 1}-s0", "page": offset + index + 1}
                for index in range(expected)
            ],
            "_word_confidences": [0.95] * expected,
        }

    monkeypatch.setattr(vision_ocr, "_annotate_part", fake_annotate)
    result = asyncio.run(vision_ocr.ocr_pdf_bytes_with_vision(b"pdf"))

    assert [page["page"] for page in result["pages"]] == list(range(1, 11))
    assert [segment["page"] for segment in result["segments"]] == list(range(1, 11))
    assert result["quality"]["mean_word_confidence"] == 0.95


def test_quality_gate_rejects_low_confidence(monkeypatch):
    monkeypatch.setenv("VISION_MIN_WORD_CONFIDENCE", "0.70")
    with pytest.raises(vision_ocr.VisionQualityError):
        vision_ocr._validate_quality(
            {
                "pages": [{"page": 1}],
                "chunks": [("Nội dung trang đủ dài để không bị trắng", {"pageSpan": {"pageStart": 1}})],
                "quality": {"mean_word_confidence": 0.5},
            },
            expected_pages=1,
        )


def test_overall_vision_timeout(monkeypatch):
    monkeypatch.setenv("VISION_OCR_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(
        vision_ocr, "split_pdf_bytes", lambda *_args, **_kwargs: [(b"part", 0, 1)]
    )

    async def slow_part(*_args):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(vision_ocr, "_annotate_part", slow_part)
    with pytest.raises(vision_ocr.VisionOcrError, match="overall timeout"):
        asyncio.run(vision_ocr.ocr_pdf_bytes_with_vision(b"pdf"))
