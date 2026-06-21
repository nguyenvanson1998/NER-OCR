import asyncio
import time

import httpx
import pytest

from app.main import app
from app.services import extraction
from app.services import google_document_ai_ocr
from app.services.layout_matching import attach_field_boxes
from app.services.timing import begin_request_timing
from app.services.timing import end_request_timing
from app.services.timing import timed_stage


def test_timing_records_success_exception_and_repeated_stages():
    timing, token = begin_request_timing("timing-test", "/test")
    try:
        with timed_stage("pdf_part", part=1):
            pass
        with timed_stage("pdf_part", part=2):
            pass
        with pytest.raises(RuntimeError):
            with timed_stage("failed_stage"):
                raise RuntimeError("expected")
        snapshot = timing.snapshot()
    finally:
        end_request_timing(token)

    assert snapshot["stage_counts"]["pdf_part"] == 2
    assert snapshot["stages"]["pdf_part"] >= 0
    assert "failed_stage" in snapshot["stages"]


def test_debug_timing_is_opt_in_and_request_id_is_propagated(monkeypatch):
    text = "TỜ TRÌNH\nSố: 12/TTr-ABC\nVề việc phê duyệt kế hoạch lựa chọn nhà thầu"

    async def request(path: str):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(path, headers={"X-Request-ID": "request-123"}, json={"text": text})

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("AGRIBANK_API_KEY", "")
    normal = asyncio.run(request("/api/llm/extract"))
    debug = asyncio.run(request("/api/llm/extract?debug_timing=true"))

    assert normal.status_code == 200
    assert normal.headers["X-Request-ID"] == "request-123"
    assert "timings" not in normal.json()
    assert "Server-Timing" not in normal.headers
    assert debug.headers["X-Request-ID"] == "request-123"
    assert "extraction_total" in debug.json()["timings"]["stages"]
    assert "Server-Timing" in debug.headers


def test_llm_execution_modes_and_adaptive_criteria(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("LLM_EXECUTION_MODE", "always")
    assert extraction.llm_execution_mode() == "off"

    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    assert extraction.llm_execution_mode() == "always"
    monkeypatch.delenv("LLM_EXECUTION_MODE")
    assert extraction.llm_execution_mode() == "adaptive"

    contract_fields = {
        key: {"value": value, "confidence": 0.9}
        for key, value in {
            "contract_name": "Hợp đồng thi công",
            "contract_number": "01/2026/HĐ",
            "signed_date": "20/06/2026",
        }.items()
    }
    should_call, reason = extraction.adaptive_llm_decision(
        {"fields": contract_fields}, "HỢP ĐỒNG THI CÔNG", "contract"
    )
    assert should_call is True
    assert reason == "contract_core_fields_3_of_5"

    internal = {
        "fields": {"title": {"value": "BIÊN BẢN NGHIỆM THU", "confidence": 0.9}},
        "document_intent": "unknown",
    }
    should_call, reason = extraction.adaptive_llm_decision(internal, "BIÊN BẢN NGHIỆM THU", "document")
    assert should_call is False
    assert reason == "local_fields_sufficient"

    internal_with_registration_reference = (
        "BIÊN BẢN HỌP\nXem xét hồ sơ có giấy chứng nhận đăng ký doanh nghiệp và mã số doanh nghiệp"
    )
    should_call, reason = extraction.adaptive_llm_decision(
        internal, internal_with_registration_reference, "document"
    )
    assert should_call is False
    assert reason == "local_fields_sufficient"


def test_document_ai_rpc_uses_timeout_and_retry_deadline(monkeypatch):
    calls = {}

    class FakeClient:
        def process_document(self, **kwargs):
            calls.update(kwargs)
            return "ok"

    monkeypatch.setenv("DOCUMENT_AI_RPC_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("DOCUMENT_AI_RETRY_DEADLINE_SECONDS", "34")
    result = google_document_ai_ocr._process_document_request(
        FakeClient(), object(), processor_kind="layout", page_offset=0
    )

    assert result == "ok"
    assert calls["timeout"] == 12
    assert calls["retry"].deadline == 34


def test_vertex_sdk_path_uses_gemini_timeout(monkeypatch):
    captured = {}

    async def fake_wait_for(awaitable, timeout):
        captured["timeout"] = timeout
        awaitable.close()
        return '{"fields": {}}'

    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "7")
    monkeypatch.setattr(extraction.asyncio, "wait_for", fake_wait_for)
    result = asyncio.run(
        extraction.call_vertex_gemini_extraction(
            {
                "project_id": "project",
                "location": "us-central1",
                "model": "gemini-test",
            },
            "prompt",
        )
    )

    assert result == {"fields": {}}
    assert captured["timeout"] == 7


def test_delta_prompt_caps_legacy_limit_and_uses_response_schema(monkeypatch):
    monkeypatch.setenv("LLM_PROMPT_MODE", "delta")
    monkeypatch.setenv("LLM_ENTITY_MAX_CHARS", "30000")
    monkeypatch.setenv("LLM_DOCUMENT_MAX_OCR_CHARS", "5000")
    monkeypatch.setenv("LLM_CONTRACT_MAX_OCR_CHARS", "10000")
    text = "\n".join(f"Dòng OCR khác nhau {index} giá trị dự án" for index in range(2000))

    assert len(extraction.select_entity_extraction_text(text, "document")) <= 5000
    assert len(extraction.select_entity_extraction_text(text, "contract")) <= 10000

    schema = extraction.build_llm_response_schema(
        "document", ["document_number", "title"]
    )
    field_properties = schema["properties"]["fields"]["properties"]
    assert set(field_properties) == {"document_number", "title"}
    payload = extraction.build_gemini_payload(
        "prompt", {"response_schema": schema, "max_output_tokens": 1536}
    )
    assert payload["generationConfig"]["responseSchema"] == schema
    assert payload["generationConfig"]["maxOutputTokens"] == 1536


def test_layout_pdf_parts_run_in_parallel_and_merge_in_page_order(monkeypatch):
    class FakeReader:
        pages = [object()] * 30

    completed = []

    def fake_process(**kwargs):
        offset = kwargs["page_offset"]
        time.sleep({0: 0.04, 14: 0.02, 28: 0.0}[offset])
        completed.append(offset)
        return {
            "text": str(offset),
            "chunks": [(str(offset), {"pageSpan": {"pageStart": offset + 1, "pageEnd": offset + 1}})],
            "pages": [{"page": offset + 1}],
            "segments": [{"id": f"s-{offset}", "page": offset + 1}],
        }

    monkeypatch.setattr(google_document_ai_ocr, "PdfReader", lambda _path: FakeReader())
    monkeypatch.setattr(
        google_document_ai_ocr,
        "_split_pdf",
        lambda *_args: [("part-1.pdf", 0), ("part-2.pdf", 14), ("part-3.pdf", 28)],
    )
    monkeypatch.setattr(google_document_ai_ocr, "_process_with_enterprise_ocr_layout", fake_process)
    monkeypatch.setenv("OCR_MAX_PARALLEL_PARTS", "2")

    result = google_document_ai_ocr.ocr_document_with_layout(
        enterprise_project_id="project",
        location="us",
        enterprise_processor_id="processor",
        file_path="document.pdf",
        max_pages=14,
    )

    assert completed != [0, 14, 28]
    assert [page["page"] for page in result["pages"]] == [1, 15, 29]
    assert result["text"] == "0\n\n14\n\n28"


def test_layout_matcher_864_segments_under_two_seconds():
    segments = []
    for index in range(864):
        text = f"Dòng nội dung hồ sơ số {index}: thông tin dự án và lựa chọn nhà thầu"
        if index % 48 == 0:
            text = f"Số văn bản: {index}/QĐ-ABC"
        segments.append(
            {
                "id": f"s{index}",
                "page": index // 72 + 1,
                "text": text,
                "bbox": {
                    "x": 0.05 + (index % 2) * 0.45,
                    "y": 0.01 * (index % 72),
                    "width": 0.4,
                    "height": 0.008,
                },
            }
        )
    fields = {
        f"field_{number}": {
            "value": f"{number * 48}/QĐ-ABC",
            "evidence": f"Số văn bản: {number * 48}/QĐ-ABC",
        }
        for number in range(18)
    }

    started = time.perf_counter()
    attach_field_boxes({"fields": fields}, segments)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0
    assert [fields[f"field_{number}"]["box"]["source_segment_id"] for number in range(18)] == [
        f"s{number * 48}" for number in range(18)
    ]
