import asyncio
import copy
import io

import httpx

from app import main
from app.main import app
from app.services import v2_pipeline
from app.services.job_queue import MemoryJobStore
from app.services.job_queue import QueuedJob
from app.services.job_queue import set_job_store
from app.worker import handle_job


OCR_TEXT = """BAN QUẢN LÝ DỰ ÁN
Số: 12/TTr-BQLDA
Hà Nội, ngày 20 tháng 06 năm 2026
TỜ TRÌNH
Về việc phê duyệt kế hoạch lựa chọn nhà thầu
Giá trị trình: 1.200.000.000 đồng
"""


def _layout():
    return {
        "text": OCR_TEXT,
        "chunks": [(OCR_TEXT, {"pageSpan": {"pageStart": 1, "pageEnd": 1}})],
        "pages": [{"page": 1, "width": 1000, "height": 1400}],
        "segments": [
            {
                "id": "p1-s0",
                "page": 1,
                "type": "line",
                "text": line,
                "bbox": {"x": 0.1, "y": 0.1 + index * 0.08, "width": 0.7, "height": 0.04},
            }
            for index, line in enumerate(OCR_TEXT.splitlines())
            if line
        ],
        "provider": "cloud_vision",
    }


def _configure_project(monkeypatch):
    monkeypatch.setenv("AGRIBANK_API_KEY", "test-key")
    monkeypatch.setenv("AGRIBANK_API_BASE_URL", "https://example.test/api/v1")


def test_v2_post_and_poll_lifecycle(monkeypatch):
    _configure_project(monkeypatch)
    store = MemoryJobStore(ttl_seconds=60)
    set_job_store(store)

    async def fake_fast_ocr(*_args, **_kwargs):
        return _layout(), "cloud_vision"

    monkeypatch.setattr("app.main.run_fast_ocr_layout", fake_fast_ocr)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v2/extractions/file?project=agribank&type=document&include_layout=true",
                files={"file": ("input.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")},
                headers={"X-Request-ID": "v2-request"},
            )
            assert response.status_code == 202
            payload = response.json()
            assert payload["revision"] == 1
            assert payload["enrichment"] == {"llm": "pending", "matching": "pending"}
            assert "match" not in payload["result"]
            assert "llm" not in payload["result"]
            assert payload["result"]["ocr"]["provider"] == "cloud_vision"
            assert "segments" in payload["result"]["layout"]
            assert response.headers["location"].endswith(payload["job_id"])
            assert response.headers["retry-after"] == "1"
            assert response.headers["x-request-id"] == "v2-request"
            assert "server-timing" in response.headers

            pending = await client.get(response.headers["location"])
            assert pending.status_code == 202

            record = await store.get(payload["job_id"])
            record.update({"status": "completed", "revision": 2})
            await store.update(payload["job_id"], record)
            completed = await client.get(response.headers["location"])
            assert completed.status_code == 200
            assert completed.json()["revision"] == 2

            missing = await client.get("/api/v2/extractions/expired-id")
            assert missing.status_code == 410

    try:
        asyncio.run(scenario())
    finally:
        set_job_store(None)


def test_enrichment_merges_without_changing_ocr_layout(monkeypatch):
    store = MemoryJobStore(ttl_seconds=60)

    async def scenario():
        record = await v2_pipeline.build_fast_job_record(
            job_id="job-1",
            request_id="request-1",
            project_key="agribank",
            project_requested="agribank",
            filename="input.pdf",
            mime_type="application/pdf",
            extraction_type="document",
            layout=_layout(),
            include_layout=True,
            ocr_provider="cloud_vision",
        )
        original_ocr = copy.deepcopy(record["result"]["ocr"])
        original_layout = copy.deepcopy(record["result"]["layout"])
        await store.create("job-1", record)

        async def fake_extract(*_args, **_kwargs):
            data = copy.deepcopy(record["_payload"]["base_extraction"])
            data["fields"]["title"]["value"] = "TỜ TRÌNH ĐÃ ENRICH"
            data["fields"]["title"]["evidence"] = "TỜ TRÌNH"
            return {"provider": "gemini_vertex_entity_extraction", "model": "gemini-test", "data": data}

        async def fake_match(extraction, *_args, **_kwargs):
            extraction["work_detail_match"] = {"status": "matched", "project": {"id": "p1"}}
            extraction["work_detail_output"] = {"title": "TỜ TRÌNH ĐÃ ENRICH"}
            return extraction

        monkeypatch.setattr(v2_pipeline, "extract_information", fake_extract)
        monkeypatch.setattr(v2_pipeline, "attach_work_detail_matches", fake_match)
        await v2_pipeline.process_enrichment_job(store, "job-1")
        final = await store.get("job-1")

        assert final["status"] == "completed"
        assert final["revision"] == 2
        assert final["result"]["ocr"] == original_ocr
        assert final["result"]["layout"] == original_layout
        assert final["result"]["match"]["status"] == "matched"

    asyncio.run(scenario())


def test_worker_acks_only_after_terminal_record_is_saved():
    events = []

    class Store(MemoryJobStore):
        async def update(self, job_id, record):
            events.append(("update", record["status"]))
            await super().update(job_id, record)

        async def ack(self, queued):
            events.append(("ack", queued.job_id))

    store = Store(ttl_seconds=60)

    async def scenario():
        await store.create(
            "broken",
            {
                "job_id": "broken",
                "status": "enriching",
                "revision": 1,
                "result": {"fast": True},
                "enrichment": {"llm": "pending", "matching": "pending"},
                "warnings": [],
            },
        )
        await handle_job(store, QueuedJob("message-1", "broken"))

    asyncio.run(scenario())

    assert events[-1] == ("ack", "broken")
    assert any(event[0] == "update" for event in events[:-1])


def test_vision_quality_failure_falls_back_to_document_ai(monkeypatch, tmp_path):
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"fake-pdf")
    monkeypatch.setenv("OCR_PROVIDER_MODE", "vision_primary")

    async def fail_vision(_content):
        raise RuntimeError("quality gate")

    monkeypatch.setattr(main, "ocr_pdf_bytes_with_vision", fail_vision)
    monkeypatch.setattr(main, "run_google_ocr_layout_for_v1", lambda *_args: _layout())

    layout, provider = asyncio.run(
        main.run_fast_ocr_layout(pdf, request_id="fallback-request")
    )

    assert layout["text"] == OCR_TEXT
    assert provider == "document_ai_fallback"
