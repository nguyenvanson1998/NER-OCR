import asyncio

import httpx
import pytest

from app.main import app
from app.services import agribank_matching


def test_llm_extract_route_returns_local_pipeline(monkeypatch):
    text = """
CÔNG TY CP ĐẦU TƯ PHÁT TRIỂN SƠN LA
CHI NHÁNH TỈNH SƠN LA
Số: 12/TTr-SL
Sơn La, ngày 15 tháng 05 năm 2026
TỜ TRÌNH
Về việc phê duyệt chủ trương đầu tư dự án: Trụ sở Công ty CP Đầu tư Phát triển Sơn La chi nhánh tỉnh Sơn La
Giá trị trình: 24.500.000.000 đồng
Giá trị phê duyệt: 24.000.000.000 đồng
"""

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/llm/extract", json={"text": text})

    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("AGRIBANK_API_KEY", "")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    response = asyncio.run(run_request())
    payload = response.json()
    extraction = payload["extraction"]

    assert response.status_code == 200
    assert payload["llm"]["provider"] == "local"
    assert extraction["pipeline"] == "local"
    assert extraction["work_detail_output"]["document_number"] == "12/TTr-SL"
    assert extraction["work_detail_output"]["project_id"] is None


def test_project_dropdown_route_returns_searchable_projects(monkeypatch):
    async def fake_projects(api_key: str):
        return [
            {
                "id": "project-1",
                "code": "P001",
                "name": "Trụ sở Agribank chi nhánh A",
                "status": "executing",
                "is_active": True,
                "extra": "hidden",
            },
            {
                "id": "project-2",
                "code": "P002",
                "name": "Dự án mở rộng chi nhánh B",
                "status": "preparing",
                "is_active": True,
            },
            {
                "id": "project-3",
                "code": "P003",
                "name": "Dự án nâng cấp chi nhánh C",
                "status": "closed",
                "is_active": True,
            },
        ]

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/work-detail/projects")

    monkeypatch.setenv("AGRIBANK_API_KEY", "test-key")
    monkeypatch.setattr("app.main.fetch_searchable_projects", fake_projects)

    response = asyncio.run(run_request())
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["count"] == 3
    statuses = {project["status"] for project in payload["projects"]}
    assert statuses == {"executing", "preparing", "closed"}
    assert payload["projects"][0] == {
        "id": "project-1",
        "code": "P001",
        "name": "Trụ sở Agribank chi nhánh A",
        "status": "executing",
        "is_active": True,
    }


def test_project_dropdown_route_missing_key(monkeypatch):
    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/work-detail/projects")

    monkeypatch.setenv("AGRIBANK_API_KEY", "")

    response = asyncio.run(run_request())
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "disabled"
    assert payload["projects"] == []


def test_project_tree_normalization_and_alias():
    assert agribank_matching.normalize_project_tree("opa") == "opa"
    assert agribank_matching.normalize_project_tree("opc") == "opc"
    assert agribank_matching.normalize_project_tree("agribank") == "agribank"
    assert agribank_matching.normalize_project_tree("argibank") == "agribank"
    with pytest.raises(ValueError):
        agribank_matching.normalize_project_tree("unknown")


def test_v1_text_extract_requires_project(monkeypatch):
    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/v1/extractions/text", json={"text": "TỜ TRÌNH"})

    monkeypatch.setenv("LLM_PROVIDER", "none")

    response = asyncio.run(run_request())
    payload = response.json()

    assert response.status_code == 422
    assert payload["error"]["code"] == "invalid_project"
    assert payload["error"]["request_id"]


def test_v1_extract_rejects_missing_text_and_file(monkeypatch):
    async def run_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            text_response = await client.post("/api/v1/extractions/text?project=opa", json={})
            file_response = await client.post("/api/v1/extractions/file?project=opa")
            return text_response, file_response

    monkeypatch.setenv("OPA_API_KEY", "opa-key")
    monkeypatch.setenv("OPA_API_BASE_URL", "https://opa.test/api/v1")

    text_response, file_response = asyncio.run(run_requests())

    assert text_response.status_code == 422
    assert text_response.json()["error"]["code"] == "invalid_text"
    assert file_response.status_code == 422
    assert file_response.json()["error"]["code"] == "missing_file"


def test_v1_text_extract_returns_compact_contract(monkeypatch):
    async def fail_projects(*args, **kwargs):
        raise AssertionError("Contract documents should not fetch work-detail trees")

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/extractions/text?project=opa&type=contract",
                json={
                    "text": """
HỢP ĐỒNG THI CÔNG XÂY DỰNG
Số: 01/2026/HĐ-XD
Hà Nội, ngày 01 tháng 02 năm 2026
Giá trị hợp đồng: 1.200.000.000 đồng
"""
                },
            )

    monkeypatch.setenv("OPA_API_KEY", "opa-key")
    monkeypatch.setenv("OPA_API_BASE_URL", "https://opa.test/api/v1")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")
    monkeypatch.setattr(agribank_matching, "fetch_searchable_projects", fail_projects)

    response = asyncio.run(run_request())
    payload = response.json()

    assert response.status_code == 200
    assert payload["project"] == {"key": "opa", "requested": None}
    assert payload["document"]["screen"] == "contract"
    assert payload["fields"]["contract_number"]["value"] == "01/2026/HĐ-XD"
    assert payload["ocr"]["page_count"] == 0
    assert "extraction" not in payload
    assert "generic_extraction" not in payload
    assert "entities" not in payload
    assert "taxonomies" not in payload
    assert "chunks" not in payload["ocr"]
    assert "work_detail_fields" not in payload
    assert "work_detail" not in payload
    assert "match" not in payload


def test_v1_text_extract_defaults_to_document(monkeypatch):
    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/extractions/text?project=opa",
                json={
                    "text": """
HỢP ĐỒNG THI CÔNG XÂY DỰNG
Số: 01/2026/HĐ-XD
Hà Nội, ngày 01 tháng 02 năm 2026
Giá trị hợp đồng: 1.200.000.000 đồng
"""
                },
            )

    monkeypatch.setenv("OPA_API_KEY", "opa-key")
    monkeypatch.setenv("OPA_API_BASE_URL", "https://opa.test/api/v1")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")

    response = asyncio.run(run_request())
    payload = response.json()

    assert response.status_code == 200
    assert payload["document"]["type"] == "document"
    assert payload["document"]["screen"] == "work_detail"
    assert "contract_number" not in payload["fields"]
    assert "document_number" in payload["fields"]


def test_v1_text_extract_rejects_invalid_type(monkeypatch):
    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/extractions/text?project=opa",
                json={"text": "TỜ TRÌNH", "type": "invoice"},
            )

    monkeypatch.setenv("OPA_API_KEY", "opa-key")
    monkeypatch.setenv("OPA_API_BASE_URL", "https://opa.test/api/v1")
    monkeypatch.setenv("LLM_PROVIDER", "none")

    response = asyncio.run(run_request())
    payload = response.json()

    assert response.status_code == 422
    assert payload["error"]["code"] == "invalid_type"


def test_v1_projects_accepts_argibank_alias_and_normalizes(monkeypatch):
    captured = {}

    async def fake_projects(api_key: str, project_tree: str = "agribank"):
        captured["api_key"] = api_key
        captured["project_tree"] = project_tree
        return [
            {
                "id": "project-1",
                "code": "P001",
                "name": "Trụ sở Agribank chi nhánh A",
                "status": "executing",
                "is_active": True,
                "hidden": "nope",
            }
        ]

    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/v1/work-detail/projects?project=argibank")

    monkeypatch.setenv("AGRIBANK_API_KEY", "agribank-key")
    monkeypatch.setenv("AGRIBANK_API_BASE_URL", "https://agribank.test/api/v1")
    monkeypatch.setattr("app.main.fetch_searchable_projects", fake_projects)

    response = asyncio.run(run_request())
    payload = response.json()

    assert response.status_code == 200
    assert payload["project"] == {"key": "agribank", "requested": "argibank"}
    assert captured == {"api_key": "agribank-key", "project_tree": "agribank"}
    assert payload["projects"] == [
        {
            "id": "project-1",
            "code": "P001",
            "name": "Trụ sở Agribank chi nhánh A",
            "status": "executing",
            "is_active": True,
        }
    ]


def test_v1_text_extract_missing_project_config_returns_503(monkeypatch):
    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/v1/extractions/text?project=opa", json={"text": "TỜ TRÌNH"})

    monkeypatch.setenv("OPA_API_KEY", "")
    monkeypatch.setenv("OPA_API_BASE_URL", "")

    response = asyncio.run(run_request())
    payload = response.json()

    assert response.status_code == 503
    assert payload["error"]["code"] == "project_tree_config_missing"
    assert set(payload["error"]["details"]["missing"]) == {"OPA_API_KEY", "OPA_API_BASE_URL"}


def test_project_tree_project_cache_is_scoped_by_tree(monkeypatch):
    calls = []

    async def fake_paginated(api_key: str, path: str, params: dict, project_tree: str = "agribank"):
        calls.append((api_key, path, project_tree))
        return [
            {
                "id": f"{project_tree}-project",
                "code": project_tree.upper(),
                "name": f"{project_tree} project",
                "status": "executing",
                "is_active": True,
            }
        ]

    agribank_matching.PROJECT_CACHE.clear()
    monkeypatch.setattr(agribank_matching, "fetch_paginated", fake_paginated)

    opa_projects = asyncio.run(agribank_matching.fetch_searchable_projects("opa-key", "opa"))
    opc_projects = asyncio.run(agribank_matching.fetch_searchable_projects("opc-key", "opc"))
    cached_opa_projects = asyncio.run(agribank_matching.fetch_searchable_projects("opa-key", "opa"))

    assert opa_projects[0]["id"] == "opa-project"
    assert opc_projects[0]["id"] == "opc-project"
    assert cached_opa_projects[0]["id"] == "opa-project"
    assert calls == [
        ("opa-key", "project/internal", "opa"),
        ("opc-key", "project/internal", "opc"),
    ]


def test_v1_file_extract_include_layout_controls_layout_payload(monkeypatch):
    text = """
CÔNG TY CP ĐẦU TƯ PHÁT TRIỂN SƠN LA
Số: 12/TTr-SL
Sơn La, ngày 15 tháng 05 năm 2026
TỜ TRÌNH
Về việc phê duyệt chủ trương đầu tư dự án: Trụ sở Công ty CP Đầu tư Phát triển Sơn La chi nhánh tỉnh Sơn La
"""

    async def fake_projects(api_key: str, project_tree: str = "agribank"):
        return []

    def fake_ocr(_file_path: str):
        return [(text, {"pageSpan": {"pageStart": 1, "pageEnd": 1}})]

    def fake_layout(_file_path: str):
        return {
            "text": text,
            "chunks": [(text, {"pageSpan": {"pageStart": 1, "pageEnd": 1}})],
            "pages": [{"page": 1, "width": 1000.0, "height": 1000.0, "unit": "px"}],
            "segments": [
                {
                    "id": "p1-s1",
                    "page": 1,
                    "type": "paragraph",
                    "text": "Số: 12/TTr-SL",
                    "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.04},
                }
            ],
        }

    async def run_file_request(include_layout: bool):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                f"/api/v1/extractions/file?project=opa&include_layout={str(include_layout).lower()}",
                files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
            )

    monkeypatch.setenv("OPA_API_KEY", "opa-key")
    monkeypatch.setenv("OPA_API_BASE_URL", "https://opa.test/api/v1")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOCAL_EXTRACTION_ENABLED", "true")
    monkeypatch.setattr(agribank_matching, "fetch_searchable_projects", fake_projects)
    monkeypatch.setattr("app.main.run_google_ocr", fake_ocr)
    monkeypatch.setattr("app.main.run_google_ocr_layout", fake_layout)

    response_without_layout = asyncio.run(run_file_request(False))
    payload_without_layout = response_without_layout.json()
    response_with_layout = asyncio.run(run_file_request(True))
    payload_with_layout = response_with_layout.json()

    assert response_without_layout.status_code == 200
    assert "layout" not in payload_without_layout
    assert "chunks" not in payload_without_layout["ocr"]
    assert "saved_as" not in payload_without_layout["file"]
    assert "box" not in payload_without_layout["fields"]["document_number"]
    assert payload_without_layout["ocr"]["page_count"] == 1

    assert response_with_layout.status_code == 200
    assert payload_with_layout["layout"] == {"pages": [{"page": 1, "width": 1000.0, "height": 1000.0, "unit": "px"}]}
    assert "segments" not in payload_with_layout["layout"]
    assert payload_with_layout["fields"]["document_number"]["box"]["bbox"] == {
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.04,
    }
