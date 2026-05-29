import asyncio

import httpx

from app.main import app


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
