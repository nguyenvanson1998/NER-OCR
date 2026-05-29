import asyncio

from app.services import agribank_matching
from app.services.extraction import extract_information


def project(project_id: str, code: str, name: str):
    return {
        "id": project_id,
        "code": code,
        "name": name,
        "status": "executing",
        "is_active": True,
    }


def task(task_id: str, name: str, status: str = "completed"):
    return {
        "id": task_id,
        "name": name,
        "status": status,
        "level": 3,
        "workflow_step": {
            "id": f"step-{task_id}",
            "name": name,
            "phase": "Giai đoạn chuẩn bị dự án",
            "order_no": 1,
        },
    }


def test_exact_project_title_match_scores_high():
    query = "Trụ sở Công ty CP Đầu tư Phát triển Sơn La chi nhánh tỉnh Sơn La"
    candidates = [
        project("p1", "4500891", "Trụ sở Công ty CP Đầu tư Phát triển Sơn La chi nhánh tỉnh Sơn La"),
        project("p2", "1000201836", "Trụ sở Agribank chi nhánh tỉnh An Giang"),
    ]

    ranked = agribank_matching.rank_projects([query], candidates)

    assert ranked[0]["id"] == "p1"
    assert ranked[0]["score"] >= 0.9


def test_similar_project_with_wrong_location_stays_below_threshold():
    query = "Trụ sở Agribank Chi nhánh huyện Ba Vì, Hà Tây I"
    candidates = [
        project("wrong", "1000201909", "Trụ sở Agribank chi nhánh huyện Dương Minh Châu, tỉnh Tây Ninh"),
        project("also_wrong", "1000201833", "Trụ sở Agribank chi nhánh Củ Chi"),
    ]

    ranked = agribank_matching.rank_projects([query], candidates)

    assert ranked[0]["score"] < 0.78


def test_three_hundred_tasks_still_pick_title_task():
    tasks = [task(f"noise-{index}", f"Công việc nghiệm thu hồ sơ số {index}") for index in range(300)]
    tasks.append(task("target", "Trình phê duyệt chủ trương đầu tư dự án - UBND Xã"))
    queries = [
        "phê duyệt chủ trương đầu tư",
        "TỜ TRÌNH Về việc phê duyệt chủ trương đầu tư dự án: Trụ sở Công ty CP Đầu tư Phát triển Sơn La",
    ]

    ranked = agribank_matching.rank_tasks(queries, tasks, "project-1")

    assert ranked[0]["id"] == "target"
    assert ranked[0]["score"] >= 0.55


def test_task_queries_use_only_title_not_llm_entities():
    """Test that task queries are built ONLY from document title, NOT from LLM entities.

    This is intentional to prevent short generic queries (like "khảo sát xây dựng")
    from matching wrong tasks with perfect scores.
    """
    extraction = {
        "generic_extraction": {
            "task_title_candidates": ["phê duyệt kế hoạch lựa chọn nhà thầu"],
            "procurement_package_candidates": ["gói thầu thi công xây dựng"],
        },
        "entities": {
            "business_actions": [{"name": "thẩm định thiết kế bản vẽ thi công"}],
            "work_items": [{"title": "lập dự toán xây dựng"}],
        },
    }

    queries = agribank_matching.build_task_queries("TỜ TRÌNH Về việc bổ sung hồ sơ", extraction)

    # Only title-based queries should be present
    assert "TỜ TRÌNH Về việc bổ sung hồ sơ" in queries
    assert "TỜ TRÌNH bổ sung hồ sơ" in queries  # without "Về việc"

    # LLM entities should NOT be in queries (intentionally excluded)
    assert "phê duyệt kế hoạch lựa chọn nhà thầu" not in queries
    assert "gói thầu thi công xây dựng" not in queries
    assert "thẩm định thiết kế bản vẽ thi công" not in queries
    assert "lập dự toán xây dựng" not in queries


def test_expand_abbreviations_replaces_known_tokens():
    expanded = agribank_matching.expand_abbreviations(
        "Trụ sở CN huyện Tam Dương - Công ty CP ĐTPT, KH LCNT gói số 1"
    )
    assert "Chi nhánh huyện Tam Dương" in expanded
    assert "Cổ phần" in expanded
    assert "Kế hoạch lựa chọn nhà thầu" in expanded


def test_expand_abbreviations_replaces_vietnamese_domain_tokens():
    expanded = agribank_matching.expand_abbreviations(
        "CĐT trình KHLCNT gói BCKT-KT, BT,HT&TĐC và TN&MT"
    )
    assert "Chủ đầu tư" in expanded
    assert "Kế hoạch lựa chọn nhà thầu" in expanded
    assert "Báo cáo Kinh tế - Kỹ thuật" in expanded
    assert "Bồi thường, Hỗ trợ và Tái định cư" in expanded
    assert "Tài nguyên và Môi trường" in expanded


def test_abbreviated_query_matches_full_project_name():
    queries = ["Trụ sở CN TP Cần Thơ"]
    candidates = [
        project("p1", "4500111", "Trụ sở Chi nhánh Thành phố Cần Thơ"),
        project("p2", "4500222", "Trụ sở Chi nhánh tỉnh An Giang"),
    ]

    ranked = agribank_matching.rank_projects(queries, candidates)

    assert ranked[0]["id"] == "p1"
    assert ranked[0]["score"] >= 0.78


def test_rank_tasks_filters_out_parent_tasks():
    tasks = [
        {**task("parent", "Giai đoạn chuẩn bị dự án"), "parent_id": None},
        {**task("child-1", "Tư vấn lập BC KTKT"), "parent_id": "parent"},
        {**task("child-2", "Thẩm tra BC KTKT"), "parent_id": "parent"},
    ]

    ranked = agribank_matching.rank_tasks(["Thẩm tra Báo cáo kinh tế kỹ thuật"], tasks, "p1")

    assert all(item["id"] != "parent" for item in ranked)
    assert ranked[0]["id"] == "child-2"


def test_attach_match_api_error_keeps_extraction(monkeypatch):
    text = """
CÔNG TY CP ĐẦU TƯ PHÁT TRIỂN SƠN LA
Số: 12/TTr-SL
TỜ TRÌNH
Về việc phê duyệt chủ trương đầu tư dự án: Trụ sở Công ty CP Đầu tư Phát triển Sơn La chi nhánh tỉnh Sơn La
"""

    async def fail_projects(api_key: str):
        raise RuntimeError("api down")

    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("AGRIBANK_API_KEY", "test-key")
    monkeypatch.setattr(agribank_matching, "fetch_searchable_projects", fail_projects)

    extraction = asyncio.run(extract_information(text))["data"]
    matched = asyncio.run(agribank_matching.attach_work_detail_matches(extraction, text))

    assert matched["fields"]["title"]["value"]
    assert matched["work_detail_match"]["status"] == "error"
    assert matched["work_detail_output"]["title"]
    assert matched["work_detail_output"]["project_id"] is None


WEAK_MATCH_TEXT = """
CÔNG TY CP ĐẦU TƯ PHÁT TRIỂN SƠN LA
Số: 12/TTr-SL
TỜ TRÌNH
Về việc phê duyệt chủ trương đầu tư dự án: Trụ sở chi nhánh tỉnh Sơn La
"""


def _setup_weak_match(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("AGRIBANK_API_KEY", "test-key")

    async def projects(api_key: str):
        return [
            project("p1", "4500891", "Trụ sở Công ty CP Đầu tư Phát triển Sơn La chi nhánh tỉnh Sơn La"),
            project("p2", "1000201836", "Trụ sở Agribank chi nhánh tỉnh An Giang"),
        ]

    async def tasks(api_key: str, project_id: str):
        return []

    monkeypatch.setattr(agribank_matching, "fetch_searchable_projects", projects)
    monkeypatch.setattr(agribank_matching, "fetch_project_tasks", tasks)


def test_llm_tiebreaker_picks_project_when_below_threshold(monkeypatch):
    _setup_weak_match(monkeypatch)
    monkeypatch.setenv("LLM_MATCH_TIEBREAKER_ENABLED", "true")
    monkeypatch.setenv("LLM_MATCH_TIEBREAKER_CONFIDENCE", "0.6")

    calls: dict[str, int] = {"project": 0, "task": 0}

    async def fake_pick_project(text, title, queries, candidates):
        calls["project"] += 1
        assert candidates, "candidates must be passed to LLM"
        return {"chosen_id": "p1", "confidence": 0.9, "reasoning": "Trùng địa danh Sơn La"}

    async def fake_pick_task(*args, **kwargs):
        calls["task"] += 1
        return None

    monkeypatch.setattr(agribank_matching, "llm_pick_project", fake_pick_project)
    monkeypatch.setattr(agribank_matching, "llm_pick_task", fake_pick_task)

    extraction = asyncio.run(extract_information(WEAK_MATCH_TEXT))["data"]
    matched = asyncio.run(agribank_matching.attach_work_detail_matches(extraction, WEAK_MATCH_TEXT))

    assert calls["project"] == 1
    assert matched["work_detail_match"]["status"].startswith("project_matched_by_llm")
    assert matched["work_detail_match"]["project"]["id"] == "p1"
    assert matched["work_detail_match"]["llm_tiebreaker"]["project"]["confidence"] == 0.9


def test_llm_tiebreaker_skips_when_confidence_low(monkeypatch):
    _setup_weak_match(monkeypatch)
    monkeypatch.setenv("LLM_MATCH_TIEBREAKER_ENABLED", "true")
    monkeypatch.setenv("LLM_MATCH_TIEBREAKER_CONFIDENCE", "0.6")

    async def fake_pick_project(text, title, queries, candidates):
        return {"chosen_id": "p1", "confidence": 0.4, "reasoning": "Không chắc"}

    monkeypatch.setattr(agribank_matching, "llm_pick_project", fake_pick_project)

    extraction = asyncio.run(extract_information(WEAK_MATCH_TEXT))["data"]
    matched = asyncio.run(agribank_matching.attach_work_detail_matches(extraction, WEAK_MATCH_TEXT))

    assert matched["work_detail_match"]["status"] == "not_matched"
    assert matched["work_detail_match"]["project"] is None
    assert matched["work_detail_match"]["llm_tiebreaker"]["project"]["confidence"] == 0.4


def test_llm_tiebreaker_disabled_via_env(monkeypatch):
    _setup_weak_match(monkeypatch)
    monkeypatch.setenv("LLM_MATCH_TIEBREAKER_ENABLED", "false")

    called = {"project": 0}

    async def fake_pick_project(*args, **kwargs):
        called["project"] += 1
        return {"chosen_id": "p1", "confidence": 0.99, "reasoning": "ignored"}

    monkeypatch.setattr(agribank_matching, "llm_pick_project", fake_pick_project)

    extraction = asyncio.run(extract_information(WEAK_MATCH_TEXT))["data"]
    matched = asyncio.run(agribank_matching.attach_work_detail_matches(extraction, WEAK_MATCH_TEXT))

    assert called["project"] == 0
    assert matched["work_detail_match"]["status"] == "not_matched"
    assert matched["work_detail_match"]["project"] is None
    assert "llm_tiebreaker" not in matched["work_detail_match"]
