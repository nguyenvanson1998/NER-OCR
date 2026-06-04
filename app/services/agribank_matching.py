import difflib
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any
from typing import Optional

import httpx

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only when optional dep is absent.
    fuzz = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - exercised only when optional dep is absent.
    TfidfVectorizer = None


DEFAULT_BASE_URL = "https://agribank-be.opa.ai.vn/api/v1"
ACTIVE_TASK_STATUSES = {"executing", "in_progress", "processing", "doing", "started"}
SEARCHABLE_PROJECT_STATUSES = {"executing", "preparing", "closed"}
PROJECT_CACHE: dict[str, tuple[float, Any]] = {}
TASK_CACHE: dict[str, tuple[float, Any]] = {}
TEXT_INDEX_CACHE: dict[str, tuple[float, Any]] = {}
PROJECT_TREE_ALIASES = {
    "agribank": "agribank",
    "argibank": "agribank",
    "opa": "opa",
    "opc": "opc",
}
PROJECT_TREE_ENV = {
    "agribank": {
        "display_name": "Agribank",
        "api_key_env": "AGRIBANK_API_KEY",
        "base_url_env": "AGRIBANK_API_BASE_URL",
        "default_base_url": DEFAULT_BASE_URL,
    },
    "opa": {
        "display_name": "OPA",
        "api_key_env": "OPA_API_KEY",
        "base_url_env": "OPA_API_BASE_URL",
        "default_base_url": None,
    },
    "opc": {
        "display_name": "OPC",
        "api_key_env": "OPC_API_KEY",
        "base_url_env": "OPC_API_BASE_URL",
        "default_base_url": None,
    },
}


@dataclass(frozen=True)
class ProjectTreeConfig:
    key: str
    requested: str
    display_name: str
    api_key_env: str
    base_url_env: str
    api_key: Optional[str]
    base_url: Optional[str]


class ProjectTreeConfigError(RuntimeError):
    def __init__(self, message: str, *, missing: Optional[list[str]] = None):
        super().__init__(message)
        self.missing = missing or []

logger = logging.getLogger("agribank_matching")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.getenv("AGRIBANK_MATCH_LOG_LEVEL", "INFO").upper())
logger.propagate = False


def normalize_project_tree(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raise ValueError("Missing project. Expected one of: opa, opc, agribank.")
    key = PROJECT_TREE_ALIASES.get(raw)
    if not key:
        allowed = ", ".join(sorted({"opa", "opc", "agribank", "argibank"}))
        raise ValueError(f"Unsupported project {value!r}. Expected one of: {allowed}.")
    return key


def get_project_tree_config(
    project_tree: Optional[str] = "agribank",
    *,
    require_config: bool = False,
    allow_default_base_url: bool = True,
    api_key_override: Optional[str] = None,
) -> ProjectTreeConfig:
    key = normalize_project_tree(project_tree)
    raw = str(project_tree or "").strip().lower()
    meta = PROJECT_TREE_ENV[key]
    env_base_url = os.getenv(str(meta["base_url_env"]))
    base_url = env_base_url or (str(meta["default_base_url"]) if allow_default_base_url and meta.get("default_base_url") else None)
    api_key = api_key_override if api_key_override is not None else os.getenv(str(meta["api_key_env"]))
    config = ProjectTreeConfig(
        key=key,
        requested=raw,
        display_name=str(meta["display_name"]),
        api_key_env=str(meta["api_key_env"]),
        base_url_env=str(meta["base_url_env"]),
        api_key=api_key,
        base_url=base_url,
    )
    if require_config:
        missing = []
        if not api_key:
            missing.append(config.api_key_env)
        if not env_base_url:
            missing.append(config.base_url_env)
        if missing:
            joined = ", ".join(missing)
            raise ProjectTreeConfigError(f"Missing project tree configuration: {joined}.", missing=missing)
    return config


ABBREVIATION_MAP: dict[str, str] = {
    "BC KTKT": "Báo cáo kinh tế kỹ thuật",
    "ANTT": "An ninh trật tự",
    "ATGT": "An toàn giao thông",
    "BQLDA": "Ban quản lý dự án",
    "BCNCTKT": "Báo cáo nghiên cứu tiền khả thi",
    "BCNCKT": "Báo cáo nghiên cứu khả thi",
    "BC ĐTM": "Báo cáo đánh giá tác động môi trường",
    "BCKT-KT": "Báo cáo Kinh tế - Kỹ thuật",
    "BCTĐ": "Báo cáo thẩm định",
    "BT,HT&TĐC": "Bồi thường, Hỗ trợ và Tái định cư",
    "BVMT": "Bảo vệ môi trường",
    "BVTC": "Bản vẽ thi công",
    "BB": "Biên bản",
    "BBMT": "Biên bản mở thầu",
    "CHCC": "Căn hộ chung cư",
    "CSPCCC": "Cảnh sát phòng cháy, chữa cháy",
    "CLCT": "Chất lượng công trình",
    "CTrĐT": "Chủ trương đầu tư",
    "CĐT": "Chủ đầu tư",
    "CBĐT": "Chuẩn bị đầu tư",
    "CBDA": "Chuẩn bị dự án",
    "CA": "Công An",
    "CT": "Công trình",
    "CTXD": "Công trình xây dựng",
    "CQ": "Cơ quan",
    "CHCN": "Cứu hộ, cứu nạn",
    "DSXH": "Danh sách xếp hạng",
    "DSNT": "Danh sách nhà thầu",
    "DVC": "Dịch vụ công",
    "DN": "Doanh nghiệp",
    "DA": "Dự án",
    "DT": "Dự toán",
    "DTCT": "Dự toán công trình",
    "ĐGHSDT": "Đánh giá hồ sơ dự thầu",
    "ĐTM": "Đánh giá tác động môi trường",
    "ĐTKQM": "Đấu thầu không qua mạng",
    "ĐTQM": "Đấu thầu qua mạng",
    "ĐTTN": "Đấu thầu trong nước",
    "ĐTQG": "Đấu thầu quốc gia",
    "ĐTQT": "Đấu thầu quốc tế",
    "ĐTC": "Đầu tư công",
    "ĐTXD": "Đầu tư xây dựng",
    "ĐVTV": "Đơn vị tư vấn",
    "GSĐG": "Giám sát, đánh giá",
    "GPMB": "Giải phóng mặt bằng",
    "HTKT": "Hạ tầng kỹ thuật",
    "HMCT": "Hạng mục công trình",
    "HTGT": "Hệ thống giao thông",
    "HTM": "Hệ thống mạng",
    "HSYC": "Hồ sơ yêu cầu",
    "KH LCNT": "Kế hoạch lựa chọn nhà thầu",
    "VPĐD": "Văn phòng đại diện",
    "LCNT": "Lựa chọn nhà thầu",
    "ĐHDA": "Điều hành dự án",
    "TMB": "Tổng mặt bằng",
    "HSMT": "Hồ sơ mời thầu",
    "HSMST": "Hồ sơ mời sơ tuyển",
    "HSDST": "Hồ sơ dự sơ tuyển",
    "HSĐX": "Hồ sơ đề xuất",
    "HSDT": "Hồ sơ dự thầu",
    "HSTK": "Hồ sơ thiết kế",
    "HĐ": "Hợp đồng",
    "HĐND": "Hội đồng nhân dân",
    "HĐTĐ": "Hội đồng thẩm định",
    "KH": "Kế hoạch",
    "KH&ĐT": "Kế hoạch và Đầu tư",
    "KHLCNT": "Kế hoạch lựa chọn nhà thầu",
    "KQTĐ": "Kết quả thẩm định",
    "KQCĐT": "Kết quả chỉ định thầu",
    "KSĐĐ": "Khảo sát, đo đạc",
    "KSTK": "Khảo sát, thiết kế",
    "KSXD": "Khảo sát xây dựng",
    "KBNN": "Kho bạc nhà nước",
    "KT-KT": "Kinh tế - kỹ thuật",
    "L/V": "Làm việc",
    "l/v": "Làm việc",
    "NLKN": "Năng lực kinh nghiệm",
    "NSNN": "Ngân sách nhà nước",
    "PA": "Phương án",
    "QTQG": "Quan trọng quốc gia",
    "QLCT": "Quản lý chất lượng",
    "QLDA": "Quản lý dự án",
    "QLĐT": "Quản lý đô thị",
    "QLNN": "Quản lý nhà nước",
    "QLSD": "Quản lý sử dụng",
    "QP": "Quốc phòng",
    "QPAN": "Quốc phòng an ninh",
    "QH": "Quy hoạch",
    "QHKT": "Quy hoạch Kiến trúc",
    "QT": "Quyết toán",
    "SXD": "Sở Xây dựng",
    "TĐC": "Tái định cư",
    "TC": "Tài chính",
    "TC-KH": "Tài chính - Kế hoạch",
    "TK": "Tài khoản",
    "TN&MT": "Tài nguyên và Môi trường",
    "TS": "Tài sản",
    "TSC": "Tài sản công",
    "TLTS": "Thanh lý tài sản",
    "TLTSC": "Thanh lý tài sản công",
    "TT": "Thanh toán",
    "TKCS": "Thiết kế cơ sở",
    "TKDT": "Thiết kế dự toán",
    "TKXD": "Thiết kế xây dựng",
    "TB": "Thông báo",
    "TBMT": "Thông báo mời thầu",
    "THDA": "Thực hiện dự án",
    "TMĐT": "Tổng mức đầu tư",
    "TVTK": "Tư vấn thiết kế",
    "UBND": "Ủy ban nhân dân",
    "UBMTTQ": "Ủy ban Mặt trận Tổ quốc",
    "UQ": "Ủy quyền",
    "XDCT": "Xây dựng công trình",
    "YCKT": "Yêu cầu kỹ thuật",
    "VĐT": "Vốn đầu tư",
    "PCCC": "Phòng cháy chữa cháy",
    "CNTT": "Công nghệ thông tin",
    "KCN": "Khu công nghiệp",
    "TNHH": "Trách nhiệm hữu hạn",
    "MTV": "Một thành viên",
    "KV": "Khu vực",
    "CN": "Chi nhánh",
    "CP": "Cổ phần",
    "TP": "Thành phố",
}

_ABBREVIATION_PATTERNS = [
    (re.compile(r"\b" + re.escape(abbr) + r"\b"), expansion)
    for abbr, expansion in sorted(ABBREVIATION_MAP.items(), key=lambda kv: -len(kv[0]))
]


def expand_abbreviations(value: str) -> str:
    if not value:
        return value
    result = value
    for pattern, expansion in _ABBREVIATION_PATTERNS:
        result = pattern.sub(expansion, result)
    return result

# STOPWORDS: Only truly generic connector words, not domain-specific keywords
# Removed domain-specific words that carry semantic meaning in construction/bidding context:
# - "phe" (phê - approve), "quyet" (quyết - decision), "dinh" (định - decision/assessment)
# - "thau" (thầu - contractor/bidding), "goi" (gói - package), "trinh" (trình - proposal)
# - "cong" (công - public/construction), "bao" (báo - report), "thong" (thông - notification)
# - "ban" (ban - department/committee), "hang" (hạng - category/class), "muc" (mục - item/category)
STOPWORDS = {
    "a",
    "agribank",
    "an",
    "cac",
    "can",
    "cho",
    "chi",
    "cp",
    "cua",
    "da",
    "de",
    "den",
    "du",
    "duoc",
    "gia",
    "gui",
    "ho",
    "huyen",
    "ke",
    "kem",
    "la",
    "lap",
    "nam",
    "ngay",
    "ngan",
    "nghiep",
    "nhanh",
    "nong",
    "noi",
    "qua",
    "so",
    "tai",
    "theo",
    "tinh",
    "tnhh",
    "thuoc",
    "to",
    "trong",
    "tru",
    "tu",
    "ty",
    "mtv",
    "va",
    "ve",
    "viec",
    "viet",
    "xa",
}


async def attach_work_detail_matches(
    extraction: dict[str, Any],
    text: str,
    project_tree: Optional[str] = "agribank",
) -> dict[str, Any]:
    if extraction.get("screen") != "work_detail":
        logger.debug("Skip matching: screen=%s", extraction.get("screen"))
        return extraction

    config = get_project_tree_config(project_tree, allow_default_base_url=True)
    fields = extraction.get("fields") if isinstance(extraction.get("fields"), dict) else {}
    title = field_value(fields, "title") or extraction.get("generic_extraction", {}).get("document_title_or_type")
    title = str(title or "").strip()
    project_queries = build_project_queries(extraction, title)
    task_queries = build_task_queries(title, extraction)
    logger.info(
        "STEP start | title=%r | project_queries=%d | task_queries=%d",
        title[:120],
        len(project_queries),
        len(task_queries),
    )
    match_info: dict[str, Any] = {
        "status": "disabled",
        "project_tree": config.key,
        "query": {
            "title": title or None,
            "project_name_candidates": project_queries[:8],
            "task_title_candidates": task_queries[:5],
        },
        "project": None,
        "task": None,
        "best_candidates": {
            "projects": [],
            "tasks": [],
        },
        "warnings": [],
    }

    api_key = config.api_key
    if not api_key:
        logger.warning("STEP skip: %s missing", config.api_key_env)
        match_info["warnings"].append(f"Missing {config.api_key_env}; skipped project/task matching.")
        extraction["work_detail_match"] = match_info
        extraction["work_detail_output"] = build_work_detail_output(extraction)
        return extraction

    try:
        if config.key == "agribank":
            projects = await fetch_searchable_projects(api_key)
        else:
            projects = await fetch_searchable_projects(api_key, config.key)
        logger.info("STEP fetch_projects | count=%d", len(projects))
    except Exception as exc:
        logger.error("STEP fetch_projects FAILED: %s", exc)
        match_info["status"] = "error"
        match_info["warnings"].append(f"Cannot fetch {config.display_name} projects: {exc}")
        extraction["work_detail_match"] = match_info
        extraction["work_detail_output"] = build_work_detail_output(extraction)
        return extraction

    project_candidates = rank_projects(project_queries, projects, cache_key=f"{config.key}:projects:searchable")
    match_info["best_candidates"]["projects"] = project_candidates[:3]
    project_match = project_candidates[0] if project_candidates else None
    project_threshold = env_float("PROJECT_MATCH_THRESHOLD", env_float("AGRIBANK_PROJECT_MATCH_THRESHOLD", 0.78))
    top_score = project_match["score"] if project_match else 0.0
    logger.info(
        "STEP rank_projects | top_score=%.4f | threshold=%.2f | top3=%s",
        top_score,
        project_threshold,
        [(c.get("code"), round(c.get("score", 0), 3)) for c in project_candidates[:3]],
    )
    if project_match and project_match["score"] >= project_threshold:
        match_info["project"] = project_match
        match_info["status"] = "project_matched"
        logger.info("STEP project matched (local) | id=%s | code=%s", project_match.get("id"), project_match.get("code"))
    else:
        tiebreaker_result: Optional[dict[str, Any]] = None
        if llm_tiebreaker_enabled():
            logger.info(
                "STEP project_tiebreaker -> LLM | top_score=%.4f < threshold=%.2f | candidates=%d",
                top_score,
                project_threshold,
                min(len(project_candidates), llm_tiebreaker_top_n()),
            )
            if config.key == "agribank":
                tiebreaker_result = await llm_pick_project(
                    text,
                    title,
                    project_queries,
                    project_candidates[: llm_tiebreaker_top_n()],
                )
            else:
                tiebreaker_result = await llm_pick_project(
                    text,
                    title,
                    project_queries,
                    project_candidates[: llm_tiebreaker_top_n()],
                    tree_name=config.display_name,
                )
            logger.info("STEP project_tiebreaker result=%s", tiebreaker_result)
            if tiebreaker_result:
                match_info.setdefault("llm_tiebreaker", {})["project"] = tiebreaker_result
                if tiebreaker_result.get("error"):
                    match_info["warnings"].append(tiebreaker_result["error"])
        else:
            logger.info("STEP project_tiebreaker SKIPPED (LLM_MATCH_TIEBREAKER_ENABLED=false)")
        confidence = float(tiebreaker_result.get("confidence", 0.0)) if tiebreaker_result and not tiebreaker_result.get("error") else 0.0
        if tiebreaker_result and not tiebreaker_result.get("error") and confidence >= llm_tiebreaker_confidence():
            chosen = next(
                (candidate for candidate in project_candidates if str(candidate.get("id")) == tiebreaker_result["chosen_id"]),
                None,
            )
            if chosen:
                project_match = chosen
                match_info["project"] = chosen
                match_info["status"] = "project_matched_by_llm"
                logger.info("STEP project matched (LLM) | id=%s | confidence=%.4f", chosen.get("id"), confidence)
        if not match_info.get("project"):
            match_info["status"] = "not_matched"
            logger.warning("STEP project NOT MATCHED | tiebreaker=%s", tiebreaker_result)
            if tiebreaker_result and not tiebreaker_result.get("error"):
                match_info["warnings"].append("LLM tiebreaker did not confirm any project candidate.")
            elif project_candidates:
                match_info["best_project_candidate"] = project_candidates[0]
                match_info["warnings"].append("Best project candidate is below threshold.")
            else:
                match_info["warnings"].append(f"No searchable projects returned by {config.display_name} API.")
            extraction["needs_review"] = True
            extraction["work_detail_match"] = match_info
            extraction["work_detail_output"] = build_work_detail_output(extraction)
            return extraction

    try:
        if config.key == "agribank":
            tasks = await fetch_project_tasks(api_key, str(project_match["id"]))
        else:
            tasks = await fetch_project_tasks(api_key, str(project_match["id"]), config.key)
        leaf_count = len(filter_leaf_tasks(tasks))
        logger.info("STEP fetch_tasks | total=%d | leaves=%d", len(tasks), leaf_count)
    except Exception as exc:
        logger.error("STEP fetch_tasks FAILED: %s", exc)
        match_info["status"] = "project_matched_task_error"
        match_info["warnings"].append(f"Cannot fetch {config.display_name} tasks: {exc}")
        extraction["work_detail_match"] = match_info
        extraction["work_detail_output"] = build_work_detail_output(extraction)
        return extraction

    task_candidates = rank_tasks(task_queries, tasks, str(project_match["id"]), project_tree=config.key)
    match_info["best_candidates"]["tasks"] = task_candidates[:3]
    task_match = task_candidates[0] if task_candidates else None
    task_threshold = env_float("TASK_MATCH_THRESHOLD", env_float("AGRIBANK_TASK_MATCH_THRESHOLD", 0.55))
    top_task_score = task_match["score"] if task_match else 0.0
    logger.info(
        "STEP rank_tasks | top_score=%.4f | threshold=%.2f | top3=%s",
        top_task_score,
        task_threshold,
        [(c.get("name", "")[:40], round(c.get("score", 0), 3)) for c in task_candidates[:3]],
    )
    if task_match and task_match["score"] >= task_threshold:
        match_info["task"] = task_match
        if match_info["status"] == "project_matched_by_llm":
            match_info["status"] = "project_matched_by_llm_task_matched"
        else:
            match_info["status"] = "matched"
        logger.info("STEP task matched (local) | id=%s | score=%.4f", task_match.get("id"), task_match["score"])
    else:
        tiebreaker_task: Optional[dict[str, Any]] = None
        if llm_tiebreaker_enabled():
            logger.info(
                "STEP task_tiebreaker -> LLM | top_score=%.4f < threshold=%.2f | candidates=%d",
                top_task_score,
                task_threshold,
                min(len(task_candidates), llm_tiebreaker_top_n()),
            )
            if config.key == "agribank":
                tiebreaker_task = await llm_pick_task(
                    text,
                    title,
                    project_match,
                    task_queries,
                    task_candidates[: llm_tiebreaker_top_n()],
                )
            else:
                tiebreaker_task = await llm_pick_task(
                    text,
                    title,
                    project_match,
                    task_queries,
                    task_candidates[: llm_tiebreaker_top_n()],
                    tree_name=config.display_name,
                )
            if tiebreaker_task:
                logger.info("STEP task_tiebreaker result=%s", tiebreaker_task)
                match_info.setdefault("llm_tiebreaker", {})["task"] = tiebreaker_task
                if tiebreaker_task.get("error"):
                    match_info["warnings"].append(tiebreaker_task["error"])
        else:
            if not llm_tiebreaker_enabled():
                logger.info("STEP task_tiebreaker SKIPPED (LLM_MATCH_TIEBREAKER_ENABLED=false)")
        task_confidence = float(tiebreaker_task.get("confidence", 0.0)) if tiebreaker_task and not tiebreaker_task.get("error") else 0.0
        if tiebreaker_task and not tiebreaker_task.get("error") and task_confidence >= llm_tiebreaker_confidence():
            chosen_task = next(
                (candidate for candidate in task_candidates if str(candidate.get("id")) == tiebreaker_task["chosen_id"]),
                None,
            )
            if chosen_task:
                match_info["task"] = chosen_task
                if match_info["status"] == "project_matched_by_llm":
                    match_info["status"] = "matched_by_llm"
                else:
                    match_info["status"] = "project_matched_task_matched_by_llm"
                match_info.setdefault("llm_tiebreaker", {})["task"] = tiebreaker_task
                logger.info("STEP task matched (LLM) | id=%s | confidence=%.4f", chosen_task.get("id"), task_confidence)
        if not match_info.get("task"):
            logger.warning("STEP task NOT MATCHED | tiebreaker=%s", tiebreaker_task)
            if tiebreaker_task and not tiebreaker_task.get("error"):
                match_info["warnings"].append("LLM tiebreaker did not confirm any task candidate.")
            elif task_candidates:
                match_info["best_task_candidate"] = task_candidates[0]
                match_info["warnings"].append("Best task candidate is below threshold.")
            else:
                match_info["warnings"].append("No tasks returned for matched project.")
            if match_info["status"] in {"project_matched", "project_matched_by_llm"}:
                match_info["status"] = (
                    "project_matched_by_llm_task_not_matched"
                    if match_info["status"] == "project_matched_by_llm"
                    else "project_matched_task_not_matched"
                )
            extraction["needs_review"] = True

    logger.info("STEP done | final_status=%s", match_info.get("status"))

    if os.getenv("AGRIBANK_MATCH_DEBUG", "false").lower() == "true":
        match_info["debug"] = {
            "searchable_project_count": len(projects),
            "task_count": len(tasks),
            "leaf_task_count": len(filter_leaf_tasks(tasks)),
        }

    extraction["work_detail_match"] = match_info
    extraction["work_detail_output"] = build_work_detail_output(extraction)
    return extraction


async def fetch_searchable_projects(api_key: str, project_tree: Optional[str] = "agribank") -> list[dict[str, Any]]:
    config = get_project_tree_config(project_tree, allow_default_base_url=True, api_key_override=api_key)
    cache_key = f"{config.key}:searchable_projects"
    cached = get_cache(PROJECT_CACHE, cache_key)
    if cached is not None:
        return cached

    projects = await fetch_paginated(api_key, "project/internal", {}, project_tree=config.key)
    projects = [
        project
        for project in projects
        if normalize_text(str(project.get("status") or "")) in SEARCHABLE_PROJECT_STATUSES
        and project.get("is_active", True)
    ]
    set_cache(PROJECT_CACHE, cache_key, projects)
    return projects


async def fetch_project_tasks(api_key: str, project_id: str, project_tree: Optional[str] = "agribank") -> list[dict[str, Any]]:
    config = get_project_tree_config(project_tree, allow_default_base_url=True, api_key_override=api_key)
    cache_key = f"{config.key}:tasks:{project_id}"
    cached = get_cache(TASK_CACHE, cache_key)
    if cached is not None:
        return cached

    tasks = await fetch_paginated(api_key, "task/internal", {"project_id": project_id}, project_tree=config.key)
    set_cache(TASK_CACHE, cache_key, tasks)
    return tasks


def llm_tiebreaker_enabled() -> bool:
    raw = os.getenv("LLM_MATCH_TIEBREAKER_ENABLED", "true").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def llm_tiebreaker_top_n() -> int:
    try:
        return max(1, int(os.getenv("LLM_MATCH_TIEBREAKER_TOP_N", "10")))
    except ValueError:
        return 10


def llm_tiebreaker_confidence() -> float:
    return env_float("LLM_MATCH_TIEBREAKER_CONFIDENCE", 0.6)


async def llm_pick_project(
    text: str,
    title: str,
    queries: list[str],
    candidates: list[dict[str, Any]],
    tree_name: str = "Agribank",
) -> Optional[dict[str, Any]]:
    if not candidates:
        return None
    prompt = build_project_tiebreaker_prompt(text, title, queries, candidates, tree_name=tree_name)
    response, error = await call_llm_tiebreaker(prompt)
    if error:
        return {"error": error}
    result = parse_tiebreaker_response(response, {str(c.get("id")) for c in candidates})
    if result is None and response is not None:
        return {"error": "LLM response could not be parsed."}
    return result


async def llm_pick_task(
    text: str,
    title: str,
    project: dict[str, Any],
    queries: list[str],
    candidates: list[dict[str, Any]],
    tree_name: str = "Agribank",
) -> Optional[dict[str, Any]]:
    if not candidates:
        return None
    prompt = build_task_tiebreaker_prompt(text, title, project, queries, candidates, tree_name=tree_name)
    response, error = await call_llm_tiebreaker(prompt)
    if error:
        return {"error": error}
    result = parse_tiebreaker_response(response, {str(c.get("id")) for c in candidates})
    if result is None and response is not None:
        return {"error": "LLM response could not be parsed."}
    return result


async def call_llm_tiebreaker(prompt: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    from app.services import extraction

    config = extraction.get_llm_config()
    if not config:
        logger.warning("LLM tiebreaker: no LLM config (check LLM_PROVIDER + GEMINI_API_KEY / Vertex creds)")
        return None, "LLM config unavailable (set LLM_PROVIDER=gemini and GEMINI_API_KEY or Vertex credentials)."
    logger.info(
        "LLM tiebreaker calling | provider=%s | auth_mode=%s | model=%s | prompt_chars=%d",
        config.get("provider"),
        config.get("auth_mode"),
        config.get("model"),
        len(prompt),
    )
    try:
        response = await extraction.call_llm_extraction(config, prompt)
        logger.info(
            "LLM tiebreaker OK | provider=%s | model=%s | response_keys=%s",
            config.get("provider"),
            config.get("model"),
            list(response.keys()) if isinstance(response, dict) else type(response).__name__,
        )
        return response, None
    except Exception as exc:
        logger.error("LLM tiebreaker call failed: %s", extraction.compact_error(exc))
        return None, f"LLM tiebreaker call failed: {extraction.compact_error(exc)}"


def parse_tiebreaker_response(
    response: Optional[dict[str, Any]],
    valid_ids: set[str],
) -> Optional[dict[str, Any]]:
    if not isinstance(response, dict):
        return None
    chosen = response.get("chosen_id")
    if chosen in (None, "", "null"):
        return None
    chosen_id = str(chosen)
    if chosen_id not in valid_ids:
        return None
    try:
        confidence = float(response.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    reasoning = str(response.get("reasoning") or "").strip()[:500] or None
    return {
        "chosen_id": chosen_id,
        "confidence": round(min(max(confidence, 0.0), 1.0), 4),
        "reasoning": reasoning,
    }


def build_project_tiebreaker_prompt(
    text: str,
    title: str,
    queries: list[str],
    candidates: list[dict[str, Any]],
    tree_name: str = "Agribank",
) -> str:
    snippet = tiebreaker_text_snippet(text)
    compact_candidates = [
        {
            "id": candidate.get("id"),
            "code": candidate.get("code"),
            "name": candidate.get("name"),
            "status": candidate.get("status"),
            "local_score": candidate.get("score"),
        }
        for candidate in candidates
    ]
    return f"""
	Bạn là tiebreaker chọn project {tree_name} cho 1 tài liệu OCR tiếng Việt.
Local matcher đã rank và đưa danh sách candidates dưới đây nhưng KHÔNG có item nào vượt threshold.
Hãy đọc tiêu đề + đoạn OCR + các query rồi quyết định project nào ĐÚNG nhất.

Quy tắc:
- chỉ được chọn id nằm trong danh sách candidates.
- Nếu KHÔNG có candidate nào hợp lý (ví dụ tên dự án trong OCR khác hẳn), trả chosen_id = null.
- confidence trong [0,1]; đặt >= 0.6 chỉ khi bằng chứng rõ.
- reasoning ngắn dưới 200 ký tự, tiếng Việt.
- Chỉ trả JSON hợp lệ, không markdown.

Title (rule local): {json.dumps(title or "", ensure_ascii=False)}

Project name candidates (sinh từ rule local): {json.dumps(queries[:6], ensure_ascii=False)}

	Candidates từ {tree_name} API (tối đa {len(compact_candidates)}):
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

Đoạn OCR đã rút gọn:
\"\"\"{snippet}\"\"\"

JSON output schema:
{{
  "chosen_id": "<id từ danh sách candidates hoặc null>",
  "confidence": 0.0,
  "reasoning": "<lý do ngắn>"
}}
""".strip()


def build_task_tiebreaker_prompt(
    text: str,
    title: str,
    project: dict[str, Any],
    queries: list[str],
    candidates: list[dict[str, Any]],
    tree_name: str = "Agribank",
) -> str:
    snippet = tiebreaker_text_snippet(text)
    compact_candidates = [
        {
            "id": candidate.get("id"),
            "name": candidate.get("name"),
            "status": candidate.get("status"),
            "workflow_step_name": candidate.get("workflow_step_name"),
            "workflow_phase": candidate.get("workflow_phase"),
            "local_score": candidate.get("score"),
        }
        for candidate in candidates
    ]
    project_info = {
        "id": project.get("id"),
        "code": project.get("code"),
        "name": project.get("name"),
    }
    return f"""
	Bạn là tiebreaker chọn task/công việc {tree_name} cho 1 tài liệu OCR đã match đúng project.
Local matcher đưa danh sách candidates dưới đây nhưng không có item nào vượt threshold.
Hãy đọc tiêu đề + đoạn OCR + các query rồi quyết định task nào ĐÚNG nhất.

Quy tắc:
- chỉ được chọn id nằm trong danh sách candidates.
- Nếu KHÔNG có task hợp lý, trả chosen_id = null.
- confidence trong [0,1]; đặt >= 0.6 chỉ khi bằng chứng rõ.
- reasoning ngắn dưới 200 ký tự, tiếng Việt.
- Chỉ trả JSON hợp lệ, không markdown.

Project đã match: {json.dumps(project_info, ensure_ascii=False)}
Title (rule local): {json.dumps(title or "", ensure_ascii=False)}

Task title candidates (sinh từ rule local): {json.dumps(queries[:6], ensure_ascii=False)}

Task candidates (tối đa {len(compact_candidates)}):
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

Đoạn OCR đã rút gọn:
\"\"\"{snippet}\"\"\"

JSON output schema:
{{
  "chosen_id": "<id từ danh sách candidates hoặc null>",
  "confidence": 0.0,
  "reasoning": "<lý do ngắn>"
}}
""".strip()


def tiebreaker_text_snippet(text: str) -> str:
    max_chars = max(1000, min(int(os.getenv("LLM_MATCH_TIEBREAKER_MAX_CHARS", "3500")), 12000))
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail_lines: list[str] = []
    keywords = ("du an", "cong trinh", "goi thau", "hang muc", "phe duyet", "tham dinh", "lua chon nha thau")
    remaining_budget = max_chars - len(head)
    for line in text[max_chars // 2 :].splitlines():
        normalized = normalize_text(line)
        if not any(keyword in normalized for keyword in keywords):
            continue
        if len(line) + 1 > remaining_budget:
            break
        tail_lines.append(line.strip())
        remaining_budget -= len(line) + 1
    if tail_lines:
        return head + "\n...\n" + "\n".join(tail_lines)
    return head


async def fetch_paginated(
    api_key: str,
    path: str,
    params: dict[str, Any],
    project_tree: Optional[str] = "agribank",
) -> list[dict[str, Any]]:
    config = get_project_tree_config(project_tree, allow_default_base_url=True, api_key_override=api_key)
    if not config.base_url:
        raise ProjectTreeConfigError(f"Missing {config.base_url_env}.", missing=[config.base_url_env])
    base_url = config.base_url.rstrip("/")
    page_size = int(os.getenv(f"{config.key.upper()}_PAGE_SIZE", os.getenv("AGRIBANK_PAGE_SIZE", "100")))
    max_pages = int(os.getenv(f"{config.key.upper()}_MAX_PAGES", os.getenv("AGRIBANK_MAX_PAGES", "10")))
    timeout = float(os.getenv(f"{config.key.upper()}_TIMEOUT_SECONDS", os.getenv("AGRIBANK_TIMEOUT_SECONDS", "15")))
    items: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for page in range(1, max_pages + 1):
            response = await client.get(
                f"{base_url}/{path.lstrip('/')}",
                params={**params, "page": page, "page_size": page_size},
                headers={"X-Api-Key": api_key},
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else {}
            page_items = data.get("items") if isinstance(data, dict) else []
            if not isinstance(page_items, list):
                break
            items.extend([item for item in page_items if isinstance(item, dict)])

            total = int(data.get("total") or len(items)) if isinstance(data, dict) else len(items)
            if len(items) >= total or len(page_items) < page_size:
                break

    return items


def find_best_project(queries: list[str], projects: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    candidates = rank_projects(queries, projects)
    return candidates[0] if candidates else None


def find_best_task(queries: list[str], tasks: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    candidates = rank_tasks(queries, tasks, "adhoc")
    return candidates[0] if candidates else None


def rank_projects(
    queries: list[str],
    projects: list[dict[str, Any]],
    cache_key: str = "agribank:projects:searchable",
) -> list[dict[str, Any]]:
    candidate_texts = [" ".join(str(project.get(key) or "") for key in ("name", "code")) for project in projects]
    scored = rank_records(queries, projects, candidate_texts, cache_key=cache_key)
    return [
        {
            "id": project.get("id"),
            "code": project.get("code"),
            "name": project.get("name"),
            "status": project.get("status"),
            **score_payload,
        }
        for project, score_payload in scored
    ]


def filter_leaf_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent_ids = {
        str(task.get("parent_id"))
        for task in tasks
        if task.get("parent_id") not in (None, "")
    }
    return [task for task in tasks if str(task.get("id")) not in parent_ids]


def rank_tasks(
    queries: list[str],
    tasks: list[dict[str, Any]],
    project_id: str,
    project_tree: str = "agribank",
) -> list[dict[str, Any]]:
    leaf_tasks = filter_leaf_tasks(tasks)
    candidate_texts = []
    for task in leaf_tasks:
        workflow_step = task.get("workflow_step") if isinstance(task.get("workflow_step"), dict) else {}
        candidate_texts.append(
            " ".join(
                str(value or "")
                for value in (
                    task.get("name"),
                    workflow_step.get("name"),
                    workflow_step.get("phase"),
                    task.get("status"),
                )
            )
        )

    scored = rank_records(queries, leaf_tasks, candidate_texts, cache_key=f"{project_tree}:tasks:leaves:{project_id}")
    results = []
    for task, score_payload in scored:
        workflow_step = task.get("workflow_step") if isinstance(task.get("workflow_step"), dict) else {}
        if normalize_text(str(task.get("status") or "")) in ACTIVE_TASK_STATUSES:
            score_payload = {**score_payload, "score": round(min(1.0, score_payload["score"] + 0.04), 4)}
        results.append(
            {
                "id": task.get("id"),
                "name": task.get("name"),
                "status": task.get("status"),
                "parent_id": task.get("parent_id"),
                "level": task.get("level"),
                "workflow_step_id": workflow_step.get("id"),
                "workflow_step_name": workflow_step.get("name"),
                "workflow_phase": workflow_step.get("phase"),
                "workflow_order_no": workflow_step.get("order_no"),
                **score_payload,
            }
        )

    sorted_results = sorted(results, key=lambda item: item["score"], reverse=True)

    # Log top 10 task matches for debugging
    logger.info("=" * 80)
    logger.info("TOP 10 TASK MATCHES | queries=%s", queries)
    logger.info("=" * 80)
    for i, task in enumerate(sorted_results[:10], 1):
        logger.info(
            "Task #%d | score=%.4f | matched_query=%r | name=%r",
            i,
            task.get("score", 0),
            task.get("matched_query", ""),
            task.get("name", "")[:80],
        )
        breakdown = task.get("score_breakdown", {})
        if breakdown:
            logger.info(
                "  Breakdown: matched_tokens=%d | longest_seq=%d | query_cov=%.2f%% | cand_cov=%.2f%%",
                breakdown.get("matched_tokens", 0),
                breakdown.get("longest_sequence", 0),
                breakdown.get("query_coverage", 0) * 100,
                breakdown.get("candidate_coverage", 0) * 100,
            )
    logger.info("=" * 80)

    return sorted_results


def build_project_queries(extraction: dict[str, Any], title: str) -> list[str]:
    generic = extraction.get("generic_extraction") if isinstance(extraction.get("generic_extraction"), dict) else {}
    entities = extraction.get("entities") if isinstance(extraction.get("entities"), dict) else {}
    queries: list[str] = []
    for key in ("project_name_candidates", "projects", "project_names", "project"):
        queries.extend(extract_text_candidates(generic.get(key)))
        queries.extend(extract_text_candidates(entities.get(key)))

    if title:
        queries.append(extract_project_tail(title) or title)
        queries.append(title)
    return unique_texts(queries)


def build_task_queries(title: str, extraction: Optional[dict[str, Any]] = None) -> list[str]:
    """Build task matching queries based ONLY on document title.

    CRITICAL DECISION: We match tasks using ONLY the document title, NOT LLM-extracted
    candidates or entities, because:

    1. Document title is the source of truth
    2. LLM-extracted task_title_candidates may be too generic (e.g., "khảo sát xây dựng")
    3. entities.work_items are often short fragments that match wrong tasks
    4. Short queries get perfect scores (query_coverage=1.0) even when semantically wrong

    Example problem (before fix):
    - Document: "QUYẾT ĐỊNH Phê duyệt thiết kế xây dựng..."
    - LLM extracts: "khảo sát xây dựng" (generic, 4 tokens)
    - Match result: "Thực hiện khảo sát xây dựng" score=1.0 (WRONG!)
    - Expected: "Quyết định phê duyệt thiết kế..." (CORRECT)

    Solution: Use only document title and its variants.

    Args:
        title: Document title (source of truth)
        extraction: DEPRECATED - kept for backward compatibility but NOT USED
    """
    # Note: extraction parameter is intentionally ignored
    # We used to extract task_title_candidates from it, but this caused
    # short generic queries to match wrong tasks
    _ = extraction  # Suppress unused parameter warning

    if not title:
        # No title available - return empty to skip task matching
        # (Project matching will handle this scenario)
        return []

    queries: list[str] = []

    # 1. Keep original title to preserve document type keywords (tờ trình, quyết định, etc.)
    # This is the PRIMARY and MOST IMPORTANT query
    queries.append(title)

    # 2. Create variant with minimal noise removal (keep document type keywords)
    cleaned_minimal = re.sub(r"\b(v/v|về việc|ve viec)\b", " ", title, flags=re.IGNORECASE)
    queries.append(cleaned_minimal)

    # 3. Create variant without project/construction site mentions
    # This helps match task names that don't repeat the project name
    without_project = re.split(
        r"\b(?:dự án|du an|công trình|cong trinh)\b\s*[:\-]?",
        cleaned_minimal,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]
    queries.append(without_project)

    # Return unique, non-empty queries
    return unique_texts([query for query in queries if query and len(query.strip()) >= 4])


def extract_text_candidates(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        results: list[str] = []
        for item in value:
            results.extend(extract_text_candidates(item))
        return results
    if isinstance(value, dict):
        results = []
        for key in ("name", "title", "value", "normalized_value", "keyword", "action", "text", "label"):
            if value.get(key) not in (None, ""):
                results.extend(extract_text_candidates(value.get(key)))
        return results
    return []


def extract_project_tail(title: str) -> Optional[str]:
    parts = re.split(r"\b(?:dự án|du an|công trình|cong trinh)\b\s*[:\-]?", title, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None
    value = parts[-1]
    value = re.split(
        r"\s+(?:địa điểm|dia diem|chủ đầu tư|chu dau tu|nguồn vốn|nguon von|gói thầu|goi thau|hạng mục|hang muc)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return value.strip(" .,:;-") or None


def best_score_for_queries(queries: list[str], candidate: str) -> tuple[float, str]:
    best = best_hybrid_score(queries, candidate, tfidf_scores={})
    return best["score"], best["matched_query"]


def match_score(query: str, candidate: str) -> float:
    return hybrid_score(query, candidate, tfidf_score=0.0)["score"]


def rank_records(
    queries: list[str],
    records: list[dict[str, Any]],
    candidate_texts: list[str],
    cache_key: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not queries or not records:
        return []

    expanded_queries = [expand_abbreviations(query) for query in queries]
    expanded_candidates = [expand_abbreviations(text) for text in candidate_texts]

    tfidf = compute_tfidf_scores(expanded_queries, expanded_candidates, cache_key)
    scored: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, record in enumerate(records):
        tfidf_scores = {query: tfidf.get((query, index), 0.0) for query in expanded_queries}
        payload = best_hybrid_score(expanded_queries, expanded_candidates[index], tfidf_scores=tfidf_scores)
        scored.append((record, payload))

    return sorted(scored, key=lambda item: item[1]["score"], reverse=True)


def detect_document_type(text: str) -> Optional[str]:
    """Detect document type from text (tờ trình, quyết định, công văn, biên bản, etc.)"""
    text_lower = text.lower()

    # Order matters: check more specific patterns first
    if re.search(r"\bquyết định|quyet dinh\b", text_lower):
        return "quyet_dinh"
    if re.search(r"\btờ trình|to trinh\b", text_lower):
        return "to_trinh"
    if re.search(r"\bcông văn|cong van\b", text_lower):
        return "cong_van"
    if re.search(r"\bbiên bản|bien ban\b", text_lower):
        return "bien_ban"
    if re.search(r"\bthông báo|thong bao\b", text_lower):
        return "thong_bao"
    if re.search(r"\bbáo cáo|bao cao\b", text_lower):
        return "bao_cao"
    if re.search(r"\bgiấy mời|giay moi\b", text_lower):
        return "giay_moi"
    if re.search(r"\bhợp đồng|hop dong\b", text_lower):
        return "hop_dong"

    return None


def compute_document_type_adjustment(query: str, candidate: str) -> float:
    """Compute bonus/penalty based on document type matching between query and candidate"""
    query_type = detect_document_type(query)
    candidate_type = detect_document_type(candidate)

    # No adjustment if either side doesn't have a detected type
    if not query_type or not candidate_type:
        return 0.0

    # Bonus if document types match
    if query_type == candidate_type:
        return 0.08

    # Penalty if document types conflict
    return -0.08


def best_hybrid_score(queries: list[str], candidate: str, tfidf_scores: dict[str, float]) -> dict[str, Any]:
    """Find best matching query for candidate.

    When multiple queries have similar scores, prefer longer queries
    as they provide more specific semantic context.
    """
    best_payload = {
        "score": 0.0,
        "matched_query": "",
        "score_breakdown": {},
    }

    for query in queries:
        payload = hybrid_score(query, candidate, tfidf_scores.get(query, 0.0))
        current_score = payload["score"]
        best_score = best_payload["score"]

        # Prefer this query if:
        # 1. Score is significantly better (>= 0.02 difference), OR
        # 2. Score is very close (< 0.02 difference) AND query is longer
        score_diff = current_score - best_score

        if score_diff >= 0.02:
            # Clear winner by score
            best_payload = {**payload, "matched_query": query}
        elif score_diff > -0.02:
            # Scores are very close (within 0.02) - prefer longer query
            current_query_tokens = len(significant_tokens(normalize_text(query)))
            best_query_tokens = len(significant_tokens(normalize_text(best_payload.get("matched_query", ""))))

            if current_query_tokens > best_query_tokens:
                # Longer query wins tie
                best_payload = {**payload, "matched_query": query}

    return best_payload


def longest_common_token_sequence(tokens1: list[str], tokens2: list[str]) -> int:
    """Find the longest common consecutive token sequence between two token lists.

    This helps prioritize matches that have long continuous sequences of matching tokens,
    which indicates stronger semantic similarity than scattered keyword matches.
    """
    if not tokens1 or not tokens2:
        return 0

    tokens2_set = set(tokens2)
    max_length = 0
    current_length = 0

    for token in tokens1:
        if token in tokens2_set:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0

    return max_length


def hybrid_score(query: str, candidate: str, tfidf_score: float) -> dict[str, Any]:
    query_norm = normalize_text(query)
    candidate_norm = normalize_text(candidate)
    if not query_norm or not candidate_norm:
        return {"score": 0.0, "score_breakdown": {}}

    query_tokens = significant_tokens(query_norm)
    candidate_tokens = significant_tokens(candidate_norm)

    # Calculate basic similarity metrics
    sequence = difflib.SequenceMatcher(None, query_norm, candidate_norm).ratio()
    rapid_score = rapid_fuzzy_score(query_norm, candidate_norm)
    doc_type_adjustment = compute_document_type_adjustment(query, candidate)

    # Short query penalty: queries with < 6 tokens are penalized
    # This prevents generic short queries from dominating long specific ones
    short_query_penalty = 0.0
    query_token_count = len(query_tokens)
    if query_token_count < 6:
        # Aggressive penalty: fewer tokens = much more penalty
        # 5 tokens: -0.05, 4 tokens: -0.10, 3 tokens: -0.15, 2 tokens: -0.20, 1 token: -0.25
        short_query_penalty = (6 - query_token_count) * 0.05

    # Handle edge case: no significant tokens
    if not query_tokens or not candidate_tokens:
        exact_score = 0.0
        if candidate_norm in query_norm:
            exact_score = 0.96
        if query_norm in candidate_norm:
            exact_score = max(exact_score, 0.92)

        weighted = 0.55 * rapid_score + 0.45 * sequence
        score = max(exact_score, weighted) + doc_type_adjustment - short_query_penalty
        return {
            "score": round(min(1.0, max(0.0, score)), 4),
            "score_breakdown": {
                "exact": round(exact_score, 4),
                "rapidfuzz": round(rapid_score, 4),
                "tfidf": round(tfidf_score, 4),
                "token": 0.0,
                "sequence": round(sequence, 4),
                "longest_sequence": 0,
                "matched_tokens": 0,
                "query_token_count": query_token_count,
                "short_query_penalty": round(short_query_penalty, 4),
                "doc_type_adjustment": round(doc_type_adjustment, 4),
            },
        }

    # Token-based scoring: prioritize more matching tokens
    intersection = query_tokens & candidate_tokens
    matched_token_count = len(intersection)
    query_coverage = matched_token_count / max(len(query_tokens), 1)
    candidate_coverage = matched_token_count / max(len(candidate_tokens), 1)
    jaccard = matched_token_count / max(len(query_tokens | candidate_tokens), 1)

    # Find longest common consecutive token sequence
    query_token_list = [t for t in query_norm.split() if t in query_tokens]
    candidate_token_list = [t for t in candidate_norm.split() if t in candidate_tokens]
    longest_seq = longest_common_token_sequence(query_token_list, candidate_token_list)

    # Enhanced exact matching with length bonus
    # REDUCED scores to avoid overconfidence
    exact_score = 0.0
    length_bonus = 0.0

    if candidate_norm in query_norm:
        # Candidate is substring of query - good match
        # Reduced from 0.92 → 0.78 to be more conservative
        base_exact = 0.78
        length_bonus = min(0.06, (matched_token_count / max(len(query_tokens), 1)) * 0.08)
        exact_score = base_exact + length_bonus

    if query_norm in candidate_norm:
        # Query is substring of candidate - also good but may have extra noise
        # Reduced from 0.88 → 0.75 to be more conservative
        base_exact = 0.75
        # Bonus for high query coverage, penalty for low candidate coverage (noise)
        length_bonus = min(0.06, query_coverage * 0.08)
        noise_penalty = max(0.0, (1.0 - candidate_coverage) * 0.08)
        exact_score = max(exact_score, base_exact + length_bonus - noise_penalty)

    # Longest sequence bonus: prefer long continuous matches
    # REDUCED bonuses to avoid overconfidence
    longest_seq_bonus = 0.0
    if longest_seq >= 8:
        longest_seq_bonus = 0.08  # Reduced from 0.12, higher threshold (6→8)
    elif longest_seq >= 5:
        longest_seq_bonus = 0.05  # Reduced from 0.08, higher threshold (4→5)
    elif longest_seq >= 3:
        longest_seq_bonus = 0.02  # Reduced from 0.04

    # Matched token count bonus: prefer matching many keywords
    # REDUCED bonuses and increased thresholds to be more conservative
    matched_count_bonus = 0.0
    if matched_token_count >= 10:
        matched_count_bonus = 0.06  # Reduced from 0.10, higher threshold (8→10)
    elif matched_token_count >= 7:
        matched_count_bonus = 0.04  # Reduced from 0.06, higher threshold (6→7)
    elif matched_token_count >= 5:
        matched_count_bonus = 0.02  # Reduced from 0.03, higher threshold (4→5)

    # Token score: balance between coverage metrics
    # Increase weight for query_coverage to prioritize matching more of the query keywords
    token_score = 0.40 * query_coverage + 0.35 * candidate_coverage + 0.25 * jaccard

    # Weighted combination of all metrics
    # Reduce overall weights to be more conservative
    weighted = (
        0.25 * rapid_score +
        0.28 * tfidf_score +
        0.38 * token_score +
        0.09 * sequence
    )

    # Combine all scoring components
    score = max(exact_score, weighted) + longest_seq_bonus + matched_count_bonus

    # REMOVED automatic boost to 0.88 - too overconfident
    # Let the score be determined by the metrics themselves
    # if candidate_coverage >= 0.85 and query_coverage >= 0.70 and matched_token_count >= 4:
    #     score = max(score, 0.88)

    # Apply document type adjustment
    score = score + doc_type_adjustment

    # Apply short query penalty (subtract from final score)
    score = score - short_query_penalty

    return {
        "score": round(min(1.0, max(0.0, score)), 4),
        "score_breakdown": {
            "exact": round(exact_score, 4),
            "rapidfuzz": round(rapid_score, 4),
            "tfidf": round(tfidf_score, 4),
            "token": round(token_score, 4),
            "sequence": round(sequence, 4),
            "longest_sequence": longest_seq,
            "longest_seq_bonus": round(longest_seq_bonus, 4),
            "matched_tokens": matched_token_count,
            "matched_count_bonus": round(matched_count_bonus, 4),
            "query_coverage": round(query_coverage, 4),
            "candidate_coverage": round(candidate_coverage, 4),
            "query_token_count": query_token_count,
            "short_query_penalty": round(short_query_penalty, 4),
            "doc_type_adjustment": round(doc_type_adjustment, 4),
        },
    }


def rapid_fuzzy_score(query_norm: str, candidate_norm: str) -> float:
    if fuzz is not None:
        return float(fuzz.WRatio(query_norm, candidate_norm)) / 100.0
    return difflib.SequenceMatcher(None, query_norm, candidate_norm).ratio()


def compute_tfidf_scores(queries: list[str], candidate_texts: list[str], cache_key: str) -> dict[tuple[str, int], float]:
    normalized_candidates = [normalize_text(text) for text in candidate_texts]
    if TfidfVectorizer is None or not normalized_candidates:
        return {}

    index = get_text_index(cache_key, normalized_candidates)
    if not index:
        return {}

    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    scores: dict[tuple[str, int], float] = {}
    for query in queries:
        query_norm = normalize_text(query)
        if not query_norm:
            continue
        query_vector = vectorizer.transform([query_norm])
        row = (query_vector @ matrix.T).toarray()[0]
        for idx, score in enumerate(row):
            scores[(query, idx)] = float(score)
    return scores


def get_text_index(cache_key: str, normalized_candidates: list[str]) -> Optional[dict[str, Any]]:
    digest = f"{cache_key}:{len(normalized_candidates)}:{hash(tuple(normalized_candidates))}"
    cached = get_cache(TEXT_INDEX_CACHE, digest)
    if cached is not None:
        return cached

    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False)
        matrix = vectorizer.fit_transform(normalized_candidates)
    except ValueError:
        return None

    index = {"vectorizer": vectorizer, "matrix": matrix}
    set_cache(TEXT_INDEX_CACHE, digest, index)
    return index


def build_work_detail_output(extraction: dict[str, Any]) -> dict[str, Any]:
    fields = (
        extraction.get("work_detail_fields")
        if isinstance(extraction.get("work_detail_fields"), dict)
        else extraction.get("fields")
        if isinstance(extraction.get("fields"), dict)
        else {}
    )
    match_info = extraction.get("work_detail_match") if isinstance(extraction.get("work_detail_match"), dict) else {}
    project = match_info.get("project") if isinstance(match_info.get("project"), dict) else {}
    task = match_info.get("task") if isinstance(match_info.get("task"), dict) else {}
    return {
        "document_number": field_value(fields, "document_number"),
        "signed_or_effective_date": field_value(fields, "signed_or_effective_date"),
        "approved_value": field_value(fields, "approved_value"),
        "submitted_value": field_value(fields, "submitted_value"),
        "issuer": field_value(fields, "issuer"),
        "notes": field_value(fields, "notes"),
        "title": field_value(fields, "title"),
        "project_id": project.get("id"),
        "project_code": project.get("code"),
        "project_name": project.get("name"),
        "project_match_score": project.get("score"),
        "task_id": task.get("id"),
        "task_name": task.get("name"),
        "task_status": task.get("status"),
        "workflow_step_id": task.get("workflow_step_id"),
        "workflow_step_name": task.get("workflow_step_name"),
        "workflow_phase": task.get("workflow_phase"),
        "task_match_score": task.get("score"),
        "needs_review": extraction.get("needs_review", False),
    }


def field_value(fields: dict[str, Any], key: str) -> Any:
    field = fields.get(key)
    if not isinstance(field, dict):
        return None
    # For date fields, prefer the human-readable display value over the ISO normalized one
    # so that backend output matches what the field card renders.
    date_keys = {
        "signed_or_effective_date",
        "signed_date",
        "performance_guarantee_end_date",
        "advance_guarantee_end_date",
    }
    if key in date_keys:
        value = field.get("value")
        if value not in (None, ""):
            return value
        return field.get("normalized_value")
    return field.get("normalized_value") if field.get("normalized_value") not in (None, "") else field.get("value")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    value = re.sub(r"[^a-z0-9/._-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def significant_tokens(normalized_value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[\s/._-]+", normalized_value)
        if len(token) >= 2 and token not in STOPWORDS and not token.isdigit()
    }


def unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = re.sub(r"\s+", " ", value).strip(" .,:;-")
        key = normalize_text(value)
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def get_cache(cache: dict[str, tuple[float, list[dict[str, Any]]]], key: str) -> Optional[list[dict[str, Any]]]:
    ttl = float(os.getenv("AGRIBANK_CACHE_TTL_SECONDS", "300"))
    item = cache.get(key)
    if not item:
        return None
    created_at, value = item
    if time.monotonic() - created_at > ttl:
        cache.pop(key, None)
        return None
    return value


def set_cache(cache: dict[str, tuple[float, list[dict[str, Any]]]], key: str, value: list[dict[str, Any]]) -> None:
    cache[key] = (time.monotonic(), value)
