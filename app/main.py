import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.services.agribank_matching import (
    ProjectTreeConfig,
    ProjectTreeConfigError,
    attach_work_detail_matches,
    fetch_searchable_projects,
    get_project_tree_config,
)
from app.services.extraction import (
    CONTRACT_FORMS,
    CONTRACTOR_GROUPS,
    apply_filename_document_number_hint,
    extract_information,
    normalize_extraction_type,
)
from app.services.google_document_ai_ocr import get_mime_type, ocr_document, ocr_document_with_layout
from app.services.layout_matching import attach_field_boxes
from app.services.ocr_cleaning import clean_ocr_chunks, clean_ocr_layout_result

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "app" / "static"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="NER OCR Information Extraction API",
    description="OCR tài liệu bằng Google Document AI và trích xuất thông tin local-first với Gemini fallback.",
    version="1.0.0",
)


class TextExtractionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="OCR text hoặc nội dung văn bản cần trích xuất.")
    type: str = Field("document", description="Extraction type: document hoặc contract.")


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        request_id: str,
        details: Any = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.details = details


@app.exception_handler(ApiError)
async def api_error_handler(_request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "request_id": exc.request_id,
            }
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/taxonomies")
def taxonomies():
    return {
        "contract_forms": CONTRACT_FORMS,
        "contractor_groups": CONTRACTOR_GROUPS,
    }


@app.get("/api/work-detail/projects")
async def work_detail_projects():
    api_key = os.getenv("AGRIBANK_API_KEY")
    if not api_key:
        return {
            "status": "disabled",
            "projects": [],
            "count": 0,
            "warnings": ["Missing AGRIBANK_API_KEY; cannot load projects."],
        }

    try:
        projects = await fetch_searchable_projects(api_key)
    except Exception as exc:
        return {
            "status": "error",
            "projects": [],
            "count": 0,
            "warnings": [f"Cannot fetch Agribank projects: {exc}"],
        }

    normalized_projects = [
        {
            "id": project.get("id"),
            "code": project.get("code"),
            "name": project.get("name"),
            "status": project.get("status"),
            "is_active": project.get("is_active", True),
        }
        for project in projects
    ]
    return {
        "status": "ok",
        "projects": normalized_projects,
        "count": len(normalized_projects),
        "warnings": [],
    }


@app.get("/api/v1/work-detail/projects")
async def v1_work_detail_projects(project: Optional[str] = None):
    request_id = new_request_id()
    config = require_v1_project_tree(project, request_id)

    try:
        projects = await fetch_searchable_projects(config.api_key or "", config.key)
    except Exception as exc:
        raise ApiError(
            502,
            "project_tree_fetch_failed",
            f"Cannot fetch {config.display_name} projects.",
            request_id=request_id,
            details={"error": str(exc)},
        ) from exc

    normalized_projects = [normalize_project_item(item) for item in projects]
    return {
        "request_id": request_id,
        "project": serialize_project_tree(config),
        "projects": normalized_projects,
        "count": len(normalized_projects),
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
@app.get("/demo-work-detail", response_class=HTMLResponse)
def demo():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/demo-layout", response_class=HTMLResponse)
def demo_layout():
    return (STATIC_DIR / "layout-demo.html").read_text(encoding="utf-8")


@app.post("/api/ocr/extract")
async def ocr_extract(
    file: Annotated[UploadFile, File(...)],
    extraction_type: Annotated[Optional[str], Query(alias="type")] = None,
):
    requested_type = parse_extraction_type(extraction_type)
    saved_path = save_upload(file)
    try:
        chunks, ocr_cleaning = clean_ocr_chunks(run_google_ocr(str(saved_path)))
        full_text = "\n\n".join(text for text, _ in chunks).strip()
        if not full_text:
            raise HTTPException(status_code=422, detail="OCR không trích được text từ file.")

        extraction = await extract_information(full_text, extraction_type=requested_type)
        extraction_data = apply_filename_document_number_hint(extraction["data"], file.filename)
        extraction_data = await attach_work_detail_matches(extraction_data, full_text)
        return {
            "file": {
                "name": file.filename,
                "mime_type": get_mime_type(str(saved_path)),
                "saved_as": saved_path.name,
            },
            "ocr": {
                "text": full_text,
                "cleaning": ocr_cleaning,
                "chunks": [
                    {
                        "text": text,
                        "page_span": metadata.get("pageSpan", {}),
                    }
                    for text, metadata in chunks
                ],
            },
            "extraction": extraction_data,
            "llm": {
                "provider": extraction["provider"],
                "model": extraction["model"],
            },
            "taxonomies": taxonomies(),
        }
    finally:
        if os.getenv("KEEP_UPLOADS", "false").lower() != "true":
            saved_path.unlink(missing_ok=True)


@app.post("/api/v1/extractions/file")
async def v1_extract_file(
    file: Annotated[Optional[UploadFile], File()] = None,
    project: Optional[str] = None,
    include_layout: bool = False,
    extraction_type: Annotated[Optional[str], Query(alias="type")] = None,
):
    request_id = new_request_id()
    config = require_v1_project_tree(project, request_id)
    requested_type = parse_extraction_type(extraction_type, request_id=request_id)
    if file is None:
        raise ApiError(422, "missing_file", "Missing upload file.", request_id=request_id)

    try:
        saved_path = save_upload(file)
    except HTTPException as exc:
        raise ApiError(
            422,
            "invalid_file",
            str(exc.detail),
            request_id=request_id,
            details={"filename": file.filename},
        ) from exc

    try:
        if include_layout:
            layout = clean_ocr_layout_result(run_google_ocr_layout_for_v1(str(saved_path), request_id))
            full_text = str(layout["text"]).strip()
            if not full_text:
                raise ApiError(422, "ocr_empty_text", "OCR did not extract text from the file.", request_id=request_id)

            extraction = await extract_information(full_text, extraction_type=requested_type)
            extraction_data = apply_filename_document_number_hint(extraction["data"], file.filename)
            extraction_data = await attach_work_detail_matches(extraction_data, full_text, config.key)
            extraction_data = attach_field_boxes(extraction_data, layout["segments"])
            return serialize_v1_extraction(
                request_id=request_id,
                config=config,
                extraction_data=extraction_data,
                llm_info=extraction,
                ocr_text=full_text,
                page_count=len(layout.get("pages") or []),
                file_info={
                    "name": file.filename,
                    "mime_type": get_mime_type(str(saved_path)),
                },
                ocr_cleaning=layout.get("cleaning"),
                layout_pages=layout.get("pages") or [],
                include_layout=True,
            )

        chunks, ocr_cleaning = clean_ocr_chunks(run_google_ocr_for_v1(str(saved_path), request_id))
        full_text = "\n\n".join(text for text, _ in chunks).strip()
        if not full_text:
            raise ApiError(422, "ocr_empty_text", "OCR did not extract text from the file.", request_id=request_id)

        extraction = await extract_information(full_text, extraction_type=requested_type)
        extraction_data = apply_filename_document_number_hint(extraction["data"], file.filename)
        extraction_data = await attach_work_detail_matches(extraction_data, full_text, config.key)
        return serialize_v1_extraction(
            request_id=request_id,
            config=config,
            extraction_data=extraction_data,
            llm_info=extraction,
            ocr_text=full_text,
            page_count=page_count_from_chunks(chunks),
            file_info={
                "name": file.filename,
                "mime_type": get_mime_type(str(saved_path)),
            },
            ocr_cleaning=ocr_cleaning,
            include_layout=False,
        )
    finally:
        if os.getenv("KEEP_UPLOADS", "false").lower() != "true":
            saved_path.unlink(missing_ok=True)


@app.post("/api/ocr/extract-layout")
async def ocr_extract_layout(
    file: Annotated[UploadFile, File(...)],
    extraction_type: Annotated[Optional[str], Query(alias="type")] = None,
):
    requested_type = parse_extraction_type(extraction_type)
    saved_path = save_upload(file)
    try:
        layout = clean_ocr_layout_result(run_google_ocr_layout(str(saved_path)))
        full_text = layout["text"].strip()
        if not full_text:
            raise HTTPException(status_code=422, detail="OCR không trích được text từ file.")

        extraction = await extract_information(full_text, extraction_type=requested_type)
        extraction_data = apply_filename_document_number_hint(extraction["data"], file.filename)
        extraction_data = await attach_work_detail_matches(extraction_data, full_text)
        extraction_data = attach_field_boxes(extraction_data, layout["segments"])
        return {
            "file": {
                "name": file.filename,
                "mime_type": get_mime_type(str(saved_path)),
                "saved_as": saved_path.name,
            },
            "ocr": {
                "text": full_text,
                "raw_text": layout.get("raw_text"),
                "cleaning": layout.get("cleaning"),
                "chunks": [
                    {
                        "text": text,
                        "page_span": metadata.get("pageSpan", {}),
                    }
                    for text, metadata in layout["chunks"]
                ],
            },
            "layout": {
                "pages": layout["pages"],
                "segments": layout["segments"],
                "removed_segments": (layout.get("cleaning") or {}).get("removed_segments", []),
            },
            "extraction": extraction_data,
            "llm": {
                "provider": extraction["provider"],
                "model": extraction["model"],
            },
            "taxonomies": taxonomies(),
        }
    finally:
        if os.getenv("KEEP_UPLOADS", "false").lower() != "true":
            saved_path.unlink(missing_ok=True)


@app.post("/api/v1/extractions/text")
async def v1_extract_text(
    payload: Annotated[Any, Body()] = None,
    project: Optional[str] = None,
    extraction_type: Annotated[Optional[str], Query(alias="type")] = None,
):
    request_id = new_request_id()
    config = require_v1_project_tree(project, request_id)
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ApiError(422, "invalid_text", "Request body must include non-empty text.", request_id=request_id)
    payload_type = payload.get("type") if isinstance(payload, dict) else None
    requested_type = parse_extraction_type(payload_type or extraction_type, request_id=request_id)

    extraction = await extract_information(text, extraction_type=requested_type)
    extraction_data = await attach_work_detail_matches(extraction["data"], text, config.key)
    return serialize_v1_extraction(
        request_id=request_id,
        config=config,
        extraction_data=extraction_data,
        llm_info=extraction,
        ocr_text=text,
        page_count=0,
    )


@app.post("/api/llm/extract")
async def llm_extract(request: TextExtractionRequest):
    requested_type = parse_extraction_type(request.type)
    extraction = await extract_information(request.text, extraction_type=requested_type)
    extraction_data = await attach_work_detail_matches(extraction["data"], request.text)
    return {
        "extraction": extraction_data,
        "llm": {
            "provider": extraction["provider"],
            "model": extraction["model"],
        },
        "taxonomies": taxonomies(),
    }


def new_request_id() -> str:
    return uuid.uuid4().hex


def parse_extraction_type(value: Optional[str], request_id: Optional[str] = None) -> str:
    try:
        return normalize_extraction_type(value)
    except ValueError as exc:
        message = "type must be either 'document' or 'contract'."
        if request_id:
            raise ApiError(
                422,
                "invalid_type",
                message,
                request_id=request_id,
                details={"type": value},
            ) from exc
        raise HTTPException(status_code=422, detail=message) from exc


def require_v1_project_tree(project: Optional[str], request_id: str) -> ProjectTreeConfig:
    try:
        return get_project_tree_config(project, require_config=True, allow_default_base_url=False)
    except ValueError as exc:
        raise ApiError(
            422,
            "invalid_project",
            str(exc),
            request_id=request_id,
            details={"allowed": ["opa", "opc", "agribank"], "aliases": {"argibank": "agribank"}},
        ) from exc
    except ProjectTreeConfigError as exc:
        raise ApiError(
            503,
            "project_tree_config_missing",
            str(exc),
            request_id=request_id,
            details={"missing": exc.missing},
        ) from exc


def serialize_project_tree(config: ProjectTreeConfig) -> dict[str, Any]:
    requested = config.requested if config.requested and config.requested != config.key else None
    return {
        "key": config.key,
        "requested": requested,
    }


def normalize_project_item(project: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": project.get("id"),
        "code": project.get("code"),
        "name": project.get("name"),
        "status": project.get("status"),
        "is_active": project.get("is_active", True),
    }


def serialize_v1_extraction(
    *,
    request_id: str,
    config: ProjectTreeConfig,
    extraction_data: dict[str, Any],
    llm_info: dict[str, Any],
    ocr_text: str,
    page_count: int,
    file_info: Optional[dict[str, Any]] = None,
    ocr_cleaning: Optional[dict[str, Any]] = None,
    layout_pages: Optional[list[dict[str, Any]]] = None,
    include_layout: bool = False,
) -> dict[str, Any]:
    screen = extraction_data.get("screen")
    response: dict[str, Any] = {
        "request_id": request_id,
        "project": serialize_project_tree(config),
        "ocr": {
            "text": ocr_text,
            "page_count": page_count,
        },
        "document": {
            "type": extraction_data.get("document_type"),
            "screen": screen,
            "intent": extraction_data.get("document_intent"),
            "needs_review": bool(extraction_data.get("needs_review", False)),
        },
        "fields": serialize_fields(extraction_data.get("fields"), include_boxes=include_layout),
        "llm": {
            "provider": llm_info.get("provider"),
            "model": llm_info.get("model"),
            "fallback_used": bool(extraction_data.get("llm_fallback_used", False)),
            "entity_extraction_used": bool(extraction_data.get("llm_entity_extraction_used", False)),
        },
    }
    if isinstance(extraction_data.get("work_detail_fields"), dict):
        response["work_detail_fields"] = serialize_fields(
            extraction_data.get("work_detail_fields"),
            include_boxes=include_layout,
        )
    if ocr_cleaning:
        response["ocr"]["cleaning"] = ocr_cleaning
    if file_info is not None:
        response["file"] = file_info
    if isinstance(extraction_data.get("work_detail_output"), dict):
        response["work_detail"] = serialize_work_detail_output(extraction_data.get("work_detail_output"))
    if isinstance(extraction_data.get("work_detail_match"), dict):
        response["match"] = serialize_work_detail_match(extraction_data.get("work_detail_match"))
    if include_layout:
        response["layout"] = {"pages": layout_pages or []}
    return response


def serialize_fields(fields: Any, *, include_boxes: bool) -> dict[str, Any]:
    if not isinstance(fields, dict):
        return {}
    serialized = {}
    for key, field in fields.items():
        if not isinstance(field, dict):
            continue
        item = {
            "label": field.get("label"),
            "value": field.get("value"),
            "normalized_value": field.get("normalized_value"),
            "confidence": field.get("confidence"),
            "evidence": field.get("evidence"),
            "source": field.get("source"),
        }
        if include_boxes and "box" in field:
            item["box"] = field.get("box")
        serialized[key] = item
    return serialized


def serialize_work_detail_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        output = {}
    keys = (
        "document_number",
        "signed_or_effective_date",
        "approved_value",
        "submitted_value",
        "issuer",
        "notes",
        "title",
    )
    return {key: output.get(key) for key in keys}


def serialize_work_detail_match(match_info: Any) -> dict[str, Any]:
    if not isinstance(match_info, dict):
        match_info = {}
    return {
        "status": match_info.get("status"),
        "project": serialize_match_project(match_info.get("project")),
        "task": serialize_match_task(match_info.get("task")),
        "warnings": match_info.get("warnings") if isinstance(match_info.get("warnings"), list) else [],
    }


def serialize_match_project(project: Any) -> Optional[dict[str, Any]]:
    if not isinstance(project, dict):
        return None
    return {
        "id": project.get("id"),
        "code": project.get("code"),
        "name": project.get("name"),
        "status": project.get("status"),
        "score": project.get("score"),
    }


def serialize_match_task(task: Any) -> Optional[dict[str, Any]]:
    if not isinstance(task, dict):
        return None
    return {
        "id": task.get("id"),
        "name": task.get("name"),
        "status": task.get("status"),
        "workflow_step_id": task.get("workflow_step_id"),
        "workflow_step_name": task.get("workflow_step_name"),
        "workflow_phase": task.get("workflow_phase"),
        "workflow_order_no": task.get("workflow_order_no"),
        "score": task.get("score"),
    }


def page_count_from_chunks(chunks: list[tuple[str, dict]]) -> int:
    pages: set[int] = set()
    for _, metadata in chunks:
        span = metadata.get("pageSpan") if isinstance(metadata, dict) else None
        if not isinstance(span, dict):
            continue
        try:
            start = int(span.get("pageStart") or 0)
            end = int(span.get("pageEnd") or start or 0)
        except (TypeError, ValueError):
            continue
        if start > 0 and end >= start:
            pages.update(range(start, end + 1))
    return len(pages)


def run_google_ocr_for_v1(file_path: str, request_id: str) -> list[tuple[str, dict]]:
    try:
        return run_google_ocr(file_path)
    except HTTPException as exc:
        raise api_error_from_http_exception(exc, request_id, upstream="ocr") from exc


def run_google_ocr_layout_for_v1(file_path: str, request_id: str) -> dict:
    try:
        return run_google_ocr_layout(file_path)
    except HTTPException as exc:
        raise api_error_from_http_exception(exc, request_id, upstream="ocr_layout") from exc


def api_error_from_http_exception(exc: HTTPException, request_id: str, *, upstream: str) -> ApiError:
    if exc.status_code == 422:
        return ApiError(422, "ocr_empty_text", str(exc.detail), request_id=request_id)
    if exc.status_code == 500:
        return ApiError(503, "ocr_config_missing", str(exc.detail), request_id=request_id)
    return ApiError(
        502,
        f"{upstream}_failed",
        str(exc.detail),
        request_id=request_id,
        details={"status_code": exc.status_code},
    )


def save_upload(file: UploadFile) -> Path:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Thiếu tên file upload.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".doc", ".docx", ".html", ".htm", ".txt"}:
        raise HTTPException(status_code=400, detail=f"Định dạng file chưa hỗ trợ: {suffix}")

    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(saved_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    return saved_path


def run_google_ocr(file_path: str) -> list[tuple[str, dict]]:
    required = {
        "GOOGLE_AI_PROJECT_ID": os.getenv("GOOGLE_AI_PROJECT_ID"),
        "GOOGLE_AI_LOCATION": os.getenv("GOOGLE_AI_LOCATION"),
        "GOOGLE_AI_PROCESSOR_ID": os.getenv("GOOGLE_AI_PROCESSOR_ID"),
        "GOOGLE_AI_PROCESSOR_VERSION": os.getenv("GOOGLE_AI_PROCESSOR_VERSION", "rc"),
        "ENTERPRISE_PROCESSOR_ID": os.getenv("ENTERPRISE_PROCESSOR_ID"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise HTTPException(status_code=500, detail=f"Thiếu cấu hình OCR: {', '.join(missing)}")

    try:
        return ocr_document(
            enterprise_project_id=required["GOOGLE_AI_PROJECT_ID"],
            layout_project_id=required["GOOGLE_AI_PROJECT_ID"],
            location=required["GOOGLE_AI_LOCATION"],
            layout_processor_id=required["GOOGLE_AI_PROCESSOR_ID"],
            layout_processor_version=required["GOOGLE_AI_PROCESSOR_VERSION"],
            enterprise_processor_id=required["ENTERPRISE_PROCESSOR_ID"],
            file_path=file_path,
            mime_type=get_mime_type(file_path),
            chunk_size=int(os.getenv("OCR_CHUNK_SIZE", "1000")),
            max_pages=int(os.getenv("OCR_MAX_PAGES_PER_REQUEST", "14")),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Lỗi Google Document AI OCR: {exc}") from exc


def run_google_ocr_layout(file_path: str) -> dict:
    required = {
        "GOOGLE_AI_PROJECT_ID": os.getenv("GOOGLE_AI_PROJECT_ID"),
        "GOOGLE_AI_LOCATION": os.getenv("GOOGLE_AI_LOCATION"),
        "ENTERPRISE_PROCESSOR_ID": os.getenv("ENTERPRISE_PROCESSOR_ID"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise HTTPException(status_code=500, detail=f"Thiếu cấu hình OCR: {', '.join(missing)}")

    try:
        return ocr_document_with_layout(
            enterprise_project_id=required["GOOGLE_AI_PROJECT_ID"],
            location=required["GOOGLE_AI_LOCATION"],
            enterprise_processor_id=required["ENTERPRISE_PROCESSOR_ID"],
            file_path=file_path,
            mime_type=get_mime_type(file_path),
            max_pages=int(os.getenv("OCR_MAX_PAGES_PER_REQUEST", "14")),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Lỗi Google Document AI OCR layout: {exc}") from exc
