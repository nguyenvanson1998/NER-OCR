import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.services.agribank_matching import attach_work_detail_matches, fetch_searchable_projects
from app.services.extraction import CONTRACT_FORMS, CONTRACTOR_GROUPS, extract_information
from app.services.google_document_ai_ocr import get_mime_type, ocr_document, ocr_document_with_layout
from app.services.layout_matching import attach_field_boxes

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


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
@app.get("/demo-work-detail", response_class=HTMLResponse)
def demo():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/demo-layout", response_class=HTMLResponse)
def demo_layout():
    return (STATIC_DIR / "layout-demo.html").read_text(encoding="utf-8")


@app.post("/api/ocr/extract")
async def ocr_extract(file: Annotated[UploadFile, File(...)]):
    saved_path = save_upload(file)
    try:
        chunks = run_google_ocr(str(saved_path))
        full_text = "\n\n".join(text for text, _ in chunks).strip()
        if not full_text:
            raise HTTPException(status_code=422, detail="OCR không trích được text từ file.")

        extraction = await extract_information(full_text)
        extraction_data = await attach_work_detail_matches(extraction["data"], full_text)
        return {
            "file": {
                "name": file.filename,
                "mime_type": get_mime_type(str(saved_path)),
                "saved_as": saved_path.name,
            },
            "ocr": {
                "text": full_text,
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


@app.post("/api/ocr/extract-layout")
async def ocr_extract_layout(file: Annotated[UploadFile, File(...)]):
    saved_path = save_upload(file)
    try:
        layout = run_google_ocr_layout(str(saved_path))
        full_text = layout["text"].strip()
        if not full_text:
            raise HTTPException(status_code=422, detail="OCR không trích được text từ file.")

        extraction = await extract_information(full_text)
        extraction_data = await attach_work_detail_matches(extraction["data"], full_text)
        extraction_data = attach_field_boxes(extraction_data, layout["segments"])
        return {
            "file": {
                "name": file.filename,
                "mime_type": get_mime_type(str(saved_path)),
                "saved_as": saved_path.name,
            },
            "ocr": {
                "text": full_text,
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


@app.post("/api/llm/extract")
async def llm_extract(request: TextExtractionRequest):
    extraction = await extract_information(request.text)
    extraction_data = await attach_work_detail_matches(extraction["data"], request.text)
    return {
        "extraction": extraction_data,
        "llm": {
            "provider": extraction["provider"],
            "model": extraction["model"],
        },
        "taxonomies": taxonomies(),
    }


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
