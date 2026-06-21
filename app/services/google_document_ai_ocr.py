import logging
import mimetypes
import os
import platform
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from pathlib import Path
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.api_core.retry import Retry
from google.cloud import documentai
from google.protobuf.json_format import MessageToDict
from pypdf import PdfReader, PdfWriter

from app.services.timing import timed_stage

logger = logging.getLogger("ner_ocr")

ONLINE_MAX_SIZE = 40 * 1024 * 1024
ONLINE_MAX_PAGES_PDF = 14


def _process_document_request(
    client: documentai.DocumentProcessorServiceClient,
    request: documentai.ProcessRequest,
    *,
    processor_kind: str,
    page_offset: int,
):
    timeout = float(os.getenv("DOCUMENT_AI_RPC_TIMEOUT_SECONDS", "60"))
    retry_deadline = float(os.getenv("DOCUMENT_AI_RETRY_DEADLINE_SECONDS", "75"))
    with timed_stage(
        "document_ai_rpc",
        processor=processor_kind,
        page_offset=page_offset,
        timeout_seconds=timeout,
    ):
        return client.process_document(
            request=request,
            timeout=timeout,
            retry=Retry(deadline=retry_deadline),
        )


def get_file_type(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")


def get_mime_type(file_path: str) -> str:
    mime_map = {
        "pdf": "application/pdf",
        "html": "text/html",
        "htm": "text/html",
        "txt": "text/plain",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "ppt": "application/vnd.ms-powerpoint",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "csv": "text/csv",
        "json": "application/json",
        "xml": "application/xml",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "bmp": "image/bmp",
        "webp": "image/webp",
    }
    extension = get_file_type(file_path)
    if not extension:
        return "application/octet-stream"
    if extension in mime_map:
        return mime_map[extension]
    system_mime_type, _ = mimetypes.guess_type(file_path)
    return system_mime_type or "application/octet-stream"


def convert_doc_to_pdf(doc_path: str, output_dir: Optional[str] = None) -> str:
    output_dir = output_dir or os.path.dirname(doc_path)
    pdf_path = os.path.join(output_dir, f"{Path(doc_path).stem}.pdf")
    system = platform.system().lower()

    if system == "linux":
        cmd = [
            "xvfb-run",
            "-a",
            "--server-args=-screen 0 1024x768x24",
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            doc_path,
        ]
    else:
        cmd = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            doc_path,
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("LibreOffice/soffice is required for DOC/DOCX OCR.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"LibreOffice conversion failed: {exc.stderr}") from exc

    if not os.path.exists(pdf_path):
        raise RuntimeError(f"Converted PDF was not found: {pdf_path}")
    return pdf_path


def _split_pdf(src_path: str, out_dir: str, max_pages: int = ONLINE_MAX_PAGES_PDF) -> list[tuple[str, int]]:
    reader = PdfReader(src_path)
    parts: list[tuple[str, int]] = []
    clean_stem = Path(src_path).stem
    for i in range(0, len(reader.pages), max_pages):
        writer = PdfWriter()
        for page_index in range(i, min(i + max_pages, len(reader.pages))):
            writer.add_page(reader.pages[page_index])
        part_path = Path(out_dir) / f"{clean_stem}_part{i // max_pages + 1}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)
        parts.append((str(part_path), i))
    return parts


def _get_text_from_anchor(text_anchor: documentai.Document.TextAnchor, text: str) -> str:
    if not text_anchor.text_segments:
        return ""
    return "".join(
        text[int(segment.start_index) : int(segment.end_index)]
        for segment in text_anchor.text_segments
    )


def _bbox_from_layout(
    layout: documentai.Document.Page.Layout,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> Optional[dict[str, float]]:
    vertices = list(layout.bounding_poly.normalized_vertices)
    is_normalized = bool(vertices)
    if not vertices:
        vertices = list(layout.bounding_poly.vertices)
    if not vertices:
        return None

    xs = [float(vertex.x) for vertex in vertices]
    ys = [float(vertex.y) for vertex in vertices]
    if not is_normalized:
        if page_width <= 0 or page_height <= 0:
            return None
        xs = [x / page_width for x in xs]
        ys = [y / page_height for y in ys]

    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(1.0, max(xs))
    y_max = min(1.0, max(ys))
    if x_max <= x_min or y_max <= y_min:
        return None
    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def _page_layout_segments(doc: documentai.Document, page_offset: int = 0) -> list[dict]:
    segments: list[dict] = []
    for page in doc.pages:
        page_number = page.page_number + page_offset
        page_width = float(page.dimension.width or 0)
        page_height = float(page.dimension.height or 0)
        page_lines = list(getattr(page, "lines", []))
        page_segments = page_lines or list(page.paragraphs) or list(page.blocks) or list(page.tokens)
        segment_type = "line" if page_lines else "paragraph" if page.paragraphs else "block" if page.blocks else "token"
        for index, item in enumerate(page_segments):
            text = re.sub(r"\s+", " ", _get_text_from_anchor(item.layout.text_anchor, doc.text)).strip()
            bbox = _bbox_from_layout(item.layout, page_width=page_width, page_height=page_height)
            if not text or not bbox:
                continue
            segments.append(
                {
                    "id": f"p{page_number}-s{index}",
                    "page": page_number,
                    "type": segment_type,
                    "text": text,
                    "bbox": bbox,
                }
            )
    return segments


def _document_pages(doc: documentai.Document, page_offset: int = 0) -> list[dict]:
    pages: list[dict] = []
    for page in doc.pages:
        dimension = page.dimension
        pages.append(
            {
                "page": page.page_number + page_offset,
                "width": float(dimension.width or 0),
                "height": float(dimension.height or 0),
                "unit": dimension.unit or "",
            }
        )
    return pages


def _process_with_enterprise_ocr(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str,
    page_offset: int = 0,
) -> list[tuple[str, dict]]:
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    )
    name = client.processor_path(project_id, location, processor_id)
    with open(file_path, "rb") as image:
        image_content = image.read()
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type),
    )
    result = _process_document_request(
        client,
        request,
        processor_kind="enterprise_ocr",
        page_offset=page_offset,
    )
    doc = result.document

    chunks: list[tuple[str, dict]] = []
    for page in doc.pages:
        page_text = _get_text_from_anchor(page.layout.text_anchor, doc.text).strip()
        if not page_text:
            continue
        page_number = page.page_number + page_offset
        metadata = {
            "content": page_text,
            "pageSpan": {"pageStart": page_number, "pageEnd": page_number},
        }
        chunks.append((page_text, metadata))
    return chunks


def _process_with_enterprise_ocr_layout(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str,
    page_offset: int = 0,
) -> dict:
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    )
    name = client.processor_path(project_id, location, processor_id)
    with open(file_path, "rb") as image:
        image_content = image.read()
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type),
    )
    result = _process_document_request(
        client,
        request,
        processor_kind="enterprise_ocr_layout",
        page_offset=page_offset,
    )
    doc = result.document

    chunks = []
    for page in doc.pages:
        page_text = _get_text_from_anchor(page.layout.text_anchor, doc.text).strip()
        if not page_text:
            continue
        page_number = page.page_number + page_offset
        chunks.append(
            (
                page_text,
                {
                    "content": page_text,
                    "pageSpan": {"pageStart": page_number, "pageEnd": page_number},
                },
            )
        )

    return {
        "text": doc.text or "\n\n".join(text for text, _ in chunks),
        "chunks": chunks,
        "pages": _document_pages(doc, page_offset=page_offset),
        "segments": _page_layout_segments(doc, page_offset=page_offset),
    }


def _process_with_layout_parser(
    project_id: str,
    location: str,
    processor_id: str,
    processor_version: str,
    file_path: str,
    mime_type: str,
    chunk_size: int,
    include_ancestor_headings: bool = True,
    page_offset: int = 0,
) -> list[tuple[str, dict]]:
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    )
    name = client.processor_version_path(project_id, location, processor_id, processor_version)
    process_options = documentai.ProcessOptions(
        layout_config=documentai.ProcessOptions.LayoutConfig(
            chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                chunk_size=chunk_size,
                include_ancestor_headings=include_ancestor_headings,
            )
        )
    )
    with open(file_path, "rb") as image:
        image_content = image.read()
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type),
        process_options=process_options,
    )
    result = _process_document_request(
        client,
        request,
        processor_kind="layout_parser",
        page_offset=page_offset,
    )

    chunks: list[tuple[str, dict]] = []
    for chunk in getattr(result.document.chunked_document, "chunks", []):
        meta = MessageToDict(chunk._pb)
        page_span = meta.get("pageSpan") or {}
        meta["pageSpan"] = {
            "pageStart": page_span.get("pageStart", 0) + page_offset,
            "pageEnd": page_span.get("pageEnd", 0) + page_offset,
        }
        chunks.append((chunk.content, meta))
    return chunks


def ocr_document(
    enterprise_project_id: str,
    layout_project_id: str,
    location: str,
    layout_processor_id: str,
    layout_processor_version: str,
    enterprise_processor_id: str,
    file_path: str,
    mime_type: Optional[str] = None,
    chunk_size: int = 1000,
    include_ancestor_headings: bool = True,
    page_offset: int = 0,
    max_pages: int = ONLINE_MAX_PAGES_PDF,
    _is_recursive_call: bool = False,
) -> list[tuple[str, dict]]:
    mime_type = mime_type or get_mime_type(file_path)
    ext = Path(file_path).suffix.lower()
    current_file_path = file_path
    temp_files_to_cleanup: list[str] = []

    try:
        if not _is_recursive_call and ext in [".doc", ".docx"]:
            temp_dir = Path(file_path).parent / "temp_conversion"
            temp_dir.mkdir(exist_ok=True)
            current_file_path = convert_doc_to_pdf(file_path, str(temp_dir))
            temp_files_to_cleanup.extend([current_file_path, str(temp_dir)])
            mime_type = "application/pdf"
            ext = ".pdf"

        if not _is_recursive_call and ext == ".pdf":
            reader = PdfReader(current_file_path)
            if len(reader.pages) > max_pages:
                all_chunks: list[tuple[str, dict]] = []
                with timed_stage("ocr_pdf_split", pages=len(reader.pages), max_pages=max_pages):
                    parts = _split_pdf(current_file_path, str(Path(current_file_path).parent), max_pages)
                for part_path, _start_page in parts:
                    temp_files_to_cleanup.append(part_path)
                workers = min(len(parts), max(1, int(os.getenv("OCR_MAX_PARALLEL_PARTS", "2"))))
                calls = [
                    partial(
                        ocr_document,
                        enterprise_project_id=enterprise_project_id,
                        layout_project_id=layout_project_id,
                        location=location,
                        layout_processor_id=layout_processor_id,
                        layout_processor_version=layout_processor_version,
                        enterprise_processor_id=enterprise_processor_id,
                        file_path=part_path,
                        mime_type="application/pdf",
                        chunk_size=chunk_size,
                        include_ancestor_headings=include_ancestor_headings,
                        page_offset=start_page,
                        max_pages=max_pages,
                        _is_recursive_call=True,
                    )
                    for part_path, start_page in parts
                ]
                with timed_stage("ocr_pdf_parts", parts=len(parts), workers=workers):
                    if workers == 1:
                        results = [call() for call in calls]
                    else:
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            futures = [executor.submit(copy_context().run, call) for call in calls]
                            results = [future.result() for future in futures]
                for chunks in results:
                    all_chunks.extend(chunks)
                return all_chunks

        if mime_type == "application/pdf":
            return _process_with_enterprise_ocr(
                project_id=enterprise_project_id,
                location=location,
                processor_id=enterprise_processor_id,
                file_path=current_file_path,
                mime_type=mime_type,
                page_offset=page_offset,
            )

        return _process_with_layout_parser(
            project_id=layout_project_id,
            location=location,
            processor_id=layout_processor_id,
            processor_version=layout_processor_version,
            file_path=current_file_path,
            mime_type=mime_type,
            chunk_size=chunk_size,
            include_ancestor_headings=include_ancestor_headings,
            page_offset=page_offset,
        )
    finally:
        for temp_file in temp_files_to_cleanup:
            temp_path = Path(temp_file)
            try:
                if temp_path.is_file():
                    temp_path.unlink()
                elif temp_path.is_dir():
                    shutil.rmtree(temp_path, ignore_errors=True)
            except Exception as exc:
                logger.warning("Failed to clean up temporary file %s: %s", temp_file, exc)


def ocr_document_with_layout(
    enterprise_project_id: str,
    location: str,
    enterprise_processor_id: str,
    file_path: str,
    mime_type: Optional[str] = None,
    page_offset: int = 0,
    max_pages: int = ONLINE_MAX_PAGES_PDF,
    _is_recursive_call: bool = False,
) -> dict:
    mime_type = mime_type or get_mime_type(file_path)
    ext = Path(file_path).suffix.lower()
    current_file_path = file_path
    temp_files_to_cleanup: list[str] = []

    try:
        if not _is_recursive_call and ext in [".doc", ".docx"]:
            temp_dir = Path(file_path).parent / "temp_conversion"
            temp_dir.mkdir(exist_ok=True)
            current_file_path = convert_doc_to_pdf(file_path, str(temp_dir))
            temp_files_to_cleanup.extend([current_file_path, str(temp_dir)])
            mime_type = "application/pdf"
            ext = ".pdf"

        if not _is_recursive_call and ext == ".pdf":
            reader = PdfReader(current_file_path)
            if len(reader.pages) > max_pages:
                merged = {"text": "", "chunks": [], "pages": [], "segments": []}
                with timed_stage("ocr_pdf_split", pages=len(reader.pages), max_pages=max_pages):
                    parts = _split_pdf(current_file_path, str(Path(current_file_path).parent), max_pages)
                for part_path, _start_page in parts:
                    temp_files_to_cleanup.append(part_path)
                workers = min(len(parts), max(1, int(os.getenv("OCR_MAX_PARALLEL_PARTS", "2"))))
                calls = [
                    partial(
                        ocr_document_with_layout,
                        enterprise_project_id=enterprise_project_id,
                        location=location,
                        enterprise_processor_id=enterprise_processor_id,
                        file_path=part_path,
                        mime_type="application/pdf",
                        page_offset=start_page,
                        max_pages=max_pages,
                        _is_recursive_call=True,
                    )
                    for part_path, start_page in parts
                ]
                with timed_stage("ocr_pdf_parts", parts=len(parts), workers=workers):
                    if workers == 1:
                        results = [call() for call in calls]
                    else:
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            futures = [executor.submit(copy_context().run, call) for call in calls]
                            results = [future.result() for future in futures]
                for part in results:
                    merged["chunks"].extend(part["chunks"])
                    merged["pages"].extend(part["pages"])
                    merged["segments"].extend(part["segments"])
                merged["text"] = "\n\n".join(text for text, _ in merged["chunks"]).strip()
                return merged

        if mime_type != "application/pdf" and not mime_type.startswith("image/"):
            raise RuntimeError("Layout demo supports PDF and image OCR files.")

        return _process_with_enterprise_ocr_layout(
            project_id=enterprise_project_id,
            location=location,
            processor_id=enterprise_processor_id,
            file_path=current_file_path,
            mime_type=mime_type,
            page_offset=page_offset,
        )
    finally:
        for temp_file in temp_files_to_cleanup:
            temp_path = Path(temp_file)
            try:
                if temp_path.is_file():
                    temp_path.unlink()
                elif temp_path.is_dir():
                    shutil.rmtree(temp_path, ignore_errors=True)
            except Exception as exc:
                logger.warning("Failed to clean up temporary file %s: %s", temp_file, exc)
