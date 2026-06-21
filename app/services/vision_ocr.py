import asyncio
import base64
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any
from typing import Optional

import httpx
from pypdf import PdfReader, PdfWriter

from app.services.timing import timed_stage


logger = logging.getLogger("ner_ocr.vision")
VISION_ENDPOINT = "https://vision.googleapis.com/v1/files:annotate"
_HTTP_CLIENT: Optional[httpx.AsyncClient] = None
_CREDENTIALS: Any = None
_CREDENTIALS_LOCK = threading.Lock()
_CIRCUIT_OPEN_UNTIL = 0.0
_CIRCUIT_FAILURES = 0


class VisionOcrError(RuntimeError):
    pass


class VisionQualityError(VisionOcrError):
    pass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _normalize_credentials_path() -> None:
    raw = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not raw:
        return
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return
    repo_path = Path(__file__).resolve().parents[2] / raw
    if repo_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(repo_path)


def _access_token() -> str:
    global _CREDENTIALS
    _normalize_credentials_path()
    import google.auth
    from google.auth.transport.requests import Request

    with _CREDENTIALS_LOCK:
        if _CREDENTIALS is None:
            _CREDENTIALS, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not _CREDENTIALS.valid or _CREDENTIALS.expired or not _CREDENTIALS.token:
            _CREDENTIALS.refresh(Request())
        return str(_CREDENTIALS.token)


def _http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(_env_float("VISION_OCR_TIMEOUT_SECONDS", 4.2)),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _HTTP_CLIENT


async def close_vision_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None and not _HTTP_CLIENT.is_closed:
        await _HTTP_CLIENT.aclose()
    _HTTP_CLIENT = None


def _circuit_available() -> bool:
    return time.monotonic() >= _CIRCUIT_OPEN_UNTIL


def _record_failure(status_code: Optional[int] = None) -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    _CIRCUIT_FAILURES += 1
    if status_code in {401, 403, 429} or status_code is None or status_code >= 500:
        if _CIRCUIT_FAILURES >= _env_int("VISION_CIRCUIT_FAILURE_THRESHOLD", 2):
            _CIRCUIT_OPEN_UNTIL = time.monotonic() + _env_float(
                "VISION_CIRCUIT_BREAKER_SECONDS", 300.0
            )


def _record_success() -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    _CIRCUIT_FAILURES = 0
    _CIRCUIT_OPEN_UNTIL = 0.0


def split_pdf_bytes(content: bytes, max_pages: int = 5) -> list[tuple[bytes, int, int]]:
    reader = PdfReader(io.BytesIO(content))
    parts: list[tuple[bytes, int, int]] = []
    for start in range(0, len(reader.pages), max_pages):
        writer = PdfWriter()
        end = min(start + max_pages, len(reader.pages))
        for page_index in range(start, end):
            writer.add_page(reader.pages[page_index])
        output = io.BytesIO()
        writer.write(output)
        parts.append((output.getvalue(), start, end - start))
    return parts


async def ocr_pdf_with_vision(file_path: str) -> dict[str, Any]:
    with open(file_path, "rb") as source:
        return await ocr_pdf_bytes_with_vision(source.read())


async def ocr_pdf_bytes_with_vision(content: bytes) -> dict[str, Any]:
    if not _circuit_available():
        raise VisionOcrError("Cloud Vision circuit breaker is open")

    max_pages = max(1, min(5, _env_int("VISION_MAX_PAGES_PER_REQUEST", 5)))
    with timed_stage("vision_pdf_split"):
        parts = await asyncio.to_thread(split_pdf_bytes, content, max_pages)
    if not parts:
        raise VisionOcrError("PDF has no pages")

    semaphore = asyncio.Semaphore(max(1, _env_int("VISION_MAX_PARALLEL_PARTS", 2)))

    async def process(part: tuple[bytes, int, int]) -> dict[str, Any]:
        part_bytes, page_offset, expected_pages = part
        async with semaphore:
            return await _annotate_part(part_bytes, page_offset, expected_pages)

    with timed_stage("vision_ocr_total", parts=len(parts)):
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(process(part) for part in parts)),
                timeout=_env_float("VISION_OCR_TIMEOUT_SECONDS", 4.2),
            )
        except asyncio.TimeoutError as exc:
            _record_failure()
            raise VisionOcrError("Cloud Vision OCR exceeded the overall timeout") from exc

    merged: dict[str, Any] = {
        "text": "",
        "chunks": [],
        "pages": [],
        "segments": [],
        "provider": "cloud_vision",
    }
    confidences: list[float] = []
    for result in results:
        merged["chunks"].extend(result["chunks"])
        merged["pages"].extend(result["pages"])
        merged["segments"].extend(result["segments"])
        confidences.extend(result.get("_word_confidences") or [])
    merged["chunks"].sort(key=lambda item: int((item[1].get("pageSpan") or {}).get("pageStart", 0)))
    merged["pages"].sort(key=lambda item: int(item.get("page", 0)))
    merged["segments"].sort(key=_segment_sort_key)
    merged["text"] = "\n\n".join(text for text, _ in merged["chunks"] if text).strip()
    merged["quality"] = {
        "mean_word_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "word_count": len(confidences),
        "page_count": len(merged["pages"]),
    }

    _validate_quality(merged, expected_pages=sum(part[2] for part in parts))
    _record_success()
    return merged


def _segment_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    segment_id = str(item.get("id") or "")
    try:
        line = int(segment_id.rsplit("s", 1)[-1])
    except ValueError:
        line = 0
    return int(item.get("page") or 0), line


async def _annotate_part(content: bytes, page_offset: int, expected_pages: int) -> dict[str, Any]:
    payload = {
        "requests": [
            {
                "inputConfig": {
                    "content": base64.b64encode(content).decode("ascii"),
                    "mimeType": "application/pdf",
                },
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["vi"]},
                "pages": list(range(1, expected_pages + 1)),
            }
        ]
    }
    try:
        token = await asyncio.to_thread(_access_token)
        with timed_stage("vision_ocr_part", page_offset=page_offset, pages=expected_pages):
            headers = {"Authorization": f"Bearer {token}"}
            quota_project = os.getenv("GOOGLE_AI_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
            if quota_project:
                headers["x-goog-user-project"] = quota_project
            response = await _http_client().post(
                VISION_ENDPOINT,
                headers=headers,
                json=payload,
            )
        if response.status_code >= 400:
            _record_failure(response.status_code)
            raise VisionOcrError(f"Cloud Vision HTTP {response.status_code}: {response.text[:240]}")
        return parse_vision_response(response.json(), page_offset, expected_pages)
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        _record_failure()
        raise VisionOcrError(f"Cloud Vision request failed: {type(exc).__name__}") from exc


def parse_vision_response(payload: dict[str, Any], page_offset: int, expected_pages: int) -> dict[str, Any]:
    file_responses = payload.get("responses") if isinstance(payload, dict) else None
    file_response = file_responses[0] if isinstance(file_responses, list) and file_responses else {}
    page_responses = file_response.get("responses") if isinstance(file_response, dict) else None
    if not isinstance(page_responses, list):
        raise VisionOcrError("Cloud Vision response is missing page responses")

    result: dict[str, Any] = {"chunks": [], "pages": [], "segments": [], "_word_confidences": []}
    for local_index, page_response in enumerate(page_responses):
        if not isinstance(page_response, dict):
            continue
        if page_response.get("error"):
            raise VisionOcrError(f"Cloud Vision page error: {page_response['error']}")
        annotation = page_response.get("fullTextAnnotation") or {}
        annotation_pages = annotation.get("pages") or []
        page_data = annotation_pages[0] if annotation_pages else {}
        page_number = page_offset + local_index + 1
        width = float(page_data.get("width") or 0)
        height = float(page_data.get("height") or 0)
        page_text = str(annotation.get("text") or "").strip()
        result["pages"].append(
            {"page": page_number, "width": width, "height": height, "unit": "px"}
        )
        result["chunks"].append(
            (page_text, {"content": page_text, "pageSpan": {"pageStart": page_number, "pageEnd": page_number}})
        )
        segments, confidences = _page_segments(page_data, page_number, width, height)
        result["segments"].extend(segments)
        result["_word_confidences"].extend(confidences)

    if len(result["pages"]) != expected_pages:
        raise VisionOcrError(
            f"Cloud Vision returned {len(result['pages'])}/{expected_pages} pages"
        )
    return result


def _page_segments(
    page: dict[str, Any], page_number: int, width: float, height: float
) -> tuple[list[dict[str, Any]], list[float]]:
    segments: list[dict[str, Any]] = []
    confidences: list[float] = []
    line_index = 0
    for block in page.get("blocks") or []:
        for paragraph in block.get("paragraphs") or []:
            words = paragraph.get("words") or []
            current_words: list[str] = []
            current_boxes: list[dict[str, float]] = []
            for word in words:
                symbols = word.get("symbols") or []
                word_text = "".join(str(symbol.get("text") or "") for symbol in symbols)
                if not word_text:
                    continue
                confidence = float(word.get("confidence") or 0.0)
                if confidence <= 0 and symbols:
                    values = [float(symbol.get("confidence") or 0.0) for symbol in symbols]
                    confidence = sum(values) / len(values)
                if confidence > 0:
                    confidences.append(confidence)
                box = _normalized_box(word.get("boundingBox") or {}, width, height)
                current_words.append(word_text)
                if box:
                    current_boxes.append(box)
                break_type = _word_break_type(symbols)
                if break_type in {"EOL_SURE_SPACE", "LINE_BREAK"}:
                    if current_words and current_boxes:
                        segments.append(
                            {
                                "id": f"p{page_number}-s{line_index}",
                                "page": page_number,
                                "type": "line",
                                "text": " ".join(current_words).strip(),
                                "bbox": _union_boxes(current_boxes),
                            }
                        )
                        line_index += 1
                    current_words, current_boxes = [], []
            if current_words and current_boxes:
                segments.append(
                    {
                        "id": f"p{page_number}-s{line_index}",
                        "page": page_number,
                        "type": "line",
                        "text": " ".join(current_words).strip(),
                        "bbox": _union_boxes(current_boxes),
                    }
                )
                line_index += 1
    return segments, confidences


def _word_break_type(symbols: list[dict[str, Any]]) -> str:
    if not symbols:
        return ""
    prop = symbols[-1].get("property") or {}
    detected_break = prop.get("detectedBreak") or {}
    return str(detected_break.get("type") or "")


def _normalized_box(poly: dict[str, Any], width: float, height: float) -> Optional[dict[str, float]]:
    vertices = poly.get("normalizedVertices") or []
    normalized = bool(vertices)
    if not vertices:
        vertices = poly.get("vertices") or []
    if not vertices:
        return None
    xs = [float(vertex.get("x") or 0.0) for vertex in vertices]
    ys = [float(vertex.get("y") or 0.0) for vertex in vertices]
    if not normalized:
        if width <= 0 or height <= 0:
            return None
        xs = [value / width for value in xs]
        ys = [value / height for value in ys]
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    return {
        "x": round(max(0.0, left), 6),
        "y": round(max(0.0, top), 6),
        "width": round(max(0.0, min(1.0, right) - max(0.0, left)), 6),
        "height": round(max(0.0, min(1.0, bottom) - max(0.0, top)), 6),
    }


def _union_boxes(boxes: list[dict[str, float]]) -> dict[str, float]:
    left = min(box["x"] for box in boxes)
    top = min(box["y"] for box in boxes)
    right = max(box["x"] + box["width"] for box in boxes)
    bottom = max(box["y"] + box["height"] for box in boxes)
    return {
        "x": round(left, 6),
        "y": round(top, 6),
        "width": round(right - left, 6),
        "height": round(bottom - top, 6),
    }


def _validate_quality(result: dict[str, Any], expected_pages: int) -> None:
    if len(result.get("pages") or []) != expected_pages:
        raise VisionQualityError("Vision page count does not match input")
    if os.getenv("VISION_QUALITY_GATE_ENABLED", "true").strip().lower() in {"false", "0", "no"}:
        return

    # Blank/image-only pages (covers, back of scans, photo plates) are accepted
    # unconditionally. Track them in `quality.short_pages` for observability
    # but never use them to reject the entire result.
    min_chars = _env_int("VISION_MIN_PAGE_TEXT_CHARS", 20)
    chunks = result.get("chunks") or []
    short_pages = [
        metadata.get("pageSpan", {}).get("pageStart")
        for text, metadata in chunks
        if len(str(text or "").strip()) < min_chars
    ]
    if short_pages:
        result.setdefault("quality", {})["short_pages"] = short_pages

    confidence = float((result.get("quality") or {}).get("mean_word_confidence") or 0.0)
    threshold = _env_float("VISION_MIN_WORD_CONFIDENCE", 0.70)
    if confidence and confidence < threshold:
        raise VisionQualityError(
            f"Vision mean word confidence {confidence:.3f} is below {threshold:.3f}"
        )
