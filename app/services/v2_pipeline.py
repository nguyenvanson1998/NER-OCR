import asyncio
import contextlib
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

from app.services.agribank_matching import attach_work_detail_matches
from app.services.extraction import apply_filename_document_number_hint
from app.services.extraction import extract_information
from app.services.extraction import normalize_result
from app.services.job_queue import JobNotFound
from app.services.job_queue import JobStore
from app.services.layout_matching import attach_field_boxes
from app.services.ocr_cleaning import clean_ocr_layout_result
from app.services.timing import timed_stage


logger = logging.getLogger("ner_ocr.v2_pipeline")


def build_pending_ocr_record(
    *,
    job_id: str,
    request_id: str,
    saved_path: str,
    file_size: int,
    project_key: str,
    project_requested: str,
    filename: str,
    mime_type: str,
    extraction_type: str,
    include_layout: bool,
) -> dict[str, Any]:
    """Initial record when OCR is deferred to the worker (large files)."""
    return {
        "job_id": job_id,
        "request_id": request_id,
        "status": "ocr_pending",
        "revision": 0,
        "result": {
            "request_id": request_id,
            "project": {
                "key": project_key,
                "requested": project_requested if project_requested != project_key else None,
            },
            "file": {"name": filename, "mime_type": mime_type, "size_bytes": file_size},
            "ocr": {"provider": None, "text": None},
            "extraction": None,
            "status": "ocr_pending",
        },
        "enrichment": {"llm": "pending", "matching": "pending"},
        "warnings": [],
        "_payload": {
            "project_key": project_key,
            "project_requested": project_requested,
            "filename": filename,
            "mime_type": mime_type,
            "extraction_type": extraction_type,
            "include_layout": include_layout,
            "saved_path": saved_path,
            "pending_ocr": True,
        },
    }


async def build_fast_job_record(
    *,
    job_id: str,
    request_id: str,
    project_key: str,
    project_requested: str,
    filename: str,
    mime_type: str,
    extraction_type: str,
    layout: dict[str, Any],
    include_layout: bool,
    ocr_provider: str,
) -> dict[str, Any]:
    text = str(layout.get("text") or "").strip()
    with timed_stage("rule_parse_fast", extraction_type=extraction_type):
        extraction_data = normalize_result(
            {}, text, payload_source="rule", extraction_type=extraction_type
        )
    extraction_data["pipeline"] = "local"
    extraction_data = apply_filename_document_number_hint(extraction_data, filename)
    with timed_stage("field_box_match_fast", segments=len(layout.get("segments") or [])):
        extraction_data = await asyncio.to_thread(
            attach_field_boxes, extraction_data, layout.get("segments") or []
        )

    result = serialize_v2_result(
        request_id=request_id,
        project_key=project_key,
        project_requested=project_requested,
        filename=filename,
        mime_type=mime_type,
        extraction_data=extraction_data,
        llm_info={"provider": "local", "model": None},
        layout=layout,
        include_layout=include_layout,
        ocr_provider=ocr_provider,
    )
    # Revision 1 is deliberately local-only. Provider/model metadata appears
    # only after enrichment has reached a terminal state.
    result.pop("llm", None)
    return {
        "job_id": job_id,
        "request_id": request_id,
        "status": "enriching",
        "revision": 1,
        "result": result,
        "enrichment": {"llm": "pending", "matching": "pending"},
        "warnings": [],
        "_payload": {
            "project_key": project_key,
            "project_requested": project_requested,
            "filename": filename,
            "mime_type": mime_type,
            "extraction_type": extraction_type,
            "include_layout": include_layout,
            "text": text,
            "layout": layout,
            "base_extraction": extraction_data,
            "ocr_provider": ocr_provider,
        },
    }


async def _run_deferred_ocr(
    store: JobStore, job_id: str, record: dict[str, Any], payload: dict[str, Any]
) -> bool:
    """Run OCR for jobs created in pending_ocr mode. Returns True if OCR succeeded."""
    saved_path = Path(str(payload.get("saved_path") or ""))
    request_id = str(record.get("request_id") or job_id)
    filename = str(payload.get("filename") or "")
    extraction_type = str(payload.get("extraction_type") or "document")
    keep_uploads = os.getenv("KEEP_UPLOADS", "false").lower() == "true"

    if not saved_path.exists():
        record["status"] = "failed"
        record.setdefault("warnings", []).append("OCR input file is missing")
        with contextlib.suppress(JobNotFound):
            await store.update(job_id, record)
        return False

    try:
        # Lazy import to avoid circular dependency (main.py imports v2_pipeline).
        from app.main import run_fast_ocr_layout  # noqa: PLC0415

        with timed_stage("ocr_fast_path_async"):
            raw_layout, ocr_provider = await run_fast_ocr_layout(
                saved_path, request_id=request_id
            )
        with timed_stage("ocr_cleaning_layout_async"):
            layout = clean_ocr_layout_result(raw_layout)
        text = str(layout.get("text") or "").strip()
        if not text:
            record["status"] = "failed"
            record.setdefault("warnings", []).append("OCR returned empty text")
            with contextlib.suppress(JobNotFound):
                await store.update(job_id, record)
            return False

        with timed_stage("rule_parse_fast_async", extraction_type=extraction_type):
            base_extraction = normalize_result(
                {}, text, payload_source="rule", extraction_type=extraction_type
            )
        base_extraction["pipeline"] = "local"
        base_extraction = apply_filename_document_number_hint(base_extraction, filename)
        with timed_stage("field_box_match_fast_async", segments=len(layout.get("segments") or [])):
            base_extraction = await asyncio.to_thread(
                attach_field_boxes, base_extraction, layout.get("segments") or []
            )

        payload.update(
            {
                "text": text,
                "layout": layout,
                "base_extraction": base_extraction,
                "ocr_provider": ocr_provider,
                "pending_ocr": False,
            }
        )
        record["_payload"] = payload
        record["status"] = "enriching"
        record["revision"] = 1
        try:
            await store.update(job_id, record)
        except JobNotFound:
            return False
        return True
    except Exception as exc:
        logger.exception("deferred OCR failed for job %s", job_id)
        record["status"] = "failed"
        record.setdefault("warnings", []).append(
            f"Deferred OCR failed: {type(exc).__name__}"
        )
        with contextlib.suppress(JobNotFound):
            await store.update(job_id, record)
        return False
    finally:
        if not keep_uploads:
            with contextlib.suppress(OSError):
                saved_path.unlink(missing_ok=True)


async def process_enrichment_job(store: JobStore, job_id: str) -> None:
    record = await store.get(job_id)
    if record is None:
        return
    if record.get("status") in {"completed", "completed_with_warnings"}:
        return
    payload = record.get("_payload") if isinstance(record.get("_payload"), dict) else {}
    if not payload:
        record["status"] = "failed"
        record.setdefault("warnings", []).append("Missing enrichment payload.")
        await store.update(job_id, record)
        return

    if payload.get("pending_ocr"):
        ok = await _run_deferred_ocr(store, job_id, record, payload)
        if not ok:
            return

    text = str(payload.get("text") or "")
    extraction_type = str(payload.get("extraction_type") or "document")
    project_key = str(payload.get("project_key") or "agribank")
    base_extraction = copy.deepcopy(payload.get("base_extraction") or {})
    warnings: list[str] = []

    async def llm_enrichment():
        return await extract_information(text, extraction_type=extraction_type)

    async def local_matching():
        return await attach_work_detail_matches(copy.deepcopy(base_extraction), text, project_key)

    with timed_stage("enrichment_parallel"):
        llm_result, matched_local = await asyncio.gather(
            llm_enrichment(), local_matching(), return_exceptions=True
        )

    if isinstance(llm_result, BaseException):
        warnings.append(f"LLM enrichment failed: {type(llm_result).__name__}")
        llm_info = {"provider": "local", "model": None}
        final_extraction = copy.deepcopy(base_extraction)
        llm_status = "failed"
    else:
        llm_info = llm_result
        final_extraction = copy.deepcopy(llm_result.get("data") or base_extraction)
        final_extraction = apply_filename_document_number_hint(
            final_extraction, str(payload.get("filename") or "")
        )
        llm_status = "completed" if llm_info.get("provider") != "local" else "skipped"

    if isinstance(matched_local, BaseException):
        warnings.append(f"Local matching failed: {type(matched_local).__name__}")
        matched_local = None

    try:
        if matched_local is not None and matching_fingerprint(base_extraction) == matching_fingerprint(final_extraction):
            final_extraction = merge_matching(final_extraction, matched_local)
        else:
            final_extraction = await attach_work_detail_matches(
                final_extraction, text, project_key
            )
        matching_status = "completed"
    except Exception as exc:
        warnings.append(f"Project/task matching failed: {type(exc).__name__}")
        matching_status = "failed"

    try:
        final_extraction = await asyncio.to_thread(
            rematch_changed_boxes,
            final_extraction,
            base_extraction,
            (payload.get("layout") or {}).get("segments") or [],
        )
    except Exception as exc:
        warnings.append(f"Box rematching failed: {type(exc).__name__}")

    with timed_stage("enrichment_response_build"):
        final_result = serialize_v2_result(
            request_id=str(record.get("request_id") or ""),
            project_key=project_key,
            project_requested=str(payload.get("project_requested") or project_key),
            filename=str(payload.get("filename") or ""),
            mime_type=str(payload.get("mime_type") or ""),
            extraction_data=final_extraction,
            llm_info=llm_info if isinstance(llm_info, dict) else {},
            layout=payload.get("layout") or {},
            include_layout=bool(payload.get("include_layout")),
            ocr_provider=str(payload.get("ocr_provider") or "unknown"),
        )
    record.update(
        {
            "status": "completed_with_warnings" if warnings else "completed",
            "revision": 2,
            "result": final_result,
            "enrichment": {"llm": llm_status, "matching": matching_status},
            "warnings": warnings,
        }
    )
    try:
        await store.update(job_id, record)
    except JobNotFound:
        return


def matching_fingerprint(extraction: dict[str, Any]) -> str:
    fields = extraction.get("fields") if isinstance(extraction.get("fields"), dict) else {}
    generic = (
        extraction.get("generic_extraction")
        if isinstance(extraction.get("generic_extraction"), dict)
        else {}
    )
    values = {
        "screen": extraction.get("screen"),
        "title": _field_value(fields, "title"),
        "work_name": _field_value(fields, "work_name"),
        "contract_name": _field_value(fields, "contract_name"),
        "project_name_candidates": generic.get("project_name_candidates"),
        "task_title_candidates": generic.get("task_title_candidates"),
        "work_item_candidates": generic.get("work_item_candidates"),
    }
    return json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)


def merge_matching(target: dict[str, Any], matched: dict[str, Any]) -> dict[str, Any]:
    for key in ("work_detail_match", "work_detail_output"):
        if key in matched:
            target[key] = copy.deepcopy(matched[key])
    if matched.get("needs_review"):
        target["needs_review"] = True
    return target


def rematch_changed_boxes(
    target: dict[str, Any], base: dict[str, Any], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reuse unchanged boxes and only score fields whose value/evidence changed."""
    pending: dict[str, Any] = {}
    for group_name in ("fields", "work_detail_fields"):
        target_group = target.get(group_name) if isinstance(target.get(group_name), dict) else {}
        base_group = base.get(group_name) if isinstance(base.get(group_name), dict) else {}
        changed: dict[str, Any] = {}
        for key, field in target_group.items():
            if not isinstance(field, dict):
                continue
            old = base_group.get(key) if isinstance(base_group.get(key), dict) else {}
            if _box_fingerprint(field) == _box_fingerprint(old):
                if old.get("box") is not None:
                    field["box"] = copy.deepcopy(old["box"])
            else:
                field.pop("box", None)
                changed[key] = field
        if changed:
            pending[group_name] = changed
    if pending:
        attach_field_boxes(pending, segments)
    return target


def _box_fingerprint(field: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(
        str(field.get(key) or "") for key in ("value", "normalized_value", "evidence")
    )


def serialize_v2_result(
    *,
    request_id: str,
    project_key: str,
    project_requested: str,
    filename: str,
    mime_type: str,
    extraction_data: dict[str, Any],
    llm_info: dict[str, Any],
    layout: dict[str, Any],
    include_layout: bool,
    ocr_provider: str,
) -> dict[str, Any]:
    fields = _serialize_fields(extraction_data.get("fields"), include_boxes=True)
    result: dict[str, Any] = {
        "request_id": request_id,
        "project": {
            "key": project_key,
            "requested": project_requested if project_requested != project_key else None,
        },
        "file": {"name": filename, "mime_type": mime_type},
        "ocr": {
            "provider": ocr_provider,
            "text": str(layout.get("text") or ""),
            "raw_text": layout.get("raw_text"),
            "page_count": len(layout.get("pages") or []),
            "cleaning": layout.get("cleaning"),
            "chunks": [
                {"text": text, "page_span": (metadata or {}).get("pageSpan", {})}
                for text, metadata in layout.get("chunks") or []
            ],
        },
        "document": {
            "type": extraction_data.get("document_type"),
            "screen": extraction_data.get("screen"),
            "intent": extraction_data.get("document_intent"),
            "needs_review": bool(extraction_data.get("needs_review", False)),
        },
        "fields": fields,
        "llm": {
            "provider": llm_info.get("provider"),
            "model": llm_info.get("model"),
            "fallback_used": bool(extraction_data.get("llm_fallback_used", False)),
        },
    }
    if isinstance(extraction_data.get("work_detail_fields"), dict):
        result["work_detail_fields"] = _serialize_fields(
            extraction_data.get("work_detail_fields"), include_boxes=True
        )
    if isinstance(extraction_data.get("work_detail_output"), dict):
        result["work_detail"] = copy.deepcopy(extraction_data["work_detail_output"])
    if isinstance(extraction_data.get("work_detail_match"), dict):
        result["match"] = _serialize_match(extraction_data["work_detail_match"])
    if include_layout:
        result["layout"] = {
            "pages": copy.deepcopy(layout.get("pages") or []),
            "segments": copy.deepcopy(layout.get("segments") or []),
            "removed_segments": copy.deepcopy(
                (layout.get("cleaning") or {}).get("removed_segments", [])
            ),
        }
    return result


def _serialize_fields(fields: Any, include_boxes: bool) -> dict[str, Any]:
    if not isinstance(fields, dict):
        return {}
    result = {}
    for key, field in fields.items():
        if not isinstance(field, dict):
            continue
        item = {
            name: field.get(name)
            for name in ("label", "value", "normalized_value", "confidence", "evidence", "source")
        }
        if include_boxes:
            item["box"] = field.get("box")
        result[key] = item
    return result


def _serialize_match(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": match.get("status"),
        "project": copy.deepcopy(match.get("project")),
        "task": copy.deepcopy(match.get("task")),
        "warnings": copy.deepcopy(match.get("warnings") or []),
    }


def _field_value(fields: dict[str, Any], key: str) -> Any:
    field = fields.get(key) if isinstance(fields.get(key), dict) else {}
    return field.get("normalized_value") if field.get("normalized_value") not in (None, "") else field.get("value")
