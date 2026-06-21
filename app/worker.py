import asyncio
import contextlib
import logging
import os
import socket
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from app.services.job_queue import JobNotFound
from app.services.job_queue import JobStore
from app.services.job_queue import QueuedJob
from app.services.job_queue import get_job_store
from app.services.timing import begin_request_timing
from app.services.timing import end_request_timing
from app.services.v2_pipeline import process_enrichment_job


logger = logging.getLogger("ner_ocr.worker")


async def handle_job(store: JobStore, queued: QueuedJob) -> None:
    """Persist a terminal state before ACK so Redis delivery remains at-least-once."""
    initial = await store.get(queued.job_id)
    request_id = str((initial or {}).get("request_id") or queued.job_id)
    timing, token = begin_request_timing(
        request_id, "/worker/enrichment", debug=False
    )
    try:
        try:
            await process_enrichment_job(store, queued.job_id)
        except Exception as exc:
            logger.exception("enrichment job %s failed", queued.job_id)
            record = await store.get(queued.job_id)
            if record is not None:
                record["status"] = "completed_with_warnings"
                record.setdefault("warnings", []).append(
                    f"Enrichment worker failed: {type(exc).__name__}"
                )
                record.setdefault("enrichment", {"llm": "failed", "matching": "failed"})
                with contextlib.suppress(JobNotFound):
                    await store.update(queued.job_id, record)
        await store.ack(queued)
    except BaseException:
        timing.log_request_complete(500, status="error")
        raise
    else:
        timing.log_request_complete(200)
    finally:
        end_request_timing(token)


async def consume(store: JobStore, slot: int) -> None:
    consumer = f"{socket.gethostname()}-{os.getpid()}-{slot}-{uuid.uuid4().hex[:8]}"
    while True:
        try:
            queued = await store.next_job(consumer)
            if queued is not None and queued.job_id:
                await handle_job(store, queued)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("worker slot %s encountered a queue/storage error", slot)
            await asyncio.sleep(1)


async def run_worker() -> None:
    store = get_job_store()
    concurrency = max(1, int(os.getenv("ENRICHMENT_WORKER_CONCURRENCY", "4")))
    try:
        await asyncio.gather(*(consume(store, slot) for slot in range(concurrency)))
    finally:
        await store.close()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(run_worker())
