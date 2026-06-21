import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from typing import Optional


logger = logging.getLogger("ner_ocr.jobs")
STREAM_NAME = "ner_ocr:enrichment"
CONSUMER_GROUP = "ner_ocr:workers"
JOB_KEY_PREFIX = "ner_ocr:job:"


class JobNotFound(KeyError):
    pass


@dataclass(frozen=True)
class QueuedJob:
    message_id: str
    job_id: str


class JobStore:
    async def create(self, job_id: str, record: dict[str, Any]) -> None:
        raise NotImplementedError

    async def get(self, job_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    async def update(self, job_id: str, record: dict[str, Any]) -> None:
        raise NotImplementedError

    async def next_job(self, consumer: str, block_ms: int = 5000) -> Optional[QueuedJob]:
        raise NotImplementedError

    async def ack(self, queued: QueuedJob) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class MemoryJobStore(JobStore):
    """Development/test backend. Production deployments should configure Redis."""

    def __init__(self, ttl_seconds: Optional[int] = None):
        self.ttl_seconds = ttl_seconds or _env_int("ENRICHMENT_JOB_TTL_SECONDS", 86400)
        self.records: dict[str, tuple[float, dict[str, Any]]] = {}
        self.queue: Optional[asyncio.Queue[QueuedJob]] = None

    async def create(self, job_id: str, record: dict[str, Any]) -> None:
        self.records[job_id] = (time.monotonic() + self.ttl_seconds, _json_copy(record))
        if self.queue is None:
            self.queue = asyncio.Queue()
        await self.queue.put(QueuedJob(message_id=job_id, job_id=job_id))

    async def get(self, job_id: str) -> Optional[dict[str, Any]]:
        item = self.records.get(job_id)
        if item is None:
            return None
        expires_at, record = item
        if time.monotonic() >= expires_at:
            self.records.pop(job_id, None)
            return None
        return _json_copy(record)

    async def update(self, job_id: str, record: dict[str, Any]) -> None:
        if await self.get(job_id) is None:
            raise JobNotFound(job_id)
        self.records[job_id] = (time.monotonic() + self.ttl_seconds, _json_copy(record))

    async def next_job(self, consumer: str, block_ms: int = 5000) -> Optional[QueuedJob]:
        del consumer
        if self.queue is None:
            self.queue = asyncio.Queue()
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=max(block_ms, 1) / 1000)
        except asyncio.TimeoutError:
            return None

    async def ack(self, queued: QueuedJob) -> None:
        del queued


class RedisJobStore(JobStore):
    def __init__(self, redis_url: str):
        try:
            import redis.asyncio as redis
        except ImportError as exc:  # pragma: no cover - depends on deployment package.
            raise RuntimeError("redis>=5 is required when REDIS_URL is configured") from exc
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = _env_int("ENRICHMENT_JOB_TTL_SECONDS", 86400)
        self._group_ready = False
        self._group_lock = asyncio.Lock()
        self._last_reclaim_at = 0.0

    async def _ensure_group(self) -> None:
        if self._group_ready:
            return
        async with self._group_lock:
            if self._group_ready:
                return
            try:
                await self.redis.xgroup_create(
                    STREAM_NAME, CONSUMER_GROUP, id="0", mkstream=True
                )
            except Exception as exc:
                if "BUSYGROUP" not in str(exc):
                    raise
            self._group_ready = True

    async def create(self, job_id: str, record: dict[str, Any]) -> None:
        key = JOB_KEY_PREFIX + job_id
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(key, json.dumps(record, ensure_ascii=False), ex=self.ttl_seconds)
            pipe.xadd(STREAM_NAME, {"job_id": job_id})
            await pipe.execute()

    async def get(self, job_id: str) -> Optional[dict[str, Any]]:
        raw = await self.redis.get(JOB_KEY_PREFIX + job_id)
        if not raw:
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None

    async def update(self, job_id: str, record: dict[str, Any]) -> None:
        key = JOB_KEY_PREFIX + job_id
        exists = await self.redis.exists(key)
        if not exists:
            raise JobNotFound(job_id)
        await self.redis.set(
            key, json.dumps(record, ensure_ascii=False), ex=self.ttl_seconds
        )

    async def next_job(self, consumer: str, block_ms: int = 5000) -> Optional[QueuedJob]:
        await self._ensure_group()
        reclaimed = await self._reclaim_abandoned(consumer)
        if reclaimed is not None:
            return reclaimed
        messages = await self.redis.xreadgroup(
            CONSUMER_GROUP,
            consumer,
            {STREAM_NAME: ">"},
            count=1,
            block=block_ms,
        )
        if not messages:
            return None
        _stream, entries = messages[0]
        if not entries:
            return None
        message_id, fields = entries[0]
        return QueuedJob(message_id=str(message_id), job_id=str(fields.get("job_id") or ""))

    async def _reclaim_abandoned(self, consumer: str) -> Optional[QueuedJob]:
        """Recover messages left pending by a crashed worker."""
        now = time.monotonic()
        interval = _env_int("ENRICHMENT_RECLAIM_INTERVAL_SECONDS", 30)
        if now - self._last_reclaim_at < interval:
            return None
        self._last_reclaim_at = now
        min_idle_ms = _env_int("ENRICHMENT_RECLAIM_IDLE_MS", 60000)
        response = await self.redis.xautoclaim(
            STREAM_NAME,
            CONSUMER_GROUP,
            consumer,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=1,
        )
        if not response or len(response) < 2:
            return None
        entries = response[1] or []
        if not entries:
            return None
        message_id, fields = entries[0]
        return QueuedJob(message_id=str(message_id), job_id=str(fields.get("job_id") or ""))

    async def ack(self, queued: QueuedJob) -> None:
        await self.redis.xack(STREAM_NAME, CONSUMER_GROUP, queued.message_id)

    async def close(self) -> None:
        await self.redis.aclose()


_STORE: Optional[JobStore] = None


def get_job_store() -> JobStore:
    global _STORE
    if _STORE is None:
        redis_url = os.getenv("REDIS_URL", "").strip()
        if redis_url:
            _STORE = RedisJobStore(redis_url)
        else:
            logger.warning("REDIS_URL is not configured; using non-durable in-memory jobs")
            _STORE = MemoryJobStore()
    return _STORE


def set_job_store(store: Optional[JobStore]) -> None:
    global _STORE
    _STORE = store


def public_job_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "_payload"}


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default
