import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from typing import Iterator
from typing import Optional


logger = logging.getLogger("ner_ocr.performance")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.getenv("PERF_LOG_LEVEL", "INFO").upper())
logger.propagate = False
_CURRENT_TIMING: ContextVar[Optional["PipelineTiming"]] = ContextVar("ner_ocr_timing", default=None)


def _safe_dimension(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:160]


def _log_event(level: int, event: str, **data: Any) -> None:
    payload = {"event": event, **{key: _safe_dimension(value) for key, value in data.items()}}
    logger.log(level, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


@dataclass
class PipelineTiming:
    request_id: str
    route: str
    debug: bool = False
    started_at: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, stage: str, duration_ms: float, status: str = "ok", **dimensions: Any) -> None:
        with self._lock:
            self.stages[stage] = self.stages.get(stage, 0.0) + duration_ms
            self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1
            cumulative_ms = (time.perf_counter() - self.started_at) * 1000

        slow_ms = env_float("PERF_SLOW_STAGE_MS", 3000.0)
        level = logging.WARNING if duration_ms >= slow_ms or status != "ok" else logging.INFO
        _log_event(
            level,
            "stage_complete",
            request_id=self.request_id,
            route=self.route,
            stage=stage,
            status=status,
            duration_ms=round(duration_ms, 3),
            cumulative_ms=round(cumulative_ms, 3),
            **dimensions,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {key: round(value, 3) for key, value in self.stages.items()}
            counts = {key: value for key, value in self.stage_counts.items() if value > 1}
        result: dict[str, Any] = {
            "request_id": self.request_id,
            "total_ms": round((time.perf_counter() - self.started_at) * 1000, 3),
            "stages": stages,
        }
        if counts:
            result["stage_counts"] = counts
        return result

    def server_timing(self) -> str:
        with self._lock:
            items = list(self.stages.items())
        return ", ".join(f"{sanitize_metric_name(name)};dur={duration:.3f}" for name, duration in items)

    def log_request_complete(self, status_code: int, status: str = "ok") -> None:
        total_ms = (time.perf_counter() - self.started_at) * 1000
        slow_ms = env_float("PERF_SLOW_REQUEST_MS", 15000.0)
        level = logging.WARNING if total_ms >= slow_ms or status != "ok" else logging.INFO
        _log_event(
            level,
            "request_complete",
            request_id=self.request_id,
            route=self.route,
            status=status,
            status_code=status_code,
            total_ms=round(total_ms, 3),
        )


def begin_request_timing(request_id: str, route: str, debug: bool = False):
    timing = PipelineTiming(request_id=request_id, route=route, debug=debug)
    token = _CURRENT_TIMING.set(timing)
    return timing, token


def end_request_timing(token: Any) -> None:
    _CURRENT_TIMING.reset(token)


def current_timing() -> Optional[PipelineTiming]:
    return _CURRENT_TIMING.get()


def current_request_id() -> Optional[str]:
    timing = current_timing()
    return timing.request_id if timing else None


def timing_snapshot() -> Optional[dict[str, Any]]:
    timing = current_timing()
    return timing.snapshot() if timing else None


@contextmanager
def timed_stage(stage: str, **dimensions: Any) -> Iterator[None]:
    timing = current_timing()
    started_at = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        if timing:
            timing.record(
                stage,
                (time.perf_counter() - started_at) * 1000,
                status="error",
                error_type=type(exc).__name__,
                **dimensions,
            )
        raise
    else:
        if timing:
            timing.record(stage, (time.perf_counter() - started_at) * 1000, **dimensions)


def record_timing_event(stage: str, duration_ms: float = 0.0, **dimensions: Any) -> None:
    timing = current_timing()
    if timing:
        timing.record(stage, duration_ms, **dimensions)


def sanitize_metric_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in value)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
