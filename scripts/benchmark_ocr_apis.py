"""Head-to-head: Cloud Vision (files:annotate) vs Document AI (process_document).

Calls the two OCR backends directly (no FastAPI / no enrichment) on the same
set of PDFs and reports per-file latency + segment counts. Both providers run
sequentially on each file so they do not compete for bandwidth.

Usage:
    python scripts/benchmark_ocr_apis.py --dir data/benchmark --runs 1

Env required (loaded from .env automatically):
    GOOGLE_AI_PROJECT_ID, GOOGLE_AI_LOCATION, ENTERPRISE_PROCESSOR_ID
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from app.services.google_document_ai_ocr import ocr_document_with_layout
from app.services.vision_ocr import close_vision_client, ocr_pdf_with_vision


@dataclass
class CallResult:
    provider: str
    file: str
    ok: bool
    duration_ms: float
    pages: int = 0
    segments: int = 0
    text_chars: int = 0
    error: str | None = None
    short_pages: list = None  # type: ignore[assignment]


async def run_vision(path: Path) -> CallResult:
    t0 = time.perf_counter()
    try:
        layout = await ocr_pdf_with_vision(str(path))
        dt = (time.perf_counter() - t0) * 1000
        return CallResult(
            provider="cloud_vision",
            file=path.name,
            ok=True,
            duration_ms=dt,
            pages=len(layout.get("pages") or []),
            segments=len(layout.get("segments") or []),
            text_chars=len(str(layout.get("text") or "")),
            short_pages=list((layout.get("quality") or {}).get("short_pages") or []),
        )
    except Exception as exc:
        return CallResult(
            provider="cloud_vision",
            file=path.name,
            ok=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {str(exc)[:160]}",
        )


def run_document_ai(path: Path) -> CallResult:
    project = os.getenv("GOOGLE_AI_PROJECT_ID")
    location = os.getenv("GOOGLE_AI_LOCATION", "us")
    processor = os.getenv("ENTERPRISE_PROCESSOR_ID") or os.getenv("GOOGLE_AI_PROCESSOR_ID")
    if not project or not processor:
        return CallResult(
            provider="document_ai",
            file=path.name,
            ok=False,
            duration_ms=0.0,
            error="missing GOOGLE_AI_PROJECT_ID / ENTERPRISE_PROCESSOR_ID",
        )
    t0 = time.perf_counter()
    try:
        layout = ocr_document_with_layout(
            enterprise_project_id=project,
            location=location,
            enterprise_processor_id=processor,
            file_path=str(path),
        )
        dt = (time.perf_counter() - t0) * 1000
        return CallResult(
            provider="document_ai",
            file=path.name,
            ok=True,
            duration_ms=dt,
            pages=len(layout.get("pages") or []),
            segments=len(layout.get("segments") or []),
            text_chars=len(str(layout.get("text") or "")),
        )
    except Exception as exc:
        return CallResult(
            provider="document_ai",
            file=path.name,
            ok=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(exc).__name__}: {str(exc)[:160]}",
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _summary(label: str, values: list[float]) -> str:
    if not values:
        return f"{label:<24} no data"
    return (
        f"{label:<24} n={len(values):>2}  "
        f"p50={_percentile(values, 50):>8.0f}  "
        f"p95={_percentile(values, 95):>8.0f}  "
        f"max={max(values):>8.0f}  "
        f"mean={statistics.fmean(values):>8.0f}"
    )


async def main_async(args) -> None:
    pdfs = sorted(p for p in args.dir.rglob("*.pdf") if p.is_file())
    if not pdfs:
        print("No PDFs in", args.dir)
        return

    results: list[CallResult] = []
    print(f"{'file':<60}  {'size MB':>7}  {'vision ms':>10}  {'docai ms':>10}  faster")
    print("-" * 110)
    for pdf in pdfs:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        vision_runs: list[CallResult] = []
        docai_runs: list[CallResult] = []
        for _ in range(args.runs):
            v = await run_vision(pdf)
            vision_runs.append(v)
            results.append(v)
            d = await asyncio.to_thread(run_document_ai, pdf)
            docai_runs.append(d)
            results.append(d)
        v_med = statistics.median(r.duration_ms for r in vision_runs)
        d_med = statistics.median(r.duration_ms for r in docai_runs)
        v_ok = all(r.ok for r in vision_runs)
        d_ok = all(r.ok for r in docai_runs)
        v_str = f"{v_med:>10.0f}" if v_ok else f"{'FAIL':>10}"
        d_str = f"{d_med:>10.0f}" if d_ok else f"{'FAIL':>10}"
        if v_ok and d_ok:
            faster = "vision" if v_med < d_med else "docai"
            ratio = max(v_med, d_med) / max(min(v_med, d_med), 1)
            faster_str = f"{faster} ({ratio:.1f}x)"
        elif v_ok:
            faster_str = "vision (docai failed)"
        elif d_ok:
            faster_str = "docai (vision failed)"
        else:
            faster_str = "both failed"
        print(f"{pdf.name[:60]:<60}  {size_mb:>7.2f}  {v_str}  {d_str}  {faster_str}")

    await close_vision_client()

    vision_ok = [r.duration_ms for r in results if r.provider == "cloud_vision" and r.ok]
    docai_ok = [r.duration_ms for r in results if r.provider == "document_ai" and r.ok]
    print("-" * 110)
    print(_summary("cloud_vision (ms)", vision_ok))
    print(_summary("document_ai  (ms)", docai_ok))

    vision_fails = [r for r in results if r.provider == "cloud_vision" and not r.ok]
    docai_fails = [r for r in results if r.provider == "document_ai" and not r.ok]
    if vision_fails:
        print(f"\nVision failures ({len(vision_fails)}):")
        for r in vision_fails:
            print(f"  - {r.file}: {r.error}")
    if docai_fails:
        print(f"\nDocument AI failures ({len(docai_fails)}):")
        for r in docai_fails:
            print(f"  - {r.file}: {r.error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1, help="Repeats per file per provider")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
