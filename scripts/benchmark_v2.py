"""Benchmark the V2 fast extraction pipeline (p50/p95/p99 latency).

Submits every PDF under --dir to ``POST /api/v2/extractions/file?debug_timing=true``,
polls each returned job until terminal, and aggregates per-stage timings from the
``timings`` block plus the ``ocr_provider`` label so Vision vs Document AI runs
are reported separately.

Usage
-----
    python scripts/benchmark_v2.py --dir data/benchmark --concurrency 8 \
        --base-url http://localhost:8000 --project agribank --type document

Prereqs: the API process and the worker (``python -m app.worker``) must be running
and reachable at --base-url.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class RunResult:
    filename: str
    ok: bool
    status: str
    ocr_provider: str
    total_ms: float
    submit_ms: float
    poll_ms: float
    stages: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    http_status: int | None = None


async def _submit(
    client: httpx.AsyncClient,
    base_url: str,
    pdf: Path,
    project: str,
    extraction_type: str,
    include_layout: bool,
) -> tuple[int, dict[str, Any]]:
    params = {
        "project": project,
        "type": extraction_type,
        "include_layout": str(include_layout).lower(),
        "debug_timing": "true",
    }
    with pdf.open("rb") as fh:
        files = {"file": (pdf.name, fh.read(), "application/pdf")}
    resp = await client.post(f"{base_url}/api/v2/extractions/file", params=params, files=files)
    try:
        body = resp.json()
    except json.JSONDecodeError:
        body = {"raw": resp.text}
    return resp.status_code, body


async def _poll(
    client: httpx.AsyncClient,
    base_url: str,
    job_id: str,
    poll_timeout_s: float,
    poll_interval_s: float,
) -> tuple[int, dict[str, Any]]:
    deadline = time.monotonic() + poll_timeout_s
    last_status = 0
    last_body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        resp = await client.get(f"{base_url}/api/v2/extractions/{job_id}")
        try:
            last_body = resp.json()
        except json.JSONDecodeError:
            last_body = {"raw": resp.text}
        last_status = resp.status_code
        if resp.status_code == 200:
            return resp.status_code, last_body
        if resp.status_code >= 400 and resp.status_code != 202:
            return resp.status_code, last_body
        await asyncio.sleep(poll_interval_s)
    return last_status, last_body


def _extract_provider(body: dict[str, Any]) -> str:
    if not isinstance(body, dict):
        return ""
    result = body.get("result") if isinstance(body.get("result"), dict) else {}
    ocr = result.get("ocr") if isinstance(result.get("ocr"), dict) else {}
    return str(
        ocr.get("provider")
        or result.get("ocr_provider")
        or body.get("ocr_provider")
        or ""
    )


async def run_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    pdf: Path,
    project: str,
    extraction_type: str,
    include_layout: bool,
    poll_timeout_s: float,
    poll_interval_s: float,
) -> RunResult:
    async with semaphore:
        t_start = time.monotonic()
        try:
            submit_status, submit_body = await _submit(
                client, base_url, pdf, project, extraction_type, include_layout
            )
        except Exception as exc:
            return RunResult(
                filename=pdf.name,
                ok=False,
                status="submit_error",
                ocr_provider="",
                total_ms=(time.monotonic() - t_start) * 1000,
                submit_ms=(time.monotonic() - t_start) * 1000,
                poll_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        t_submitted = time.monotonic()
        submit_ms = (t_submitted - t_start) * 1000

        if submit_status != 202:
            return RunResult(
                filename=pdf.name,
                ok=False,
                status="submit_http_error",
                ocr_provider=_extract_provider(submit_body),
                total_ms=submit_ms,
                submit_ms=submit_ms,
                poll_ms=0.0,
                http_status=submit_status,
                error=json.dumps(submit_body)[:300],
            )

        job_id = submit_body.get("job_id") or submit_body.get("id")
        if not job_id:
            return RunResult(
                filename=pdf.name,
                ok=False,
                status="missing_job_id",
                ocr_provider="",
                total_ms=submit_ms,
                submit_ms=submit_ms,
                poll_ms=0.0,
                http_status=submit_status,
                error=json.dumps(submit_body)[:300],
            )

        poll_status, poll_body = await _poll(
            client, base_url, job_id, poll_timeout_s, poll_interval_s
        )
        t_done = time.monotonic()
        poll_ms = (t_done - t_submitted) * 1000
        total_ms = (t_done - t_start) * 1000
        ok = poll_status == 200 and poll_body.get("status") in {"completed", "completed_with_warnings"}

        timings_block = poll_body.get("timings") or submit_body.get("timings") or {}
        stages = timings_block.get("stages") if isinstance(timings_block, dict) else None
        if not isinstance(stages, dict):
            stages = {}

        return RunResult(
            filename=pdf.name,
            ok=ok,
            status=str(poll_body.get("status") or "unknown"),
            ocr_provider=_extract_provider(poll_body) or _extract_provider(submit_body),
            total_ms=total_ms,
            submit_ms=submit_ms,
            poll_ms=poll_ms,
            stages={k: float(v) for k, v in stages.items() if isinstance(v, (int, float))},
            http_status=poll_status,
            error=None if ok else json.dumps({"status": poll_body.get("status"), "warnings": poll_body.get("warnings")})[:300],
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _summarize(label: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {"label": label, "count": 0}
    return {
        "label": label,
        "count": len(values),
        "p50_ms": round(_percentile(values, 50), 2),
        "p95_ms": round(_percentile(values, 95), 2),
        "p99_ms": round(_percentile(values, 99), 2),
        "max_ms": round(max(values), 2),
        "mean_ms": round(statistics.fmean(values), 2),
    }


def _print_summary(results: list[RunResult]) -> None:
    ok_runs = [r for r in results if r.ok]
    print("\n" + "=" * 80)
    print(f"Total: {len(results)}   OK: {len(ok_runs)}   Failed: {len(results) - len(ok_runs)}")
    print("=" * 80)

    if not ok_runs:
        print("No successful runs to summarize.")
        return

    provider_counts: dict[str, int] = {}
    for r in ok_runs:
        provider_counts[r.ocr_provider or "unknown"] = provider_counts.get(r.ocr_provider or "unknown", 0) + 1
    print("OCR provider distribution:")
    for prov, count in sorted(provider_counts.items()):
        share = 100 * count / len(ok_runs)
        print(f"  {prov:30s} {count:5d}  ({share:5.1f}%)")

    print("\nEnd-to-end latency (client-side, ms):")
    print(json.dumps(_summarize("total_e2e", [r.total_ms for r in ok_runs]), indent=2))

    print("\nPer-provider end-to-end:")
    for prov in sorted(provider_counts):
        provider_runs = [r.total_ms for r in ok_runs if (r.ocr_provider or "unknown") == prov]
        print(json.dumps(_summarize(prov, provider_runs), indent=2))

    stage_keys: set[str] = set()
    for r in ok_runs:
        stage_keys.update(r.stages.keys())
    if stage_keys:
        print("\nServer-side stage timings (from debug_timing block, ms):")
        for stage in sorted(stage_keys):
            stage_values = [r.stages[stage] for r in ok_runs if stage in r.stages]
            if stage_values:
                print(json.dumps(_summarize(stage, stage_values), indent=2))


def _write_csv(path: Path, results: list[RunResult]) -> None:
    stage_keys: set[str] = set()
    for r in results:
        stage_keys.update(r.stages.keys())
    fieldnames = [
        "filename", "ok", "status", "http_status", "ocr_provider",
        "total_ms", "submit_ms", "poll_ms", "error",
        *sorted(stage_keys),
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "filename": r.filename,
                "ok": r.ok,
                "status": r.status,
                "http_status": r.http_status or "",
                "ocr_provider": r.ocr_provider,
                "total_ms": round(r.total_ms, 2),
                "submit_ms": round(r.submit_ms, 2),
                "poll_ms": round(r.poll_ms, 2),
                "error": r.error or "",
            }
            for stage in stage_keys:
                row[stage] = round(r.stages.get(stage, 0.0), 2)
            writer.writerow(row)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="Directory with PDFs")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--project", default="agribank")
    parser.add_argument("--type", dest="extraction_type", default="document")
    parser.add_argument("--include-layout", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1, help="Loop dataset N times")
    parser.add_argument("--poll-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"--dir not found: {args.dir}", file=sys.stderr)
        return 2
    pdfs = sorted(p for p in args.dir.rglob("*.pdf") if p.is_file())
    if not pdfs:
        print(f"No PDFs in {args.dir}", file=sys.stderr)
        return 2

    targets: list[Path] = []
    for _ in range(max(1, args.iterations)):
        targets.extend(pdfs)
    print(f"Submitting {len(targets)} runs ({len(pdfs)} unique x {args.iterations} iter) at concurrency={args.concurrency}")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    timeout = httpx.Timeout(args.poll_timeout + 30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            run_one(
                client, semaphore, args.base_url, pdf,
                args.project, args.extraction_type, args.include_layout,
                args.poll_timeout, args.poll_interval,
            )
            for pdf in targets
        ]
        results = await asyncio.gather(*tasks)

    _print_summary(results)
    if args.csv:
        _write_csv(args.csv, results)
        print(f"\nCSV written to {args.csv}")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
