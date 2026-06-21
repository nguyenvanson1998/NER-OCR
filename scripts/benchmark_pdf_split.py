"""Benchmark PDF-into-N-page-chunks split across pypdf / pikepdf / PyMuPDF.

Usage:
    python scripts/benchmark_pdf_split.py --dir data/benchmark --max-pages 5

Reports per-file wall time per library and total across the dataset. Splitters
all produce in-memory bytes for each chunk so they can be fed straight to the
Vision/Document AI clients without touching disk.
"""

from __future__ import annotations

import argparse
import io
import statistics
import time
from pathlib import Path


def split_with_pypdf(content: bytes, max_pages: int) -> list[bytes]:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(content))
    parts: list[bytes] = []
    for start in range(0, len(reader.pages), max_pages):
        writer = PdfWriter()
        for i in range(start, min(start + max_pages, len(reader.pages))):
            writer.add_page(reader.pages[i])
        out = io.BytesIO()
        writer.write(out)
        parts.append(out.getvalue())
    return parts


def split_with_pikepdf(content: bytes, max_pages: int) -> list[bytes]:
    import pikepdf

    parts: list[bytes] = []
    with pikepdf.open(io.BytesIO(content)) as src:
        total = len(src.pages)
        for start in range(0, total, max_pages):
            with pikepdf.new() as dst:
                for i in range(start, min(start + max_pages, total)):
                    dst.pages.append(src.pages[i])
                out = io.BytesIO()
                dst.save(out)
                parts.append(out.getvalue())
    return parts


def split_with_pymupdf(content: bytes, max_pages: int) -> list[bytes]:
    import fitz  # PyMuPDF

    parts: list[bytes] = []
    src = fitz.open(stream=content, filetype="pdf")
    try:
        total = src.page_count
        for start in range(0, total, max_pages):
            dst = fitz.open()
            try:
                dst.insert_pdf(src, from_page=start, to_page=min(start + max_pages, total) - 1)
                parts.append(dst.tobytes())
            finally:
                dst.close()
    finally:
        src.close()
    return parts


SPLITTERS = {
    "pypdf": split_with_pypdf,
    "pikepdf": split_with_pikepdf,
    "pymupdf": split_with_pymupdf,
}


def run(path: Path, max_pages: int, repeats: int) -> dict[str, dict]:
    raw = path.read_bytes()
    results: dict[str, dict] = {}
    for name, fn in SPLITTERS.items():
        durations: list[float] = []
        parts_count = 0
        total_out_bytes = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            parts = fn(raw, max_pages)
            durations.append((time.perf_counter() - t0) * 1000)
            parts_count = len(parts)
            total_out_bytes = sum(len(p) for p in parts)
        results[name] = {
            "p50_ms": round(statistics.median(durations), 2),
            "min_ms": round(min(durations), 2),
            "max_ms": round(max(durations), 2),
            "parts": parts_count,
            "out_kb": round(total_out_bytes / 1024, 1),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    pdfs = sorted(p for p in args.dir.rglob("*.pdf") if p.is_file())
    if not pdfs:
        print("No PDFs found")
        return

    header = f"{'file':<60}  {'pypdf':>10}  {'pikepdf':>10}  {'pymupdf':>10}  {'parts':>6}  {'size KB':>8}"
    print(header)
    print("-" * len(header))
    totals = dict.fromkeys(SPLITTERS, 0.0)
    for pdf in pdfs:
        try:
            res = run(pdf, args.max_pages, args.repeats)
        except Exception as exc:
            print(f"{pdf.name[:60]:<60}  ERROR: {type(exc).__name__}: {exc}")
            continue
        name = pdf.name[:60]
        size_kb = round(pdf.stat().st_size / 1024, 1)
        parts = res["pypdf"]["parts"]
        print(
            f"{name:<60}  {res['pypdf']['p50_ms']:>10.2f}  {res['pikepdf']['p50_ms']:>10.2f}  "
            f"{res['pymupdf']['p50_ms']:>10.2f}  {parts:>6d}  {size_kb:>8.1f}"
        )
        for k, v in res.items():
            totals[k] += v["p50_ms"]

    print("-" * len(header))
    print(
        f"{'TOTAL (sum of medians, ms)':<60}  "
        f"{totals['pypdf']:>10.2f}  {totals['pikepdf']:>10.2f}  {totals['pymupdf']:>10.2f}"
    )


if __name__ == "__main__":
    main()
