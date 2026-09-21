"""Sharded parallel partition: split PDF into page ranges (+1 page overlap),
fan out one process per shard, offset page numbers, drop overlap dupes, merge.

Usage on EC2:
    python3.11 shard_compare.py <s3uri> <strategy> <workers>
Output:
    s3://.../out/<stem>/shard_<strategy>_elements.json + shard_<strategy>_metrics.json
"""
import json
import math
import os
import statistics
import sys
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import boto3

BUCKET = "rag-extract-405633560616-use1"
TMP = "/tmp/shards"
PROGRESS_LOG = "/tmp/shard_progress.log"
s3 = boto3.client("s3", region_name="us-east-1")


def download(uri: str, dest: str):
    parts = uri[5:].split("/", 1)
    s3.download_file(parts[0], urllib.parse.unquote(parts[1]), dest)


def upload(local: str, key: str):
    s3.upload_file(local, BUCKET, key)


def make_shards(pdf: str, workers: int, first: int = 1, last: int = 0,
                per_shard: int = 0):
    """Split pages [first, last] into ranges with 1-page overlap.

    per_shard>0 forces small shards (granular progress) instead of
    one-shard-per-worker. Returns [(start, path)].
    """
    from pypdf import PdfReader, PdfWriter

    os.makedirs(TMP, exist_ok=True)
    reader = PdfReader(pdf)
    total = len(reader.pages)
    last = last or total
    span = last - first + 1
    per = per_shard or math.ceil(span / workers)
    shards = []
    i, start = 0, first
    while start <= last:
        end = min(start + per - 1, last)
        # overlap: include one extra page (fully covered by next shard)
        file_end = min(end + 1, last)
        w = PdfWriter()
        for p in range(start - 1, file_end):
            w.add_page(reader.pages[p])
        path = f"{TMP}/shard_{i:02d}_p{start}-{file_end}.pdf"
        with open(path, "wb") as f:
            w.write(f)
        shards.append({"idx": i, "start": start, "end": end,
                       "path": path, "last": end == last})
        i += 1
        start = end + 1
    return shards, total


def work(shard: dict, strategy: str):
    """Partition one shard. Returns element dicts with LOCAL page numbers."""
    from unstructured.partition.pdf import partition_pdf

    t0 = time.time()
    els = partition_pdf(
        filename=shard["path"], strategy=strategy,
        infer_table_structure=True, languages=["en"],
    )
    out = []
    for e in els:
        out.append({
            "type": type(e).__name__,
            "page": getattr(e.metadata, "page_number", None),
            "text": getattr(e, "text", "") or "",
            "html": getattr(e.metadata, "text_as_html", None),
        })
    return shard, out, round(time.time() - t0, 1)


PADDLE_AGENT = (
    "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"
)


def warm_model_cache():
    """Download layout + table models ONCE serially — workers must hit disk
    cache, never the network (8x concurrent unauthenticated HF downloads
    get rate-limited and kill every worker)."""
    from huggingface_hub import snapshot_download
    from unstructured_inference.models.base import get_model

    print("warming yolox layout model cache...", flush=True)
    get_model("yolox")
    print("warming table-transformer cache...", flush=True)
    snapshot_download("microsoft/table-transformer-structure-recognition")
    print("model cache warm", flush=True)


def warm_ocr_cache():
    """Download PaddleOCR models ONCE serially so workers don't race on it."""
    import PIL.Image

    os.environ["OCR_AGENT"] = PADDLE_AGENT
    from unstructured.partition.utils.ocr_models.paddle_ocr import (
        OCRAgentPaddle,
    )

    agent = OCRAgentPaddle("en")
    agent.get_text_from_image(PIL.Image.new("RGB", (100, 100), "white"))
    print("paddle OCR cache warm", flush=True)


def merge(shards, results):
    """Offset pages to global numbering; drop overlap-page dupes exactly."""
    merged = []
    boundary_dropped = 0
    for shard, els, _ in sorted(results, key=lambda r: r[0]["idx"]):
        for e in els:
            if e["page"] is None:
                continue
            glob = shard["start"] + e["page"] - 1
            # overlap page == next shard's start page -> next shard owns it
            if not shard["last"] and glob == shard["end"] + 1:
                boundary_dropped += 1
                continue
            e["page"] = glob
            merged.append(e)
    merged.sort(key=lambda e: e["page"])
    return merged, boundary_dropped


def metrics(strategy: str, wall: float, worker_secs: list, els: list,
            first: int, last: int, dropped: int):
    lens = [len(e["text"]) for e in els]
    types = Counter(e["type"] for e in els)
    pages = sorted({e["page"] for e in els})
    tables = [e for e in els if e["type"] in ("Table", "TableChunk")]
    with_cells = sum(1 for t in tables if t["html"] and "<td" in t["html"])
    cells = [t["html"].count("<td") + t["html"].count("<th") for t in tables if t["html"]]
    return {
        "strategy": strategy,
        "wall_seconds": round(wall, 1),
        "sum_worker_seconds": round(sum(worker_secs), 1),
        "elements": len(els),
        "types": dict(types),
        "median_len": statistics.median(lens) if lens else 0,
        "crumbs_lt100": sum(1 for l in lens if l < 100),
        "pages_covered": len(pages),
        "pages_range": [first, last],
        "pages_missing": sorted(set(range(first, last + 1)) - set(pages)),
        "titles": types.get("Title", 0),
        "footnotes": types.get("Footnote", 0) + types.get("Footnotes", 0),
        "tables": len(tables),
        "tables_with_cells": with_cells,
        "avg_cells_per_table": round(sum(cells) / len(cells), 1) if cells else 0,
        "boundary_dupes_dropped": dropped,
    }


def main():
    src, strategy, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    # optional: pages "START-END" (e.g. "1-20"), per-shard size for progress
    pages = sys.argv[4] if len(sys.argv) > 4 else ""
    per_shard = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    first, last = (int(x) for x in pages.split("-")) if pages else (1, 0)
    fname = src.rsplit("/", 1)[-1]
    stem = fname.rsplit(".", 1)[0]
    local = f"/tmp/{fname}"
    print(f"downloading {src} ...", flush=True)
    download(src, local)

    t0 = time.time()
    shards, total = make_shards(local, workers, first, last, per_shard)
    last_page = last or total
    print(f"shards: {len(shards)} (pages {shards[0]['start']}-{shards[-1]['end']} "
          f"of {total}, 1-page overlap)", flush=True)
    if strategy == "hi_res":
        warm_model_cache()
        if "paddle" in os.getenv("OCR_AGENT", "").lower():
            warm_ocr_cache()
        # Caches are fully warm above — go offline so workers never touch
        # HF Hub (huggingface_hub revalidates even cache hits with HEAD
        # requests; unauthenticated bursts get rate-limited). Inherited by
        # spawned children.
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("HF_HUB_OFFLINE=1 (workers use disk cache only)", flush=True)

    # spawn (not fork): parent has paddle/torch thread pools loaded from the
    # cache-warm step, and forking a multithreaded process kills children.
    # Spawned children boot clean and load models from warm disk cache.
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(work, s, strategy): s for s in shards}
        results, errors, pages_done = [], [], 0
        # fresh progress log for live polling (SSM buffers stdout till end)
        open(PROGRESS_LOG, "w").write(
            f"started {len(shards)} shards strategy={strategy}\n")
        for f in as_completed(futs):
            s = futs[f]
            try:
                shard, els, secs = f.result()
                rate = (s["end"] - s["start"] + 1) / max(secs, 0.1)
                pages_done += s["end"] - s["start"] + 1
                line = (f"  shard {s['idx']}: pages {s['start']}-{s['end']} done "
                        f"in {secs}s ({len(els)} els, {rate:.1f} pg/s) "
                        f"[pages {pages_done}/{last_page} | "
                        f"{time.time()-t0:.0f}s elapsed]")
                print(line, flush=True)
                open(PROGRESS_LOG, "a").write(line + "\n")
                results.append((shard, els, secs))
            except Exception as e:
                errors.append({"shard": s["idx"], "pages": [s["start"], s["end"]],
                               "error": f"{type(e).__name__}: {e}"})
        if errors:
            print("SHARD ERRORS: " + json.dumps(errors), flush=True)
            if not results:
                raise RuntimeError(f"all shards failed: {errors}")
    worker_secs = [r[2] for r in results]

    merged, dropped = merge(shards, results)
    wall = time.time() - t0
    m = metrics(strategy, wall, worker_secs, merged, first, last_page, dropped)
    tag = f"p{first}-{last_page}" if pages else "full"
    print(json.dumps(m, indent=1), flush=True)

    with open(f"/tmp/{stem}_shard_{strategy}_{tag}_elements.json", "w") as f:
        json.dump(merged, f)
    upload(f"/tmp/{stem}_shard_{strategy}_{tag}_elements.json",
           f"out/{stem}/shard_{strategy}_{tag}_elements.json")
    with open(f"/tmp/{stem}_shard_{strategy}_{tag}_metrics.json", "w") as f:
        json.dump(m, f, indent=1)
    upload(f"/tmp/{stem}_shard_{strategy}_{tag}_metrics.json",
           f"out/{stem}/shard_{strategy}_{tag}_metrics.json")
    print(f"results: s3://{BUCKET}/out/{stem}/", flush=True)


if __name__ == "__main__":
    main()
