"""Run fast vs hi_res partition on one PDF, save metrics + elements JSON.

Usage on EC2:
    python3.11 compare.py s3://rag-extract-405633560616-use1/in/<file>.pdf
Output:
    s3://.../out/<file>/fast_elements.json, hires_elements.json, metrics.json
    + metric table printed to stdout.
"""
import json
import statistics
import sys
import time
import urllib.parse
from collections import Counter

import boto3

BUCKET = "rag-extract-405633560616-use1"
s3 = boto3.client("s3", region_name="us-east-1")


def parse_s3(uri: str):
    assert uri.startswith("s3://"), uri
    parts = uri[5:].split("/", 1)
    return parts[0], urllib.parse.unquote(parts[1])


def download(uri: str, dest: str):
    b, k = parse_s3(uri)
    s3.download_file(b, k, dest)


def upload(local: str, key: str):
    s3.upload_file(local, BUCKET, key)


def run_strategy(path: str, strategy: str):
    from unstructured.partition.pdf import partition_pdf

    t0 = time.time()
    els = partition_pdf(
        filename=path, strategy=strategy, infer_table_structure=True
    )
    dt = time.time() - t0
    lens = [len(getattr(e, "text", "") or "") for e in els]
    types = Counter(type(e).__name__ for e in els)
    pages = sorted(
        {getattr(e.metadata, "page_number", None) for e in els}
        - {None}
    )
    tables = sum(
        1 for e in els if type(e).__name__ in ("Table", "TableChunk")
    )
    return {
        "strategy": strategy,
        "wall_seconds": round(dt, 1),
        "elements": len(els),
        "types": dict(types),
        "min_len": min(lens) if lens else 0,
        "median_len": statistics.median(lens) if lens else 0,
        "crumbs_lt100": sum(1 for l in lens if l < 100),
        "pages": len(pages),
        "tables": tables,
    }, [
        {
            "type": type(e).__name__,
            "page": getattr(e.metadata, "page_number", None),
            "text": (getattr(e, "text", "") or "")[:500],
        }
        for e in els
    ]


def main():
    src = sys.argv[1]
    only = sys.argv[2:] or ("fast", "hi_res")
    fname = src.rsplit("/", 1)[-1]
    stem = fname.rsplit(".", 1)[0]
    local = f"/tmp/{fname}"
    print(f"downloading {src} ...", flush=True)
    download(src, local)

    out = {}
    for strategy in ("fast", "hi_res"):
        if strategy not in only:
            continue
        print(f"--- strategy={strategy} ---", flush=True)
        metrics, elements = run_strategy(local, strategy)
        out[strategy] = metrics
        with open(f"/tmp/{stem}_{strategy}_elements.json", "w") as f:
            json.dump(elements, f)
        upload(
            f"/tmp/{stem}_{strategy}_elements.json",
            f"out/{stem}/{strategy}_elements.json",
        )
        print(json.dumps(metrics, indent=1), flush=True)

    with open(f"/tmp/{stem}_metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    upload(f"/tmp/{stem}_metrics.json", f"out/{stem}/metrics.json")

    if set(only) == {"fast", "hi_res"}:
        f, h = out["fast"], out["hi_res"]
        print("\n=== DELTA (hi_res vs fast) ===", flush=True)
        print(f"wall time:      {f['wall_seconds']}s -> {h['wall_seconds']}s", flush=True)
        print(f"elements:       {f['elements']} -> {h['elements']}", flush=True)
        print(f"crumbs (<100):  {f['crumbs_lt100']} -> {h['crumbs_lt100']}", flush=True)
        print(f"median len:     {f['median_len']} -> {h['median_len']}", flush=True)
        print(f"tables found:   {f['tables']} -> {h['tables']}", flush=True)
    print(f"results: s3://{BUCKET}/out/{stem}/", flush=True)


if __name__ == "__main__":
    main()
