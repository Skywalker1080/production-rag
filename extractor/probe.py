"""Single-process hi_res probe on 2 pages: surface the REAL traceback."""
import os
import sys
import traceback

os.environ.setdefault(
    "OCR_AGENT",
    "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle",
)
from pypdf import PdfReader, PdfWriter
from unstructured.partition.pdf import partition_pdf

SRC = "/tmp/jiofin financial report.pdf"
if not os.path.exists(SRC):
    import boto3

    boto3.client("s3", region_name="us-east-1").download_file(
        "rag-extract-405633560616-use1",
        "in/jiofin financial report.pdf", SRC)
    print("downloaded", flush=True)

r = PdfReader(SRC)
w = PdfWriter()
for p in range(2):
    w.add_page(r.pages[p])
with open("/tmp/probe.pdf", "wb") as f:
    w.write(f)
print("probe pdf ready", flush=True)

try:
    els = partition_pdf(filename="/tmp/probe.pdf", strategy="hi_res",
                        infer_table_structure=True, languages=["en"])
    print(f"PROBE OK elements={len(els)}", flush=True)
    for e in els[:5]:
        print(f"  {type(e).__name__} pg={e.metadata.page_number} "
              f"{(e.text or '')[:80]!r}", flush=True)
except BaseException:
    traceback.print_exc()
    print("PROBE FAILED", flush=True)
