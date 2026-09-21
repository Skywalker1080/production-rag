"""Pre-download + cache all local models so uploads never wait on the network.

Run once after install (and after any unstructured version bump):
    uv run python -m app.warmup

What it warms:
  1. YOLOX layout model (unstructuredio/yolo_x_layout) — hi_res multi-column parsing
  2. Table Transformer (microsoft/...) — infer_table_structure=True
  3. End-to-end hi_res partition on a throwaway 1-page PDF (proves the path works)

Notes:
  - HuggingFace Hub caches to ~/.cache/huggingface permanently; downloads are
    one-time. This script just forces them upfront.
  - Bedrock models are API-side (no download); connectivity is checked instead.
  - Optional: set HF_TOKEN in .env to avoid Hub rate limits (auto-picked-up).
"""
import os
import tempfile

from app import config  # noqa: F401  (loads .env)
from app.logging_setup import setup_logging, step_logger

setup_logging()
log = step_logger("warmup")

TABLE_MODEL = os.getenv(
    "TABLE_MODEL_ID", "microsoft/table-transformer-structure-recognition"
)


def warm_layout_model() -> None:
    from unstructured_inference.models.base import get_model

    log.info("warming layout model (unstructuredio/yolo_x_layout)…")
    get_model("yolox")
    log.info("layout model ready (disk-cached from now on)")


def warm_table_model() -> None:
    from transformers import (
        DetrImageProcessor,
        TableTransformerForObjectDetection,
    )

    log.info(f"warming table model ({TABLE_MODEL})…")
    DetrImageProcessor.from_pretrained(TABLE_MODEL)
    TableTransformerForObjectDetection.from_pretrained(TABLE_MODEL)
    log.info("table model ready (disk-cached from now on)")


def warm_partition_path() -> None:
    from pypdf import PdfWriter
    from unstructured.partition.pdf import partition_pdf

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        w = PdfWriter()
        w.add_blank_page(612, 792)
        w.write(f)
        tmp = f.name
    try:
        log.info("warming end-to-end hi_res partition on throwaway PDF…")
        els = partition_pdf(
            filename=tmp, strategy="hi_res", infer_table_structure=True
        )
        log.info(f"partition path OK (elements={len(els)})")
    finally:
        os.unlink(tmp)


def check_bedrock() -> None:
    from app import rag

    log.info("checking Bedrock clients construct (no $ spent)…")
    rag.get_embeddings()
    rag.get_llm()
    log.info(
        f"Bedrock OK (llm={config.BEDROCK_MODEL_ID}, "
        f"embed={config.BEDROCK_EMBED_MODEL_ID}, region={config.AWS_REGION})"
    )


def main() -> None:
    log.info("warmup start")
    warm_layout_model()
    warm_table_model()
    warm_partition_path()
    check_bedrock()
    log.info("warmup done — uploads will no longer wait on model downloads")


if __name__ == "__main__":
    main()
