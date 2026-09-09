"""Dispatcher: read once, identify, guard, route, extract.

Owns everything format-agnostic (I/O, empty/size guards, content sniff,
registry lookup). Extractors own only their format. Content sniff always
wins over the file extension; disagreements are logged, never silent.
"""

from __future__ import annotations

from pathlib import Path

import magic

from production_rag.common.logging import get_logger, stage
from production_rag.indexing.base import Extractor
from production_rag.indexing.document import Document
from production_rag.indexing.errors import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from production_rag.indexing.text import TextExtractor

MAX_BYTES = 100 * 1024 * 1024  # draft cap; config later (plan 0001)

logger = get_logger(__name__)

REGISTRY: dict[str, Extractor] = {}


def _register(extractor: Extractor) -> None:
    for kind in extractor.kinds:
        REGISTRY[kind] = extractor


_register(TextExtractor())

_EXTENSION_FAMILY = {
    ".txt": "text",
    ".md": "markdown",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


def identify(raw: bytes) -> tuple[str, str]:
    """Sniff `raw` content → (kind, mime). Never trusts the extension."""
    mime = magic.from_buffer(raw, mime=True)
    if mime == "application/pdf":
        return "pdf", mime
    if mime.startswith("text/"):
        return "text", mime
    return "unknown", mime


def route(kind: str, file_name: str, mime: str) -> Extractor:
    """Registry lookup; unknown kinds fail loudly (extension point)."""
    extractor = REGISTRY.get(kind)
    if extractor is None:
        raise UnsupportedFormatError(file_name, mime)
    return extractor


def ingest(path: str | Path, encoding: str | None = None) -> Document:
    """Public entry: file path → cleaned `Document`. Raises, never partial."""
    target = Path(path)
    raw = target.read_bytes()
    with stage("identifier", component="indexing"):
        if len(raw) == 0:
            logger.warning("reject: empty file", extra={"file_name": target.name})
            raise EmptyFileError(target.name)
        if len(raw) > MAX_BYTES:
            logger.warning(
                "reject: file too large",
                extra={"file_name": target.name, "byte_size": len(raw)},
            )
            raise FileTooLargeError(target.name, len(raw), MAX_BYTES)
        kind, mime = identify(raw)
        hinted = _EXTENSION_FAMILY.get(target.suffix.lower())
        if hinted is not None and hinted != kind and kind != "unknown":
            logger.warning(
                "extension disagrees with content; trusting content",
                extra={
                    "file_name": target.name,
                    "extension": target.suffix,
                    "detected_mime": mime,
                },
            )
        extractor = route(kind, target.name, mime)
    return extractor.extract(target, raw, mime, encoding)
