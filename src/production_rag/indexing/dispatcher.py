"""Dispatcher: read once, identify, guard, route, extract.

Owns everything format-agnostic (I/O, empty/size guards, content sniff,
registry lookup). Extractors own only their format. Content sniff always
wins over the file extension; disagreements are logged, never silent.
"""

from __future__ import annotations

import re
from pathlib import Path

import magic
import requests

from production_rag.common.logging import get_logger, stage
from production_rag.indexing.base import Extractor
from production_rag.indexing.document import Document
from production_rag.indexing.errors import (
    EmptyFileError,
    FetchError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from production_rag.indexing.html import HtmlExtractor
from production_rag.indexing.markdown import MdExtractor
from production_rag.indexing.pdf import PdfExtractor
from production_rag.indexing.text import TextExtractor

MAX_BYTES = 100 * 1024 * 1024  # draft cap; config later (plan 0001)
FETCH_TIMEOUT_S = 10
_STREAM_CHUNK = 65536

logger = get_logger(__name__)

REGISTRY: dict[str, Extractor] = {}


def _register(extractor: Extractor) -> None:
    for kind in extractor.kinds:
        REGISTRY[kind] = extractor


_register(TextExtractor())
_register(HtmlExtractor())
_register(PdfExtractor())
_register(MdExtractor())

_EXTENSION_FAMILY = {
    ".txt": "text",
    ".md": "markdown",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


_HTML_DOCUMENT_MARKER = re.compile(rb"(?i)<\s*html[\s>]|<!doctype\s+html")
_SNIFF_WINDOW = 8192


def _looks_like_html_document(raw: bytes) -> bool:
    """Document-level markup present in the head window?

    Boolean presence test only (no parsing, no nesting risk). latin-1
    decodes any bytes; markers are pure ASCII so the decode is exact.
    """
    return _HTML_DOCUMENT_MARKER.search(raw[:_SNIFF_WINDOW]) is not None


def identify(raw: bytes, filename: str = "") -> tuple[str, str]:
    """Sniff `raw` content → (kind, mime).

    Content leads; the extension only breaks ties a sniff cannot win:
    libmagic misreads code-heavy HTML as e.g. text/x-python, so a generic
    text/* sniff on a .html/.htm file routes html. A document-level
    markup marker overrides any generic sniff (even under a .txt name):
    TextExtractor leaking tags over markup hurts more than HtmlExtractor
    normalizing whitespace on plain text. Residual risk — prose quoting
    a full HTML document — is accepted and logged.
    """
    mime = magic.from_buffer(raw, mime=True)
    if mime == "application/pdf":
        return "pdf", mime
    if mime == "text/html":
        return "html", mime
    if mime == "text/markdown":
        return "markdown", mime
    if mime.startswith("text/"):
        if _looks_like_html_document(raw):
            return "html", mime
        suffix = Path(filename).suffix.lower()
        if suffix in (".html", ".htm"):
            return "html", mime
        if suffix == ".md":
            return "markdown", mime
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
    return _finalize(target.name, target.suffix, target.read_bytes(), encoding)


def ingest_url(url: str, encoding: str | None = None) -> Document:
    """Public entry: http(s) URL → cleaned `Document`.

    Fetches with timeout and streaming cap, then rejoins the shared
    guard-route-extract tail: the sniff (not Content-Type) decides the
    extractor. Raises `FetchError` on network failure, never partial.
    """
    from urllib.parse import urlparse

    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError(url, f"disallowed scheme: {parts.scheme or 'none'}")
    hint = Path(parts.path.rsplit("/", 1)[-1]).name
    display = hint or parts.netloc
    with stage("fetch", component="indexing"):
        try:
            response = requests.get(url, stream=True, timeout=FETCH_TIMEOUT_S)
            response.raise_for_status()
            buf = bytearray()
            for chunk in response.iter_content(_STREAM_CHUNK):
                buf.extend(chunk)
                if len(buf) > MAX_BYTES:
                    raise FileTooLargeError(display, len(buf), MAX_BYTES)
            raw = bytes(buf)
        except FileTooLargeError:
            raise
        except requests.RequestException as exc:
            raise FetchError(url, type(exc).__name__) from exc
    return _finalize(display, Path(hint).suffix, raw, encoding, file_path=url)


def _finalize(
    display_name: str,
    suffix: str,
    raw: bytes,
    encoding: str | None,
    file_path: str | None = None,
) -> Document:
    """Shared guard-route-extract tail for file and URL ingest."""
    with stage("identifier", component="indexing"):
        if len(raw) == 0:
            logger.warning("reject: empty file", extra={"file_name": display_name})
            raise EmptyFileError(display_name)
        if len(raw) > MAX_BYTES:
            logger.warning(
                "reject: file too large",
                extra={"file_name": display_name, "byte_size": len(raw)},
            )
            raise FileTooLargeError(display_name, len(raw), MAX_BYTES)
        kind, mime = identify(raw, display_name)
        hinted = _EXTENSION_FAMILY.get(suffix.lower())
        if hinted is not None and hinted != kind and kind != "unknown":
            logger.warning(
                "extension disagrees with content; trusting content",
                extra={
                    "file_name": display_name,
                    "extension": suffix,
                    "detected_mime": mime,
                },
            )
        extractor = route(kind, display_name, mime)
    return extractor.extract(
        Path(display_name), raw, mime, encoding, file_path=file_path
    )
