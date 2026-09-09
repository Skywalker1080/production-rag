"""PDF normalizer: pages in, cleaned `Document` out.

Accumulates EVERY page in order (the MVP read only the last). Pages
with no text are skipped; a document with no text at all is scanned —
loud failure, never a silent empty Document. Encrypted files fail
chained; no password attempts in Phase-1.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import ClassVar

from pypdf import PdfReader

from production_rag.common.logging import stage
from production_rag.indexing.document import Document, make_metadata
from production_rag.indexing.errors import UnsupportedFormatError


class PdfExtractor:
    """Extractor for the `pdf` kind (pdf uploads)."""

    kinds: ClassVar[tuple[str, ...]] = ("pdf",)

    def extract(
        self,
        path: Path,
        raw: bytes,
        mime: str,
        encoding: str | None = None,
        file_path: str | None = None,
    ) -> Document:
        del encoding  # PDFs carry their own encoding; override unused.
        with stage("pdf_normalize", component="indexing"):
            if mime != "application/pdf":
                raise UnsupportedFormatError(path.name, mime)
            try:
                reader = PdfReader(BytesIO(raw))
                # Broad catch is deliberate: pypdf fails many ways
                # (encrypted, corrupt, truncated) and every one must be
                # loud, chained, and typed as UnsupportedFormatError.
                texts = [(page.extract_text() or "") for page in reader.pages]
            except Exception as exc:
                raise UnsupportedFormatError(path.name, mime) from exc
            content = "\n\n".join(text for text in texts if text.strip())
            if not content:
                raise UnsupportedFormatError(
                    path.name, f"{mime} (no extractable text)"
                )
            return Document(
                content=content,
                metadata=make_metadata(
                    path, raw, mime, "pdf-text", file_path=file_path
                ),
            )
