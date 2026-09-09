"""Source-level record every extractor returns.

`Document` is frozen (plan 0001 section 3.3): nothing mutates it after
extraction. Chunk-born facts live on the future `Chunk` type, linked by
`doc_id`. Home is indexing until vector-DB ingest (CONTEXT.md).
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class DocumentMetadata(BaseModel):
    """Provenance subset every extractor emits, plus optional semantic
    fields (plan 0004 section 3.2) that only frontmatter-bearing formats
    set. Strict model stays `forbid`: new fields are added here, never
    smuggled through.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    file_name: str
    file_path: str
    sha256: str
    byte_size: int
    detected_mime: str
    encoding: str
    title: str | None = None
    tags: list[str] | None = None
    language: str | None = None
    doc_date: str | None = None


class Document(BaseModel):
    """Immutable normalized source record."""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: DocumentMetadata


def make_metadata(
    path: Path,
    raw: bytes,
    mime: str,
    encoding: str,
    title: str | None = None,
    tags: list[str] | None = None,
    language: str | None = None,
    doc_date: str | None = None,
    file_path: str | None = None,
) -> DocumentMetadata:
    """Assemble metadata identically for every extractor; semantic
    fields stay `None` unless the caller sets them. `file_path`
    overrides the resolved local path (URL sources record the URL)."""
    return DocumentMetadata(
        doc_id=uuid.uuid4().hex,
        file_name=path.name,
        file_path=file_path if file_path is not None else str(path.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_size=len(raw),
        detected_mime=mime,
        encoding=encoding,
        title=title,
        tags=tags,
        language=language,
        doc_date=doc_date,
    )
