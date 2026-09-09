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
    """Provenance subset the Text Extractor emits (plan 0001 section 3.1)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    file_name: str
    file_path: str
    sha256: str
    byte_size: int
    detected_mime: str
    encoding: str


class Document(BaseModel):
    """Immutable normalized source record."""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: DocumentMetadata


def make_metadata(
    path: Path, raw: bytes, mime: str, encoding: str
) -> DocumentMetadata:
    """Assemble the provenance subset identically for every extractor."""
    return DocumentMetadata(
        doc_id=uuid.uuid4().hex,
        file_name=path.name,
        file_path=str(path.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_size=len(raw),
        detected_mime=mime,
        encoding=encoding,
    )
