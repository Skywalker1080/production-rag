"""Source-level record every extractor returns.

`Document` is frozen (plan 0001 section 3.3): nothing mutates it after
extraction. Chunk-born facts live on the future `Chunk` type, linked by
`doc_id`. Home is indexing until vector-DB ingest (CONTEXT.md).
"""

from __future__ import annotations

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
