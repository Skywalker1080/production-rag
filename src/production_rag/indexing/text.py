"""Plain-text normalizer: bytes in, cleaned `Document` out."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import ClassVar

from production_rag.common.logging import stage
from production_rag.indexing.base import decode
from production_rag.indexing.document import Document, DocumentMetadata
from production_rag.indexing.errors import UnsupportedFormatError


class TextExtractor:
    """Extractor for the `text` kind (txt uploads)."""

    kinds: ClassVar[tuple[str, ...]] = ("text",)

    def extract(
        self, path: Path, raw: bytes, mime: str, encoding: str | None = None
    ) -> Document:
        with stage("txt_normalize", component="indexing"):
            if not mime.startswith("text/"):
                raise UnsupportedFormatError(path.name, mime)
            content, detected = decode(raw, path.name, encoding)
            return Document(
                content=content,
                metadata=DocumentMetadata(
                    doc_id=uuid.uuid4().hex,
                    file_name=path.name,
                    file_path=str(path.resolve()),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    byte_size=len(raw),
                    detected_mime=mime,
                    encoding=detected,
                ),
            )
