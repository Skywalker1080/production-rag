"""Markdown normalizer: frontmatter split, allowlisted, body kept.

Frontmatter is author config — only allowlisted citation/search fields
cross into metadata; everything else is dropped, never into content,
never into metadata. Missing or malformed blocks degrade to
whole-file-as-body (frontmatter is optional by format design).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import frontmatter

from production_rag.common.logging import stage
from production_rag.indexing.base import decode
from production_rag.indexing.document import Document, make_metadata
from production_rag.indexing.errors import UnsupportedFormatError

#: Frontmatter key → DocumentMetadata field. Nothing else crosses.
FRONTMATTER_ALLOWLIST = {
    "title": "title",
    "tags": "tags",
    "language": "language",
    "date": "doc_date",
}


class MdExtractor:
    """Extractor for the `markdown` kind (md uploads)."""

    kinds: ClassVar[tuple[str, ...]] = ("markdown",)

    def extract(
        self, path: Path, raw: bytes, mime: str, encoding: str | None = None
    ) -> Document:
        with stage("md_normalize", component="indexing"):
            if not mime.startswith("text/"):
                raise UnsupportedFormatError(path.name, mime)
            content, detected = decode(raw, path.name, encoding)
            try:
                post = frontmatter.loads(content)
            except Exception:
                post = None
            if post is None:
                body, fields = content, {}
            else:
                body, fields = post.content, self._allowlisted(
                    post.metadata or {}
                )
            if not body.strip():
                raise UnsupportedFormatError(
                    path.name, f"{mime} (no extractable text)"
                )
            return Document(
                content=body,
                metadata=make_metadata(path, raw, mime, detected, **fields),
            )

    @staticmethod
    def _allowlisted(meta: dict[str, Any]) -> dict[str, Any]:
        """Keep allowlisted keys, coerced; drop everything else silently."""
        fields: dict[str, Any] = {}
        for key, field in FRONTMATTER_ALLOWLIST.items():
            if key not in meta:
                continue
            value = meta[key]
            if field == "tags":
                if isinstance(value, list):
                    fields[field] = [str(item) for item in value]
                else:
                    fields[field] = [str(value)]
            elif isinstance(value, (str, int, float)):
                fields[field] = str(value)
            # Other shapes (dicts, nested) are dropped: no blanks, no junk.
        return fields
