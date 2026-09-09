"""HTML normalizer: markup in, cleaned `Document` out.

bs4 + html5lib repairs broken trees (never regex); `script`/`style`/
`noscript` are dropped; Unstructured turns the repaired tree into the
shared markdown shape. Unstructured imports lazily so its heavy
transitives never touch dispatcher or txt paths. Table fidelity is
best-effort (issue #8 owns it).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from bs4 import BeautifulSoup

from production_rag.common.logging import stage
from production_rag.indexing.base import decode
from production_rag.indexing.document import Document, make_metadata
from production_rag.indexing.errors import UnsupportedFormatError

_DROP_TAGS = ("script", "style", "noscript")


class HtmlExtractor:
    """Extractor for the `html` kind (html uploads)."""

    kinds: ClassVar[tuple[str, ...]] = ("html",)

    def extract(
        self, path: Path, raw: bytes, mime: str, encoding: str | None = None
    ) -> Document:
        with stage("html_normalize", component="indexing"):
            # Any text/* is safe: tag-free input passes through unchanged,
            # markup gets repaired and stripped. Non-text never arrives —
            # the dispatcher routes it elsewhere. `mime` is recorded
            # verbatim in metadata for traceability.
            if not mime.startswith("text/"):
                raise UnsupportedFormatError(path.name, mime)
            text, detected = decode(raw, path.name, encoding)
            soup = BeautifulSoup(text, "html5lib")
            for tag in soup(_DROP_TAGS):
                tag.decompose()
            content = self._to_markdown(str(soup))
            if not content.strip():
                raise UnsupportedFormatError(
                    path.name, f"{mime} (no extractable text)"
                )
            return Document(
                content=content,
                metadata=make_metadata(path, raw, mime, detected),
            )

    @staticmethod
    def _to_markdown(repaired_html: str) -> str:
        """Unstructured elements → markdown-ish text (lazy import)."""
        from unstructured.partition.html import partition_html

        parts = []
        for element in partition_html(text=repaired_html):
            text = (element.text or "").strip()
            if not text:
                continue
            if element.category == "Title":
                parts.append(f"# {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts)
