"""Extractor interface and shared decode helper.

Every format normalizer implements `Extractor`; the dispatcher routes by
`kinds`. Normalizers are pure: they receive already-read bytes, do no
I/O, and import no heavy per-format libraries at module level.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from charset_normalizer import from_bytes

from production_rag.indexing.document import Document
from production_rag.indexing.errors import UnsupportedFormatError


@runtime_checkable
class Extractor(Protocol):
    """One format in, one cleaned `Document` out."""

    kinds: ClassVar[tuple[str, ...]]

    def extract(
        self,
        path: Path,
        raw: bytes,
        mime: str,
        encoding: str | None = None,
        file_path: str | None = None,
    ) -> Document:
        """Normalize `raw` into a `Document`; raise, never partial.

        `file_path` overrides the recorded source location (URL ingest
        records the URL instead of a resolved local path).
        """
        ...


def decode(
    raw: bytes, file_name: str, encoding: str | None = None
) -> tuple[str, str]:
    """Decode `raw` to text.

    Explicit `encoding` wins (strict): short single-byte texts are
    information-theoretically ambiguous between legacy encodings, so a
    caller with out-of-band knowledge overrides the guess. Otherwise
    UTF-8 first (strict, the standard form — deterministic, never
    guessed), detector only on `UnicodeDecodeError`. Returns (text,
    encoding name); raises `UnsupportedFormatError` naming the file when
    nothing is detectable or decodable.
    """
    if encoding is not None:
        try:
            return raw.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError) as exc:
            raise UnsupportedFormatError(file_name, encoding) from exc
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    match = from_bytes(raw).best()
    if match is None or not match.encoding:
        raise UnsupportedFormatError(file_name, "unknown")
    try:
        return str(match), str(match.encoding)
    except Exception as exc:
        raise UnsupportedFormatError(file_name, "undecodable") from exc
