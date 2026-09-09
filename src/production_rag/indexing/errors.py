"""Indexing failure taxonomy.

One base type so callers catch `IndexingError`; one subtype per
reject reason so operators and tests can distinguish them. Every error
names the file it broke on.
"""

from __future__ import annotations


class IndexingError(Exception):
    """Base for all indexing failures; never raised directly."""

    def __init__(self, file_name: str, detail: str = "") -> None:
        self.file_name = file_name
        super().__init__(f"{file_name}: {detail}" if detail else file_name)


class EmptyFileError(IndexingError):
    """0-byte input rejected."""

    def __init__(self, file_name: str) -> None:
        super().__init__(file_name, "empty file rejected")


class FileTooLargeError(IndexingError):
    """Input over MAX_BYTES rejected."""

    def __init__(self, file_name: str, byte_size: int, limit: int) -> None:
        self.byte_size = byte_size
        self.limit = limit
        super().__init__(
            file_name, f"size {byte_size} bytes exceeds limit {limit}"
        )


class UnsupportedFormatError(IndexingError):
    """Content the pipeline cannot normalize."""

    def __init__(self, file_name: str, detected_mime: str) -> None:
        self.detected_mime = detected_mime
        super().__init__(file_name, f"unsupported content: {detected_mime}")


class FetchError(IndexingError):
    """URL fetch failed: bad scheme, network error, HTTP error."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(url, f"fetch failed: {reason}")
