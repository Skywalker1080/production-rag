"""URL ingest tests (plan 0005, stdlib unittest).

Network mocked at the seam: no real HTTP in tests.
"""

import unittest
from pathlib import Path
from unittest import mock

import requests

from production_rag.indexing import dispatcher
from production_rag.indexing.document import Document
from production_rag.indexing.errors import (
    FetchError,
    FileTooLargeError,
    IndexingError,
)


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size: int = 65536):
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]


def _html_bytes() -> bytes:
    return (
        b"<html><body><h1>Remote Post</h1>"
        b"<p>Fetched paragraph.</p></body></html>"
    )


class FetchTests(unittest.TestCase):
    def _get(self, response=None, error=None):
        if error is not None:
            return mock.patch.object(
                dispatcher.requests, "get", side_effect=error
            )
        return mock.patch.object(
            dispatcher.requests, "get", return_value=response
        )

    def test_html_url_rejoins_html_path_cleaned(self) -> None:
        with self._get(_FakeResponse(_html_bytes())):
            doc = dispatcher.ingest_url("https://example.com/post.html")
        self.assertIsInstance(doc, Document)
        self.assertIn("Remote Post", doc.content)
        self.assertNotIn("<h1>", doc.content)
        self.assertEqual(doc.metadata.file_path, "https://example.com/post.html")

    def test_pdf_bytes_over_url_route_pdf(self) -> None:
        raw = Path("tests/fixtures/pdf/sample.pdf").read_bytes()
        with self._get(_FakeResponse(raw)):
            doc = dispatcher.ingest_url("https://example.com/doc.pdf")
        self.assertIn("opening sentence", doc.content)

    def test_http_error_is_fetch_error(self) -> None:
        with self._get(_FakeResponse(b"nope", status=404)):
            with self.assertRaises(FetchError):
                dispatcher.ingest_url("https://example.com/missing")

    def test_network_error_chains_fetch_error(self) -> None:
        with self._get(error=requests.ConnectionError("dns down")):
            with self.assertRaises(FetchError) as ctx:
                dispatcher.ingest_url("https://example.com/x")
        self.assertIsInstance(ctx.exception.__cause__, requests.ConnectionError)

    def test_file_scheme_never_touches_network(self) -> None:
        with mock.patch.object(
            dispatcher.requests, "get"
        ) as fake_get:
            with self.assertRaises(FetchError):
                dispatcher.ingest_url("file:///etc/passwd")
        fake_get.assert_not_called()

    def test_oversized_stream_rejected_past_cap(self) -> None:
        with mock.patch.object(dispatcher, "MAX_BYTES", 10):
            with self._get(_FakeResponse(b"x" * 64)):
                with self.assertRaises(FileTooLargeError):
                    dispatcher.ingest_url("https://example.com/big")

    def test_fetch_error_is_indexing_error(self) -> None:
        self.assertTrue(issubclass(FetchError, IndexingError))


if __name__ == "__main__":
    unittest.main()
