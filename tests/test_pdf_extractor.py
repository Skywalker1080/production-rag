"""PDF Extractor tests (plan 0003, stdlib unittest).

Seams: identify() pdf mapping, registry conformance, PdfExtractor
output contract, ingest() end-to-end on fixtures.
"""

import unittest
from pathlib import Path

from production_rag.indexing import dispatcher
from production_rag.indexing.base import Extractor
from production_rag.indexing.dispatcher import identify
from production_rag.indexing.document import Document
from production_rag.indexing.errors import UnsupportedFormatError
from production_rag.indexing.pdf import PdfExtractor


class IdentifyTests(unittest.TestCase):
    def test_pdf_bytes_map_to_pdf_kind(self) -> None:
        raw = Path("tests/fixtures/sample.pdf").read_bytes()
        kind, mime = identify(raw)
        self.assertEqual(kind, "pdf")
        self.assertEqual(mime, "application/pdf")

    def test_pdf_kind_is_registered(self) -> None:
        extractor = dispatcher.REGISTRY["pdf"]
        self.assertIsInstance(extractor, Extractor)
        self.assertIn("pdf", extractor.kinds)


class PdfExtractorTests(unittest.TestCase):
    def test_two_pages_accumulate_in_order(self) -> None:
        raw = Path("tests/fixtures/sample.pdf").read_bytes()
        doc = PdfExtractor().extract(
            Path("sample.pdf"), raw, "application/pdf"
        )
        self.assertIsInstance(doc, Document)
        first = doc.content.index("opening sentence")
        second = doc.content.index("closes it")
        self.assertLess(first, second)
        self.assertEqual(doc.metadata.detected_mime, "application/pdf")

    def test_scanned_pdf_fails_loudly(self) -> None:
        raw = Path("tests/fixtures/sample_scanned.pdf").read_bytes()
        with self.assertRaises(UnsupportedFormatError) as ctx:
            PdfExtractor().extract(
                Path("sample_scanned.pdf"), raw, "application/pdf"
            )
        self.assertIn("no extractable text", str(ctx.exception))

    def test_locked_pdf_fails_chained_not_hanging(self) -> None:
        raw = Path("tests/fixtures/sample_locked.pdf").read_bytes()
        with self.assertRaises(UnsupportedFormatError) as ctx:
            PdfExtractor().extract(
                Path("sample_locked.pdf"), raw, "application/pdf"
            )
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_mixed_pages_keep_text_skip_blank(self) -> None:
        raw = Path("tests/fixtures/sample_mixed.pdf").read_bytes()
        doc = PdfExtractor().extract(
            Path("sample_mixed.pdf"), raw, "application/pdf"
        )
        self.assertIn("Only this page has words", doc.content)

    def test_ingest_pdf_fixture_end_to_end(self) -> None:
        doc = dispatcher.ingest("tests/fixtures/sample.pdf")
        self.assertEqual(doc.metadata.file_name, "sample.pdf")
        self.assertIn("opening sentence", doc.content)


if __name__ == "__main__":
    unittest.main()
