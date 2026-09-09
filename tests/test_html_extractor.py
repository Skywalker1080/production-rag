"""HTML Extractor tests (plan 0002, stdlib unittest).

Seams: identify() kind mapping, registry conformance, HtmlExtractor
output contract, ingest() end-to-end on fixtures.
"""

import unittest
from pathlib import Path
from unittest import mock

from production_rag.indexing import dispatcher
from production_rag.indexing.base import Extractor
from production_rag.indexing.dispatcher import identify
from production_rag.indexing.document import Document
from production_rag.indexing.errors import UnsupportedFormatError
from production_rag.indexing.html import HtmlExtractor


class IdentifyTests(unittest.TestCase):
    def test_html_bytes_map_to_html_kind(self) -> None:
        kind, mime = identify(b"<html><body><p>hi</p></body></html>")
        self.assertEqual(kind, "html")
        self.assertEqual(mime, "text/html")

    def test_plain_text_still_maps_to_text_kind(self) -> None:
        kind, _ = identify(b"just words")
        self.assertEqual(kind, "text")

    def test_generic_sniff_with_html_extension_breaks_toward_html(self) -> None:
        raw = b"<html><body><p>code-heavy page</p></body></html>"
        with mock.patch.object(
            dispatcher.magic, "from_buffer", return_value="text/x-python"
        ):
            kind, mime = identify(raw, "page.html")
        self.assertEqual(kind, "html")
        self.assertEqual(mime, "text/x-python")

    def test_generic_sniff_without_html_extension_stays_text(self) -> None:
        raw = b"print('hi')"
        with mock.patch.object(
            dispatcher.magic, "from_buffer", return_value="text/x-python"
        ):
            kind, _ = identify(raw, "script.txt")
        self.assertEqual(kind, "text")


class HtmlExtractorTests(unittest.TestCase):
    def test_wellformed_page_returns_cleaned_document(self) -> None:
        raw = Path("tests/fixtures/html/happy_page.html").read_bytes()
        doc = HtmlExtractor().extract(
            Path("happy_page.html"), raw, "text/html"
        )
        self.assertIsInstance(doc, Document)
        self.assertIn("Revenue grew 12 percent", doc.content)
        self.assertIn("Expenses", doc.content)
        self.assertNotIn("<", doc.content)
        self.assertNotIn("must not appear", doc.content)
        self.assertNotIn("color: red", doc.content)
        self.assertEqual(doc.metadata.detected_mime, "text/html")

    def test_malformed_nesting_repairs_without_loss(self) -> None:
        raw = Path("tests/fixtures/html/unhappy_malformed.html").read_bytes()
        doc = HtmlExtractor().extract(
            Path("unhappy_malformed.html"), raw, "text/html"
        )
        for sentence in (
            "Broken Report",
            "First half of the story",
            "Second half of the story",
            "bold tail",
        ):
            self.assertIn(sentence, doc.content)

    def test_page_without_visible_text_fails_loudly(self) -> None:
        raw = b"<html><body><script>only code</script></body></html>"
        with self.assertRaises(UnsupportedFormatError):
            HtmlExtractor().extract(Path("empty.html"), raw, "text/html")

    def test_table_text_present_without_fidelity_claim(self) -> None:
        raw = Path("tests/fixtures/html/happy_page.html").read_bytes()
        doc = HtmlExtractor().extract(
            Path("happy_page.html"), raw, "text/html"
        )
        self.assertIn("Q3", doc.content)

    def test_code_sniffed_html_ingests_cleaned_end_to_end(self) -> None:
        # Regression: libmagic reads code-heavy HTML as text/x-python;
        # the tiebreak must still deliver a tag-free Document.
        import tempfile

        raw = Path("tests/fixtures/html/happy_page.html").read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "page.html"
            page.write_bytes(raw)
            with mock.patch.object(
                dispatcher.magic, "from_buffer", return_value="text/x-python"
            ):
                doc = dispatcher.ingest(page)
        self.assertNotIn("<", doc.content)
        self.assertIn("Revenue grew 12 percent", doc.content)

    def test_non_text_mime_rejected_at_extractor(self) -> None:
        with self.assertRaises(UnsupportedFormatError):
            HtmlExtractor().extract(
                Path("doc.pdf"), b"%PDF-1.4", "application/pdf"
            )

    def test_registered_html_extractor_satisfies_protocol(self) -> None:
        extractor = dispatcher.REGISTRY["html"]
        self.assertIsInstance(extractor, Extractor)
        self.assertIn("html", extractor.kinds)

    def test_ingest_html_fixture_end_to_end(self) -> None:
        doc = dispatcher.ingest("tests/fixtures/html/happy_page.html")
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.metadata.file_name, "happy_page.html")

    def test_unhappy_code_heavy_blog_ingests_tag_free(self) -> None:
        # Reporter's real file: libmagic mis-sniffs it, tiebreak routes
        # html. Locks the #9 fix to real bytes, not mocks.
        doc = dispatcher.ingest("tests/fixtures/html/unhappy_code_heavy.html")
        self.assertNotIn("<html", doc.content)
        self.assertIn("Demystifying Deep Learning", doc.content)

    def test_happy_blog_ingests_end_to_end(self) -> None:
        doc = dispatcher.ingest("tests/fixtures/html/happy_blog.html")
        self.assertIn("Backpropagation", doc.content)


if __name__ == "__main__":
    unittest.main()
