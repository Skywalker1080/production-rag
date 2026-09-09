"""Markdown Extractor tests (plan 0004, stdlib unittest).

Seams: identify() markdown mapping, frontmatter allowlist boundary,
registry conformance, ingest() end-to-end on fixtures.
"""

import unittest
from pathlib import Path

from production_rag.indexing import dispatcher
from production_rag.indexing.base import Extractor
from production_rag.indexing.dispatcher import identify
from production_rag.indexing.document import Document, DocumentMetadata
from production_rag.indexing.markdown import MdExtractor


class IdentifyTests(unittest.TestCase):
    def test_md_bytes_map_to_markdown_kind(self) -> None:
        raw = Path("tests/fixtures/sample_doc.md").read_bytes()
        kind, mime = identify(raw, "sample_doc.md")
        self.assertEqual(kind, "markdown")
        self.assertEqual(mime, "text/plain")

    def test_plain_text_md_extension_routes_markdown(self) -> None:
        kind, _ = identify(b"no frontmatter at all", "notes.md")
        self.assertEqual(kind, "markdown")


class MetadataExtensionTests(unittest.TestCase):
    def test_optional_semantic_fields_default_none(self) -> None:
        meta = DocumentMetadata(
            doc_id="x",
            file_name="a.md",
            file_path="/tmp/a.md",
            sha256="0" * 64,
            byte_size=1,
            detected_mime="text/plain",
            encoding="utf-8",
        )
        self.assertIsNone(meta.title)
        self.assertIsNone(meta.tags)
        self.assertIsNone(meta.language)
        self.assertIsNone(meta.doc_date)

    def test_markdown_kind_is_registered(self) -> None:
        extractor = dispatcher.REGISTRY["markdown"]
        self.assertIsInstance(extractor, Extractor)
        self.assertIn("markdown", extractor.kinds)


class MdExtractorTests(unittest.TestCase):
    def test_frontmatter_crosses_allowlist_only(self) -> None:
        raw = Path("tests/fixtures/sample_doc.md").read_bytes()
        doc = MdExtractor().extract(Path("sample_doc.md"), raw, "text/plain")
        self.assertEqual(doc.metadata.title, "RAG Notes")
        self.assertEqual(doc.metadata.tags, ["rag", "indexing"])
        self.assertEqual(doc.metadata.language, "en")
        self.assertEqual(doc.metadata.doc_date, "2026-09-01")

    def test_config_keys_leak_nowhere(self) -> None:
        raw = Path("tests/fixtures/sample_doc.md").read_bytes()
        doc = MdExtractor().extract(Path("sample_doc.md"), raw, "text/plain")
        self.assertNotIn("draft", doc.content)
        self.assertNotIn("hidden-slug", doc.content)
        self.assertNotIn("---", doc.content)
        self.assertIn("First real paragraph", doc.content)

    def test_no_frontmatter_ingests_with_nones(self) -> None:
        raw = Path("tests/fixtures/sample_plain.md").read_bytes()
        doc = MdExtractor().extract(Path("sample_plain.md"), raw, "text/plain")
        self.assertIn("No Frontmatter Here", doc.content)
        self.assertIsNone(doc.metadata.title)
        self.assertIsNone(doc.metadata.tags)

    def test_malformed_frontmatter_degrades_to_body(self) -> None:
        raw = b"---\ntitle: [unclosed\n\nBody survives.\n"
        doc = MdExtractor().extract(Path("broken.md"), raw, "text/plain")
        self.assertIn("Body survives", doc.content)

    def test_ingest_md_fixture_end_to_end(self) -> None:
        doc = dispatcher.ingest("tests/fixtures/sample_doc.md")
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.metadata.file_name, "sample_doc.md")
        self.assertEqual(doc.metadata.title, "RAG Notes")


if __name__ == "__main__":
    unittest.main()
