"""Text Extractor tests (plan 0001, stdlib unittest).

Seams: public ingest(), TextExtractor.extract(), registry conformance,
error taxonomy, component log lines.
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from production_rag.common import logging as plog
from production_rag.indexing import dispatcher
from production_rag.indexing.base import Extractor, decode
from production_rag.indexing.document import Document, DocumentMetadata
from production_rag.indexing.errors import (
    EmptyFileError,
    FileTooLargeError,
    IndexingError,
    UnsupportedFormatError,
)
from production_rag.indexing.text import TextExtractor


class ErrorTaxonomyTests(unittest.TestCase):
    def test_each_failure_has_its_own_type_under_one_base(self) -> None:
        for error in (EmptyFileError, FileTooLargeError, UnsupportedFormatError):
            self.assertTrue(issubclass(error, IndexingError))

    def test_failures_carry_file_context(self) -> None:
        error = EmptyFileError("sample.txt")
        self.assertIn("sample.txt", str(error))


def _metadata(**overrides: object) -> DocumentMetadata:
    base: dict = {
        "doc_id": "abc123",
        "file_name": "sample.txt",
        "file_path": "C:/tmp/sample.txt",
        "sha256": "0" * 64,
        "byte_size": 5,
        "detected_mime": "text/plain",
        "encoding": "utf_8",
    }
    base.update(overrides)
    return DocumentMetadata(**base)


class DocumentTests(unittest.TestCase):
    def test_document_is_frozen_after_extraction(self) -> None:
        doc = Document(content="hello", metadata=_metadata())
        with self.assertRaises(Exception):
            doc.content = "mutated"  # type: ignore[misc]

    def test_document_rejects_incomplete_provenance(self) -> None:
        with self.assertRaises(Exception):
            Document(content="hello", metadata={"doc_id": "abc123"})  # type: ignore[arg-type]


class DecodeTests(unittest.TestCase):
    def test_explicit_encoding_resolves_ambiguous_legacy_bytes(self) -> None:
        # Short single-byte texts are ambiguous between legacy encodings
        # (verified: this input misdetects), so the override must win.
        raw = "Café, résumé, naïve.".encode("latin-1")
        text, encoding = decode(raw, "probe.txt", encoding="latin-1")
        self.assertEqual(text, "Café, résumé, naïve.")
        self.assertEqual(encoding, "latin-1")

    def test_unknown_explicit_encoding_fails_loudly(self) -> None:
        with self.assertRaises(UnsupportedFormatError) as ctx:
            decode(b"abc", "probe.txt", encoding="not-a-codec")
        self.assertIn("probe.txt", str(ctx.exception))

    def test_utf8_bytes_decode_without_guessing(self) -> None:
        text, encoding = decode("plain ascii 日本語".encode("utf-8"), "a.txt")
        self.assertEqual(text, "plain ascii 日本語")
        self.assertEqual(encoding, "utf-8")


class TextExtractorTests(unittest.TestCase):
    def test_extract_returns_document_with_provenance(self) -> None:
        raw = "hello indexing".encode("utf-8")
        doc = TextExtractor().extract(Path("greet.txt"), raw, "text/plain")
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.content, "hello indexing")
        self.assertEqual(doc.metadata.file_name, "greet.txt")

    def test_extract_rejects_non_text_mime(self) -> None:
        with self.assertRaises(UnsupportedFormatError) as ctx:
            TextExtractor().extract(
                Path("img.txt"), b"\x89PNG\r\n\x1a\nxxxx", "image/png"
            )
        self.assertIn("image/png", str(ctx.exception))


class IngestTests(unittest.TestCase):
    def test_ingest_utf8_fixture_end_to_end(self) -> None:
        doc = dispatcher.ingest("tests/fixtures/txt/happy_utf8.txt")
        self.assertIsInstance(doc, Document)
        self.assertIn("Financial Time-Series", doc.content)
        self.assertIn("日本語", doc.content)
        self.assertEqual(doc.metadata.file_name, "happy_utf8.txt")

    def test_ingest_latin1_fixture_with_override(self) -> None:
        doc = dispatcher.ingest("tests/fixtures/txt/unhappy_latin1.txt", encoding="latin-1")
        self.assertIn("Café, résumé, naïve", doc.content)
        self.assertEqual(doc.metadata.encoding, "latin-1")

    def test_ingest_routes_by_content_not_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tricky = Path(tmp) / "report.pdf"
            tricky.write_bytes("plain text in disguise".encode("utf-8"))
            doc = dispatcher.ingest(tricky)
        self.assertEqual(doc.content, "plain text in disguise")

    def test_ingest_empty_file_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.txt"
            empty.write_bytes(b"")
            with self.assertRaises(EmptyFileError):
                dispatcher.ingest(empty)

    def test_ingest_oversized_file_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.txt"
            big.write_bytes(b"x" * 16)
            with mock.patch.object(dispatcher, "MAX_BYTES", 15):
                with self.assertRaises(FileTooLargeError):
                    dispatcher.ingest(big)

    def test_ingest_binary_names_detected_mime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "notes.txt"
            fake.write_bytes(bytes.fromhex("89504e470d0a1a0a") + b"fakepng")
            with self.assertRaises(UnsupportedFormatError) as ctx:
                dispatcher.ingest(fake)
        self.assertIn("octet-stream", str(ctx.exception))

    def test_every_registered_extractor_satisfies_protocol(self) -> None:
        for extractor in dispatcher.REGISTRY.values():
            self.assertIsInstance(extractor, Extractor)
            self.assertTrue(extractor.kinds)

    def test_break_emits_component_stage_error(self) -> None:
        buf = io.StringIO()
        plog.configure(level="DEBUG", stream=buf, force=True)
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "notes.txt"
            fake.write_bytes(bytes.fromhex("89504e470d0a1a0a") + b"fakepng")
            with self.assertRaises(UnsupportedFormatError):
                dispatcher.ingest(fake)
        import json

        errors = [
            json.loads(line)
            for line in buf.getvalue().splitlines()
            if "stage_error" in line
        ]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["component"], "indexing")


if __name__ == "__main__":
    unittest.main()
