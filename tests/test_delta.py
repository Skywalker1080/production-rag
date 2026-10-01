"""Unit tests for page-level delta indexing (app.delta).

Run: .venv\\Scripts\\python.exe -m unittest discover -s tests -v
Stdlib only — no pytest needed.
"""
import unittest

from app.delta import (
    chunk_point_id,
    is_legacy_point,
    page_hash,
    plan_delta,
)


class TestPageHash(unittest.TestCase):
    def test_stable(self):
        self.assertEqual(page_hash("hello world"), page_hash("hello world"))

    def test_whitespace_insensitive(self):
        self.assertEqual(page_hash("hello   world\n"), page_hash("hello world"))

    def test_differs_on_content(self):
        self.assertNotEqual(page_hash("page one"), page_hash("page two"))


class TestChunkPointId(unittest.TestCase):
    def test_stable_prose(self):
        a = chunk_point_id("pdf_docs", "doc.pdf", "p", "1", 0)
        b = chunk_point_id("pdf_docs", "doc.pdf", "p", "1", 0)
        self.assertEqual(a, b)

    def test_page_scoped(self):
        a = chunk_point_id("pdf_docs", "doc.pdf", "p", "1", 0)
        b = chunk_point_id("pdf_docs", "doc.pdf", "p", "2", 0)
        self.assertNotEqual(a, b)

    def test_table_scoped_by_hash(self):
        a = chunk_point_id("pdf_docs", "doc.pdf", "t", "abc123", 0)
        b = chunk_point_id("pdf_docs", "doc.pdf", "t", "abc123", 0)
        c = chunk_point_id("pdf_docs", "doc.pdf", "t", "zzz999", 0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestPlanDelta(unittest.TestCase):
    def _units(self, pages):
        """pages: {page_key: text} -> old/new unit maps."""
        return {
            ("p", k): {"hash": page_hash(t), "ids": [f"id-{k}-{i}" for i in range(2)]}
            for k, t in pages.items()
        }

    def test_identical_reupload_all_skip(self):
        old = self._units({"1": "alpha", "2": "beta"})
        new = {k: {"hash": v["hash"]} for k, v in old.items()}
        plan = plan_delta(old, new)
        self.assertEqual(plan["skip"], set(old))
        self.assertEqual(plan["changed"], set())
        self.assertEqual(plan["removed"], set())

    def test_one_page_edited(self):
        old = self._units({"1": "alpha", "2": "beta"})
        new = {
            ("p", "1"): {"hash": page_hash("alpha")},
            ("p", "2"): {"hash": page_hash("beta EDITED")},
        }
        plan = plan_delta(old, new)
        self.assertEqual(plan["skip"], {("p", "1")})
        self.assertEqual(plan["changed"], {("p", "2")})
        self.assertEqual(plan["removed"], set())

    def test_page_added(self):
        old = self._units({"1": "alpha"})
        new = {
            ("p", "1"): {"hash": page_hash("alpha")},
            ("p", "2"): {"hash": page_hash("brand new")},
        }
        plan = plan_delta(old, new)
        self.assertEqual(plan["skip"], {("p", "1")})
        self.assertEqual(plan["changed"], {("p", "2")})

    def test_page_removed(self):
        old = self._units({"1": "alpha", "2": "beta"})
        new = {("p", "1"): {"hash": page_hash("alpha")}}
        plan = plan_delta(old, new)
        self.assertEqual(plan["removed"], {("p", "2")})

    def test_table_change_leaves_prose(self):
        old = {
            ("p", "1"): {"hash": page_hash("alpha"), "ids": ["a"]},
            ("t", "h1"): {"hash": "h1", "ids": ["t0"]},
        }
        new = {
            ("p", "1"): {"hash": page_hash("alpha")},
            ("t", "h2"): {"hash": "h2"},
        }
        plan = plan_delta(old, new)
        self.assertEqual(plan["skip"], {("p", "1")})
        self.assertEqual(plan["changed"], {("t", "h2")})
        self.assertEqual(plan["removed"], {("t", "h1")})


class TestLegacy(unittest.TestCase):
    def test_missing_hash_is_legacy(self):
        self.assertTrue(is_legacy_point({"page": 1}))
        self.assertFalse(is_legacy_point({"page": 1, "content_hash": "abc"}))


if __name__ == "__main__":
    unittest.main()
