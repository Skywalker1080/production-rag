# Plan 0004: Markdown Extractor + Frontmatter Formatter (indexing, md path)

Status: Accepted. Source of truth for issue #6. Fourth format on the
0001 Protocol. The new ground here is frontmatter: YAML config block
parsed, allowlisted into metadata, never leaked downstream.

## 1. Objective

`.md` upload → cleaned `Document`: body preserved as-is (already
markdown), frontmatter split off and filtered against an allowlist so
config never reaches content, metadata, or the LLM.

## 2. Scope

In: `MdExtractor`, markdown kind routing, frontmatter parse + allowlist,
optional metadata extension, tests.
Out: full semantic enrichment (language detection, entity extraction —
#8 territory), markdown linting/reformatting (body passes through).

## 3. Contract

```python
class MdExtractor:
    kinds: ClassVar[tuple[str, ...]] = ("markdown",)
    def extract(self, path, raw, mime, encoding=None) -> Document: ...
```

Same provenance subset, same errors, `stage("md_normalize",
component="indexing")`. Immutability boundary unchanged.

### 3.1 Frontmatter allowlist (decided)

Initial list: `title`, `tags`, `language`, `date`. Everything else is
dropped — never into content, never into metadata. Rationale: frontmatter
is author config (draft flags, slugs, layout); only citation/search
fields (groups 1+4) cross the boundary.

### 3.2 Metadata extension

`DocumentMetadata` gains optional fields (all default `None`, strict
model stays `forbid`): `title: str | None`, `tags: list[str] |
None`, `language: str | None`, `doc_date: str | None`. Allowlisted
frontmatter maps onto these; absent frontmatter leaves them `None`.
Shared by all extractors (txt/html/pdf simply never set them).

## 4. Pipeline steps

1. **Route**: dispatcher `identify()` maps `text/markdown` → `markdown`
   kind, plus the html-style tiebreak: generic `text/*` sniff on `.md`
   routes markdown (same asymmetry argument — `MdExtractor` passes
   plain text through; `TextExtractor` would leak frontmatter as
   content). Registry gains `"markdown": MdExtractor()`.
2. **Decode**: shared helper (utf-8 first, override, detector).
3. **Split**: `python-frontmatter` load. Missing/damaged block → whole
   file is body (no fail — frontmatter is optional by format design).
4. **Filter**: keep allowlisted keys, map to metadata fields; drop the
   rest. Body (post object `.content`) is the Document content —
   frontmatter bytes never enter it.
5. **Metadata + Document**: shared `make_metadata()` extended with the
   four optional fields.

## 5. Dependencies (`uv add`)

- `python-frontmatter` (light; pyyaml already transitive).

## 6. Files

- `src/production_rag/indexing/markdown.py` — `MdExtractor`,
  `FRONTMATTER_ALLOWLIST`.
- `src/production_rag/indexing/document.py` — four optional fields +
  extended `make_metadata()`.
- `src/production_rag/indexing/dispatcher.py` — markdown kind rule +
  registry line.
- `tests/fixtures/sample_doc.md` (frontmatter incl. a `draft: true`
  trap key), `tests/fixtures/sample_plain.md` (no frontmatter).
- `tests/test_md_extractor.py` — cases below.

## 7. Tests (stdlib unittest)

- Frontmatter doc → `title`/`tags` in metadata, `draft` nowhere (not
  in content, not in metadata).
- Body excludes the `---` block (content starts at first heading).
- No-frontmatter md → ingests, optionals all `None`.
- Malformed frontmatter (unclosed `---`) → whole file treated as body,
  no crash.
- Registry conformance extends to `MdExtractor`.
- Ingest end-to-end on both fixtures; `detected_mime` recorded.

## 8. Acceptance (issue #6)

- [ ] Frontmatter parsed, kept in metadata not content.
- [ ] Body normalized to same markdown contract.
- [ ] Missing frontmatter still ingests.
