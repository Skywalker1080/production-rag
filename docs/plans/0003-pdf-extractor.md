# Plan 0003: PDF Extractor (indexing, pdf path)

Status: Accepted. Source of truth for issue #4. Third format on the
0001 Protocol; dispatcher already maps `application/pdf` → `pdf` kind,
so the only dispatcher change is one registry line.

## 1. Objective

`.pdf` upload → cleaned `Document` under the same contract: multi-page
order preserved, scanned/image-only fails loudly, academic 2-col
handled by pypdf's extraction order (no custom column logic).

## 2. Scope

In: `PdfExtractor`, registry line, pypdf page accumulation, encrypted /
scanned loud failures, provenance metadata, tests.
Out: image/OCR extraction, nested-table fidelity (#8), password support
(encrypted → loud fail, no decrypt attempt).

## 3. Contract

```python
class PdfExtractor:
    kinds: ClassVar[tuple[str, ...]] = ("pdf",)
    def extract(self, path, raw, mime, encoding=None) -> Document: ...
```

Same provenance subset (`detected_mime=application/pdf`; `encoding`
records `pdf-text` — PDFs carry their own encoding, no detector
involved). Same errors, `stage("pdf_normalize", component="indexing")`.

## 4. Pipeline steps

1. **Route**: registry gains `"pdf": PdfExtractor()` — no dispatcher
   logic change (identify already emits the kind).
2. **Read from memory**: `PdfReader(BytesIO(raw))` — pure normalizer,
   no second file read. Explicitly accumulates EVERY page (fixes the
   MVP `sample.py` last-page-only overwrite bug).
3. **Per-page text**: join non-empty pages with `"\n\n"`, order
   preserved. Pages yielding `None`/empty are skipped; if NOTHING is
   extracted (scanned document) → `UnsupportedFormatError` — never a
   silent empty Document. Encrypted/corrupt → chained
   `UnsupportedFormatError` from the pypdf error (no password attempts).
4. **Metadata + Document**: identical subset via shared
   `make_metadata()`.

## 5. Dependencies (`uv add`)

- None for runtime (`pypdf` already direct).
- `--dev reportlab`: fixture generation only (script writes text pages
  pypdf's writer cannot). Never imported by runtime code.

## 6. Files

- `src/production_rag/indexing/pdf.py` — `PdfExtractor`.
- `src/production_rag/indexing/dispatcher.py` — registry line only.
- `tests/fixtures/make_pdf_fixtures.py` — generates `sample.pdf`
  (2 text pages), `sample_scanned.pdf` (blank pages), `sample_locked.pdf`
  (user-password encrypted). Generated binaries committed; script kept
  for regeneration.
- `tests/test_pdf_extractor.py` — cases below.

## 7. Tests (stdlib unittest)

- Two-page order: page-1 sentence precedes page-2 sentence in content.
- Scanned fixture → `UnsupportedFormatError`, MIME named.
- Encrypted fixture → `UnsupportedFormatError` chained (no hang on
  password prompt).
- Mixed content (text page + blank page) → Document with text page kept.
- Registry conformance extends to `PdfExtractor`.
- Metadata `detected_mime` is `application/pdf`.
- Ingest end-to-end on `sample.pdf`.

## 8. Acceptance (issue #4)

- [ ] Reads a `.pdf` file and returns a cleaned `Document`.
- [ ] Scanned/image-only PDF fails loudly with clear reason.
- [ ] Multi-page ordering preserved.
