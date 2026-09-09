# Plan 0006: Common MD Formatter (indexing, pre-LLM stage)

Status: Accepted. Source of truth for issue #11. Completes the
architecture diagram's last box (identifier → normalizer → MD
Formatter): every `Document` gets one deterministic cleanup before
chunking/LLM, so downstream always sees a single markdown shape.

## 1. Objective

`format_document(doc) -> Document`: conservative, idempotent,
meaning-preserving cleanup. Whitespace and control noise go; not one
word changes.

## 2. Scope

In: shared `format_text` + `format_document`, wiring into the shared
tail, idempotency/behavior tests.
Out: semantic restructuring, re-parsing through Unstructured (html
already had its pass — double-parsing risks loss for zero gain),
table reflowing (#8 owns tables).

## 3. Contract

```python
format_text(text: str) -> str       # pure string function
format_document(doc: Document) -> Document  # new Document, same metadata
```

Rules (in order): `\r\n` → `\n`; strip ASCII control chars except
`\n` and `\t`; strip trailing whitespace per line; collapse 3+
newlines to 2; strip leading/trailing blank lines. `Document` is
frozen, so the formatter builds a new instance reusing the metadata
object. Runs inside `stage("md_format", component="indexing")`.

## 4. Pipeline position

Wired at the end of dispatcher `_finalize()`: extract → format →
return. One-line tail change; all sources (file, URL) and formats get
it automatically. No extractor changes.

## 5. Dependencies

- None (stdlib `re`).

## 6. Files

- `src/production_rag/indexing/md_formatter.py`
- `src/production_rag/indexing/dispatcher.py` — tail call (small).
- `tests/test_md_formatter.py` — cases below.

## 7. Tests (stdlib unittest)

- Idempotency on all four format outputs (real fixtures).
- Control chars (`\x00`, `\x1b`) removed, words intact.
- `line  \n\n\n\nline` → `line\n\nline`.
- CRLF input → LF output.
- Metadata object identical after formatting (same provenance).
- Empty-after-cleanup (whitespace-only doc) → returns stripped empty
  content, no raise (emptiness was the guards' job at ingest).

## 8. Acceptance (issue #11)

- [ ] Idempotent across all formats.
- [ ] Drops noise only, never words.
- [ ] Wired into the shared tail.
