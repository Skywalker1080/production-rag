# Plan 0001: Text Extractor (indexing, txt path)

Status: Accepted. Source of truth for issue #2. Later plans are numbered
0002, 0003, ... in this folder.

## 1. Objective

First shippable indexing component: take a `.txt` upload from bytes on
disk to a normalized `Document`. Proves the
identify → normalize → Document path that html/md/pdf will reuse.

## 2. Scope

In: txt files only, magic gate, empty/size guards, encoding detection +
decode, provenance metadata, error taxonomy, logging, tests.
Out: md/html/pdf normalizers, markdownify/Unstructured pass, chunking,
language detection (`language` field marked TBD, not emitted).

## 3. Contract

```python
extract_text(path: str | Path) -> Document
```

- Input: path to a candidate `.txt` file.
- Output: `Document(content=<decoded str>, metadata=<provenance subset>)`.
- Raises, never returns partials: `EmptyFileError`, `FileTooLargeError`,
  `UnsupportedFormatError`. Original exceptions chained (`raise ... from`).

### 3.1 Provenance subset emitted (group 1, partial)

`doc_id` (uuid4 hex), `file_name`, `file_path` (absolute resolved),
`sha256` (hex of raw bytes, for update detection), `byte_size`,
`detected_mime` (from magic gate), `encoding` (from detector).

Deferred to their stages: timestamps (ingest-time, added by dispatcher
#7), structural fields (chunking), semantic fields (later enrichment).

### 3.2 Models

Pydantic v2 (consistent with MVP `sample.py`). `Document` and metadata
live in `indexing/document.py` — Document home is indexing until
vector-DB ingest (CONTEXT.md).

### 3.3 Immutability boundary (decided)

`Document` is a frozen source-level record: it is never mutated after
extraction. Rationale: each stage only knows its own facts — the
extractor cannot know pages or chunk boundaries, so positional metadata
(`chunk_id`, index, prev/next, page, headings) can only be born in the
chunker. It will live on a future `Chunk` type (plan 0002), each chunk
carrying `doc_id` as parent link. Citation-critical fields (`file_name`,
source) are denormalized onto chunks so retrieval hits stay
self-contained; full provenance stays on the parent. Re-chunking =
delete chunks for `doc_id` and re-run; the parent `Document` is untouched.

## 4. Pipeline steps (in order)

1. **Read once**: `Path.read_bytes()`. Single read; everything downstream
   works from memory (no double-open).
2. **Magic gate** (`stage("identifier", component="indexing")`):
   `python-magic` on raw bytes. Accept text/* (and `inode/x-empty`,
   empty handled next). Anything else → `UnsupportedFormatError` with
   detected MIME. This runs BEFORE decoding — the detector mislabels
   binary as text (verified: PNG header → big5hkscs).
3. **Guards**: 0 bytes → `EmptyFileError`; `> MAX_BYTES` (100 MiB,
   module constant, config later) → `FileTooLargeError`. Fail fast,
   before decode.
4. **Decode** (`stage("txt_normalize", component="indexing")`):
   `charset_normalizer`, best match, decode from memory. Undecodable →
   `UnsupportedFormatError` chained from decoder error.
5. **Metadata + Document**: sha256 over raw bytes, assemble subset,
   return `Document`.

## 5. Logging

All work inside `stage(..., component="indexing")` so any break emits
`stage_error` naming component + stage. Guards log a WARNING line with
`file_name` and reason before raising (operators see rejects without
digging tracebacks).

## 6. Dependencies (`uv add`)

- `charset-normalizer` (direct; today only transitive via requests).
- `python-magic` (locked per user).
- Windows caveat: libmagic DLL required. If import fails on Windows,
  `uv add python-magic-bin` (bundles the DLL). No code change either way.

## 7. Files

- `src/production_rag/indexing/document.py` — `Document`, metadata model.
- `src/production_rag/indexing/errors.py` — `IndexingError` base +
  `EmptyFileError`, `FileTooLargeError`, `UnsupportedFormatError`.
- `src/production_rag/indexing/text.py` — `extract_text`, `MAX_BYTES`.
- `tests/test_text_extractor.py` — cases below.

## 8. Tests (stdlib unittest)

- utf-8 fixture decodes (`sample_japanese_utf8.txt`).
- latin-1 fixture transcodes (`sample_latin1.txt`), café/resumé intact.
- 0-byte tmp file → `EmptyFileError`.
- Oversized input → `FileTooLargeError` (monkeypatched small limit, not
  a real 100 MiB file).
- Binary-with-.txt-extension → `UnsupportedFormatError`, MIME named.
- Metadata: sha256 matches fixture bytes; `doc_id` unique per call.
- Logs: break inside stage emits `stage_error` with
  `component=indexing`.

## 9. Acceptance (issue #2)

- [ ] Reads a `.txt` file and returns a `Document`.
- [ ] Normalization applied per Document contract.
- [ ] Corrupt/empty/unreadable file fails loudly, not silently.
