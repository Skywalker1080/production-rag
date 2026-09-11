# Plan 0007: Policy-Shaped Structure Retention + Chunking

Status: Accepted. Source of truth for issue #12. Domain: company
policy handbook, single-tenant (RBAC out). Builds on formatted
`Document`s (plan 0006 runs first); single structural chunker, no
per-format chunkers, no Block layer, no LangChain dep.

## 1. Objective

Retain what policy documents guarantee (headings, tables with
explaining context, versions/dates) through extraction, then cut
retrieval-ready `Chunk`s: atomic tables with context, heading paths,
deterministic IDs, neighbors, enforced budgets.

## 2. Scope

In: header/footer suppression, table-boundary markers at extraction,
version/date heuristics, Chunk model + engine + structural chunker,
policy fixtures, intrinsic eval.
Out: OCR/vision (#8), RBAC, semantic chunking, LangChain splitters,
page reconstruction, retrieval metrics (arrive with retrieval pillar).

## 3. Stage A — retain structure at extraction

A1. **Repeated header/footer suppression** (pdf path): lines repeated
verbatim on 3+ pages (e.g. "Confidential — ACME — Page N") are
boilerplate — drop them in `PdfExtractor` before assembly. Page numbers
vary, so match with `\d+`-masked comparison. Applies to pdf only (html
nav handled by tag stripping; txt/md have no such repetition).
A2. **Table-boundary markers**: normalizers must not flatten tables
silently. md tables already textual — detect pipe-row runs. html:
`HtmlExtractor` wraps each `<table>` region in
`<!-- table:start -->` / `<!-- table:end -->` markers around its text
(comments survive unstructured as separators; formatter keeps them).
pdf: no structure available — mark nothing, tables stay prose (honest
limitation, recorded).
A3. **Version/date heuristics** (best-effort, never blocking):
`Effective: <date>` / `Version X.Y` header patterns populate `doc_date`
plus new optional `doc_version`; absent → `None`, never guessed.
Frontmatter `date` keeps precedence for md.

## 4. Stage B — Chunk model + engine + chunker

B1. **Model** (`indexing/chunking.py`, strict + frozen like Document):
`ChunkMetadata`: `chunk_id`, `doc_id`, `chunk_index`, `heading_path:
list[str]`, `chunk_strategy: str`, `token_count: int`,
`prev_chunk_id`/`next_chunk_id: str | None`. `Chunk`: `content`,
`metadata`. IDs deterministic: `sha256(doc_id:index:content)`.
B2. **Protocol + engine**: `ChunkingStrategy.chunk(doc) ->
list[Chunk]`; `ChunkingEngine(strategy)` owns IDs, neighbor linking,
`stage("chunk", component="indexing")` logging, post-validation that
`max_tokens` holds. One impl now: `StructureAwareRecursiveChunker`.
B3. **Chunking rules**: paragraphs are the base unit (universal across
formats); md heading hierarchy → paths, html flat `#` → single-element
boundary (never fabricated hierarchy); fenced code + marked table
regions atomic unless over max; table bundle = table + preceding
paragraph (+ heading path prepended to every chunk content); overlap at
block boundaries; oversized block fallback sentence → char (stdlib,
no LangChain); budgets via `ChunkingConfig(target=500, max=700,
overlap=100)` measured with tiktoken cl100k (`uv add tiktoken`).

## 5. Dependencies (`uv add`)

- `tiktoken` (measuring stick; cl100k approximates future embedders).

## 6. Files

- `src/production_rag/indexing/pdf.py` — footer/header suppression.
- `src/production_rag/indexing/html.py` — table markers.
- `src/production_rag/indexing/document.py` — `doc_version` optional.
- `src/production_rag/indexing/chunking.py` — model, Protocol, engine,
  chunker, config.
- `tests/fixtures/policy/` — `leave_policy.md` (hierarchy + table +
  frontmatter date), `travel_policy.html` (styled + nav + table),
  `notice_policy.pdf` (generated with repeated footer + split table).
- `tests/test_chunking.py` — heading paths, atomic bundles with
  context, footer absence, determinism (same bytes twice → same IDs),
  max enforcement, overlap sharing, strategy substitution.
- `tests/test_policy_fixtures.py` — generator/regen notes if binaries.

## 7. Tests (stdlib unittest)

- md hierarchy → `["Leave Policy", "Casual Leave"]`-style paths.
- Table chunk contains table + paragraph above; no table split across
  chunks (search all chunks: table rows co-occur).
- Footer line ("Page N of M" pattern) in zero chunks.
- Same Document twice → identical `chunk_id`s; neighbors link both ways.
- 2000-token paragraph → all chunks ≤ max, ordered, lossless join.
- Engine accepts a stub strategy unchanged (substitution).
- Fixture policy QA spot checks (3 hand-written Q→section pairs).

## 8. Acceptance (issue #12)

- [ ] Repeated headers/footers suppressed, never chunked.
- [ ] Tables atomic with above/below context.
- [ ] Heading paths on chunks; deterministic IDs; neighbors linked.
- [ ] Version/date filterable metadata.
- [ ] Budgets enforced with stdlib fallback, tiktoken-measured.
