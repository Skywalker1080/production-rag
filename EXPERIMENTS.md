# Experiment Log — PDF RAG on Financial Reports

Running lab notebook for this project. Each entry: hypothesis, setup, metrics,
verdict. Source of truth for the write-up article.

Corpus: `jiofin financial report.pdf` (8 MB, 143 pages) unless noted.
Stack end-state: EC2 `hi_res` parse → page-grouped markdown chunks (tables as
captioned grids, headers repeated per chunk) → bge-m3 dense + BM25 sparse →
Qdrant RRF hybrid → Bedrock GLM 5 (`zai.glm-5`) with role-grounded prompt.

## E1 — fast vs hi_res, single process (c7i.xlarge)

Hypothesis: `hi_res` quality is worth its cost on a 143-page financial report.
Setup: `extractor/compare.py` on EC2, `strategy=fast`, `infer_table_structure=True`.

| Metric (fast) | Value |
|---|---|
| Wall time | 104 s |
| Elements | 13,231 |
| Median element length | 11 chars |
| Crumbs (<100 chars) | 10,969 / 13,231 (**83%**) |
| Tables detected | **0** |
| Pages covered | 143 / 143 |

Verdict: fast is quick but blind — 83% crumbs, zero tables on a financial
report. Quality unacceptable against the P0 baseline (tables + row/col
relationships non-negotiable).

## E2 — sharded hi_res, pages 1–20 (c7i.4xlarge, 8 workers)

Setup: `extractor/shard_compare.py` — pypdf page shards + 1-page overlap,
`ProcessPoolExecutor(spawn)`, page-offset merge, deterministic overlap
page-level dedupe. Fixes hit along the way: serial HF-Hub cache warm (8×
concurrent unauthenticated downloads get rate-limited); spawn-not-fork
(forking multithreaded paddle/torch kills children); tesseract via
conda-forge (AL2023 has no package; `process_file_with_ocr` hardcodes
tesseract, ignores `OCR_AGENT=paddle`); `HF_HUB_OFFLINE=1` after warming
(HF revalidates even cache hits).

| Metric (hi_res, 20 pp) | Value |
|---|---|
| Wall time | 57 s |
| Elements | 377 |
| Median length | 121 chars |
| Crumbs | 45% |
| Tables (with cell HTML) | 16 (15 with cells, ~26 cells avg) |
| Titles | 39, pages 20/20, 0 missing, 73 overlap dupes dropped |

Verdict: tables recovered. Proceed to full doc.

## E3 — full-doc hi_res, 143 pages, 8 workers, 5-min hypothesis

Hypothesis: full report in ~4 min. Result: **FAILED** — wall **667 s (~11 min)**,
29× fewer seconds than single-core would take at 92% parallel efficiency
(4,934 worker-seconds). No scheduling problem left on CPU: unit cost is
~30 s/page of layout inference + OCR + table-transformer.

| Metric (hi_res, full) | Value |
|---|---|
| Elements | 8,135 |
| Median length | 20 chars, crumbs 74% |
| Tables (with cells, ~49 cells avg) | 277 (274) |
| Titles 648, footnotes 0, pages 143/143, 995 dupes dropped | — |

Verdict: quality passes P0 except footnotes (0 found — needs manual
spot-check); latency fails any SLA on CPU. Only cheaper-work-per-page moves
the needle from here (quantized layout model, hybrid digital/fast routing,
GPU). Footnote detection stays open.

## E4 — chunking: crumbs → page-grouped docs (DeepLOB paper)

Finding: `elements` mode shatters papers (628 elements, median 10 chars);
`split_documents` splits each element independently, so crumbs persist into
Qdrant; page metadata key is `page_number`, not `page` (all citations "p. ?");
95 Header/Footer elements pollute retrieval.
Fix: drop Header/Footer, regroup into one doc per page with normalized `page`,
then split. Re-uploads delete the file's old points first (idempotent).
Measured: 628 crumbs → 12 page-docs → 94 chunks, median 811 chars, pages 1–12.

## E5 — embeddings providers (all hit walls except local)

- Bedrock Titan v2: works, but ~4.8 s/call sequential → ~50 min for 638
  chunks; 4-way parallel + throttle-only backoff → ~9 min; 8-way parallel
  hits account `ThrottlingException`.
- Gemini `gemini-embedding-001`: batched path (27 requests for 1,312 chunks)
  built, but free-tier **1,000 req/day quota blown by earlier runs** — dead
  until reset.
- Cohere Embed v3 (`cohere.embed-english-v3`, 96 texts/call): blocked by
  `INVALID_PAYMENT_INSTRUMENT` — Marketplace subscription needs a valid
  payment instrument on the account. Code kept (`EMBED_PROVIDER=cohere`).
- Ollama (local, unlimited): nomic-embed-text baseline; bake-off below.

## E6 — table structure survival (the FY2025/FY2026 bug)

Symptom: model reported FY2026 revenue but hedged on FY2025 ("values not
clearly aligned"). Parser innocent — p.110 `text_as_html` has perfect
column alignment. Cause: we indexed flat `text`, which linearizes FY26
values, unit, FY25 header, FY25 values into one ambiguous stream.
Fix chain (each verified by re-ingest + re-ask):
1. Table HTML instead of flat text → retrieval collapsed (tag soup).
2. HTML→markdown grid + natural-language caption (columns + row labels) →
   p.110 retrievable but header split from rows by the 1,000-char splitter.
3. Tables as standalone docs, row-split with header+caption repeated per
   chunk + FY aliases (`FY2026` ↔ `year ended 31st March, 2026`) →
   p.110 rank still ~25/30 on dense (flat score band 0.64–0.71).
4. `top_k` 4 → 8 (1,289 chunks need more recall).
Root cause of ranking failure → E7.

## E7 — dense bake-off (same corpus, same 3 probes, retrieval-only)

Probes: revenue FY25/26 (gold p.110), standalone PAT FY26 (gold p.70),
service-revenue recognition (gold p.114). Metric: gold-page rank in top-30.

| Model | Dim | Revenue rank | PAT rank | Narrative rank | Ingest 1,312 ch |
|---|---|---|---|---|---|
| nomic-embed-text | 768 | ~25+ | — | — | ~25 s |
| bge-m3 | 1024 | 7 | 2 | 1 | ~80 s |
| qwen3-embedding:4b | 2560 | (preempted) | — | — | ~10 min |

Notes: bge-m3 clearly best of measured. qwen run preempted before scoring;
also exposed Qdrant 32 MB payload cap (2560-dim × 1,300 points ≈ 43 MB) →
upserts now batched (`UPSERT_BATCH_SIZE=200`).

## E8 — hybrid BM25 + RRF (bge-m3 dense)

Setup: `fastembed` BM25 (`Qdrant/bm25`), collection v3 with named vectors
`dense` + `bm25`, dual prefetch (20+20) + `FusionQuery(RRF)`, `top_k=8`.
Ranks (same probes): **p.110: 7 → 1**, p.70: 2, p.114: 1 — all top-4.
End-to-end answer now returns consolidated FY26 ₹3,513.26 cr / FY25
₹2,042.91 cr (p.110) plus standalone figures (p.81), correctly attributed —
the exact query that failed through E1–E7.

## E9 — RAGAS eval harness + baseline (bge-m3 + hybrid, Atlas prompt)

Setup: `evals/golden.py` (8 answer + 2 refusal probes on JioFin), judge =
Bedrock GLM 5, judge embeddings = local bge-m3. `uv run python
evals/run_evals.py [--probes ids]`; results JSON under `evals/results/`
with auto-delta vs previous run. Notes: ragas pinned `<0.4` (0.4 drops
LangChain judges); `langchain-community<0.4` (0.4.x removed the vertexai
module ragas imports).

| Metric | Baseline |
|---|---|
| faithfulness | 0.886 |
| answer_relevancy | 0.907 |
| context_precision | 0.721 |
| context_recall | 0.875 |
| refusals | 2/2 PASS |

Weakest axis is context_precision (0.72) — retrieval ranking still Nos. 1
lever for the next experiment round.

## E10 — cross-encoder rerank (bge-reranker-v2-m3), before/after

Prerequisite check first: recall audit over golden answer probes — all gold
pages present in hybrid top-30, but dividend (rank 10), total-expenses (10),
finance-costs (28) stranded outside the top-8 the LLM reads. Rerank is the
right surgery (retrieval already finds them).
Setup: hybrid RRF top-30 → `CrossEncoder(BAAI/bge-reranker-v2-m3)` rescore →
top-8 → LLM. `RERANK` toggle, `RERANK_TOPN=30`. Model cached in HF hub,
~1–3 s per query on 4 GB VRAM.

| Metric | BEFORE (no rerank) | AFTER (rerank) | Δ |
|---|---|---|---|
| context_precision | 0.736 | **0.926** | **+0.190** |
| answer_relevancy | 0.933 | 0.941 | +0.008 |
| faithfulness | 0.773 | 0.710 | −0.063 |
| context_recall | 0.875 | 0.750 | −0.125 |
| refusals | 1/2* | 2/2 | — |

*BEFORE refuse-poem "FAIL" was a rule-check miss (valid decline lacking our
markers), not a model failure — markers widened after.
Caveats: finance-costs now answers correctly from p.58 instead of gold p.110
(valid alternate citation, penalized by reference-bound recall); judge noise
is real (±0.11 faithfulness swings observed between identical-code runs), so
the faithfulness/recall dips need a repeat run before calling them signal.
Headline stands: precision +0.19, the exact axis reranking targets.

## Open items

- Footnote extraction unverified (0 across runs) — manual audit vs known pages.
- GPU quota case (`CASE_OPENED`, 4 vCPU G-family) pending — unblocks GPU
  hi_res measurement.
- qwen3-embedding:4b dense numbers never scored (rerunnable in one command).
- Celery + fleet parallelization parked (Phase 2).
- Per-upload strategy toggle + table toggle ideas parked.
