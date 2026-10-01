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

## E11 — user golden set (20 probes, GLM judge + deterministic checks)

Setup: `evals/run_user_eval.py` over `data/Eval Set` (2×10, manually verified
answers, physical-page citations). Per-probe disk cache
(`results/.cache_user_golden.json`, config-fingerprinted) after crashes forced
two full reruns. Guards added along the way: bge-m3 emits NaN for specific
token combos (Ollama 500s) → BM25-only retrieval fallback; `p1-q10` has no
`pdf_page` (unanswerable probe) → page checks tolerate missing gold.

| Metric | User set (20) |
|---|---|
| faithfulness | 0.906 |
| answer_relevancy | 0.756 |
| context_precision | 0.594 |
| context_recall | 0.800 |
| number_match | 0.65 (13/20) |
| page_exact / page±1 | 0.65 / 0.65 |

Misses cluster on pages 51/52/90/117/129 + number mismatches on q1/q4/q5/q6
— open triage: (a) right answer, neighboring duplicate page (standalone vs
consolidated tables repeat content), vs (b) truly absent. Precision 0.59
confirms E9's signal: ranking is still the #1 lever.

## E12 — query rewriting / expansion (deterministic, retrieval-only)

Hypothesis: queries use shorthand (FY26, PAT, ECL) while docs use long
forms, so appending long forms to the query should lift gold-page ranks.
Setup: `expand_query()` — FY patterns → "FY2026 year ended 31st March
2026", abbreviations (PAT→profit after tax, ECL→expected credit loss,
…) appended, original kept first. 5 miss-cluster probes
(p1-q5, p2-q2, p2-q3, p2-q4, p2-q9), `_hybrid_search` top-30,
prefetch 30, valid pages = gold + E11 alternates. No LLM (no $ cost).

| Probe | base rank | +full expansion | +abbrev-only |
|---|---|---|---|
| p1-q5 (Chairman) | 1 | 1 | 1 |
| p2-q2 (dividend) | 1 | 1 | 1 |
| p2-q3 (PAT) | 1 | **14** | 1 |
| p2-q4 (net loan) | 4 | 4 | 4 |
| p2-q9 (ECL stages) | 15 | 14–15 (noise) | 15 |

Verdict: **DROPPED, code reverted.** Full expansion regresses p2-q3
1→14; everything else ties or noise. Root cause is an interaction with
our own E6 fix: FY aliases were baked into *every* table caption, so
appending more FY terms boosts all table chunks equally and drowns the
discriminative signal ("profit after tax 1,560.90"). Abbrev-only variant
restores rank 1, proving the FY terms are the poison — but abbrev-only
is never *better* than baseline, so nothing ships.
Side finding: p2-q4 ("total net loan at March 31, 2026…") trips the
bge-m3 NaN overflow on the *base* query too (BM25-only fallback both
arms) — another instance of the E11 NaN class, not caused by expansion.

## E13 — TOP_K 8 → 12 (end-to-end, 5 probes)

Hypothesis (from HANDOFF §4.2): more context raises faithfulness or
adds noise. Setup: same 5 miss-cluster probes as E12,
`rag.query(top_k=12)` vs cached top-8 answers, rerank ON (30→k),
deterministic number/page checks. No code change (top_k param).

| Probe | top-8 pages | top-8 num | top-12 pages | top-12 num |
|---|---|---|---|---|
| p1-q5 | 55,81,83,104,109,110,111,140 | n/a (entity) | +59,60,65,67 | n/a (entity) |
| p2-q2 | 3,68,70,94,99,110,111,140 | True | +82 | True |
| p2-q3 | 30,58,70,124,139 | True | +89,137 | True |
| p2-q4 | 88,95,101,124,132,139 | **False** (quoted 224.07) | +92,**131** | **True** |
| p2-q9 | 85,87,106,115,118,131,139 | True | +37 | True |

Verdict: **SHIP (default TOP_K 8→12).** p2-q4 fixed — p.131 with the
correct loan table (25,710.80) makes the cut and the LLM quotes it;
no probe regressed, no noise observed on these 5. Cost: +50% LLM
context tokens; rerank cost unchanged (still scores 30). Caveats:
faithfulness not RAGAS-measured on this subset (judge noise ±0.1);
side observation — CPU rerank of 30 docs takes 60–185 s/query and
dominates end-to-end latency, worth a Prometheus look before any
RERANK_TOPN increase.

## E14 — Rerank truncation proof: LLM-as-judge + end-to-end (512 tokens)

Hypothesis (from HANDOFF Lever 1): "Most relevance signal is in the first
512 tokens; bge-reranker-v2-m3 supports 8192 but the tail adds compute, not
signal." If true, truncating docs to 512 tokens before rerank cuts O(n^2)
attention cost on long table chunks with minimal accuracy loss.

Setup: 20-question user golden set (data/Eval Set, 2x10). For each question,
hybrid RRF retrieves top-30 docs. GLM-5 (Bedrock Converse, same model as RAG)
judges each (question, doc) pair's relevance on a 1-5 scale, twice:
- Arm `full`  — judge sees the full doc content.
- Arm `trunc` — judge sees only the first 512 tokens (bge-m3 tokenizer).
Phase 1: per-question Spearman rho across the 30 docs + top-8 Jaccard.
Phase 2: top-8 by each arm's judge scores -> generate -> RAGAS + deterministic.
Cache: per-(q, doc_hash, arm) in `evals/results/.cache_truncation_exp.json`.
Retries: throttle-only, max 3, exp backoff 2/4/8s. Heartbeat every 15s.
Script: `evals/run_truncation_exp.py` + wrapper `run_truncation_exp.ps1`.

Doc length distribution: 17/20 questions had >=1 doc > 512 tokens (89 docs
truncated total, ~15% of the pool). Max token length: 1925 (p1-q10).

| Phase 1 (LLM judge) | Value |
|---|---|
| Mean Spearman rho (all docs) | **0.7788** |
| Mean top-8 Jaccard | **0.7121** |
| Mean rho on actually-truncated docs only | 0.5539 |
| Questions | 20 |

| Phase 2 (RAGAS) | full | trunc | delta |
|---|---|---|---|
| faithfulness | 0.9093 | 0.9385 | +0.0292 |
| answer_relevancy | 0.8898 | 0.8074 | **-0.0824** |
| context_precision | 0.8912 | 0.8296 | **-0.0616** |
| context_recall | 0.875 | 0.875 | 0.0 |

| Phase 2 (deterministic) | full | trunc |
|---|---|---|
| number_match | 0.80 | 0.80 |
| page_exact | 0.95 | 0.90 |
| page_pm1 | 0.95 | 0.90 |

Verdict: **PARTIALLY SUPPORTED.** The broad relevance signal IS in the
first 512 tokens (rho 0.78, top-8 Jaccard 0.71), but it is not complete.
On the docs that were actually truncated (token_len > 512), rho drops to
0.55 — the tail carries real signal for ~15% of the pool, mostly long
table chunks. End-to-end impact is small but non-zero: answer_relevancy
-8%, context_precision -6%, 1 question lost page_match (p1-q7). The
faithfulness +3% and context_recall 0% are within RAGAS judge noise
(+-0.1, see E10). Net: truncating to 512 tokens trades ~6-8% on the
retrieval-quality axes for the O(n^2) rerank speedup. Acceptable for the
latency goal (23s -> 8-10s) if the 1-question page regression is tolerated;
otherwise a higher cutoff (e.g. 1024 tokens) should be tested.

Outliers worth noting:
- p1-q10 (rho=0.0, max_tok=1925): longest-doc question; truncation fully
  reorders the judge ranking. Both arms fail this probe (it is the
  unanswerable FY2021-22 question) so no end-to-end impact.
- p2-q10 (cut_rho=-0.1391): truncated-doc scores anti-correlate with
  full-doc scores. The tail flips the judge's assessment for some docs.
- p1-q9 (rho=1.0, n_cut=10): perfect correlation even with 10 truncated
  docs — when the relevant content is in the head, truncation is free.

Caveats: LLM judge noise is real (E10 measured +-0.11 faithfulness swings).
The 0.78 rho is a single-run point estimate; a repeat run would tighten
the band. The judge sees the doc as text (no table structure metadata),
so the "tail signal" may partly be the judge reading table rows that
the cross-encoder would also weight.

## Open items

- Footnote extraction unverified (0 across runs) — manual audit vs known pages.
- GPU quota case (`CASE_OPENED`, 4 vCPU G-family) pending — unblocks GPU
  hi_res measurement.
- qwen3-embedding:4b dense numbers never scored (rerunnable in one command).
- Celery + fleet parallelization parked (Phase 2).
- Per-upload strategy toggle + table toggle ideas parked.
- E14 extension: fp16 reranker (Lever 2) — rerun E14 truncation with the
  reranker in fp16 to measure the combined speedup vs accuracy trade.
- E14 extension: test 1024-token cutoff as a middle ground (512 loses
  ~6-8% on retrieval quality; 1024 may recover most of it at half the
  rerank speedup).
