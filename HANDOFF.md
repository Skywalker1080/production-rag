# Project Handoff — PDF RAG on Financial Reports

Date: 2026-09-22. Read this + `EXPERIMENTS.md` (E1–E11 lab log) to continue.
Repo: `C:\Projects\RAG` (git log is the logbook; `data/`, `.env`, logs ignored).

## 1. System snapshot

- **Pipeline:** EC2 `hi_res` parse (8,135 elements, S3) → page-grouped markdown
  chunks (tables as captioned grids, headers repeated) → bge-m3 dense (1024)
  + BM25 sparse → local Qdrant RRF hybrid → rerank top-30→8
  (`bge-reranker-v2-m3`) → Bedrock GLM 5 (`zai.glm-5`, Atlas role prompt).
- **Corpus indexed:** `jiofin financial report.pdf`, 1,312 chunks, hybrid on.
- **Rerank:** ON (`RERANK=true`). Toggle via `.env`.
- **UI:** React `frontend/` served from FastAPI `:8000` (rebuild `dist/` after
  UI edits). Legacy debug page at `app/static/` is fallback only.
- **Eval:** `uv run python evals/run_evals.py` (own 10-probe set),
  `uv run python evals/run_user_eval.py` (user 20-probe set, cached+resumable).
  Judge = GLM 5, judge-embeddings = local bge-m3.
- **Infra:** EC2 `i-09f713876327b16b5` STOPPED (keep EBS). S3
  `rag-extract-405633560616-use1` (in/ + out/). GPU quota case OPEN.
  Qdrant + Ollama run locally via docker/daemon.

## 2. Metrics dashboard (latest)

| Suite | faithful | relevancy | precision | recall | extra |
|---|---|---|---|---|---|
| Own 10, rerank ON (E10) | 0.710 | 0.941 | **0.926** | 0.750 | refusals 2/2 |
| User 20 (E11) | **0.906** | 0.756 | 0.594 | 0.800 | numbers 0.65, pages 0.65 |

Judge noise is ±0.1 run-to-run (observed) — treat small deltas as noise.

## 3. Findings that constrain next work

1. **Ranking, not retrieval, is the bottleneck.** All gold pages reach top-30;
   misses sit at ranks 10–28. Precision 0.59–0.74 across suites.
2. **Hybrid + rerank compose:** p.110 revenue query went rank ~25 → 7 (hybrid)
   → 1 (rerank). Keep both.
3. **Table chunks need three properties** (all implemented): markdown grid
   (not HTML soup, not flat text), NL caption + FY aliases, header repeated
   per row-split chunk. Removing any one regresses p.110 out of top-10.
4. **Provider walls mapped:** Titan throttles (~1.6 s/call); Gemini free tier
   = 1,000 req/day (blown once already); Cohere needs Marketplace payment
   instrument (code ready, `EMBED_PROVIDER=cohere`); bge-m3 emits NaN for rare
   token combos → BM25-only fallback exists in `_hybrid_search`.
5. **hi_res CPU floor:** ~30 s/page single-core, 92% parallel efficiency at
   8 workers, 143 pp = 667 s. Footnotes: 0 detected (unverified).
6. **Miss cluster (E11):** pages 51/52/90/117/129 — neighbors retrieved.
   Untriage: offset vs duplicate-content vs truly-absent.

## 4. Next actions (priority order)

1. **Miss triage (30 min, no code):** hand-check 2–3 of {p1-q5, p2-q4, p2-q9}
   answers vs gold. Decides gold-tolerance fix vs retrieval fix.
2. **TOP_K sweep 8→12→16** (`TOP_K` env, evals rerun each): tests whether more
   context raises faithfulness or adds noise. No re-ingest needed.
3. **Footnote audit:** find a known footnote page, check parse output
   (`dump_tables.py` pattern), close the last P0 gap.
4. **qwen3-embedding:4b scoring:** one command bake-off rerun (upsert-batch
   fix already in) if dense quality is questioned again.
5. **GPU quota:** case open; when approved, measure hi_res on `g4dn.xlarge`.
6. **Phase 2 (only if volume demands):** Celery fleet, per-upload strategy
   toggle, ColBERT late-interaction.

## 5. Commands

```powershell
uv run uvicorn app.main:app --reload --port 8000   # backend (:8000 = UI)
docker compose up -d                                # qdrant
ollama serve                                        # embeddings daemon
uv run python evals/run_evals.py                    # own suite
uv run python evals/run_user_eval.py                # user suite (resumable)
aws ec2 start-instances --instance-ids i-09f713876327b16b5  # extractor box
```

Secrets live only in `.env` (never committed): `AWS_BEARER_TOKEN_BEDROCK`,
`GOOGLE_API_KEY`, `HF_TOKEN` (optional).
