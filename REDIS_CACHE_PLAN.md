# Redis Semantic Cache — Implementation Plan

Status: **PLANNED, not yet implemented.** Decisions locked: full-flush
invalidation on any ingest/clear; implement on next session.

## Goal

Cache LLM responses keyed on the **question embedding**. On `/api/ask`,
do a vector search against Redis; if the closest cached question is above
a cosine-similarity threshold, return its cached answer immediately.
Otherwise run the full RAG pipeline and store the result asynchronously.

## Flow

```
/api/ask (question)
  └─ rag.query(question)
       ├─ 1. embed question ONCE (reused by cache + Qdrant)
       ├─ 2. cache.lookup(question, embedding)
       │     └─ RedisVL vector search, cosine sim ≥ CACHE_THRESHOLD → HIT
       │         HIT  → return (cached answer, cached sources)        [done]
       │         MISS ↓
       ├─ 3. retrieve (Qdrant hybrid) → rerank → generate (existing)
       ├─ 4. fire-and-forget cache.store(question, embedding, answer, sources)
       └─ return (answer, sources)
```

Cache key = question embedding only (not question+context). One embed
serves both the cache lookup and the Qdrant retrieve, so no extra embed
cost on a miss.

## Components

| File | Change | Notes |
|---|---|---|
| `app/cache.py` | **NEW** | RedisVL `SearchIndex`, `lookup()`, `store()`, `clear()`, `ensure_index()` |
| `app/rag.py` | **EDIT** | `query()` wraps with cache; refactor `_hybrid_search` to accept precomputed `dense_vec` so we don't embed twice |
| `app/config.py` | **EDIT** | `CACHE_*` env vars |
| `app/metrics.py` | **EDIT** | `CACHE_HITS`, `CACHE_MISSES`, `CACHE_STORES` counters + `cache_lookup` latency |
| `docker-compose.yml` | **EDIT** | add `redis` service (redis-stack with vector search) |
| `requirements.txt` / `pyproject.toml` | **EDIT** | `redisvl>=0.4`, `redis>=5.0` |
| `.env.example` | **EDIT** | document `CACHE_*` |

## `app/cache.py` shape

```python
# Singleton SearchIndex. Schema: question(text), answer(text),
# sources(text=JSON), embedding(vector, dims=embedding_dim()),
# created_at(numeric), source(text tag for invalidation).
def ensure_index()           # create/overwrite if dims changed (mirrors ensure_collection)
def get_index() -> SearchIndex
def lookup(question, embedding) -> (answer, sources) | None
def store(question, embedding, answer, sources, source_tags)  # non-blocking
def clear()                   # flush cache (called on ingest / clear_collection)
```

- `lookup`: `VectorQuery(return_fields=[...])` → take top-1 → convert
  cosine **distance** to similarity (`1 - distance`) → if
  `sim ≥ CACHE_THRESHOLD` return payload, else `None`.
- `store`: submitted to a module-level `ThreadPoolExecutor(max_workers=1)`
  — fire-and-forget, never blocks the response, errors logged not raised.
  `sources` serialized as JSON. TTL via `index.load(..., ttl=CACHE_TTL_SECONDS)`.

## Config additions (`app/config.py`)

```python
CACHE_ENABLED      = os.getenv("CACHE_ENABLED", "true").lower() == "true"
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_INDEX        = os.getenv("CACHE_INDEX", "rag_cache")
CACHE_THRESHOLD    = float(os.getenv("CACHE_THRESHOLD", "0.92"))   # cosine sim
CACHE_TTL_SECONDS  = int(os.getenv("CACHE_TTL_SECONDS", "604800"))  # 7 days
CACHE_TOP_K        = int(os.getenv("CACHE_TOP_K", "1"))             # candidates to fetch
```

Threshold 0.92 is a starting point — tunable. bge-m3 separates
paraphrases around 0.85–0.95; 0.92 is conservative (few false hits,
decent recall). Tune against the eval suite.

## `query()` modification (the core edit)

```python
def query(question, top_k=None):
    k = top_k or config.TOP_K
    log = step_logger("query")

    # embed once — reused by cache + Qdrant
    q_vec = _finite(get_embeddings().embed_query(question), "query")

    # 1. cache lookup
    if config.CACHE_ENABLED:
        with metrics.time_step("cache_lookup"):
            hit = cache.lookup(question, q_vec)
        if hit is not None:
            metrics.CACHE_HITS.inc()
            log.info("step=cache HIT")
            return hit
        metrics.CACHE_MISSES.inc()
        log.info("step=cache MISS")

    # 2. normal pipeline (pass q_vec into _hybrid_search to avoid re-embed)
    docs = _hybrid_search(question, fetch_k, dense_vec=q_vec)
    ... rerank ...
    ... generate ...
    answer, sources = resp.content, [...]

    # 3. fire-and-forget store
    if config.CACHE_ENABLED:
        cache.store(question, q_vec, answer, sources, source_tags=...)
    return answer, sources
```

## docker-compose.yml addition

```yaml
  redis:
    image: redis/redis-stack:latest   # RediSearch + vector search built in
    container_name: rag-redis
    ports: ["6379:6379"]
    volumes: [redis_data:/data]
    restart: unless-stopped
```

Add `redis_data` to the `volumes:` block.

## Edge cases & decisions

1. **Dimension mismatch on provider switch** — `ensure_index()` detects
   `embedding_dim()` change and recreates the index (mirrors the existing
   `ensure_collection()` pattern in `rag.py:201`). Old cached answers
   dropped, same as Qdrant.
2. **Corpus invalidation — FULL FLUSH (DECIDED)** — on `_reset_source(name)`
   and `clear_collection()`, call `cache.clear()`. v1 = full flush (simple,
   safe). v2 (follow-up) = tag cache entries with source names, invalidate
   only matching entries via Redis query filter.
3. **Stale answers after re-upload** — TTL is the safety net; full-clear on
   ingest is the immediate guarantee.
4. **Async store failure** — logged via `step_logger("cache").exception`,
   never raises, response already returned. Cache is a pure optimization;
   a store failure must not break a successful answer.
5. **Concurrency / duplicate stores** — if two identical questions race,
   both miss, both store. Redis upserts by key (idempotent) — no
   corruption, last writer wins. Fine.
6. **Threshold tuning** — start 0.92, re-run
   `uv run python evals/run_user_eval.py` and watch faithfulness. If
   false hits appear, raise to 0.95; if recall poor, lower to 0.88. The
   eval suite already exists and will catch regressions.
7. **NaN embedding** — `_finite()` already sanitizes bge-m3 NaN dims; the
   same sanitized vector goes to both cache and Qdrant, so cache key is
   consistent.

## Implementation order

1. Add Redis to `docker-compose.yml` + `docker compose up -d redis`
2. Add `redisvl` to `requirements.txt` / `pyproject.toml` + `uv sync`
3. `app/config.py` — `CACHE_*` vars
4. `app/metrics.py` — cache counters
5. `app/cache.py` — full module + `ensure_index` mirroring `ensure_collection`
6. `app/rag.py` — refactor `_hybrid_search(question, k, dense_vec=None)`,
   wrap `query()` with lookup/store, hook `clear()` into `_reset_source` +
   `clear_collection`
7. `.env.example` — document new vars
8. Verify: `GET /health` (add `cache_ok`), run eval suite, check
   `rag_cache_*` metrics at `:9090`

## Verification

- Hit path: ask same question twice → second call's `cache_lookup` step
  latency drops to ~ms, `generate` step absent in `/metrics`.
- Threshold boundary: paraphrase question → still hits; unrelated
  question → misses.
- Eval regression: `uv run python evals/run_user_eval.py` — faithfulness
  must not drop (if it does, threshold too low).
- Prometheus: `rag_cache_hits_total` / `rag_cache_misses_total` /
  `rag_cache_store_seconds`.
