"""Qdrant-backed semantic cache: question embedding -> (answer, sources).

No extra infra (no Redis): a second Qdrant collection holding one point per
answered question. Lookup is a dense cosine search; a hit returns the cached
answer without running retrieval + LLM generation.

Invalidation (v1): full flush on any ingest / clear. TTL is enforced as a
timestamp filter on lookup (Qdrant has no native per-point TTL) + best-effort
cleanup of expired points on store.
"""
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from app import config
from app.logging_setup import step_logger

_executor = ThreadPoolExecutor(max_workers=1)


def _dim() -> int:
    # Single source of truth lives in rag.embedding_dim(); lazy import
    # avoids a module-level circular import (rag imports this module).
    from app import rag

    return rag.embedding_dim()


def _client():
    from app import rag

    return rag.get_qdrant_client()


def _collection() -> str:
    return config.CACHE_COLLECTION


def ensure_cache_collection() -> None:
    """Create the cache collection, recreating it when the dense dim changed.

    Mirrors rag.ensure_collection(): switching embedding providers makes old
    vectors unreadable, so a mismatch wipes + recreates rather than failing.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse

    log = step_logger("cache")
    client = _client()
    need = _dim()
    name = _collection()
    ok = False
    if client.collection_exists(name):
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            dense = vectors.get("dense")
            ok = getattr(dense, "size", None) == need
        if not ok:
            log.warning(
                f"cache collection layout mismatch (need dense={need}) "
                "— recreating cache collection, old cached answers dropped"
            )
            client.delete_collection(name)
    if not ok:
        log.info(
            f"creating cache collection {name} "
            f"(dense dim={need}, provider={config.EMBED_PROVIDER})"
        )
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(size=need, distance=Distance.COSINE)
            },
        )
        try:
            client.create_payload_index(
                collection_name=name,
                field_name="created_at",
                field_schema=PayloadSchemaType.FLOAT,
            )
        except (UnexpectedResponse, Exception):
            # Index creation races / older servers: non-fatal, filter still works.
            pass


def _ttl_filter(now: float):
    if config.CACHE_TTL_SECONDS <= 0:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="created_at",
                range=Range(gte=now - config.CACHE_TTL_SECONDS),
            )
        ]
    )


def lookup(question_embedding: list) -> tuple[str, list] | None:
    """Return (answer, sources) on a semantic hit, else None.

    Never raises: cache is a pure optimization, any failure is a miss.
    Qdrant COSINE returns similarity directly (1.0 = identical).
    """
    log = step_logger("cache")
    if not config.CACHE_ENABLED:
        return None
    try:
        client = _client()
        name = _collection()
        if not client.collection_exists(name):
            return None
        res = client.query_points(
            collection_name=name,
            query=list(question_embedding),
            using="dense",
            query_filter=_ttl_filter(time.time()),
            limit=max(config.CACHE_TOP_K, 1),
            with_payload=True,
        )
        if not res.points:
            return None
        top = res.points[0]
        score = float(top.score)
        if score < config.CACHE_THRESHOLD:
            return None
        payload = top.payload or {}
        answer = payload.get("answer")
        if not answer:
            return None
        try:
            sources = json.loads(payload.get("sources", "[]"))
        except (json.JSONDecodeError, TypeError):
            sources = []
        log.info(
            f"step=cache HIT score={score:.4f} "
            f"q={str(payload.get('question', ''))[:80]!r}"
        )
        return answer, sources
    except Exception:
        log.exception("cache lookup failed (treating as MISS)")
        return None


def _store_sync(
    question: str,
    question_embedding: list,
    answer: str,
    sources: list,
    source_tags: list | None = None,
) -> None:
    log = step_logger("cache")
    ensure_cache_collection()
    now = time.time()
    norm = " ".join(question.strip().lower().split())
    point_id = uuid.uuid5(
        uuid.NAMESPACE_URL, f"{_collection()}:{norm}").hex
    payload = {
        "question": question,
        "answer": answer,
        "sources": json.dumps(sources or []),
        "created_at": now,
        "source_tags": source_tags or [],
    }
    _client().upsert(
        collection_name=_collection(),
        points=[
            PointStruct(
                id=point_id,
                vector={"dense": list(question_embedding)},
                payload=payload,
            )
        ],
    )
    try:
        from app import metrics as _m

        _m.CACHE_STORES.inc()
    except Exception:
        pass
    # Best-effort expiry of stale points (lookup filter is the real guarantee).
    if config.CACHE_TTL_SECONDS > 0:
        try:
            from qdrant_client.http.models import FilterSelector

            _client().delete(
                collection_name=_collection(),
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="created_at",
                                range=Range(
                                    lt=now - config.CACHE_TTL_SECONDS),
                            )
                        ]
                    )
                ),
            )
        except Exception:
            pass
    log.info(f"step=cache STORE q={question[:80]!r}")


def store(question, question_embedding, answer, sources, source_tags=None) -> None:
    """Fire-and-forget write: never blocks the response, never raises."""
    if not config.CACHE_ENABLED:
        return
    if not answer:
        return
    try:
        _executor.submit(
            _store_sync, question, list(question_embedding),
            answer, sources, source_tags or [])
    except Exception:
        step_logger("cache").exception("cache store submit failed")


def clear() -> None:
    """Full flush (called on ingest / clear_collection). Synchronous."""
    log = step_logger("cache")
    try:
        client = _client()
        if client.collection_exists(_collection()):
            client.delete_collection(_collection())
            log.info("step=cache CLEAR done (full flush)")
    except Exception:
        log.exception("cache clear failed")
