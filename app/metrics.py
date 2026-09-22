"""Prometheus metrics: latency per pipeline step + request/error counters.

Import-safe: creating metrics at import time, served from /metrics.
Steps mirror rag.query(): retrieve -> rerank -> generate,
plus ingest: load_pdf -> split -> embed -> upsert.
"""
import time
from contextlib import contextmanager
from functools import wraps

from prometheus_client import Counter, Gauge, Histogram

# Latency per step (seconds). Buckets cover 10ms (BM25) to 5min (hi_res ingest).
STEP_LATENCY = Histogram(
    "rag_step_seconds",
    "Latency per RAG pipeline step",
    labelnames=("step",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)

REQUESTS = Counter(
    "rag_requests_total",
    "API requests",
    labelnames=("endpoint", "status"),
)

ERRORS = Counter(
    "rag_errors_total",
    "Pipeline errors",
    labelnames=("step",),
)

CHUNKS_INDEXED = Counter(
    "rag_chunks_indexed_total",
    "Chunks upserted to Qdrant",
)

QDRANT_UP = Gauge(
    "rag_qdrant_up",
    "1 if Qdrant reachable on last health check, else 0",
)


@contextmanager
def time_step(step: str):
    """Time a pipeline step; on exception count an error then re-raise."""
    start = time.perf_counter()
    try:
        yield
    except Exception:
        ERRORS.labels(step=step).inc()
        raise
    finally:
        STEP_LATENCY.labels(step=step).observe(time.perf_counter() - start)


def timed(step: str):
    """Decorator version of time_step for functions."""

    def deco(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            with time_step(step):
                return fn(*args, **kwargs)

        return inner

    return deco
