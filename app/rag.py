"""LangChain RAG service: PDF -> chunks -> Bedrock embeddings -> Qdrant -> Bedrock GLM 5.

Auth: prefers Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) if set,
falls back to classic AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (SigV4).
"""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_aws.utils import create_aws_client
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app import config
from app.delta import chunk_point_id, is_legacy_point, page_hash, plan_delta
from app.logging_setup import step_logger


def _bearer_key() -> str:
    return (config.AWS_BEARER_TOKEN_BEDROCK or "").strip()


def _is_throttle(exc: BaseException) -> bool:
    """Only retry rate limits — never mask auth/model errors with backoff."""
    name = type(exc).__name__.lower()
    code = ""
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = str(resp.get("Error", {}).get("Code", "")).lower()
    text = f"{name} {code} {exc}".lower()
    return (
        "throttl" in text
        or "toomany" in text
        or "rate_exceeded" in text
        or "resourceexhausted" in text.replace(" ", "").replace("_", "")
        or " 429" in f" {text}"
    )


def _retry_throttle(fn):
    """Throttle-only retry decorator (never masks auth/model errors)."""
    return retry(
        retry=retry_if_exception(_is_throttle),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, max=60),
        reraise=True,
    )(fn)


def embedding_dim() -> int:
    """Vector size for the active provider (must match Qdrant collection)."""
    if config.EMBED_PROVIDER == "ollama":
        return config.OLLAMA_EMBED_DIM
    if config.EMBED_PROVIDER == "gemini":
        return config.GEMINI_EMBED_DIM  # gemini-embedding-001, truncated
    return 1024  # titan-embed-text-v2, cohere-embed-v3


def get_embeddings():
    """Cohere (Bedrock batch), Gemini, local Ollama, or Bedrock Titan."""
    if config.EMBED_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBED_MODEL,
            task_type="retrieval_document",
            output_dimensionality=config.GEMINI_EMBED_DIM,
        )
    if config.EMBED_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    if config.EMBED_PROVIDER == "cohere":
        # Queries are single texts: langchain's cohere path is fine.
        # Ingest batching bypasses it via _cohere_batch_invoke below.
        return _get_bedrock_embeddings(config.COHERE_EMBED_MODEL_ID)
    return _get_bedrock_embeddings(config.BEDROCK_EMBED_MODEL_ID)


def _get_bedrock_embeddings(model_id: str) -> BedrockEmbeddings:
    key = _bearer_key()
    if key:
        # BedrockEmbeddings has no bedrock_api_key field in this version,
        # so build a bearer-authed boto3 client and inject it.
        client = create_aws_client(
            service_name="bedrock-runtime",
            region_name=config.AWS_REGION,
            api_key=SecretStr(key),
        )
        return BedrockEmbeddings(
            client=client,
            model_id=model_id,
            region_name=config.AWS_REGION,
        )
    return BedrockEmbeddings(
        model_id=model_id,
        region_name=config.AWS_REGION,
    )


def _bedrock_runtime_client():
    """Raw boto3 bedrock-runtime client (bearer or SigV4) for batch calls."""
    key = _bearer_key()
    if key:
        return create_aws_client(
            service_name="bedrock-runtime",
            region_name=config.AWS_REGION,
            api_key=SecretStr(key),
        )
    import boto3

    return boto3.client("bedrock-runtime", region_name=config.AWS_REGION)


@retry(
    retry=retry_if_exception(_is_throttle),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, max=60),
    reraise=True,
)
def _cohere_batch_invoke(texts: list) -> list:
    """One InvokeModel with up to ~96 texts (Cohere native batching)."""
    import json

    client = _bedrock_runtime_client()
    resp = client.invoke_model(
        modelId=config.COHERE_EMBED_MODEL_ID,
        body=json.dumps({"texts": texts, "input_type": "search_document"}),
    )
    return json.loads(resp["body"].read())["embeddings"]


def get_llm() -> ChatBedrockConverse:
    # GLM 5 (zai.glm-5) is served via the Bedrock Converse API.
    # NOTE: legacy ChatBedrock uses the old InvokeModel path and raises
    # "Provider zai model does not support chat" — ChatBedrockConverse works.
    key = _bearer_key()
    kwargs: dict = {
        "model_id": config.BEDROCK_MODEL_ID,
        "region_name": config.AWS_REGION,
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    if key:
        kwargs["bedrock_api_key"] = key
    return ChatBedrockConverse(**kwargs)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=config.QDRANT_COLLECTION,
        embedding=get_embeddings(),
        vector_name="dense",
    )


_bm25_model = None


def get_bm25():
    """Local BM25 sparse encoder (no server, no quota). Singleton."""
    global _bm25_model
    if _bm25_model is None:
        from fastembed import SparseTextEmbedding

        _bm25_model = SparseTextEmbedding("Qdrant/bm25")
    return _bm25_model


def _to_sparse(vec) -> "SparseVector":
    from qdrant_client.http.models import SparseVector

    return SparseVector(
        indices=[int(i) for i in vec.indices],
        values=[float(v) for v in vec.values],
    )


def ensure_collection() -> None:
    """Create the collection (dense + BM25 sparse), recreating it when the
    dense dim changed or the layout is legacy (unnamed single vector).

    Switching providers makes old vectors unreadable, so a mismatch wipes +
    recreates rather than failing obscurely.
    """
    from qdrant_client.http.models import SparseVectorParams

    log = step_logger("qdrant")
    client = get_qdrant_client()
    need = embedding_dim()
    ok = False
    if client.collection_exists(config.QDRANT_COLLECTION):
        info = client.get_collection(config.QDRANT_COLLECTION)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            dense = vectors.get("dense")
            sparse = info.config.params.sparse_vectors or {}
            ok = getattr(dense, "size", None) == need and "bm25" in sparse
        if not ok:
            log.warning(
                f"collection layout mismatch (need dense={need}+bm25) "
                "— recreating collection, old vectors dropped"
            )
            client.delete_collection(config.QDRANT_COLLECTION)
    if not ok:
        log.info(
            f"creating collection {config.QDRANT_COLLECTION} "
            f"(dense dim={need} + bm25 sparse, provider={config.EMBED_PROVIDER})"
        )
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(size=need, distance=Distance.COSINE)
            },
            sparse_vectors_config={"bm25": SparseVectorParams()},
        )


SKIP_CATEGORIES = {"Header", "Footer", "PageBreak", "Advertisement"}


def html_table_to_markdown(html: str) -> str:
    """Convert a <table> grid to GitHub markdown (row/col alignment kept,
    ~3x fewer tokens than raw HTML, embedding-friendly plain words)."""
    from html.parser import HTMLParser

    rows: list = []
    cur_row: list = []
    cur_cell: list = []
    cell_span = 1
    in_cell = False

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            nonlocal in_cell, cell_span
            if tag == "tr":
                cur_row.clear()
            elif tag in ("td", "th"):
                in_cell = True
                cur_cell.clear()
                cell_span = 1
                for k, v in attrs:
                    if k == "colspan":
                        try:
                            cell_span = max(int(v), 1)
                        except ValueError:
                            pass

        def handle_data(self, data):
            if in_cell:
                cur_cell.append(data.strip())

        def handle_endtag(self, tag):
            nonlocal in_cell
            if tag in ("td", "th"):
                in_cell = False
                text = " ".join(t for t in cur_cell if t)
                cur_row.extend([text] * cell_span)
            elif tag == "tr":
                rows.append(list(cur_row))

    P().feed(html)
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # Natural-language caption: grids alone embed poorly ("| 1,901.87 |"),
    # so name the columns + row labels up front for retrieval.
    headers = [c for c in rows[0] if c][:6]
    labels = [r[0] for r in rows[1:] if r and r[0]][:15]
    caption = "Table"
    if headers:
        caption += " with columns: " + " | ".join(headers)
    if labels:
        caption += ". Row labels include: " + "; ".join(labels)
    # Year aliases: queries say "FY2026" while headers say "year ended 31st
    # March, 2026" (sometimes OCR-garbled). Index both forms.
    import re

    years = sorted({m.group(0) for m in re.finditer(r"(?:19|20)\d{2}", caption)})
    if years:
        caption += ". Fiscal years mentioned: " + ", ".join(f"FY{y}" for y in years)
    out = [caption,
           "| " + " | ".join(rows[0]) + " |",
           "|" + "|".join(["---"] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def group_page_docs(items: list, name: str) -> list:
    """Group (category, page, text, html) rows into one doc per page.

    Drops running Header/Footer junk and normalizes page numbers, so crumbs
    (median ~10 chars) become page-sized docs before splitting.
    Table elements contribute their HTML grid instead of flat text, so
    row/column alignment survives into chunks for the LLM.
    """
    by_page: dict = {}
    order: list = []
    tables: list = []
    for category, page, text, html in items:
        if category in SKIP_CATEGORIES:
            continue
        if page is None:
            page = "?"
        # Tables stand alone (never merged into page prose) so the
        # row-splitter below can repeat headers in every chunk.
        if category == "Table" and html:
            grid = html_table_to_markdown(html)
            if grid.strip():
                tables.append(
                    Document(
                        page_content=grid,
                        metadata={"source": name, "page": page, "table": True},
                    )
                )
            continue
        body = (text or "").strip()
        if not body:
            continue
        if page not in by_page:
            by_page[page] = []
            order.append(page)
        by_page[page].append(body)
    docs = [
        Document(
            page_content="\n".join(by_page[p]),
            metadata={"source": name, "page": p, "table": False},
        )
        for p in order
    ]
    return docs + tables


def split_table_doc(doc, max_rows: int = 12):
    """Split a markdown-grid doc by rows, repeating caption+header in each
    chunk so split tables never lose their FY column labels."""
    lines = doc.page_content.split("\n")
    head, rest = lines[:3], [l for l in lines[3:] if l.strip()]
    chunks = []
    for i in range(0, len(rest), max_rows):
        chunks.append(
            Document(
                page_content="\n".join(head + rest[i : i + max_rows]),
                metadata=dict(doc.metadata),
            )
        )
    return chunks or [doc]


def _reset_source(name: str) -> None:
    """Ensure collection + drop this source's old points (re-upload replaces)."""
    ensure_collection()
    get_qdrant_client().delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=name))]
        ),
    )
    # Corpus changed -> cached answers may be stale. v1: full flush.
    try:
        from app import cache as _cache

        _cache.clear()
    except Exception:
        step_logger("cache").exception("cache clear on ingest failed")


def split_docs(docs: list, name: str, emit, log) -> list:
    """Split page docs into chunks (no Qdrant writes)."""
    try:
        emit("split", "splitting into chunks…")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(
            [d for d in docs if not d.metadata.get("table")]
        )
        for t in (d for d in docs if d.metadata.get("table")):
            chunks.extend(split_table_doc(t))
        emit("split", f"split into {len(chunks)} chunks")
    except Exception:
        log.exception(f"FAILED step=split file={name}")
        raise

    for d in chunks:
        d.metadata["source"] = name
    return chunks


def embed_and_upsert(chunks: list, name: str, emit, log) -> int:
    """Embed chunks + upsert to Qdrant. Returns count."""
    if not chunks:
        emit("done", "no extractable text — nothing to index")
        return 0

    # Step: embed (parallel) + upsert to Qdrant
    # Remote calls take time each; ThreadPool cuts it ~MAX_WORKERS x.
    # Point IDs are deterministic so re-ingesting overwrites, never dupes.
    try:
        total = len(chunks)
        embedding = get_embeddings()
        vectors: list = [None] * total

        @retry(
            retry=retry_if_exception(_is_throttle),
            stop=stop_after_attempt(6),
            wait=wait_exponential(multiplier=2, max=30),
            reraise=True,
        )
        def _embed_one(text: str):
            return _finite(embedding.embed_query(text), "chunk")

        if config.EMBED_PROVIDER in ("gemini", "cohere"):
            # Batched providers (1 request / N texts). Slow + steady:
            # small sequential batches + pacing sleeps stay under quotas
            # instead of hammering 1-call-per-chunk with retries.
            import time as _time

            if config.EMBED_PROVIDER == "cohere":
                batch = config.COHERE_BATCH_SIZE

                def _batch_fn(texts: list):
                    return _cohere_batch_invoke(texts)
            else:
                batch = config.EMBED_BATCH_SIZE

                @_retry_throttle
                def _batch_fn(texts: list):
                    return embedding.embed_documents(texts)

            done = 0
            emit("upsert", f"embedding {total} chunks (batches of {batch})…")

            for s in range(0, total, batch):
                vecs = _batch_fn([c.page_content for c in chunks[s : s + batch]])
                vecs = [_finite(v, f"batch@{s}") for v in vecs]
                vectors[s : s + len(vecs)] = vecs
                done += len(vecs)
                emit("upsert", f"embedded {done}/{total}…")
                _time.sleep(config.EMBED_PACING_SECS)
        else:
            workers = config.EMBED_WORKERS
            emit("upsert", f"embedding {total} chunks ({workers} parallel)…")

            def _embed(i: int):
                vectors[i] = _embed_one(chunks[i].page_content)
                return i

            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for i in ex.map(_embed, range(total)):
                    done += 1
                    if done % 25 == 0 or done == total:
                        emit("upsert", f"embedded {done}/{total}…")

        emit("upsert", f"upserting {total} points to {config.QDRANT_COLLECTION}…")
        client = get_qdrant_client()
        # BM25 sparse vectors: local, fast, batched (no server, no quota).
        sparse_vecs = []
        if config.HYBRID_SEARCH and total:
            emit("upsert", "encoding BM25 sparse vectors…")
            bm25 = get_bm25()
            for s in range(0, total, 256):
                batch = [c.page_content for c in chunks[s : s + 256]]
                sparse_vecs.extend(_to_sparse(v) for v in bm25.passage_embed(batch))
        # Qdrant caps request payloads (~32 MB): batch the upsert, critical
        # for wide vectors (2560-dim x 1300 chunks ≈ 43 MB in one shot).
        # IDs are stable per unit+index (see app.delta) so re-ingesting
        # overwrites the same points, never dupes.
        assigned = _assign_point_ids(chunks, name)
        for s in range(0, total, config.UPSERT_BATCH_SIZE):
            points = [
                PointStruct(
                    id=assigned[i][1],
                    vector=(
                        {"dense": vectors[i], "bm25": sparse_vecs[i]}
                        if sparse_vecs
                        else {"dense": vectors[i]}
                    ),
                    payload={
                        "page_content": chunks[i].page_content,
                        "metadata": chunks[i].metadata,
                        "content_hash": chunks[i].metadata.get("content_hash"),
                    },
                )
                for i in range(s, min(s + config.UPSERT_BATCH_SIZE, total))
            ]
            client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    except Exception:
        log.exception(f"FAILED step=upsert (embeddings or Qdrant?) file={name}")
        raise

    emit("done", f"ingested {len(chunks)} chunks")
    try:
        from app import metrics as _m

        _m.CHUNKS_INDEXED.inc(len(chunks))
    except Exception:
        pass
    return len(chunks)


def split_and_upsert(docs: list, name: str, emit, log) -> int:
    """Backward-compat wrapper: split then embed+upsert (no reset)."""
    chunks = split_docs(docs, name, emit, log)
    return embed_and_upsert(chunks, name, emit, log)


class _FullFallback(Exception):
    """Diff/fetch unavailable -> caller runs the legacy full path."""


def _tag_doc_hashes(docs: list) -> None:
    """Stamp content_hash on each page/table doc (survives splitting)."""
    for d in docs:
        md = d.metadata if isinstance(d.metadata, dict) else {}
        d.metadata = md
        md["content_hash"] = page_hash(d.page_content or "")


def _unit_key_of(chunk) -> tuple:
    """Prose -> ("p", page); table grid -> ("t", hash12)."""
    md = chunk.metadata or {}
    if md.get("table"):
        return ("t", str(md.get("content_hash") or "?")[:12])
    return ("p", str(md.get("page")))


def _assign_point_ids(chunks: list, name: str) -> list:
    """[(chunk, stable_pid)] with per-unit occurrence indices."""
    counters: dict = {}
    out = []
    for c in chunks:
        key = _unit_key_of(c)
        i = counters.get(key, 0)
        counters[key] = i + 1
        kind, ukey = key
        out.append((c, chunk_point_id(
            config.QDRANT_COLLECTION, name, kind, ukey, i)))
    return out


def _fetch_source_units(name: str) -> dict:
    """Scroll existing points for source -> {unit_key: {hashes, ids, legacy}}."""
    client = get_qdrant_client()
    units: dict = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(
                    key="metadata.source", match=MatchValue(value=name))]
            ),
            limit=512,
            offset=offset,
            # NB: page/table live under nested "metadata"; page_content is
            # excluded to keep the scroll light (hashes suffice for diffing).
            with_payload=["metadata", "content_hash"],
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            md = pl.get("metadata") or {}
            legacy = is_legacy_point(pl)
            if md.get("table"):
                key = ("t", str(pl.get("content_hash") or "?")[:12])
            else:
                key = ("p", str(md.get("page")))
            u = units.setdefault(key, {"hashes": set(), "ids": [], "legacy": False})
            if pl.get("content_hash"):
                u["hashes"].add(pl["content_hash"])
            u["ids"].append(str(p.id))
            if legacy:
                u["legacy"] = True
        if offset is None:
            break
    return units


def _delete_point_ids(ids: set) -> None:
    get_qdrant_client().delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=PointIdsList(points=list(ids)),
    )


def _ingest_delta(chunks: list, name: str, emit, log) -> int:
    """Embed+upsert only changed units; delete stale ids after. Returns total.

    Raises _FullFallback when diffing is impossible (fresh/legacy source,
    scroll failure) so the caller runs the legacy full path.
    """
    try:
        old = _fetch_source_units(name)
    except Exception as e:
        raise _FullFallback(f"scroll failed: {e}")
    if not old:
        raise _FullFallback("fresh source")
    if any(u["legacy"] for u in old.values()):
        raise _FullFallback("legacy points (one full reset, then delta)")

    new_groups: dict = {}
    for c in chunks:
        new_groups.setdefault(_unit_key_of(c), []).append(c)
    old_view = {
        k: {"hash": next(iter(u["hashes"])) if len(u["hashes"]) == 1 else None,
            "ids": u["ids"]}
        for k, u in old.items()
    }
    new_view = {k: {"hash": v[0].metadata.get("content_hash")} for k, v in new_groups.items()}
    plan = plan_delta(old_view, new_view)
    # Skip only when hash AND chunk count match (dup "?" pages stay correct).
    skip = {k for k in plan["skip"]
            if len(old[k]["ids"]) == len(new_groups[k])}
    changed_keys = plan["changed"] | (plan["skip"] - skip)
    to_embed = [c for k in changed_keys for c in new_groups.get(k, [])]
    skipped_n = sum(len(new_groups[k]) for k in skip)

    if not to_embed and not plan["removed"]:
        total_old = sum(len(u["ids"]) for u in old.values())
        emit("done", f"no changes — {total_old} chunks already indexed")
        return total_old

    if to_embed:
        embed_and_upsert(to_embed, name, emit, log)
    else:
        emit("upsert", "delta: nothing to embed, removing stale points…")
    if skipped_n:
        emit("upsert", f"delta: skipped {skipped_n} unchanged chunks")

    new_ids = {pid for _, pid in _assign_point_ids(chunks, name)}
    old_ids = {i for u in old.values() for i in u["ids"]}
    stale = old_ids - new_ids
    if stale:
        emit("upsert", f"delta: deleting {len(stale)} stale points…")
        _delete_point_ids(stale)
    # Corpus changed -> cached answers may be stale (mirrors _reset_source).
    try:
        from app import cache as _cache

        _cache.clear()
    except Exception:
        step_logger("cache").exception("cache clear on delta ingest failed")
    emit("done", f"delta ingested {len(to_embed)} new chunks, "
                 f"{skipped_n} unchanged, {len(new_ids)} total")
    return len(new_ids)


def _make_emit(log, name, on_step):
    def emit(step: str, detail: str):
        log.info(f"step={step} {detail} file={name}")
        if on_step:
            on_step(step, detail)

    return emit


def ingest_pdf(
    pdf_path: str,
    source_name: str | None = None,
    on_step: Callable[[str, str], None] | None = None,
) -> int:
    """Load a PDF locally, page-group, split, embed, upsert. Returns count.

    Consistency: old Qdrant points for this source are deleted only AFTER
    a successful parse+split, so a failed big-file retry keeps the old index.
    Delta: re-uploads embed only changed pages (stable point IDs); falls
    back to the full path for fresh/legacy sources or scroll failures.
    """
    name = source_name or os.path.basename(pdf_path)
    log = step_logger("ingest")
    log.info(f"start file={name} path={pdf_path}")
    emit = _make_emit(log, name, on_step)
    try:
        emit("load_pdf", f"parsing PDF (strategy={config.UNSTRUCTURED_STRATEGY})…")
        loader = UnstructuredPDFLoader(
            pdf_path,
            mode="elements",
            strategy=config.UNSTRUCTURED_STRATEGY,
            infer_table_structure=True,
        )
        elements = loader.load()
        rows = [
            (
                d.metadata.get("category"),
                d.metadata.get("page_number", d.metadata.get("page")),
                d.page_content,
                d.metadata.get("text_as_html"),
            )
            for d in elements
        ]
        docs = group_page_docs(rows, name)
        emit("load_pdf", f"parsed {len(docs)} pages ({len(elements)} raw elements)")
        _tag_doc_hashes(docs)
        chunks = split_docs(docs, name, emit, log)
    except Exception:
        log.exception(f"FAILED step=load_pdf file={name}")
        raise
    if not chunks:
        log.info(f"empty parse file={name} — keeping old index")
        emit("done", "no extractable text — nothing to index")
        return 0
    try:
        ensure_collection()
    except Exception:
        log.exception(f"FAILED step=ensure_collection file={name} (Qdrant?)")
        raise
    try:
        return _ingest_delta(chunks, name, emit, log)
    except _FullFallback as e:
        log.info(f"delta unavailable ({e}) — full ingest file={name}")
    try:
        # Old index is replaced only once the replacement is ready.
        _reset_source(name)
    except Exception:
        log.exception(f"FAILED step=ensure_collection file={name} (Qdrant?)")
        raise
    return embed_and_upsert(chunks, name, emit, log)


def ingest_pdf_from_s3(
    s3_bucket: str,
    s3_key: str,
    source_name: str,
    on_step: Callable[[str, str], None] | None = None,
    expected_sha256: str | None = None,
) -> int:
    """Download staged PDF from S3 to temp, optionally SHA256-check, ingest.

    Backend never buffers the upload; the worker pulls from S3 (source of
    truth). Temp file is always cleaned up.
    """
    import hashlib
    import tempfile

    from app.logging_setup import step_logger as _sl

    log = _sl("ingest")
    emit = _make_emit(log, source_name, on_step)
    emit("load_pdf", f"downloading s3://{s3_bucket}/{s3_key}…")
    from app.uploads import s3_client as _s3c

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path = tmp.name
    tmp.close()
    try:
        _s3c().download_file(s3_bucket, s3_key, tmp_path)
        if expected_sha256:
            h = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(blk)
            if h.hexdigest() != expected_sha256.lower():
                raise ValueError("SHA256 mismatch after S3 download.")
        return ingest_pdf(tmp_path, source_name=source_name, on_step=on_step)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def ingest_elements(
    items: list,
    source_name: str,
    on_step: Callable[[str, str], None] | None = None,
) -> int:
    """Ingest pre-parsed elements (e.g. hi_res JSON from the EC2 worker).

    items: [{type, page, text, html?}, ...]. Returns chunk count.
    """
    name = source_name
    log = step_logger("ingest")
    log.info(f"start pre-parsed file={name} elements={len(items)}")
    emit = _make_emit(log, name, on_step)
    try:
        rows = [(e.get("type"), e.get("page"), e.get("text"), e.get("html"))
                for e in items]
        docs = group_page_docs(rows, name)
        emit("load_pdf", f"grouped {len(items)} elements into {len(docs)} pages")
        _tag_doc_hashes(docs)
        chunks = split_docs(docs, name, emit, log)
    except Exception:
        log.exception(f"FAILED step=load_pdf file={name}")
        raise
    try:
        if chunks:
            _reset_source(name)
    except Exception:
        log.exception(f"FAILED step=ensure_collection file={name} (Qdrant?)")
        raise
    return embed_and_upsert(chunks, name, emit, log)

SYSTEM_PROMPT = (
    "You are Atlas, a RAG specialist. Your ONLY job is answering questions "
    "about company annual reports using retrieved excerpts. Nothing else.\n"
    "\n"
    "SCOPE\n"
    "- Answer questions grounded in company annual reports / filings.\n"
    "- If the question is unrelated to annual-report content (general chat, "
    "coding, advice, other domains), decline politely in one sentence and "
    "say what you do cover.\n"
    "\n"
    "GROUNDING (non-negotiable)\n"
    "- Use ONLY the retrieved context below. Never use outside knowledge, "
    "never guess, never fill gaps with plausible-sounding detail.\n"
    "- Every factual claim must trace to a cited excerpt. Cite as "
    "[filename, p. N] after the claim.\n"
    "- Numbers: quote exactly as stated — value, currency/unit, and period "
    "(FY, quarter, date). Do NOT convert currencies, annualize, or compute "
    "ratios unless the user explicitly asks; if you do compute, show the "
    "inputs with citations.\n"
    "- If excerpts conflict, present each side with its citation instead of "
    "picking one silently.\n"
    "\n"
    "HONESTY\n"
    "- If the context does not contain the answer, say plainly: "
    "'The retrieved excerpts do not contain this information.' Then add what "
    "the excerpts DO say on the closest related point, if anything.\n"
    "- If you are unsure or confused by fragmented/ambiguous excerpts, say "
    "which part is unclear and why, rather than smoothing over it.\n"
    "- Never state uncertainty as fact. Prefer 'not determinable from the "
    "provided excerpts' over hedging filler.\n"
    "\n"
    "STYLE\n"
    "- Be detailed and precise: directly answer first, then supporting "
    "detail with citations. Use short paragraphs or bullets.\n"
    "- Keep table figures aligned to their row/column labels — a bare value "
    "without its label is meaningless."
)


def _build_rag_message(context: str, question: str) -> str:
    return (
        "RETRIEVED CONTEXT (excerpts from company annual reports — "
        "your only source of truth):\n"
        "----------------------------------------\n"
        f"{context}\n"
        "----------------------------------------\n"
        "\n"
        "USER QUESTION (answer this question ONLY based on the context above, "
        "following your role instructions):\n"
        f"{question}"
    )


def _hybrid_search(question: str, k: int, dense_vec: list | None = None):
    """Dense (bge-m3) + BM25 sparse retrieval fused with RRF.

    Returns (docs, debug) where debug records each hit's origin ranks.
    Pass dense_vec to reuse the query embedding (cache + retrieve share it).
    """
    from langchain_core.documents import Document
    from qdrant_client.http.models import Fusion, FusionQuery, Prefetch

    client = get_qdrant_client()
    pre = config.HYBRID_PREFETCH
    try:
        dense_vec = _finite(
            dense_vec if dense_vec is not None
            else get_embeddings().embed_query(question),
            "query",
        )
    except Exception:
        # bge-m3 can emit NaN for specific token combos (fp16 overflow);
        # Ollama then 500s. Fall back to BM25-only rather than dying.
        step_logger("query").warning("dense embed failed, BM25-only fallback")
        dense_vec = None
    sparse_vec = _to_sparse(next(iter(get_bm25().query_embed(question))))
    if dense_vec is None:
        res = client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=sparse_vec, using="bm25", limit=k, with_payload=True,
        )
    else:
        res = client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=pre),
                Prefetch(query=sparse_vec, using="bm25", limit=pre),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            with_payload=True,
        )
    docs = [
        Document(
            page_content=p.payload.get("page_content", ""),
            metadata=p.payload.get("metadata", {}),
        )
        for p in res.points
    ]
    return docs, {"fused": len(docs), "prefetch_each": pre}


def _finite(vec: list, where: str) -> list:
    """bge-m3 rarely emits NaN/Inf for certain inputs (Ollama 500s on them).
    Zero those dims and log — a slightly degraded vector beats a dead query.
    """
    import math

    bad = [i for i, x in enumerate(vec)
           if isinstance(x, float) and (math.isnan(x) or math.isinf(x))]
    if bad:
        step_logger("embed").warning(
            f"non-finite dims in {where}: {len(bad)}/{len(vec)} — zeroed")
        vec = [0.0 if i in set(bad) else x for i, x in enumerate(vec)]
    return vec


_reranker_model = None
_reranker_device: str | None = None


def _resolve_rerank_device() -> str:
    """Resolve the rerank device: auto -> cuda if available, else cpu."""
    import torch

    choice = config.RERANK_DEVICE
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _reranker():
    """bge cross-encoder singleton (GPU if available)."""
    global _reranker_model, _reranker_device
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        _reranker_device = _resolve_rerank_device()
        log = step_logger("rerank")
        log.info(f"loading reranker {config.RERANK_MODEL} on {_reranker_device}")
        _reranker_model = CrossEncoder(config.RERANK_MODEL, device=_reranker_device)
        log.info(f"reranker loaded on {_reranker_device}")
    return _reranker_model


def warm_reranker() -> None:
    """Load the reranker + a dummy predict so the first query doesn't pay it."""
    import torch

    model = _reranker()
    with torch.inference_mode():
        model.predict([("warmup", "text")])
    log = step_logger("warmup")
    log.info(f"reranker warm on {_reranker_device}")


def _rerank(question: str, docs: list, k: int, log) -> list:
    """Cross-encoder rescore of candidate docs, keep top-k."""
    import time as _time
    import torch

    t0 = _time.time()
    model = _reranker()
    with torch.inference_mode():
        scores = model.predict(
            [(question, d.page_content) for d in docs],
            batch_size=config.RERANK_BATCH_SIZE,
        )
    ranked = sorted(zip(scores, docs), key=lambda p: float(p[0]), reverse=True)
    top = [d for _, d in ranked[:k]]
    best = [(round(float(s), 4), d.metadata.get("page")) for s, d in ranked[:5]]
    log.info(f"step=rerank done kept={len(top)}/{len(docs)} "
             f"in {_time.time()-t0:.1f}s device={_reranker_device} "
             f"batch={config.RERANK_BATCH_SIZE} top5={best}")
    return top

def query(question: str, top_k: int | None = None):
    """Retrieve + generate. Returns (answer, sources).

    Semantic cache (Qdrant): the question is embedded ONCE and the vector is
    reused by both the cache lookup and the hybrid retrieve, so a miss costs
    no extra embedding call. A hit returns without any LLM call.
    """
    from app import metrics

    k = top_k or config.TOP_K
    log = step_logger("query")
    log.info(f"start q={question[:120]!r} top_k={k} hybrid={config.HYBRID_SEARCH} "
             f"rerank={config.RERANK} cache={config.CACHE_ENABLED}")

    # Step 0: embed once — reused by cache + Qdrant.
    q_vec = None
    if config.CACHE_ENABLED or config.HYBRID_SEARCH:
        try:
            q_vec = _finite(get_embeddings().embed_query(question), "query")
        except Exception:
            step_logger("query").warning(
                "query embed failed, cache skipped / BM25-only fallback")
            q_vec = None

    # Step 1: semantic cache lookup (Qdrant cosine >= threshold -> HIT).
    if config.CACHE_ENABLED and q_vec is not None:
        from app import cache as _cache

        with metrics.time_step("cache_lookup"):
            hit = _cache.lookup(q_vec)
        if hit is not None:
            metrics.CACHE_HITS.inc()
            log.info("step=cache HIT")
            return hit
        metrics.CACHE_MISSES.inc()
        log.info("step=cache MISS")

    # Step 2: retrieve similar chunks from Qdrant
    try:
        fetch_k = max(k, config.RERANK_TOPN) if config.RERANK else k
        with metrics.time_step("retrieve"):
            if config.HYBRID_SEARCH:
                docs, dbg = _hybrid_search(question, fetch_k, dense_vec=q_vec)
            else:
                store = get_vector_store()
                docs = store.similarity_search(question, k=fetch_k)
                dbg = {"fused": len(docs), "prefetch_each": 0}
        log.info(f"step=retrieve done hits={len(docs)} {dbg}")
        if config.RERANK and len(docs) > k:
            with metrics.time_step("rerank"):
                docs = _rerank(question, docs, k, log)
    except Exception:
        log.exception("FAILED step=retrieve (embeddings or Qdrant?)")
        raise

    context = "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')} | page: {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    # Step 2: generate answer with Bedrock LLM
    try:
        log.info(f"step=generate model={config.BEDROCK_MODEL_ID}")
        llm = get_llm()
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", _build_rag_message(context, question)),
        ]
        with metrics.time_step("generate"):
            resp = llm.invoke(messages)
        log.info("step=generate done")
    except Exception:
        log.exception(f"FAILED step=generate model={config.BEDROCK_MODEL_ID}")
        raise

    sources = [
        {
            "content": d.page_content,
            "metadata": {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
            },
        }
        for d in docs
    ]

    # Step 4: fire-and-forget cache store (never blocks / never raises).
    if config.CACHE_ENABLED and q_vec is not None:
        try:
            from app import cache as _cache

            source_tags = sorted(
                {str(d.metadata.get("source", "unknown")) for d in docs})
            _cache.store(question, q_vec, resp.content, sources,
                         source_tags=source_tags)
        except Exception:
            log.exception("cache store failed (non-fatal)")

    return resp.content, sources


def collection_count() -> int:
    client = get_qdrant_client()
    if not client.collection_exists(config.QDRANT_COLLECTION):
        return 0
    info = client.get_collection(config.QDRANT_COLLECTION)
    return info.points_count or 0


def clear_collection() -> None:
    client = get_qdrant_client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)
    # Corpus wiped -> cached answers are stale. v1: full flush.
    try:
        from app import cache as _cache

        _cache.clear()
    except Exception:
        step_logger("cache").exception("cache clear on clear_collection failed")
