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
    PointStruct,
    VectorParams,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app import config
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
    )


def ensure_collection() -> None:
    """Create the Qdrant collection, recreating it if the embedding dim changed.

    Switching providers (Titan 1024-dim <-> nomic 768-dim) makes old vectors
    unreadable, so a mismatch wipes + recreates rather than failing obscurely.
    """
    log = step_logger("qdrant")
    client = get_qdrant_client()
    need = embedding_dim()
    if client.collection_exists(config.QDRANT_COLLECTION):
        info = client.get_collection(config.QDRANT_COLLECTION)
        vectors = info.config.params.vectors
        have = getattr(vectors, "size", None)
        if have != need:
            log.warning(
                f"dim mismatch (collection={have} vs {config.EMBED_PROVIDER}={need}) "
                "— recreating collection, old vectors dropped"
            )
            client.delete_collection(config.QDRANT_COLLECTION)
        else:
            return
    log.info(
        f"creating collection {config.QDRANT_COLLECTION} "
        f"(dim={need}, provider={config.EMBED_PROVIDER})"
    )
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=need, distance=Distance.COSINE),
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


def split_and_upsert(docs: list, name: str, emit, log) -> int:
    """Split page docs into chunks, embed (parallel), upsert. Returns count."""
    # Step: split into chunks (tables split by row, headers repeated)
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
            return embedding.embed_query(text)

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
        # Qdrant caps request payloads (~32 MB): batch the upsert, critical
        # for wide vectors (2560-dim x 1300 chunks ≈ 43 MB in one shot).
        for s in range(0, total, config.UPSERT_BATCH_SIZE):
            points = [
                PointStruct(
                    id=uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{config.QDRANT_COLLECTION}:{name}:{i}",
                    ).hex,
                    vector=vectors[i],
                    payload={
                        "page_content": chunks[i].page_content,
                        "metadata": chunks[i].metadata,
                    },
                )
                for i in range(s, min(s + config.UPSERT_BATCH_SIZE, total))
            ]
            client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    except Exception:
        log.exception(f"FAILED step=upsert (embeddings or Qdrant?) file={name}")
        raise

    emit("done", f"ingested {len(chunks)} chunks")
    return len(chunks)


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
    """Load a PDF locally, page-group, split, embed, upsert. Returns count."""
    name = source_name or os.path.basename(pdf_path)
    log = step_logger("ingest")
    log.info(f"start file={name} path={pdf_path}")
    emit = _make_emit(log, name, on_step)
    try:
        _reset_source(name)
    except Exception:
        log.exception(f"FAILED step=ensure_collection file={name} (Qdrant?)")
        raise
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
    except Exception:
        log.exception(f"FAILED step=load_pdf file={name}")
        raise
    return split_and_upsert(docs, name, emit, log)


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
        _reset_source(name)
    except Exception:
        log.exception(f"FAILED step=ensure_collection file={name} (Qdrant?)")
        raise
    try:
        rows = [(e.get("type"), e.get("page"), e.get("text"), e.get("html"))
                for e in items]
        docs = group_page_docs(rows, name)
        emit("load_pdf", f"grouped {len(items)} elements into {len(docs)} pages")
    except Exception:
        log.exception(f"FAILED step=load_pdf file={name}")
        raise
    return split_and_upsert(docs, name, emit, log)

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


def query(question: str, top_k: int | None = None):
    """Retrieve + generate. Returns (answer, sources)."""
    k = top_k or config.TOP_K
    log = step_logger("query")
    log.info(f"start q={question[:120]!r} top_k={k}")

    # Step 1: retrieve similar chunks from Qdrant
    try:
        store = get_vector_store()
        docs = store.similarity_search(question, k=k)
        log.info(f"step=retrieve done hits={len(docs)}")
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
