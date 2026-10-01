"""FastAPI backend serving the UI + RAG API (PDF only)."""
import os
import shutil

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, jobs, metrics, rag, uploads
from app.logging_setup import setup_logging, step_logger

setup_logging()
log = step_logger("api")

app = FastAPI(title="Simple PDF RAG (LangChain + Qdrant + Bedrock GLM 5)")


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    from fastapi.responses import PlainTextResponse
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    import time

    start = time.perf_counter()
    try:
        resp = await call_next(request)
        status = "ok" if resp.status_code < 400 else "error"
    except Exception:
        metrics.REQUESTS.labels(
            endpoint=request.url.path, status="error").inc()
        raise
    elapsed = time.perf_counter() - start
    if request.url.path.startswith("/api/"):
        metrics.REQUESTS.labels(
            endpoint=request.url.path, status=status).inc()
        metrics.STEP_LATENCY.labels(
            step="http_" + request.url.path.replace("/api/", "")).observe(elapsed)
    return resp

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.DATA_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class UploadInitRequest(BaseModel):
    filename: str
    size_bytes: int
    sha256: str | None = None


class UploadCompletePart(BaseModel):
    part_number: int
    etag: str


class UploadCompleteRequest(BaseModel):
    parts: list[UploadCompletePart]
    sha256: str | None = None


@app.get("/health")
def health():
    qdrant_ok = False
    try:
        c = rag.get_qdrant_client()
        c.get_collections()
        qdrant_ok = True
    except Exception:
        log.exception("health check: Qdrant unreachable")
        qdrant_ok = False
    metrics.QDRANT_UP.set(1 if qdrant_ok else 0)
    cache_ok = False
    cache_points = 0
    if qdrant_ok and config.CACHE_ENABLED:
        try:
            from app import cache as _cache

            cc = _cache._client()
            if cc.collection_exists(config.CACHE_COLLECTION):
                cache_ok = True
                cache_points = (
                    cc.get_collection(config.CACHE_COLLECTION).points_count or 0)
            else:
                # No cache collection yet = healthy (nothing cached), not an error.
                cache_ok = True
        except Exception:
            log.exception("health check: cache collection unreachable")
            cache_ok = False
    return {
        "status": "ok",
        "qdrant_ok": qdrant_ok,
        "qdrant_url": config.QDRANT_URL,
        "llm_model": config.BEDROCK_MODEL_ID,
        "embed_model": config.BEDROCK_EMBED_MODEL_ID,
        "chunks": rag.collection_count() if qdrant_ok else 0,
        "cache_enabled": config.CACHE_ENABLED,
        "cache_ok": cache_ok,
        "cache_collection": config.CACHE_COLLECTION,
        "cache_threshold": config.CACHE_THRESHOLD,
        "cache_points": cache_points,
    }


@app.post("/api/upload", status_code=202)
def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accept a PDF, queue ingestion, return 202 + job_id immediately."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        log.warning(f"upload rejected (not a pdf): {file.filename}")
        raise HTTPException(400, "Only PDF files are supported.")
    log.info(f"upload received: {file.filename}")
    dest = os.path.join(config.DATA_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = jobs.create_job(file.filename)
    background_tasks.add_task(jobs.run_ingest_job, job_id, dest, file.filename)
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued", "filename": file.filename},
    )


@app.post("/api/uploads/init")
def uploads_init(req: UploadInitRequest):
    """Start an S3 multipart session; returns presigned PUT urls per part."""
    try:
        sess = uploads.init_session(req.filename, req.size_bytes, req.sha256)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        log.exception("uploads init FAILED (S3/config?)")
        raise HTTPException(500, str(e))
    return {
        **sess,
        "threshold_mb": config.S3_UPLOAD_THRESHOLD_MB,
        "sha256_required": True,
    }


@app.get("/api/uploads/{upload_id}")
def uploads_status(upload_id: str):
    sess = uploads.get_session(upload_id)
    if not sess:
        raise HTTPException(404, "Unknown upload id.")
    try:
        uploaded = uploads.list_uploaded_parts(upload_id)
    except Exception:
        uploaded = []
    return {**sess, "uploaded_parts": uploaded}


@app.post("/api/uploads/{upload_id}/complete", status_code=202)
def uploads_complete(upload_id: str, req: UploadCompleteRequest, background_tasks: BackgroundTasks):
    sess = uploads.get_session(upload_id)
    if not sess:
        raise HTTPException(404, "Unknown upload id.")
    # Enforce locked decision: SHA256 required.
    claimed = (req.sha256 or "").lower() or (sess.get("sha256_client") or "").lower()
    if not claimed:
        raise HTTPException(400, "sha256 is required for big uploads.")
    try:
        staged = uploads.complete_session(
            upload_id,
            [{"part_number": p.part_number, "etag": p.etag} for p in req.parts],
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        log.exception(f"uploads complete FAILED id={upload_id}")
        raise HTTPException(500, str(e))
    job_id = jobs.create_job(staged["filename"])
    uploads._set(upload_id, job_id=job_id, status=uploads.STATUS_PARSING)
    background_tasks.add_task(
        jobs.run_ingest_s3_job,
        job_id, staged["s3_bucket"], staged["s3_key"],
        staged["filename"], upload_id, claimed,
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued",
                 "filename": staged["filename"], "upload_id": upload_id},
    )


@app.post("/api/uploads/{upload_id}/abort")
def uploads_abort(upload_id: str):
    sess = uploads.get_session(upload_id)
    if not sess:
        raise HTTPException(404, "Unknown upload id.")
    try:
        uploads.abort_session(upload_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"aborted": True, "upload_id": upload_id}


@app.post("/api/uploads/{upload_id}/refresh")
def uploads_refresh(upload_id: str):
    """Fresh presigned urls for missing parts (resume after failure/expiry)."""
    sess = uploads.get_session(upload_id)
    if not sess:
        raise HTTPException(404, "Unknown upload id.")
    if sess["status"] not in (uploads.STATUS_INITIATED, uploads.STATUS_FAILED):
        raise HTTPException(400, f"Cannot refresh from status={sess['status']}.")
    try:
        return uploads.refresh_urls(upload_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        log.exception(f"uploads refresh FAILED id={upload_id}")
        raise HTTPException(500, str(e))


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id.")
    return job


@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty.")
    log.info(f"ask received: {req.question[:120]!r}")
    try:
        answer, sources = rag.query(req.question, top_k=req.top_k)
    except Exception as e:
        log.exception("ask FAILED (see query steps above)")
        raise HTTPException(500, f"Query failed: {e}")
    log.info(f"ask done sources={len(sources)}")
    return {"answer": answer, "sources": sources}


@app.get("/api/stats")
def stats():
    try:
        count = rag.collection_count()
    except Exception as e:
        raise HTTPException(500, f"Qdrant not reachable: {e}")
    files = (
        [f for f in os.listdir(config.DATA_DIR) if f.lower().endswith(".pdf")]
        if os.path.isdir(config.DATA_DIR)
        else []
    )
    return {"chunks": count, "files": files}


@app.delete("/api/documents")
def delete_documents():
    try:
        rag.clear_collection()
    except Exception as e:
        raise HTTPException(500, f"Clear failed: {e}")
    return {"cleared": True}


# Serve UI (must be last so /api/* routes win).
# Prefer the React build (frontend/dist) when present; fall back to the
# legacy debug page (app/static). React Router safe: SPA fallback to index.
REACT_DIST = os.path.join(
    os.path.dirname(BASE_DIR), "frontend", "dist"
)


def _spa_file(path: str):
    full = os.path.join(REACT_DIST, path.lstrip("/") or "index.html")
    if os.path.isfile(full):
        return FileResponse(full)
    return FileResponse(os.path.join(REACT_DIST, "index.html"))


if os.path.isfile(os.path.join(REACT_DIST, "index.html")):
    log.info(f"serving React UI from {REACT_DIST}")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Unknown API route.")
        return _spa_file(path)

elif os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/", include_in_schema=False)
def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"msg": "UI not found. API is running. See /docs"}
