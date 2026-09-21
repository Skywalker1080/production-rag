"""FastAPI backend serving the UI + RAG API (PDF only)."""
import os
import shutil

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, jobs, rag
from app.logging_setup import setup_logging, step_logger

setup_logging()
log = step_logger("api")

app = FastAPI(title="Simple PDF RAG (LangChain + Qdrant + Bedrock GLM 5)")

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
    return {
        "status": "ok",
        "qdrant_ok": qdrant_ok,
        "qdrant_url": config.QDRANT_URL,
        "llm_model": config.BEDROCK_MODEL_ID,
        "embed_model": config.BEDROCK_EMBED_MODEL_ID,
        "chunks": rag.collection_count() if qdrant_ok else 0,
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
