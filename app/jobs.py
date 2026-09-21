"""In-memory ingest jobs: upload returns 202 + job_id, UI polls for progress."""
import threading
import uuid
from datetime import datetime, timezone

from app import rag
from app.logging_setup import step_logger

log = step_logger("jobs")

_lock = threading.Lock()
_jobs: dict = {}


STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Map internal pipeline steps to 0-100 progress for the UI.
STEP_PROGRESS = {
    "queued": 5,
    "load_pdf": 20,
    "split": 45,
    "upsert": 70,
    "done": 100,
    "failed": 100,
}


def create_job(filename: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "status": STATUS_QUEUED,  # queued -> processing -> completed | failed
            "step": "queued",
            "detail": "waiting to start",
            "progress": STEP_PROGRESS["queued"],
            "chunks": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    log.info(f"job created id={job_id} file={filename}")
    return job_id


def _progress_for(step: str, detail: str) -> int:
    """Refine upsert progress from 'embedded X/Y' details when available."""
    if step == "upsert":
        try:
            parts = detail.split()
            idx = parts.index("embedded")
            done, total = parts[idx + 1].split("/")
            return 50 + round(45 * int(done) / max(int(total), 1))
        except Exception:
            pass
    return STEP_PROGRESS.get(step, 10)


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _update(job_id: str, **fields):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
            _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


def run_ingest_job(job_id: str, pdf_path: str, filename: str) -> None:
    """Background worker: runs the pipeline, publishing each step to the job."""
    _update(job_id, status=STATUS_PROCESSING, step="load_pdf",
            detail="parsing PDF layout…",
            progress=_progress_for("load_pdf", ""))

    def on_step(step: str, detail: str):
        _update(job_id, status=STATUS_PROCESSING, step=step, detail=detail,
                progress=_progress_for(step, detail))
        log.info(f"job id={job_id} step={step} {detail}")

    try:
        n = rag.ingest_pdf(pdf_path, source_name=filename, on_step=on_step)
        _update(
            job_id, status=STATUS_COMPLETED, step="done",
            detail=f"ingested {n} chunks", chunks=n,
            progress=_progress_for("done", ""),
        )
        log.info(f"job done id={job_id} file={filename} chunks={n}")
    except Exception as e:
        _update(
            job_id, status=STATUS_FAILED, step="failed",
            detail="failed — see logs/rag.log for the step + traceback",
            error=f"{type(e).__name__}: {e}",
            progress=_progress_for("failed", ""),
        )
        log.exception(f"job failed id={job_id} file={filename}")
