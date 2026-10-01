"""S3 multipart staging for big PDFs + SQLite upload-session manifest.

Flow (direct-to-S3, consistency-first):
  init    -> create AWS MPU, presigned PUT urls per part, row status=initiated
  PUTs    -> browser PUTs parts straight to S3 (16MB, retry per-part)
  complete-> CompleteMultipartUpload, HEAD verify size, SHA256 check,
             row status=staged (S3 is source of truth from here)
  worker  -> downloads from S3 to temp, parses, upserts; old Qdrant
             points deleted only AFTER successful parse (see rag.py).

Small files (< threshold) keep using POST /api/upload unchanged.
"""
import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app import config
from app.logging_setup import step_logger

log = step_logger("uploads")

_lock = threading.Lock()
_db_init_done = False

STATUS_INITIATED = "initiated"
STATUS_STAGED = "staged"
STATUS_PARSING = "parsing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS upload_sessions (
  upload_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  s3_bucket TEXT NOT NULL,
  s3_key TEXT NOT NULL,
  s3_upload_id TEXT NOT NULL,
  total_size INTEGER NOT NULL,
  part_size INTEGER NOT NULL,
  total_parts INTEGER NOT NULL,
  sha256_client TEXT,
  status TEXT NOT NULL,
  job_id TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def _db() -> sqlite3.Connection:
    global _db_init_done
    os.makedirs(os.path.dirname(os.path.abspath(config.UPLOAD_DB_PATH)) or ".", exist_ok=True)
    conn = sqlite3.connect(config.UPLOAD_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with _lock:
        if not _db_init_done:
            conn.execute(SCHEMA)
            conn.commit()
            _db_init_done = True
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(name: str) -> str:
    base = os.path.basename(name or "document.pdf").strip() or "document.pdf"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return base[:180] or "document.pdf"


def s3_client():
    """Real S3 client. Uses standard AWS credential chain (IAM keys in env)."""
    kwargs: dict = {"region_name": config.AWS_REGION}
    if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = config.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = config.AWS_SECRET_ACCESS_KEY
    return boto3.client("s3", **kwargs)


def _require_bucket() -> str:
    bucket = (config.S3_STAGING_BUCKET or "").strip()
    if not bucket:
        raise RuntimeError(
            "S3_STAGING_BUCKET is not set. Add it to .env "
            "(e.g. S3_STAGING_BUCKET=rag-extract-405633560616-use1)."
        )
    return bucket


def compute_parts(total_size: int, part_size: int) -> int:
    return max(1, -(-total_size // part_size))


def init_session(filename: str, size_bytes: int, sha256_client: str | None) -> dict:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    if size_bytes <= 0:
        raise ValueError("size_bytes must be > 0.")
    bucket = _require_bucket()
    part_size = config.S3_PART_SIZE_MB * 1024 * 1024
    total_parts = compute_parts(size_bytes, part_size)
    if total_parts > 10000:
        raise ValueError(f"File too large: {total_parts} parts exceeds S3 limit of 10000.")

    safe = safe_filename(filename)
    doc_id = uuid.uuid4().hex[:12]
    upload_id = uuid.uuid4().hex[:12]
    s3_key = f"staged/{doc_id}/{safe}"

    s3 = s3_client()
    try:
        mpu = s3.create_multipart_upload(
            Bucket=bucket, Key=s3_key, ContentType="application/pdf"
        )
    except (ClientError, BotoCoreError) as e:
        log.exception(f"init FAILED file={filename} bucket={bucket}")
        raise RuntimeError(f"S3 create_multipart_upload failed: {e}")

    s3_upload_id = mpu["UploadId"]
    urls: list[dict] = []
    try:
        for part in range(1, total_parts + 1):
            url = s3.generate_presigned_url(
                "upload_part",
                Params={
                    "Bucket": bucket,
                    "Key": s3_key,
                    "UploadId": s3_upload_id,
                    "PartNumber": part,
                },
                ExpiresIn=config.S3_PRESIGNED_EXPIRY_SECS,
            )
            urls.append({"part_number": part, "url": url})
    except (ClientError, BotoCoreError) as e:
        try:
            s3.abort_multipart_upload(Bucket=bucket, Key=s3_key, UploadId=s3_upload_id)
        except Exception:
            pass
        raise RuntimeError(f"S3 presign failed: {e}")

    conn = _db()
    now = _now()
    conn.execute(
        "INSERT INTO upload_sessions (upload_id, doc_id, filename, s3_bucket, s3_key,"
        " s3_upload_id, total_size, part_size, total_parts, sha256_client,"
        " status, job_id, error, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            upload_id, doc_id, safe, bucket, s3_key, s3_upload_id,
            size_bytes, part_size, total_parts, (sha256_client or "").lower() or None,
            STATUS_INITIATED, None, None, now, now,
        ),
    )
    conn.commit()
    conn.close()
    log.info(f"init id={upload_id} doc={doc_id} file={safe} parts={total_parts} size={size_bytes}")
    return {
        "upload_id": upload_id,
        "doc_id": doc_id,
        "s3_key": s3_key,
        "part_size": part_size,
        "total_parts": total_parts,
        "urls": urls,
    }


def get_session(upload_id: str) -> dict | None:
    conn = _db()
    row = conn.execute(
        "SELECT * FROM upload_sessions WHERE upload_id=?", (upload_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _set(upload_id: str, **fields):
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in fields)
    conn = _db()
    conn.execute(
        f"UPDATE upload_sessions SET {sets} WHERE upload_id=?",
        (*fields.values(), upload_id),
    )
    conn.commit()
    conn.close()


def list_uploaded_parts(upload_id: str) -> list[int]:
    """Parts already on S3 (for resume). Returns part numbers."""
    sess = get_session(upload_id)
    if not sess:
        raise ValueError("Unknown upload id.")
    s3 = s3_client()
    parts: list[int] = []
    key_marker = 0
    while True:
        resp = s3.list_parts(
            Bucket=sess["s3_bucket"], Key=sess["s3_key"],
            UploadId=sess["s3_upload_id"], PartNumberMarker=key_marker,
        )
        for p in resp.get("Parts", []):
            parts.append(int(p["PartNumber"]))
        if resp.get("IsTruncated"):
            key_marker = resp.get("NextPartNumberMarker", 0)
        else:
            break
    return parts


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256_s3(sess: dict, tmp_path: str) -> None:
    """Download-then-hash check against client SHA256 (locked decision)."""
    expected = (sess.get("sha256_client") or "").lower()
    if not expected:
        return  # client omitted hash: size check below is the only gate
    actual = _sha256_file(tmp_path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: expected {expected}, got {actual}.")


def complete_session(upload_id: str, parts: list[dict]) -> dict:
    sess = get_session(upload_id)
    if not sess:
        raise ValueError("Unknown upload id.")
    if sess["status"] == STATUS_STAGED:
        return sess  # idempotent re-complete
    if sess["status"] not in (STATUS_INITIATED, STATUS_FAILED):
        raise ValueError(f"Cannot complete from status={sess['status']}.")

    # S3 requires parts sorted by PartNumber with ETags from PUT responses.
    sorted_parts = sorted(parts, key=lambda p: int(p["part_number"]))
    if len(sorted_parts) != int(sess["total_parts"]):
        raise ValueError(
            f"Expected {sess['total_parts']} parts, got {len(sorted_parts)}."
        )
    s3 = s3_client()
    try:
        s3.complete_multipart_upload(
            Bucket=sess["s3_bucket"],
            Key=sess["s3_key"],
            UploadId=sess["s3_upload_id"],
            MultipartUpload={
                "Parts": [
                    {"PartNumber": int(p["part_number"]), "ETag": p["etag"]}
                    for p in sorted_parts
                ]
            },
        )
    except (ClientError, BotoCoreError) as e:
        _set(upload_id, status=STATUS_FAILED, error=str(e))
        raise RuntimeError(f"S3 complete failed: {e}")

    # Size gate: HEAD must match init size (catches truncated completes).
    try:
        head = s3.head_object(Bucket=sess["s3_bucket"], Key=sess["s3_key"])
        if int(head.get("ContentLength", -1)) != int(sess["total_size"]):
            _set(upload_id, status=STATUS_FAILED,
                  error="Size mismatch after S3 complete.")
            raise ValueError("Size mismatch after S3 complete.")
    except (ClientError, BotoCoreError) as e:
        _set(upload_id, status=STATUS_FAILED, error=str(e))
        raise RuntimeError(f"S3 HEAD failed: {e}")

    _set(upload_id, status=STATUS_STAGED, error=None)
    log.info(f"staged id={upload_id} key={sess['s3_key']} size={sess['total_size']}")
    sess = get_session(upload_id)
    assert sess is not None
    return sess


def abort_session(upload_id: str) -> None:
    sess = get_session(upload_id)
    if not sess:
        raise ValueError("Unknown upload id.")
    try:
        s3_client().abort_multipart_upload(
            Bucket=sess["s3_bucket"], Key=sess["s3_key"],
            UploadId=sess["s3_upload_id"],
        )
    except Exception:
        pass  # already completed/aborted server-side; still mark locally
    _set(upload_id, status=STATUS_FAILED, error="aborted by client")


def refresh_urls(upload_id: str) -> dict:
    """Fresh presigned PUT urls for parts not yet on S3 (resume)."""
    sess = get_session(upload_id)
    if not sess:
        raise ValueError("Unknown upload id.")
    try:
        done = set(list_uploaded_parts(upload_id))
    except Exception as e:
        raise RuntimeError(f"S3 list_parts failed: {e}")
    s3 = s3_client()
    urls: list[dict] = []
    try:
        for part in range(1, int(sess["total_parts"]) + 1):
            if part in done:
                continue
            url = s3.generate_presigned_url(
                "upload_part",
                Params={
                    "Bucket": sess["s3_bucket"],
                    "Key": sess["s3_key"],
                    "UploadId": sess["s3_upload_id"],
                    "PartNumber": part,
                },
                ExpiresIn=config.S3_PRESIGNED_EXPIRY_SECS,
            )
            urls.append({"part_number": part, "url": url})
    except (ClientError, BotoCoreError) as e:
        raise RuntimeError(f"S3 presign failed: {e}")
    if sess["status"] == STATUS_FAILED:
        _set(upload_id, status=STATUS_INITIATED, error=None)
    return {
        "upload_id": upload_id,
        "part_size": sess["part_size"],
        "total_parts": sess["total_parts"],
        "uploaded_parts": sorted(done),
        "urls": urls,
    }
