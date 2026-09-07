"""Centralized structured logging for the pipeline.

Every log line is JSON with ``request_id`` and ``stage`` attached, so a
broken chain is traceable: filter by ``request_id`` to see the full path,
the ``stage_error`` line marks exactly where it broke.

Stdlib only. Usage::

    from production_rag.common.logging import get_logger, new_request_id, stage

    logger = get_logger(__name__)
    new_request_id()
    with stage("identifier"):
        ...  # enter/exit lines emitted automatically, errors logged + reraised
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TextIO

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "production_rag_request_id", default=None
)
_stage: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "production_rag_stage", default=None
)

_configured = False

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class _JsonFormatter(logging.Formatter):
    """Render records as single-line JSON with chain context attached."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id.get(),
            "stage": getattr(record, "stage_override", None) or _stage.get(),
        }
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key != "stage_override":
                payload[key] = value
        return json.dumps(payload, default=str)


def configure(
    level: str | None = None, stream: TextIO | None = None, force: bool = False
) -> None:
    """Configure the root handler once. Safe to call repeatedly."""
    global _configured
    if _configured and not force:
        return
    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel((level or os.getenv("LOG_LEVEL", "INFO")).upper())
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a pipeline logger; configures the root handler on first use."""
    configure()
    return logging.getLogger(name)


def new_request_id() -> str:
    """Start a new chain: generate, bind, and return its id."""
    value = uuid.uuid4().hex[:12]
    _request_id.set(value)
    return value


def set_request_id(value: str) -> None:
    """Continue an existing chain (e.g. a celery worker picking up a job)."""
    _request_id.set(value)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Mark a pipeline stage; logs enter/exit and pinpoints failures.

    On exception an ERROR ``stage_error`` line (with traceback) is logged
    for this stage, then the exception is reraised unchanged.
    """
    logger = get_logger("production_rag.stage")
    token = _stage.set(name)
    start = time.perf_counter()
    logger.info("stage_enter", extra={"stage_override": name})
    try:
        yield
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "stage_error: %s (%s)",
            type(exc).__name__,
            exc,
            exc_info=True,
            extra={"stage_override": name, "duration_ms": duration_ms},
        )
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "stage_exit", extra={"stage_override": name, "duration_ms": duration_ms}
    )
    _stage.reset(token)


def log_stage(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator version of :func:`stage` using the function name."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with stage(func.__name__):
            return func(*args, **kwargs)

    return wrapper
