"""Centralized component-level logging for the pipeline.

Every log line is JSON with ``request_id``, ``component``, and ``stage``
attached, so a broken chain is traceable: filter by ``request_id`` to see
the full path, the ``stage_error`` line (with its ``component``) marks
exactly which component and stage broke.

Stdlib only. Usage::

    from production_rag.common.logging import get_logger, new_request_id, stage

    logger = get_logger(__name__)
    new_request_id()
    with stage("identifier", component="indexing"):
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
_component: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "production_rag_component", default=None
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
            "component": getattr(record, "component_override", None)
            or _component.get(),
            "stage": getattr(record, "stage_override", None) or _stage.get(),
        }
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in (
                "stage_override",
                "component_override",
            ):
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


def set_component(name: str) -> None:
    """Bind the owning component (indexing, storage, retrieval, ...).

    Prefer passing ``component=`` to :func:`stage`, which binds and resets
    automatically; use this only for long-lived component entry points.
    """
    _component.set(name)


@contextmanager
def stage(name: str, component: str | None = None) -> Iterator[None]:
    """Mark a pipeline stage; logs enter/exit and pinpoints failures.

    ``component`` binds the owning pillar for the block (e.g.
    ``component="indexing"``) and is reset on exit, so the ``stage_error``
    line always names the component where the chain broke. On exception an
    ERROR ``stage_error`` line (with traceback) is logged, then the
    exception is reraised unchanged.
    """
    logger = get_logger("production_rag.stage")
    stage_token = _stage.set(name)
    component_token = _component.set(component) if component is not None else None
    start = time.perf_counter()
    logger.info("stage_enter")
    try:
        yield
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "stage_error: %s (%s)",
            type(exc).__name__,
            exc,
            exc_info=True,
            extra={"duration_ms": duration_ms},
        )
        _stage.reset(stage_token)
        if component_token is not None:
            _component.reset(component_token)
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("stage_exit", extra={"duration_ms": duration_ms})
    _stage.reset(stage_token)
    if component_token is not None:
        _component.reset(component_token)


def log_stage(
    func: Callable[..., Any] | None = None, *, component: str | None = None
) -> Callable[..., Any]:
    """Decorator version of :func:`stage` using the function name.

    Usable bare (``@log_stage``) or with a component
    (``@log_stage(component="indexing")``).
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with stage(target.__name__, component=component):
                return target(*args, **kwargs)

        return wrapper

    return decorate(func) if func is not None else decorate
