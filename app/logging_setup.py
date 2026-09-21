"""Centralized logging: console + dedicated rotating log file."""
import logging
import os
from logging.handlers import RotatingFileHandler

from app import config


class _DefaultStep(logging.Filter):
    """External loggers (unstructured) have no `step` — default it to '-'."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "step"):
            record.step = "-"
        return True


def setup_logging() -> logging.Logger:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = logging.getLogger("rag")
    logger.setLevel(config.LOG_LEVEL)
    if logger.handlers:
        return logger  # already configured (e.g. uvicorn reload)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(step)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        config.LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(_DefaultStep())

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.addFilter(_DefaultStep())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    # Route unstructured's own loggers into the same file + console so you
    # can see what the PDF partitioner is doing (strategy, pages, fallbacks).
    # transformers stays at WARNING to avoid per-weight spam.
    for name in ("unstructured", "unstructured_inference"):
        ulog = logging.getLogger(name)
        ulog.setLevel(config.LOG_LEVEL)
        ulog.addHandler(file_handler)
        ulog.addHandler(console_handler)
        ulog.propagate = False
    logging.getLogger("transformers").setLevel(logging.WARNING)
    return logger


log = logging.getLogger("rag")


def step_logger(step: str) -> logging.LoggerAdapter:
    """Logger adapter that injects the pipeline step name into every record."""
    return logging.LoggerAdapter(log, {"step": step})
