"""Manual test entry point: file path -> cleaned Document summary.

Usage:
    uv run python ingest.py <path> [--encoding latin-1] [--full]

Exit codes: 0 ok, 1 rejected by the pipeline (reason on stderr),
2 bad CLI usage / unreadable file.
"""

from __future__ import annotations

import argparse
import json
import sys

from production_rag.common import logging as plog
from production_rag.common.logging import get_logger, new_request_id
from production_rag.indexing import dispatcher
from production_rag.indexing.errors import IndexingError

# force: importing the dispatcher already configured the default
# (console-only) handler; the entry point owns the final config.
plog.configure(file="logs/pipeline.jsonl", force=True)

PREVIEW_CHARS = 500

logger = get_logger("ingest_cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest one file to a Document.")
    parser.add_argument("path", help="file to ingest")
    parser.add_argument("--encoding", default=None, help="explicit decode override")
    parser.add_argument("--full", action="store_true", help="print full content")
    args = parser.parse_args(argv)

    # Windows consoles default to cp1252; fixtures contain日本語.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
    request_id = new_request_id()
    logger.info("manual ingest start", extra={"file_name": args.path})
    try:
        doc = dispatcher.ingest(args.path, encoding=args.encoding)
    except IndexingError as exc:
        print(f"REJECTED [{request_id}]: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"NOT FOUND [{request_id}]: {args.path}", file=sys.stderr)
        return 2

    print(f"OK [{request_id}]")
    print(json.dumps(doc.metadata.model_dump(), indent=2))
    content = doc.content if args.full else doc.content[:PREVIEW_CHARS]
    print(f"--- content ({len(doc.content)} chars) ---")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
