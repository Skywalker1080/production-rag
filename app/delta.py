"""Pure helpers for page-level delta indexing (stdlib only, no langchain).

Units:
  prose page -> key ("p", page)     identity = page number, change = text hash
  table grid -> key ("t", hash[:12]) identity = content hash (position-free)

Point IDs are stable per unit+index so unchanged units are never rewritten:
  prose: uuid5("{collection}:{source}:p:{page}:{i}")
  table: uuid5("{collection}:{source}:t:{hash12}:{i}")
"""
import hashlib
import re
import uuid

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def page_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def chunk_point_id(collection: str, source: str, kind: str, unit_key: str, idx: int) -> str:
    """Stable point id. kind: 'p' (unit_key=page) or 't' (unit_key=hash12).

    Dashed canonical form: Qdrant returns ids dashed, so diffing requires
    the same format on both sides (hex never equals its dashed twin).
    """
    return str(uuid.uuid5(
        uuid.NAMESPACE_URL, f"{collection}:{source}:{kind}:{unit_key}:{idx}"
    ))


def is_legacy_point(payload: dict) -> bool:
    """Old-scheme points carry no content_hash -> one full reset, then delta."""
    return not (payload or {}).get("content_hash")


def plan_delta(old_units: dict, new_units: dict) -> dict:
    """old: key -> {"hash", "ids"}; new: key -> {"hash"}.

    Returns {"skip", "changed", "removed"} as sets of keys.
    A unit is skipped only when the key exists on both sides with equal hash.
    """
    old_keys, new_keys = set(old_units), set(new_units)
    removed = old_keys - new_keys
    skip, changed = set(), set()
    for key in old_keys & new_keys:
        if old_units[key].get("hash") == new_units[key].get("hash"):
            skip.add(key)
        else:
            changed.add(key)
    changed |= new_keys - old_keys
    return {"skip": skip, "changed": changed, "removed": removed}
