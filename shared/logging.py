"""Logging helpers for the POC."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("tatva.poc.logging")


def _break_cycles(obj: Any, _seen: set[int] | None = None) -> Any:
    """Return a shallow copy of *obj* with circular references replaced by a sentinel."""
    if _seen is None:
        _seen = set()
    if isinstance(obj, dict):
        obj_id = id(obj)
        if obj_id in _seen:
            return "<circular ref>"
        _seen.add(obj_id)
        result = {k: _break_cycles(v, _seen) for k, v in obj.items()}
        _seen.discard(obj_id)
        return result
    if isinstance(obj, (list, tuple)):
        obj_id = id(obj)
        if obj_id in _seen:
            return "<circular ref>"
        _seen.add(obj_id)
        items = [_break_cycles(v, _seen) for v in obj]
        _seen.discard(obj_id)
        return type(obj)(items) if isinstance(obj, tuple) else items
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(payload, indent=2, sort_keys=True)
    except (ValueError, RecursionError):
        logger.warning("Circular reference in payload for %s — stripping cycles", path.name)
        text = json.dumps(_break_cycles(payload), indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

