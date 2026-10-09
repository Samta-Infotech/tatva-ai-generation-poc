"""PBQ template presets loaded from the Django application source.

The POC intentionally reuses the six canonical PBQ presets from
candidate-screening-backend-django/questions/services/pbq_presets.py so the UI
and generated starter tree match the platform templates.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.settings import ROOT_DIR
from runtime.workspace import validate_relative_path
from shared.schemas import FileEntry


DJANGO_PRESETS_PATH = (
    ROOT_DIR.parent / "candidate-screening-backend-django" / "questions" / "services" / "pbq_presets.py"
)


@lru_cache(maxsize=1)
def _load_django_presets() -> tuple[dict[str, Any], ...]:
    spec = importlib.util.spec_from_file_location("tatva_django_pbq_presets", DJANGO_PRESETS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load PBQ presets from {DJANGO_PRESETS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.PRESETS)


def all_pbq_templates() -> list[dict[str, Any]]:
    return [
        {
            "slug": preset["slug"],
            "name": preset["name"],
            "description": preset["description"],
            "stack_labels": list(preset.get("stack_labels") or []),
            "runtime_profile": preset["runtime_profile"],
            "role": preset["role"],
            "source_roots": list(preset.get("source_roots") or []),
            "test_roots": list(preset.get("test_roots") or []),
            "test_contract": preset.get("test_contract") or "",
            "default_tree": template_tree(preset["slug"]),
            "locked_files": sorted(template_locked_paths(preset["slug"])),
        }
        for preset in _load_django_presets()
    ]


def get_pbq_template(slug: str) -> dict[str, Any]:
    for preset in _load_django_presets():
        if preset["slug"] == slug:
            return preset
    choices = ", ".join(sorted(preset["slug"] for preset in _load_django_presets()))
    raise ValueError(f"Unknown PBQ template '{slug}'. Supported: {choices}")


def template_starter_files(slug: str) -> list[FileEntry]:
    preset = get_pbq_template(slug)
    files: list[FileEntry] = []
    for raw in preset.get("files") or []:
        path = validate_relative_path(str(raw.get("path") or ""))
        kind = str(raw.get("kind") or "file")
        content = str(raw.get("content") or "")
        files.append(FileEntry(path=path, content=content, kind=kind))
    return files


def template_locked_paths(slug: str) -> set[str]:
    preset = get_pbq_template(slug)
    return {
        validate_relative_path(str(raw.get("path") or ""))
        for raw in preset.get("files") or []
        if str(raw.get("kind") or "file") == "file" and str(raw.get("access") or "candidate") == "locked"
    }


def template_tree(slug: str) -> list[str]:
    entries: list[str] = []
    for entry in template_starter_files(slug):
        path = entry.path.rstrip("/")
        entries.append(f"{path}/" if entry.kind == "directory" else path)
    return sorted(set(entries))


def template_generation_context(slug: str) -> dict[str, Any]:
    preset = get_pbq_template(slug)
    return {
        "slug": preset["slug"],
        "name": preset["name"],
        "description": preset["description"],
        "stack_labels": list(preset.get("stack_labels") or []),
        "runtime_profile": preset["runtime_profile"],
        "role": preset["role"],
        "source_roots": list(preset.get("source_roots") or []),
        "test_roots": list(preset.get("test_roots") or []),
        "test_contract": preset.get("test_contract") or "",
        "starter_files": [entry.to_dict() for entry in template_starter_files(slug)],
    }
