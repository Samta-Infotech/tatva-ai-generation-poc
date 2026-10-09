"""Workspace materialization helpers."""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

from shared.schemas import FileEntry


def validate_relative_path(raw_path: str) -> str:
    value = str(raw_path or "").strip().replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value.startswith("./"):
        raise ValueError(f"Unsafe workspace path: {raw_path}")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"Unsafe workspace path: {raw_path}")
    return value


def materialize(files: list[FileEntry], destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in files:
        relative = validate_relative_path(entry.path)
        target = destination / relative
        if entry.kind == "directory":
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry.content, encoding="utf-8")


def overlay(base: list[FileEntry], changes: list[FileEntry]) -> list[FileEntry]:
    by_path = {item.path: item for item in base}
    for change in changes:
        by_path[change.path] = change
    return list(by_path.values())


def tree(root: Path) -> list[str]:
    if not root.exists():
        return []
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if any(part in {".pytest_cache", "__pycache__", "node_modules", "target"} for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        entries.append(f"{rel}/" if path.is_dir() else rel)
    return entries
