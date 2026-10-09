"""Shared POC schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CommandResult:
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileEntry:
    path: str
    content: str
    kind: str = "file"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class BenchmarkMetric:
    name: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)

