"""PBQ graph schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from shared.schemas import FileEntry


@dataclass
class PBQRequest:
    runtime_profile: str
    difficulty: str
    experience: str
    duration: int
    prompt: str
    template_slug: str = ""


@dataclass
class PBQDesign:
    title: str
    behavioral_contract: list[str]
    scaffolding_strategy: str
    candidate_freedom: list[str]
    difficulty_notes: list[str]


@dataclass
class PBQBundle:
    requirements: dict[str, Any]
    design: PBQDesign
    starter_files: list[FileEntry] = field(default_factory=list)
    public_tests: list[FileEntry] = field(default_factory=list)
    private_tests: list[FileEntry] = field(default_factory=list)
    reference_solution: list[FileEntry] = field(default_factory=list)
    mutations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
