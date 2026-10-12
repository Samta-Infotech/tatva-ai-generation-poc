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
        return {
            "requirements": self.requirements,
            "design": asdict(self.design),
            "starter_files": [f.to_dict() for f in self.starter_files],
            "public_tests": [f.to_dict() for f in self.public_tests],
            "private_tests": [f.to_dict() for f in self.private_tests],
            "reference_solution": [f.to_dict() for f in self.reference_solution],
            "mutations": self.mutations,
        }
