"""MCQ graph schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MCQRequest:
    skill: str
    topic: str
    difficulty: str
    experience: str


@dataclass
class DifficultyBlueprint:
    requested_difficulty: str
    target_experience: str
    minimum_reasoning_steps: int
    minimum_concepts: int
    direct_recall_allowed: bool
    scenario_based: bool
    requires_code_analysis: bool
    distractor_quality: str


@dataclass
class MCQOption:
    id: str
    option_text: str
    is_correct: bool = False


@dataclass
class MCQ:
    question: str
    options: list[MCQOption]
    correct_answer: str
    explanation: str
    skills: list[str]
    concepts: list[str]
    difficulty: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

