"""MCQ graph state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcq.schemas import DifficultyBlueprint, MCQ, MCQRequest
from shared.metrics import Metrics


@dataclass
class MCQState:
    request: MCQRequest
    output_dir: Path
    metrics: Metrics = field(default_factory=Metrics)
    blueprint: DifficultyBlueprint | None = None
    mcq: MCQ | None = None
    solver: dict[str, Any] = field(default_factory=dict)
    distractors: dict[str, Any] = field(default_factory=dict)
    difficulty_judgement: dict[str, Any] = field(default_factory=dict)
    final: dict[str, Any] = field(default_factory=dict)

