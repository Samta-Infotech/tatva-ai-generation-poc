from __future__ import annotations

from mcq.schemas import DifficultyBlueprint
from mcq.state import MCQState
from models.gateway import ModelGateway


def create_difficulty_blueprint(state: MCQState) -> MCQState:
    difficulty = state.request.difficulty.lower()
    state.blueprint = DifficultyBlueprint(
        requested_difficulty=difficulty,
        target_experience=state.request.experience,
        minimum_reasoning_steps={"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2),
        minimum_concepts={"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2),
        direct_recall_allowed=difficulty == "easy",
        scenario_based=difficulty in {"medium", "hard"},
        requires_code_analysis=difficulty == "hard",
        distractor_quality={"easy": "basic", "medium": "plausible", "hard": "high"}.get(difficulty, "plausible"),
    )
    return state


def validate_difficulty(state: MCQState, gateway: ModelGateway) -> MCQState:
    assert state.blueprint is not None
    state.difficulty_judgement = gateway.judge(
        "Independently classify the MCQ difficulty from the question and options.",
        context={"difficulty": state.request.difficulty, "blueprint": state.blueprint.__dict__},
    )
    return state

