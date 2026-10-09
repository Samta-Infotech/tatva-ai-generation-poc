from __future__ import annotations

from models.gateway import ModelGateway
from pbq.state import PBQState


def validate_pbq_difficulty(state: PBQState, gateway: ModelGateway) -> PBQState:
    assert state.bundle is not None
    state.difficulty = gateway.judge(
        "Classify the PBQ difficulty from its behavioral contract and workspace scaffolding.",
        context={
            "requested_difficulty": state.request.difficulty,
            "difficulty": state.request.difficulty,
            "file_count": len(state.bundle.starter_files) + len(state.bundle.reference_solution),
            "contract": state.bundle.design.behavioral_contract,
        },
    )
    state.validation["difficulty_matches_request"] = (
        state.difficulty.get("predicted_difficulty") == state.request.difficulty
    )
    return state

