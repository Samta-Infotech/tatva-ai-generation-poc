from __future__ import annotations

from mcq.state import MCQState
from models.gateway import ModelGateway


def independent_solver(state: MCQState, gateway: ModelGateway) -> MCQState:
    assert state.mcq is not None
    state.solver = gateway.solve(
        "Solve the question from question text and options only. Do not use the generator answer.",
        question=state.mcq.question,
        options=[option.__dict__ for option in state.mcq.options],
    )
    return state

