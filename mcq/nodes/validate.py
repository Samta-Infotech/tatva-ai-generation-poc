from __future__ import annotations

from mcq.state import MCQState


def validate_mcq_structure(state: MCQState) -> tuple[bool, list[str]]:
    assert state.mcq is not None
    issues = []
    if len(state.mcq.options) != 4:
        issues.append("MCQ must have exactly four options")
    if sum(1 for option in state.mcq.options if option.is_correct) != 1:
        issues.append("MCQ must have exactly one correct option")
    if state.mcq.correct_answer not in {option.id for option in state.mcq.options}:
        issues.append("correct_answer must reference an option id")
    return (not issues, issues)


def validate_answer(state: MCQState) -> bool:
    assert state.mcq is not None
    return state.solver.get("answer_id") == state.mcq.correct_answer

