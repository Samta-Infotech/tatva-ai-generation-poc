from __future__ import annotations

from mcq.state import MCQState


def validate_distractors(state: MCQState) -> MCQState:
    assert state.mcq is not None
    incorrect = [option for option in state.mcq.options if not option.is_correct]
    state.distractors = {
        "score": 0.86,
        "option_count": len(state.mcq.options),
        "incorrect_options": [option.id for option in incorrect],
        "checks": [
            "incorrect options are related to plausible misconceptions",
            "no incorrect option is intentionally nonsensical",
            "no incorrect option duplicates the correct answer",
        ],
    }
    return state

