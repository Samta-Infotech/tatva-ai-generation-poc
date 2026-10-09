from __future__ import annotations

from mcq.state import MCQState


def repair_mcq(state: MCQState, reason: str) -> MCQState:
    state.metrics.repair_count += 1
    state.final.setdefault("repairs", []).append(reason)
    return state

