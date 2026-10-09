from __future__ import annotations

from pbq.state import PBQState


def validate_pbq_design(state: PBQState) -> tuple[bool, list[str]]:
    assert state.bundle is not None
    issues = []
    if len(state.bundle.design.behavioral_contract) < 3:
        issues.append("behavioral contract is too thin")
    if "single file" in state.bundle.design.scaffolding_strategy.lower():
        issues.append("design assumes one editable file")
    return (not issues, issues)


def repair_pbq_design(state: PBQState, issues: list[str]) -> PBQState:
    assert state.bundle is not None
    state.metrics.repair_count += 1
    state.bundle.design.behavioral_contract.append("External behavior must be testable without matching reference file layout.")
    state.bundle.design.difficulty_notes.append(f"Repair applied: {', '.join(issues)}")
    return state

