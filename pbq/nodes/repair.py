from __future__ import annotations

from pbq.state import PBQState


def repair_reference(state: PBQState) -> PBQState:
    state.metrics.repair_count += 1
    state.validation.setdefault("repairs", []).append("reference execution failed; no automatic code rewrite in fixture POC")
    return state


def repair_tests(state: PBQState) -> PBQState:
    state.metrics.repair_count += 1
    state.validation.setdefault("repairs", []).append("mutation score below threshold; no automatic test rewrite in fixture POC")
    return state

