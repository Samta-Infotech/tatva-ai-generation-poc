from __future__ import annotations

from pbq.state import PBQState
from runtime.workspace import materialize, overlay, tree


def generate_candidate_workspace(state: PBQState) -> PBQState:
    assert state.bundle is not None
    candidate = state.output_dir / "candidate_workspace"
    materialize(state.bundle.starter_files + state.bundle.public_tests, candidate)
    state.validation["candidate_workspace_tree"] = tree(candidate)
    return state


def materialize_reference_workspace(state: PBQState) -> PBQState:
    assert state.bundle is not None
    reference = state.output_dir / "reference_workspace"
    files = overlay(state.bundle.starter_files, state.bundle.reference_solution)
    materialize(files + state.bundle.public_tests + state.bundle.private_tests, reference)
    state.validation["reference_workspace_tree"] = tree(reference)
    return state

