from __future__ import annotations

from pbq.state import PBQState
from runtime.mutation_runner import run_mutations


def validate_test_strength(state: PBQState) -> PBQState:
    assert state.runtime_profile is not None
    assert state.bundle is not None
    state.mutation = run_mutations(
        profile=state.runtime_profile,
        starter_files=state.bundle.starter_files,
        reference_solution=state.bundle.reference_solution,
        private_tests=state.bundle.private_tests,
        mutations=state.bundle.mutations,
        extra_env=state.runtime_env,
    )
    state.validation["mutation_threshold"] = 0.8
    state.validation["tests_strong_enough"] = state.mutation["mutation_score"] >= 0.8
    return state

