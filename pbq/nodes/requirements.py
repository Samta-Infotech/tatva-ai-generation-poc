from __future__ import annotations

from pbq.state import PBQState


def analyze_pbq_requirements(state: PBQState) -> PBQState:
    profile = state.runtime_profile
    assert profile is not None
    state.bundle.requirements.update(
        {
            "prompt": state.request.prompt,
            "runtime_profile": profile.to_dict(),
            "requested_difficulty": state.request.difficulty,
            "experience": state.request.experience,
            "duration_minutes": state.request.duration,
            "candidate_workspace_policy": "full executable workspace; candidates may create files under source roots",
        }
    )
    return state

