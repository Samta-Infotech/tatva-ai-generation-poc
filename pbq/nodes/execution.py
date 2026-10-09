from __future__ import annotations

from collections.abc import Callable

from pbq.state import PBQState
from runtime.executor import ensure_npm_cache, ensure_runtime_venv, inject_npm_cache, run_command


def execute_reference(
    state: PBQState,
    on_progress: Callable[[str], None] | None = None,
) -> PBQState:
    assert state.runtime_profile is not None
    reference = state.output_dir / "reference_workspace"

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    extra_env: dict[str, str] | None = None
    deps_ready = False
    dep_manager = state.runtime_profile.metadata.get("dependency_manager")

    if dep_manager == "pip" and state.runtime_profile.allowed_packages:
        _progress("Setting up runtime environment (cached)...")
        extra_env = ensure_runtime_venv(
            state.runtime_profile.id,
            state.runtime_profile.allowed_packages,
        )
        state.runtime_env = extra_env
        deps_ready = True

    elif dep_manager == "npm":
        _progress("Setting up runtime environment (cached)...")
        node_modules_cache = ensure_npm_cache(state.runtime_profile.id, reference)
        inject_npm_cache(node_modules_cache, reference)
        state.runtime_env = {"_npm_cache": str(node_modules_cache)}
        deps_ready = True

    commands = []

    if state.runtime_profile.install_command and not deps_ready:
        _progress(f"Running: {state.runtime_profile.install_command}")
        commands.append(
            run_command(state.runtime_profile.install_command, cwd=reference, extra_env=extra_env).to_dict()
        )

    if state.runtime_profile.validation_command:
        _progress(f"Running: {state.runtime_profile.validation_command}")
        commands.append(
            run_command(state.runtime_profile.validation_command, cwd=reference, extra_env=extra_env).to_dict()
        )

    _progress(f"Running: {state.runtime_profile.test_command}")
    commands.append(run_command(state.runtime_profile.test_command, cwd=reference, extra_env=extra_env).to_dict())

    state.validation["reference_execution"] = commands
    state.validation["reference_passed"] = all(
        (not item.get("skipped")) and item.get("exit_code") == 0 for item in commands
    )
    return state
