"""Mutation testing for generated PBQ workspaces."""

from __future__ import annotations

import tempfile
from pathlib import Path

from config.runtime_profiles import RuntimeProfile
from runtime.executor import inject_npm_cache, run_command
from runtime.workspace import materialize, overlay
from shared.schemas import FileEntry


def _is_infrastructure_failure(result: object) -> bool:
    exit_code = getattr(result, "exit_code", None)
    stderr = str(getattr(result, "stderr", "") or "").lower()
    return bool(
        getattr(result, "skipped", False)
        or getattr(result, "timed_out", False)
        or exit_code in {None, 127}
        or "not found" in stderr
        or "executable not found" in stderr
    )


def run_mutations(
    *,
    profile: RuntimeProfile,
    starter_files: list[FileEntry],
    reference_solution: list[FileEntry],
    private_tests: list[FileEntry],
    mutations: list[dict],
    extra_env: dict[str, str] | None = None,
) -> dict:
    npm_cache_path: Path | None = None
    pip_env: dict[str, str] | None = None
    if extra_env:
        cache_str = extra_env.get("_npm_cache")
        if cache_str:
            npm_cache_path = Path(cache_str)
        else:
            pip_env = extra_env

    results = []
    caught = 0
    conclusive = 0
    inconclusive = 0
    for mutation in mutations:
        files = overlay(starter_files, reference_solution)
        files = overlay(files, [FileEntry(**item) for item in mutation["changes"]])
        with tempfile.TemporaryDirectory(prefix="tatva-mutant-") as tmp:
            workspace = Path(tmp)
            materialize(files + private_tests, workspace)

            if npm_cache_path and npm_cache_path.is_dir():
                inject_npm_cache(npm_cache_path, workspace)
                result = run_command(profile.test_command, cwd=workspace)
            elif pip_env:
                result = run_command(profile.test_command, cwd=workspace, extra_env=pip_env)
            elif profile.install_command:
                install = run_command(profile.install_command, cwd=workspace)
                if _is_infrastructure_failure(install) or install.exit_code != 0:
                    result = install
                else:
                    result = run_command(profile.test_command, cwd=workspace)
            else:
                result = run_command(profile.test_command, cwd=workspace)

        infrastructure_failure = _is_infrastructure_failure(result)
        killed = (not infrastructure_failure) and result.exit_code != 0
        if infrastructure_failure:
            inconclusive += 1
        else:
            conclusive += 1
        caught += 1 if killed else 0
        results.append(
            {
                "name": mutation["name"],
                "killed": killed,
                "inconclusive": infrastructure_failure,
                "execution": result.to_dict(),
            }
        )
    score = caught / conclusive if conclusive else 0.0
    return {
        "mutation_score": score,
        "mutations_caught": caught,
        "total_mutations": len(mutations),
        "conclusive_mutations": conclusive,
        "inconclusive_mutations": inconclusive,
        "results": results,
    }
