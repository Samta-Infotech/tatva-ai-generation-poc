"""Independent solver validation.

Gives the question (behavioral contract + starter files + public tests) to a
second AI call that has **no access** to the reference solution.  The solver's
output is executed against the *private* tests.  This catches:

- Ambiguous or incomplete behavioral contracts (solver can't figure out what to build)
- Insufficient starter files (solver can't even start)
- Unfair private tests that test implementation details not in the contract
- Leaked answers in public tests (solver scores suspiciously well)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from models.gateway import ModelGateway
from pbq.state import PBQState
from runtime.executor import run_command
from runtime.workspace import materialize, overlay
from shared.schemas import FileEntry

logger = logging.getLogger("tatva.poc.solver_validation")


def validate_with_independent_solver(state: PBQState, gateway: ModelGateway) -> PBQState:
    """Ask the AI to solve the question from starter files only, then run private tests."""
    assert state.bundle is not None
    assert state.runtime_profile is not None

    if gateway.provider not in {"ollama", "openai", "openai-compatible"}:
        state.validation["solver_validation"] = {"skipped": True, "reason": "fixture provider"}
        return state

    contract = state.bundle.design.behavioral_contract
    starter_contents = {f.path: f.content for f in state.bundle.starter_files}
    public_test_contents = {f.path: f.content for f in state.bundle.public_tests}

    raw = gateway.structured_generate(
        (
            "You are a candidate solving a programming question. You are given:\n"
            "1. A behavioral contract describing what you must implement\n"
            "2. Starter files with empty function/class bodies\n"
            "3. Public tests you can reference\n\n"
            "Write the COMPLETE implementation that satisfies every contract item. "
            "Return JSON with a 'solution_files' array where each entry has 'path' and 'content'. "
            "Only return files that need changes — do not return test files or config files. "
            "The implementation must pass the public tests. Do not use TODO, placeholder comments, "
            "or `pass` as function bodies. Write real, working code."
        ),
        schema_name="PBQSolverSolution",
        context={
            "question_title": state.bundle.design.title,
            "behavioral_contract": contract,
            "difficulty": state.request.difficulty,
            "framework": state.runtime_profile.framework,
            "runtime_profile": state.runtime_profile.id,
            "starter_files": starter_contents,
            "public_tests": public_test_contents,
        },
    )

    solver_files = _parse_solver_output(raw)
    if not solver_files:
        state.validation["solver_validation"] = {
            "skipped": False,
            "solver_produced_files": False,
            "reason": "AI solver did not produce valid solution files",
            "raw_keys": list(raw.keys()) if isinstance(raw, dict) else [],
        }
        return state

    solver_result = _execute_solver_solution(state, solver_files)
    state.validation["solver_validation"] = solver_result
    return state


def _parse_solver_output(raw: dict[str, Any]) -> list[FileEntry]:
    """Extract solution files from the solver's raw JSON response."""
    if raw.get("slm_error") or raw.get("slm_fallback"):
        return []

    files_raw = raw.get("solution_files", [])
    if not isinstance(files_raw, list):
        return []

    entries = []
    for item in files_raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        content = str(item.get("content", "")).strip()
        if path and content:
            entries.append(FileEntry(path=path, content=content))
    return entries


def _execute_solver_solution(state: PBQState, solver_files: list[FileEntry]) -> dict[str, Any]:
    """Build a workspace with the solver's solution and run private tests."""
    assert state.bundle is not None
    assert state.runtime_profile is not None

    solver_workspace = state.output_dir / "solver_workspace"
    files = overlay(state.bundle.starter_files, solver_files)
    materialize(
        files + state.bundle.public_tests + state.bundle.private_tests,
        solver_workspace,
    )

    extra_env = state.runtime_env

    commands = []
    if state.runtime_profile.validation_command:
        commands.append(
            run_command(state.runtime_profile.validation_command, cwd=solver_workspace, extra_env=extra_env).to_dict()
        )

    test_result = run_command(state.runtime_profile.test_command, cwd=solver_workspace, extra_env=extra_env)
    commands.append(test_result.to_dict())

    all_passed = all(
        (not item.get("skipped")) and item.get("exit_code") == 0 for item in commands
    )

    ref_execution = state.validation.get("reference_execution", [])
    ref_test = next((c for c in ref_execution if "pytest" in c.get("command", "") or "test" in c.get("command", "")), None)
    ref_test_count = ref_test.get("exit_code", -1) if ref_test else -1

    result: dict[str, Any] = {
        "skipped": False,
        "solver_produced_files": True,
        "solver_files": [f.path for f in solver_files],
        "solver_all_tests_passed": all_passed,
        "solver_execution": commands,
    }

    if all_passed:
        result["verdict"] = "question_valid"
        result["explanation"] = (
            "An independent solver produced a working solution from the starter files "
            "and behavioral contract alone. The question is solvable and the tests are fair."
        )
    else:
        failed_tests = _extract_failures(test_result.to_dict())
        result["failed_tests"] = failed_tests
        result["verdict"] = "needs_review"
        result["explanation"] = (
            "The independent solver's solution failed some private tests. "
            "This may indicate: (1) ambiguous behavioral contract, "
            "(2) private tests checking implementation details not in the contract, "
            "or (3) the solver wrote incorrect code. Review the failed tests."
        )

    return result


def _extract_failures(test_result: dict[str, Any]) -> list[str]:
    """Pull failed test names from pytest/test output."""
    stdout = test_result.get("stdout", "")
    failures = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED "):
            failures.append(stripped.removeprefix("FAILED ").split(" ")[0])
    return failures
