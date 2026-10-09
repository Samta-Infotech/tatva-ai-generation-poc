"""FastAPI backend for the TATVA AI Generation POC UI and API."""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("tatva.poc")

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config.runtime_profiles import all_runtime_profiles, get_runtime_profile
from config.pbq_templates import all_pbq_templates, get_pbq_template, template_starter_files
from config import settings
from shared import progress as prog
from config.settings import ROOT_DIR
from main import PBQ_BENCHMARK_PROMPTS
from mcq.graph import finalize_mcq, run_mcq_graph
from mcq.nodes.difficulty import validate_difficulty
from mcq.nodes.distractors import validate_distractors
from mcq.nodes.solve import independent_solver
from mcq.nodes.validate import validate_answer, validate_mcq_structure
from mcq.schemas import DifficultyBlueprint, MCQ, MCQOption, MCQRequest
from mcq.state import MCQState
from models.gateway import ModelGateway
from pbq.graph import (
    ensure_pbq_runtime_support,
    finalize_pbq,
    normalize_pbq_design_from_slm_metadata,
    repair_pbq_from_validation_feedback,
    run_pbq_graph,
)
from pbq.nodes.difficulty import validate_pbq_difficulty
from pbq.nodes.execution import execute_reference
from pbq.nodes.mutation import validate_test_strength
from pbq.nodes.workspace import generate_candidate_workspace, materialize_reference_workspace
from pbq.schemas import PBQBundle, PBQDesign, PBQRequest
from pbq.state import PBQState
from runtime.workspace import validate_relative_path
from shared.logging import write_json
from shared.metrics import Metrics
from shared.schemas import FileEntry
from ui import INDEX_HTML, _load_artifact, _results_payload, _run_mcq_benchmark, _run_pbq_benchmark


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="TATVA AI Generation POC",
    version="0.1.0",
    description="FastAPI layer for running PBQ/MCQ generation graphs and inspecting artifacts.",
)

# Thread pool for background PBQ/MCQ graph runs so the API stays responsive.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="poc-graph")


@app.on_event("startup")
def _startup() -> None:
    tasks_dir = settings.OUTPUT_DIR / "tasks"
    prog.configure(tasks_dir)
    logger.info("TATVA POC API started — tasks dir: %s", tasks_dir)


class PBQRunRequest(BaseModel):
    template: str = Field(default="django-sqlite-starter")
    runtime: str = Field(default="django-sqlite-py312")
    difficulty: str = Field(default="hard")
    experience: str = Field(default="4-6 years")
    duration: int = Field(default=90, ge=1)
    prompt: str = Field(default=PBQ_BENCHMARK_PROMPTS["django-sqlite-py312"])
    design_only: bool = Field(default=False)


class MCQRunRequest(BaseModel):
    skill: str = Field(default="Python")
    topic: str = Field(default="concurrency")
    difficulty: str = Field(default="hard")
    experience: str = Field(default="4-6 years")


class PBQValidateRequest(BaseModel):
    runtime: str = Field(default="django-sqlite-py312")


class PBQRepairRequest(BaseModel):
    runtime: str = Field(default="django-sqlite-py312")
    reason: str = Field(default="Fix the PBQ using the latest validation errors.")


class PBQFileSaveRequest(BaseModel):
    runtime: str = Field(default="django-sqlite-py312")
    section: str
    path: str
    content: str


class PBQDesignSaveRequest(BaseModel):
    runtime: str = Field(default="django-sqlite-py312")
    title: str | None = None
    prompt: str | None = None
    behavioral_contract: list[str] | None = None
    scaffolding_strategy: str | None = None
    candidate_freedom: list[str] | None = None
    difficulty_notes: list[str] | None = None


class MCQValidateRequest(BaseModel):
    difficulty: str = Field(default="hard")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/profiles")
def profiles() -> dict[str, list[dict[str, Any]]]:
    return {"profiles": [profile.to_dict() for profile in all_runtime_profiles()]}


@app.get("/api/templates")
def templates() -> dict[str, list[dict[str, Any]]]:
    return {"templates": all_pbq_templates()}


@app.get("/api/template-files/{slug}")
def template_files(slug: str) -> dict[str, Any]:
    try:
        entries = template_starter_files(slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "slug": slug,
        "files": [{"path": e.path, "content": e.content, "kind": e.kind} for e in entries],
    }


@app.get("/api/task/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    task = prog.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return task.to_dict()


@app.get("/api/progress/{runtime}")
def get_progress(runtime: str) -> dict[str, Any]:
    task = prog.get_runtime_task(runtime)
    if task is None:
        raise HTTPException(status_code=404, detail=f"No active task for runtime '{runtime}'.")
    return task.to_dict()


@app.get("/api/results")
def results() -> dict[str, Any]:
    return _results_payload()


@app.get("/api/slm/status")
def slm_status() -> dict[str, Any]:
    gateway = ModelGateway(provider=settings.MODEL_PROVIDER)
    provider = settings.MODEL_PROVIDER
    is_fixture = provider == "fixture"
    return {
        "provider": provider,
        "is_fixture_mode": is_fixture,
        "fixture_mode_note": (
            "Running in fixture mode — no real AI is called. "
            "Set MODEL_PROVIDER=ollama or openai in .env to use an actual SLM/LLM."
        ) if is_fixture else None,
        "ollama_base_url_configured": bool(settings.OLLAMA_BASE_URL),
        "ollama_model": settings.OLLAMA_MODEL,
        "generator_model": settings.GENERATOR_MODEL,
        "judge_model": settings.JUDGE_MODEL,
        "solver_model": settings.SOLVER_MODEL,
        "repair_model": settings.REPAIR_MODEL,
        "effective_generator_model": gateway.generator_model,
        "effective_judge_model": gateway.judge_model,
        "effective_solver_model": gateway.solver_model,
        "effective_repair_model": gateway.repair_model,
        "openai_base_url_configured": bool(settings.OPENAI_BASE_URL),
        "openai_model": settings.OPENAI_MODEL,
        "openai_api_key_configured": bool(settings.OPENAI_API_KEY),
        "cloudflare_access_configured": bool(
            settings.OLLAMA_CF_ACCESS_CLIENT_ID and settings.OLLAMA_CF_ACCESS_CLIENT_SECRET
        ),
    }


@app.post("/api/slm/test")
def slm_test() -> dict[str, Any]:
    gateway = ModelGateway(provider=settings.MODEL_PROVIDER)
    result = gateway.structured_generate(
        "Return JSON confirming the SLM gateway is reachable.",
        schema_name="SLMHealthCheck",
        context={"expected": "ok"},
    )
    return {"status": "ok" if not result.get("slm_error") else "failed", "result": result}


@app.get("/api/artifact")
def artifact(path: str) -> dict[str, Any]:
    try:
        return {"path": path, "content": _load_artifact(path)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/review-packet")
def review_packet(runtime: str = "django-sqlite-py312") -> dict[str, Any]:
    return _review_packet_payload(runtime)


@app.get("/api/pbq-workspace")
def pbq_workspace(runtime: str = "django-sqlite-py312") -> dict[str, Any]:
    return _pbq_workspace_payload(runtime)


@app.post("/api/pbq-workspace/file")
def save_pbq_workspace_file(payload: PBQFileSaveRequest) -> dict[str, Any]:
    bundle, output_dir = _load_saved_pbq_bundle(payload.runtime, ensure_support=True)
    section = _section_name(payload.section)
    path = validate_relative_path(payload.path)
    files = list(getattr(bundle, section))
    for index, entry in enumerate(files):
        if entry.path == path:
            files[index] = FileEntry(path=path, content=payload.content, kind=entry.kind or "file")
            break
    else:
        files.append(FileEntry(path=path, content=payload.content, kind="file"))
    setattr(bundle, section, files)
    _write_pbq_bundle(output_dir, bundle)
    return _pbq_workspace_payload(payload.runtime)


@app.post("/api/pbq-workspace/design")
def save_pbq_workspace_design(payload: PBQDesignSaveRequest) -> dict[str, Any]:
    bundle, output_dir = _load_saved_pbq_bundle(payload.runtime, ensure_support=True)
    if payload.title is not None:
        bundle.design.title = payload.title
    if payload.behavioral_contract is not None:
        bundle.design.behavioral_contract = [item for item in payload.behavioral_contract if item.strip()]
    if payload.scaffolding_strategy is not None:
        bundle.design.scaffolding_strategy = payload.scaffolding_strategy
    if payload.candidate_freedom is not None:
        bundle.design.candidate_freedom = [item for item in payload.candidate_freedom if item.strip()]
    if payload.difficulty_notes is not None:
        bundle.design.difficulty_notes = [item for item in payload.difficulty_notes if item.strip()]
    if payload.prompt is not None:
        bundle.requirements["prompt"] = payload.prompt
    _write_pbq_bundle(output_dir, bundle)
    return _pbq_workspace_payload(payload.runtime)


@app.post("/api/pbq")
def run_pbq(payload: PBQRunRequest) -> dict[str, Any]:
    template = get_pbq_template(payload.template)
    runtime = template["runtime_profile"]
    task_id = uuid.uuid4().hex[:10]
    steps = prog.DESIGN_ONLY_STEPS if payload.design_only else prog.GENERATE_STEPS
    task = prog.create_task(task_id, "generate", runtime, steps)
    _executor.submit(_run_generate_task, task, payload, runtime)
    return {"task_id": task_id, "runtime": runtime, "status": "accepted"}


@app.post("/api/mcq")
def run_mcq(payload: MCQRunRequest) -> dict[str, Any]:
    state = run_mcq_graph(
        MCQRequest(
            skill=payload.skill,
            topic=payload.topic,
            difficulty=payload.difficulty,
            experience=payload.experience,
        )
    )
    return {"result": state.final, "output_dir": str(state.output_dir.relative_to(ROOT_DIR))}


@app.post("/api/validate-pbq")
def validate_pbq(payload: PBQValidateRequest) -> dict[str, Any]:
    task_id = uuid.uuid4().hex[:10]
    task = prog.create_task(task_id, "validate", payload.runtime, prog.VALIDATE_STEPS)
    _executor.submit(_run_validate_task, task, payload.runtime, False, "")
    return {"task_id": task_id, "runtime": payload.runtime, "status": "accepted"}


@app.post("/api/regenerate-pbq-from-validation")
def regenerate_pbq_from_validation(payload: PBQRepairRequest) -> dict[str, Any]:
    task_id = uuid.uuid4().hex[:10]
    task = prog.create_task(task_id, "regenerate", payload.runtime, prog.REGENERATE_STEPS)
    _executor.submit(_run_validate_task, task, payload.runtime, True, payload.reason)
    return {"task_id": task_id, "runtime": payload.runtime, "status": "accepted"}


@app.post("/api/validate-mcq")
def validate_mcq(payload: MCQValidateRequest) -> dict[str, Any]:
    state = _validate_saved_mcq(payload.difficulty)
    return {"result": state.final, "output_dir": str(state.output_dir.relative_to(ROOT_DIR))}


@app.post("/api/benchmark-pbq")
def benchmark_pbq() -> dict[str, Any]:
    return {"summary": _run_pbq_benchmark()}


@app.post("/api/benchmark-mcq")
def benchmark_mcq() -> dict[str, Any]:
    return {"summary": _run_mcq_benchmark()}


def _make_progress_callback(task: prog.TaskProgress):
    """Return a callback that routes run_pbq_graph progress events into task steps."""
    def _cb(step_key: str, status: str, msg: str = "") -> None:
        step = task.get_step(step_key)
        if step is None:
            logger.debug("unknown step key '%s' for task %s", step_key, task.task_id)
            return
        if status == "start":
            step.start(msg)
            logger.info("task %s [%s] → %s", task.task_id, step_key, msg or "started")
        elif status == "done":
            step.done(msg)
            logger.info("task %s [%s] ✓ %s (%.1fs)", task.task_id, step_key, msg or "", step.elapsed_seconds() or 0)
        elif status == "error":
            step.error(msg)
            logger.error("task %s [%s] ✗ %s", task.task_id, step_key, msg)
        elif status == "skip":
            step.skip(msg)
            logger.info("task %s [%s] skipped: %s", task.task_id, step_key, msg)
        elif status == "update":
            # Message-only update for a step already running — does not change timing
            if msg:
                step.message = msg
            logger.info("task %s [%s] update: %s", task.task_id, step_key, msg)
        prog._persist(task)
    return _cb


def _run_generate_task(task: prog.TaskProgress, payload: PBQRunRequest, runtime: str) -> None:
    logger.info("generate task %s started: runtime=%s difficulty=%s design_only=%s", task.task_id, runtime, payload.difficulty, payload.design_only)
    try:
        state = run_pbq_graph(
            PBQRequest(
                runtime_profile=runtime,
                difficulty=payload.difficulty,
                experience=payload.experience,
                duration=payload.duration,
                prompt=payload.prompt,
                template_slug=payload.template,
            ),
            on_progress=_make_progress_callback(task),
            design_only=payload.design_only,
        )
        task.result = state.final
        task.overall_status = "done"
        prog._persist(task)
        logger.info("generate task %s done: status=%s", task.task_id, state.final.get("final_status"))
    except Exception as exc:
        logger.exception("generate task %s failed: %s", task.task_id, exc)
        task.abort_remaining(str(exc))


def _run_validate_task(
    task: prog.TaskProgress,
    runtime: str,
    repair_with_ai: bool,
    repair_reason: str,
) -> None:
    op = task.operation  # "validate" or "regenerate"
    logger.info("%s task %s started: runtime=%s", op, task.task_id, runtime)
    cb = _make_progress_callback(task)
    try:
        cb("load_bundle", "start", "Loading saved PBQ bundle...")
        bundle, output_dir = _load_saved_pbq_bundle(runtime, ensure_support=True)
        _write_pbq_bundle(output_dir, bundle)
        existing = _read_json(output_dir / "pbq_result.json") if (output_dir / "pbq_result.json").is_file() else {}
        difficulty = str(existing.get("requested_difficulty") or "hard")
        state = PBQState(
            request=PBQRequest(
                runtime_profile=runtime,
                difficulty=difficulty,
                experience="stored validation",
                duration=90,
                prompt=str((bundle.requirements or {}).get("prompt") or "Stored PBQ validation"),
            ),
            output_dir=output_dir,
            metrics=Metrics(),
        )
        state.runtime_profile = get_runtime_profile(runtime)
        state.bundle = bundle
        cb("load_bundle", "done", f"Loaded {len(bundle.starter_files)} starter files")

        gateway = ModelGateway(provider=settings.MODEL_PROVIDER, metrics=state.metrics)
        reason = repair_reason or "Fix using latest validation feedback"

        if repair_with_ai:
            # Regenerate path: AI repair first, then rebuild and test
            cb("build_workspace", "start", "Rebuilding workspaces...")
            state = generate_candidate_workspace(state)
            state = materialize_reference_workspace(state)
            cb("build_workspace", "done")

            cb("ai_repair_ref", "start", "Repairing PBQ with AI...")
            state = repair_pbq_from_validation_feedback(state, gateway, reason)
            cb("ai_repair_ref", "done")
        else:
            # Validate path: just build workspaces, no AI repair yet
            step = task.get_step("ai_repair_ref")
            if step:
                step.skip("validate-only mode")

            cb("build_workspace", "start", "Building candidate and reference workspaces...")
            state = generate_candidate_workspace(state)
            state = materialize_reference_workspace(state)
            cb("build_workspace", "done")

        cb("execute_tests", "start", "Running install, validation, and test commands...")
        _exec_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        _exec_fut = _exec_pool.submit(execute_reference, state)
        try:
            state = _exec_fut.result(timeout=settings.EXECUTE_REF_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.warning("execute_reference timed out after %ss in %s task", settings.EXECUTE_REF_TIMEOUT_SECONDS, op)
            state.validation["reference_passed"] = False
            state.validation.setdefault("reference_execution", []).append(
                {"command": "wall-clock-timeout", "exit_code": None, "timed_out": True, "skipped": False}
            )
        except Exception as exc:
            logger.exception("execute_reference failed in %s task: %s", op, exc)
            state.validation["reference_passed"] = False
        finally:
            _exec_pool.shutdown(wait=False)
        ref_passed = state.validation.get("reference_passed", False)
        cb("execute_tests", "done", f"Reference {'passed' if ref_passed else 'failed'}")
        logger.info("%s task %s: reference_passed=%s", op, task.task_id, ref_passed)

        cb("mutation_testing", "start", "Running mutation testing...")
        state = validate_test_strength(state)
        score = state.validation.get("mutation_score", "n/a")
        tests_strong = state.validation.get("tests_strong_enough", False)
        cb("mutation_testing", "done", f"Score: {score}")
        logger.info("%s task %s: mutation_score=%s tests_strong=%s", op, task.task_id, score, tests_strong)

        if repair_with_ai and not tests_strong:
            cb("ai_repair_tests", "start", "Tests weak — repairing with AI...")
            state = repair_pbq_from_validation_feedback(
                state, gateway, repair_reason or "private tests did not catch mutations"
            )
            if state.validation.get("reference_passed"):
                state = validate_test_strength(state)
            cb("ai_repair_tests", "done")
        else:
            step = task.get_step("ai_repair_tests")
            if step and step.status == "pending":
                step.skip("not needed" if tests_strong else "validate-only mode")

        cb("validate_difficulty", "start", "Validating difficulty with AI judge...")
        state = validate_pbq_difficulty(state, gateway)
        cb("validate_difficulty", "done")

        cb("finalize", "start", "Writing artifacts and review packet...")
        state = finalize_pbq(state)
        final_status = state.final.get("final_status", "unknown")
        cb("finalize", "done", f"Status: {final_status}")

        task.result = state.final
        task.overall_status = "done"
        prog._persist(task)
        logger.info("%s task %s done: status=%s", op, task.task_id, final_status)
    except Exception as exc:
        logger.exception("%s task %s failed: %s", op, task.task_id, exc)
        task.abort_remaining(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TATVA AI Generation POC FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    reload_kwargs = {}
    if args.reload:
        reload_kwargs["reload_excludes"] = ["outputs"]
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload, **reload_kwargs)


def _read_json(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _file_entries(raw_items: list[dict[str, Any]]) -> list[FileEntry]:
    return [FileEntry(path=item["path"], content=item.get("content", ""), kind=item.get("kind", "file")) for item in raw_items]


def _bundle_from_json(data: dict[str, Any]) -> PBQBundle:
    design = data.get("design") or {}
    return PBQBundle(
        requirements=data.get("requirements") or {},
        design=PBQDesign(
            title=str(design.get("title") or "Stored PBQ"),
            behavioral_contract=[str(item) for item in design.get("behavioral_contract") or []],
            scaffolding_strategy=str(design.get("scaffolding_strategy") or ""),
            candidate_freedom=[str(item) for item in design.get("candidate_freedom") or []],
            difficulty_notes=[str(item) for item in design.get("difficulty_notes") or []],
        ),
        starter_files=_file_entries(data.get("starter_files") or []),
        public_tests=_file_entries(data.get("public_tests") or []),
        private_tests=_file_entries(data.get("private_tests") or []),
        reference_solution=_file_entries(data.get("reference_solution") or []),
        mutations=list(data.get("mutations") or []),
    )


PBQ_FILE_SECTIONS = {
    "starter": "starter_files",
    "starter_files": "starter_files",
    "public": "public_tests",
    "public_tests": "public_tests",
    "private": "private_tests",
    "private_tests": "private_tests",
    "reference": "reference_solution",
    "reference_solution": "reference_solution",
}


def _section_name(raw: str) -> str:
    try:
        return PBQ_FILE_SECTIONS[raw]
    except KeyError as exc:
        choices = ", ".join(sorted(PBQ_FILE_SECTIONS))
        raise HTTPException(status_code=400, detail=f"Unknown PBQ section '{raw}'. Use one of: {choices}") from exc


def _pbq_output_dir(runtime: str):
    direct = settings.OUTPUT_DIR / "pbq" / runtime
    if (direct / "pbq_bundle.json").is_file():
        return direct
    benchmark = settings.OUTPUT_DIR / "benchmarks" / "pbq" / runtime
    if (benchmark / "pbq_bundle.json").is_file():
        return benchmark
    raise HTTPException(status_code=404, detail=f"No generated PBQ found for runtime '{runtime}'. Run PBQ first.")


def _load_saved_pbq_bundle(runtime: str, *, ensure_support: bool = False) -> tuple[PBQBundle, Path]:
    output_dir = _pbq_output_dir(runtime)
    bundle = _bundle_from_json(_read_json(output_dir / "pbq_bundle.json"))
    bundle = normalize_pbq_design_from_slm_metadata(bundle)
    if ensure_support:
        bundle = ensure_pbq_runtime_support(bundle, get_runtime_profile(runtime))
    return bundle, output_dir


def _write_pbq_bundle(output_dir: Path, bundle: PBQBundle) -> None:
    write_json(output_dir / "pbq_bundle.json", bundle.to_dict())


def _file_payload(section: str, files: list[FileEntry]) -> list[dict[str, str]]:
    return [
        {"section": section, "path": item.path, "content": item.content, "kind": item.kind}
        for item in sorted(files, key=lambda entry: entry.path)
    ]


def _pbq_workspace_payload(runtime: str) -> dict[str, Any]:
    bundle, output_dir = _load_saved_pbq_bundle(runtime, ensure_support=True)
    _write_pbq_bundle(output_dir, bundle)
    result_path = output_dir / "pbq_result.json"
    profile = get_runtime_profile(runtime)
    return {
        "runtime": runtime,
        "output_dir": _safe_relative(output_dir),
        "design": {
            "title": bundle.design.title,
            "behavioral_contract": bundle.design.behavioral_contract,
            "scaffolding_strategy": bundle.design.scaffolding_strategy,
            "candidate_freedom": bundle.design.candidate_freedom,
            "difficulty_notes": bundle.design.difficulty_notes,
        },
        "prompt": str((bundle.requirements or {}).get("prompt") or ""),
        "approved_dependencies": profile.allowed_packages,
        "files": (
            _file_payload("starter_files", bundle.starter_files)
            + _file_payload("public_tests", bundle.public_tests)
            + _file_payload("private_tests", bundle.private_tests)
            + _file_payload("reference_solution", bundle.reference_solution)
        ),
        "mutations": bundle.mutations,
        "result": _read_json(result_path) if result_path.is_file() else None,
    }


REVIEW_PACKET_FILES = (
    "QUESTION.md",
    "CANDIDATE_TREE.txt",
    "REFERENCE_TREE.txt",
    "STARTER_FILES.md",
    "REFERENCE_SOLUTION.md",
    "PUBLIC_TESTS.md",
    "PRIVATE_TESTS.md",
    "VALIDATION_OUTPUT.md",
)


def _safe_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()


def _review_packet_payload(runtime: str) -> dict[str, Any]:
    output_dir = _pbq_output_dir(runtime)
    packet_dir = output_dir / "review_packet"
    if not packet_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"No review packet found for runtime '{runtime}'. Generate or validate the PBQ first.",
        )
    files: dict[str, str] = {}
    missing: list[str] = []
    for name in REVIEW_PACKET_FILES:
        path = packet_dir / name
        if path.is_file():
            files[name] = path.read_text(encoding="utf-8")
        else:
            missing.append(name)
    result_path = output_dir / "pbq_result.json"
    return {
        "runtime": runtime,
        "output_dir": _safe_relative(output_dir),
        "review_packet_path": _safe_relative(packet_dir),
        "result": _read_json(result_path) if result_path.is_file() else None,
        "files": files,
        "missing": missing,
    }


def _validate_saved_pbq(runtime: str, *, repair_with_ai: bool = False, repair_reason: str = "") -> PBQState:
    bundle, output_dir = _load_saved_pbq_bundle(runtime, ensure_support=True)
    _write_pbq_bundle(output_dir, bundle)
    existing = _read_json(output_dir / "pbq_result.json") if (output_dir / "pbq_result.json").is_file() else {}
    difficulty = str(existing.get("requested_difficulty") or "hard")
    state = PBQState(
        request=PBQRequest(
            runtime_profile=runtime,
            difficulty=difficulty,
            experience="stored validation",
            duration=90,
            prompt=str((bundle.requirements or {}).get("prompt") or "Stored PBQ validation"),
        ),
        output_dir=output_dir,
        metrics=Metrics(),
    )
    state.runtime_profile = get_runtime_profile(runtime)
    state.bundle = bundle
    gateway = ModelGateway(provider=settings.MODEL_PROVIDER, metrics=state.metrics)
    state = generate_candidate_workspace(state)
    state = materialize_reference_workspace(state)
    state = execute_reference(state)
    if repair_with_ai and not state.validation.get("reference_passed"):
        state = repair_pbq_from_validation_feedback(
            state,
            gateway,
            repair_reason or "reference execution failed during saved PBQ validation",
        )
    state = validate_test_strength(state)
    if repair_with_ai and not state.validation.get("tests_strong_enough"):
        state = repair_pbq_from_validation_feedback(
            state,
            gateway,
            repair_reason or "private tests did not catch mutations during saved PBQ validation",
        )
        if state.validation.get("reference_passed"):
            state = validate_test_strength(state)
    state = validate_pbq_difficulty(state, gateway)
    state = finalize_pbq(state)
    return state


def _mcq_output_dir(difficulty: str):
    direct = settings.OUTPUT_DIR / "mcq" / difficulty
    if (direct / "mcq_result.json").is_file():
        return direct
    benchmark = settings.OUTPUT_DIR / "benchmarks" / "mcq" / difficulty
    if (benchmark / "mcq_result.json").is_file():
        return benchmark
    raise HTTPException(status_code=404, detail=f"No generated MCQ found for difficulty '{difficulty}'. Run MCQ first.")


def _validate_saved_mcq(difficulty: str) -> MCQState:
    output_dir = _mcq_output_dir(difficulty)
    existing = _read_json(output_dir / "mcq_result.json")
    raw_mcq = existing.get("mcq") or {}
    raw_blueprint = existing.get("difficulty_blueprint") or {}
    request = MCQRequest(
        skill=str(existing.get("skill") or (raw_mcq.get("skills") or ["Python"])[0]),
        topic=str(existing.get("topic") or raw_mcq.get("metadata", {}).get("topic") or "general"),
        difficulty=str(existing.get("requested_difficulty") or difficulty),
        experience=str(raw_blueprint.get("target_experience") or "stored validation"),
    )
    state = MCQState(request=request, output_dir=output_dir, metrics=Metrics())
    state.blueprint = DifficultyBlueprint(
        requested_difficulty=str(raw_blueprint.get("requested_difficulty") or request.difficulty),
        target_experience=str(raw_blueprint.get("target_experience") or request.experience),
        minimum_reasoning_steps=int(raw_blueprint.get("minimum_reasoning_steps") or 1),
        minimum_concepts=int(raw_blueprint.get("minimum_concepts") or 1),
        direct_recall_allowed=bool(raw_blueprint.get("direct_recall_allowed", False)),
        scenario_based=bool(raw_blueprint.get("scenario_based", False)),
        requires_code_analysis=bool(raw_blueprint.get("requires_code_analysis", False)),
        distractor_quality=str(raw_blueprint.get("distractor_quality") or "unknown"),
    )
    state.mcq = MCQ(
        question=str(raw_mcq.get("question") or ""),
        options=[
            MCQOption(
                id=str(item.get("id")),
                option_text=str(item.get("option_text")),
                is_correct=bool(item.get("is_correct")),
            )
            for item in raw_mcq.get("options") or []
        ],
        correct_answer=str(raw_mcq.get("correct_answer") or ""),
        explanation=str(raw_mcq.get("explanation") or ""),
        skills=[str(item) for item in raw_mcq.get("skills") or [request.skill]],
        concepts=[str(item) for item in raw_mcq.get("concepts") or []],
        difficulty=str(raw_mcq.get("difficulty") or request.difficulty),
        metadata=dict(raw_mcq.get("metadata") or {}),
    )
    gateway = ModelGateway(provider=settings.MODEL_PROVIDER, metrics=state.metrics)
    ok, issues = validate_mcq_structure(state)
    if not ok:
        state.final.setdefault("repairs", []).append("; ".join(issues))
        state.metrics.repair_count += 1
    state = independent_solver(state, gateway)
    if not validate_answer(state):
        state.final.setdefault("repairs", []).append("independent solver disagreed with generator answer")
        state.metrics.repair_count += 1
    state = validate_distractors(state)
    state = validate_difficulty(state, gateway)
    state = finalize_mcq(state)
    return state


if __name__ == "__main__":
    main()
