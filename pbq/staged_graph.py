"""Staged, dependency-aware PBQ generation using LangGraph.

Replaces the one-shot _generate_slm_pbq_bundle() with a multi-node
LangGraph StateGraph that generates each file individually, keeping
each SLM call within the 7B model's token budget.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from config import settings
from config.runtime_profiles import RuntimeProfile
from models.gateway import ModelGateway
from pbq.graph_helpers import (
    ai_generation_failed_bundle,
    assert_prompt_domain_alignment,
    design_from_slm,
    ensure_profile_support_files,
    extract_user_prompt,
    framework_generation_rules,
    merge_template_starter_files,
    prompt_scoped_design,
    safe_file_entries,
    safe_mutations,
    template_context_for_request,
)
from pbq.schemas import PBQBundle, PBQDesign, PBQRequest
from pbq.state import PBQState
from shared.schemas import FileEntry

logger = logging.getLogger("tatva.poc.staged_graph")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class StagedGenState(TypedDict, total=False):
    # Inputs
    request: PBQRequest
    runtime_profile: RuntimeProfile
    output_dir: str

    # Progressive artifacts
    feature_plan: dict[str, Any]
    question_spec: dict[str, Any]
    file_manifest: list[dict[str, Any]]
    reference_files: list[dict[str, str]]
    starter_files: list[dict[str, str]]
    public_tests: list[dict[str, str]]
    private_tests: list[dict[str, str]]
    mutations: list[dict[str, Any]]
    design: dict[str, Any]

    # Persistence tracking
    generated_artifacts: list[dict[str, Any]]

    # Loop control
    current_ref_index: int
    current_starter_index: int
    ref_repair_attempts: int
    starter_repair_attempts: int

    # Error tracking
    stage_errors: list[dict[str, Any]]
    failed: bool
    failure_reason: str


# ---------------------------------------------------------------------------
# Non-persisted runtime context (gateway, callbacks) passed via closure
# ---------------------------------------------------------------------------

class _RuntimeContext:
    __slots__ = ("gateway", "on_progress")

    def __init__(self, gateway: ModelGateway, on_progress: Callable | None = None):
        self.gateway = gateway
        self.on_progress = on_progress

    def progress(self, message: str) -> None:
        if self.on_progress:
            self.on_progress("design", "update", message)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_MAX_CONSECUTIVE_TIMEOUTS = 2


def _call_with_timeout(fn: Callable[..., Any], *args: Any, timeout: int | None = None, **kwargs: Any) -> Any:
    """Call *fn* with a hard wall-clock timeout. Returns the result or raises TimeoutError.

    Uses a fresh single-use pool so a timed-out call cannot block
    subsequent calls (a shared single-worker pool deadlocks because
    ``future.cancel()`` cannot interrupt a running Python thread).
    """
    effective = timeout or settings.PBQ_STAGE_FILE_TIMEOUT_SECONDS
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=effective)
    except FuturesTimeoutError:
        future.cancel()
        raise TimeoutError(f"SLM call exceeded {effective}s wall-clock timeout")
    finally:
        pool.shutdown(wait=False)


def _topological_sort(manifest: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    """Sort manifest entries of a given role by depends_on (Kahn's algorithm)."""
    items = [entry for entry in manifest if entry.get("role") == role]
    by_path: dict[str, dict[str, Any]] = {entry["path"]: entry for entry in items}
    in_degree: dict[str, int] = {entry["path"]: 0 for entry in items}
    for entry in items:
        for dep in entry.get("depends_on") or []:
            if dep in in_degree:
                in_degree[entry["path"]] += 1

    queue: deque[str] = deque(path for path, deg in in_degree.items() if deg == 0)
    ordered: list[dict[str, Any]] = []
    while queue:
        path = queue.popleft()
        ordered.append(by_path[path])
        for entry in items:
            if path in (entry.get("depends_on") or []):
                in_degree[entry["path"]] -= 1
                if in_degree[entry["path"]] == 0:
                    queue.append(entry["path"])

    if len(ordered) != len(items):
        cycle_paths = [p for p, d in in_degree.items() if d > 0]
        raise ValueError(f"Dependency cycle detected in file manifest: {cycle_paths}")
    return ordered


def _interface_summary(files: list[dict[str, str]], paths: list[str]) -> str:
    """Extract function/class signatures from generated files for dependency context."""
    summaries = []
    for f in files:
        if f["path"] not in paths:
            continue
        content = f["content"]
        lines = content.splitlines()
        sig_lines = []
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith(("def ", "class ", "async def ", "function ", "export function ",
                                     "export default function ", "export class ", "module.exports"))
                    or stripped.startswith(("public ", "private ", "protected "))
                    and ("(" in stripped or "{" in stripped)):
                sig_lines.append(line.rstrip())
            elif stripped.startswith(('"""', "'''")) and len(sig_lines) < 30:
                sig_lines.append(line.rstrip())
        if sig_lines:
            summaries.append(f"// {f['path']}\n" + "\n".join(sig_lines[:20]))
    return "\n\n".join(summaries) if summaries else "(no dependencies)"


def _syntax_check_python(content: str) -> str | None:
    """Return error message if Python content has syntax errors, else None."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp.name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return result.stderr.strip()
    return None


def _syntax_check_js(content: str) -> str | None:
    """Basic JS/JSX syntax check — look for obvious parse issues."""
    open_braces = content.count("{") - content.count("}")
    open_parens = content.count("(") - content.count(")")
    open_brackets = content.count("[") - content.count("]")
    issues = []
    if abs(open_braces) > 1:
        issues.append(f"unbalanced braces (delta={open_braces})")
    if abs(open_parens) > 1:
        issues.append(f"unbalanced parens (delta={open_parens})")
    if abs(open_brackets) > 1:
        issues.append(f"unbalanced brackets (delta={open_brackets})")
    return "; ".join(issues) if issues else None


def _validate_file_content(path: str, content: str, language: str) -> str | None:
    """Validate a single file's content. Returns error or None."""
    if not content.strip() and path not in {"app/__init__.py", "config/__init__.py", "__init__.py"}:
        return f"File content is empty: {path}"
    if language == "python" and path.endswith(".py"):
        return _syntax_check_python(content)
    if path.endswith((".js", ".jsx", ".ts", ".tsx")):
        return _syntax_check_js(content)
    return None


def _strip_to_starter(content: str, language: str) -> str:
    """Strip function/method bodies from reference code to create a starter file. Pure Python, no SLM."""
    if language == "python":
        lines = content.split("\n")
        result: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith(("def ", "async def ")):
                result.append(line)
                indent = len(line) - len(stripped)
                body_indent = indent + 4
                # Check for docstring on next line
                i += 1
                if i < len(lines):
                    next_stripped = lines[i].lstrip()
                    if next_stripped.startswith(('"""', "'''")):
                        # Include the docstring
                        quote = next_stripped[:3]
                        result.append(lines[i])
                        if next_stripped.count(quote) < 2:
                            i += 1
                            while i < len(lines) and quote not in lines[i]:
                                result.append(lines[i])
                                i += 1
                            if i < len(lines):
                                result.append(lines[i])
                                i += 1
                        else:
                            i += 1
                result.append(" " * body_indent + "# TODO: implement this function")
                result.append(" " * body_indent + "pass")
                # Skip original body
                while i < len(lines):
                    if not lines[i].strip():
                        i += 1
                        continue
                    body_stripped = lines[i].lstrip()
                    current_indent = len(lines[i]) - len(body_stripped)
                    if current_indent < body_indent and body_stripped:
                        break
                    i += 1
            else:
                result.append(line)
                i += 1
        return "\n".join(result)
    # JS/JSX — simpler: replace function bodies with empty blocks
    if language in ("javascript", "typescript"):
        import re as _re
        def _replace_body(match: _re.Match) -> str:
            return match.group(0).split("{")[0] + "{\n  // TODO: implement\n}"
        result_str = re.sub(r'(function\s+\w+\s*\([^)]*\)\s*)\{[^}]*\}', _replace_body, content)
        return result_str
    return content


def _is_boilerplate(path: str, profile_id: str) -> bool:
    """True if this file should be generated deterministically, not via SLM."""
    boilerplate_paths = {"__init__.py", "app/__init__.py", "config/__init__.py"}
    if path in boilerplate_paths:
        return True
    if profile_id == "django-sqlite-py312" and path in {
        "manage.py", "config/settings.py", "config/urls.py", "pytest.ini",
    }:
        return True
    return False


def _persist_file(output_dir: str, path: str, content: str, role: str) -> dict[str, Any]:
    """Write a generated file to the staged/ directory and return a tracking record."""
    staged_dir = Path(output_dir) / "staged" / role
    target = staged_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": path, "role": role, "persisted_at": str(target)}


def _resume_artifacts(output_dir: str, role: str) -> list[dict[str, str]]:
    """Load previously generated files for a role from staged/ directory."""
    staged_dir = Path(output_dir) / "staged" / role
    if not staged_dir.is_dir():
        return []
    files = []
    for file_path in sorted(staged_dir.rglob("*")):
        if file_path.is_file():
            rel = str(file_path.relative_to(staged_dir))
            files.append({"path": rel, "content": file_path.read_text(encoding="utf-8"), "kind": "file"})
    return files


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _make_node_feature_plan(ctx: _RuntimeContext):
    def node_feature_plan(state: StagedGenState) -> dict[str, Any]:
        ctx.progress("AI: planning features (1/8)...")
        request = state["request"]
        profile = state["runtime_profile"]
        full_template_context = template_context_for_request(request)
        template_context = None
        if full_template_context:
            template_context = {
                "name": full_template_context.get("name"),
                "runtime_profile": full_template_context.get("runtime_profile"),
                "note": "This template is ONLY the runtime scaffold. The DOMAIN comes from user_prompt_only.",
            }
        user_prompt_only = extract_user_prompt(request.prompt)
        try:
            raw = _call_with_timeout(
                ctx.gateway.structured_generate,
                (
                    "Analyze the user prompt and create a PBQ feature plan before writing code. "
                    "CRITICAL: The 'user_prompt_only' field is the ACTUAL task requirement — the title, domain, entities, "
                    "and features MUST be derived from it. The selected template is ONLY the framework/runtime scaffold "
                    "(e.g. Django, FastAPI); do NOT use template names, descriptions, or default examples as the task domain. "
                    "If user_prompt_only says 'create excel upload functionality', the domain is excel/file upload, NOT todo or cart. "
                    "Think like an assessment designer. Choose the smallest set of realistic features that match "
                    "the requested difficulty, experience level, duration, and runtime. Return strict JSON with: "
                    "domain_summary, entities, features, primary_workflow, validation_rules, public_test_focus, "
                    "private_test_focus, mutation_ideas, and candidate_scope. Do not generate code."
                ),
                schema_name="PBQFeaturePlan",
                context={
                    "prompt": request.prompt,
                    "user_prompt_only": user_prompt_only,
                    "difficulty": request.difficulty,
                    "experience": request.experience,
                    "duration": request.duration,
                    "runtime_profile": profile.to_dict(),
                    "selected_template": template_context,
                },
                num_predict=settings.PBQ_STAGE_FEATURE_PLAN_TOKENS,
            )
        except TimeoutError as exc:
            raw = {"slm_error": str(exc)}
        if raw.get("slm_error") or raw.get("slm_fallback"):
            return {
                "feature_plan": {"planning_error": raw},
                "failed": True,
                "failure_reason": f"Feature plan generation failed: {raw.get('slm_error', 'fallback')}",
            }
        return {"feature_plan": raw}
    return node_feature_plan


def _make_node_question_spec(ctx: _RuntimeContext):
    def node_question_spec(state: StagedGenState) -> dict[str, Any]:
        if state.get("failed"):
            return {}
        ctx.progress("AI: generating question spec (2/8)...")
        request = state["request"]
        profile = state["runtime_profile"]
        plan = state.get("feature_plan", {})
        rules = framework_generation_rules(profile)

        try:
            raw = _call_with_timeout(
                ctx.gateway.structured_generate,
                (
                    "Based on the feature plan AND the user_prompt_only field, generate a question specification and file manifest. "
                    "CRITICAL: The title and behavioral_contract MUST reflect the user_prompt_only domain, NOT a generic todo/cart/blog app. "
                    "Return strict JSON with: title (string), behavioral_contract (list of 3-5 requirement strings), "
                    "scaffolding_strategy (string), file_manifest (list of file descriptors). "
                    "Each file_manifest entry must have: path (string), role (one of 'reference', 'starter', "
                    "'test_public', 'test_private'), depends_on (list of file paths this file imports from), "
                    "description (one-line string describing the file's purpose). "
                    "Order files so dependencies come before dependents. "
                    "IMPORTANT: Keep the workspace minimal. Include ONLY source code files (.py, .js, .jsx, .java). "
                    "Do NOT include documentation, YAML, markdown, or config files in the manifest. "
                    "Maximum 2-3 reference source files (plus __init__.py). Keep it minimal: one models/entities file and one services/logic file. "
                    f"Framework rules: {rules}"
                ),
                schema_name="PBQQuestionSpec",
                context={
                    "prompt": request.prompt,
                    "user_prompt_only": extract_user_prompt(request.prompt),
                    "difficulty": request.difficulty,
                    "experience": request.experience,
                    "runtime_profile": profile.to_dict(),
                    "feature_plan": plan,
                },
                num_predict=settings.PBQ_STAGE_QUESTION_SPEC_TOKENS,
            )
        except TimeoutError as exc:
            raw = {"slm_error": str(exc)}
        if raw.get("slm_error") or raw.get("slm_fallback"):
            return {
                "question_spec": raw,
                "failed": True,
                "failure_reason": f"Question spec generation failed: {raw.get('slm_error', 'fallback')}",
            }

        manifest = raw.get("file_manifest")
        if not isinstance(manifest, list) or not manifest:
            return {
                "question_spec": raw,
                "failed": True,
                "failure_reason": "Question spec missing file_manifest",
            }

        # Validate and clean manifest entries
        cleaned_manifest = []
        for entry in manifest:
            if not isinstance(entry, dict) or "path" not in entry:
                continue
            entry.setdefault("role", "reference")
            entry.setdefault("depends_on", [])
            entry.setdefault("description", "")
            # Filter out non-source files (docs, configs the SLM shouldn't generate)
            path = str(entry.get("path", ""))
            if path.endswith((".yml", ".yaml", ".md", ".txt", ".json", ".xml")) and entry["role"] == "reference":
                logger.info("Filtering non-source manifest entry: %s", path)
                continue
            cleaned_manifest.append(entry)
        manifest = cleaned_manifest

        # Cap reference files — fewer files = fewer slow SLM calls
        MAX_REF_FILES = 3
        ref_entries = [e for e in manifest if e.get("role") == "reference"]
        if len(ref_entries) > MAX_REF_FILES:
            logger.warning("Manifest has %d reference files, capping to %d", len(ref_entries), MAX_REF_FILES)
            kept_paths = {e["path"] for e in ref_entries[:MAX_REF_FILES]}
            manifest = [e for e in manifest if e.get("role") != "reference" or e["path"] in kept_paths]

        try:
            _topological_sort(manifest, "reference")
        except ValueError as exc:
            return {
                "question_spec": raw,
                "file_manifest": manifest,
                "failed": True,
                "failure_reason": str(exc),
            }

        design_data = {
            "title": raw.get("title", ""),
            "behavioral_contract": raw.get("behavioral_contract", []),
            "scaffolding_strategy": raw.get("scaffolding_strategy", ""),
        }
        return {
            "question_spec": raw,
            "file_manifest": manifest,
            "design": design_data,
            "current_ref_index": 0,
            "ref_repair_attempts": 0,
        }
    return node_question_spec


def _make_node_gen_reference_file(ctx: _RuntimeContext):
    def node_gen_reference_file(state: StagedGenState) -> dict[str, Any]:
        if state.get("failed"):
            return {}
        manifest = state.get("file_manifest", [])
        ref_entries = _topological_sort(manifest, "reference")
        idx = state.get("current_ref_index", 0)

        if idx >= len(ref_entries):
            return {}

        entry = ref_entries[idx]
        path = entry["path"]
        profile = state["runtime_profile"]
        existing_refs = state.get("reference_files", [])
        stage_errors = list(state.get("stage_errors", []))
        artifacts = list(state.get("generated_artifacts", []))
        output_dir = state.get("output_dir", "")

        ctx.progress(f"AI: generating reference file {idx + 1}/{len(ref_entries)} ({path}) (3/8)...")

        # Check resume
        if output_dir:
            resumed = _resume_artifacts(output_dir, "reference")
            for r in resumed:
                if r["path"] == path:
                    logger.info("Resumed reference file from disk: %s", path)
                    new_refs = list(existing_refs) + [{"path": path, "content": r["content"], "kind": "file"}]
                    return {
                        "reference_files": new_refs,
                        "current_ref_index": idx + 1,
                        "ref_repair_attempts": 0,
                        "generated_artifacts": artifacts + [{"path": path, "role": "reference", "persisted_at": "resumed"}],
                    }

        # Boilerplate files — generate without SLM
        if _is_boilerplate(path, profile.id):
            content = ""
            new_refs = list(existing_refs) + [{"path": path, "content": content, "kind": "file"}]
            if output_dir:
                artifacts.append(_persist_file(output_dir, path, content, "reference"))
            return {
                "reference_files": new_refs,
                "current_ref_index": idx + 1,
                "ref_repair_attempts": 0,
                "generated_artifacts": artifacts,
            }

        # Build minimal context: spec + dependency interfaces only
        dep_paths = entry.get("depends_on") or []
        dep_context = _interface_summary(existing_refs, dep_paths)
        spec = state.get("question_spec", {})
        contract = spec.get("behavioral_contract", [])

        try:
            raw = _call_with_timeout(
                ctx.gateway.structured_generate,
                (
                    f"Generate the complete source code for file '{path}'. "
                    f"File description: {entry.get('description', 'source file')}. "
                    "Return strict JSON with keys: path (string), content (string with the full file source code), kind ('file'). "
                    "The content must be complete, runnable code — no placeholders, TODOs, or pass-only functions. "
                    f"Behavioral contract: {json.dumps(contract)}. "
                    f"Framework rules: {framework_generation_rules(profile)}"
                ),
                schema_name="PBQStagedFile",
                context={
                    "file_path": path,
                    "file_description": entry.get("description", ""),
                    "file_role": "reference",
                    "dependency_interfaces": dep_context,
                    "runtime_profile": profile.to_dict(),
                    "question_title": spec.get("title", ""),
                    "user_prompt": extract_user_prompt(state["request"].prompt),
                },
                num_predict=settings.PBQ_STAGE_REFERENCE_FILE_TOKENS,
            )
        except TimeoutError as exc:
            logger.warning("Timeout generating reference file %s: %s", path, exc)
            raw = {"slm_error": str(exc)}

        is_timeout = "timeout" in str(raw.get("slm_error", "")).lower()
        content = _extract_file_content(raw, path)
        if content is None:
            repair_attempts = state.get("ref_repair_attempts", 0)
            if is_timeout and repair_attempts >= 1:
                # Timeout on a retry — skip this file immediately, don't burn more time
                stage_errors.append({"stage": "gen_reference_file", "file": path, "error": "Timeout on retry, skipping", "attempt": repair_attempts})
                consecutive_skips = sum(1 for e in stage_errors if e.get("stage") == "gen_reference_file" and "skip" in e.get("error", "").lower())
                if consecutive_skips >= _MAX_CONSECUTIVE_TIMEOUTS:
                    logger.error("SLM unresponsive: %d consecutive file timeouts, aborting pipeline", consecutive_skips)
                    return {
                        "failed": True,
                        "failure_reason": f"SLM unresponsive: {consecutive_skips} consecutive file generation timeouts",
                        "stage_errors": stage_errors,
                    }
                return {
                    "current_ref_index": idx + 1,
                    "ref_repair_attempts": 0,
                    "stage_errors": stage_errors,
                }
            if repair_attempts < settings.PBQ_STAGE_FILE_REPAIR_MAX:
                error_msg = f"SLM returned invalid response for {path}: {json.dumps(raw)[:500]}"
                stage_errors.append({"stage": "gen_reference_file", "file": path, "error": error_msg, "attempt": repair_attempts + 1})
                return {"ref_repair_attempts": repair_attempts + 1, "stage_errors": stage_errors}
            stage_errors.append({"stage": "gen_reference_file", "file": path, "error": "Max repair attempts exceeded", "attempt": repair_attempts})
            return {
                "current_ref_index": idx + 1,
                "ref_repair_attempts": 0,
                "stage_errors": stage_errors,
            }

        # Validate syntax
        syntax_error = _validate_file_content(path, content, profile.language)
        if syntax_error:
            repair_attempts = state.get("ref_repair_attempts", 0)
            if repair_attempts < settings.PBQ_STAGE_FILE_REPAIR_MAX:
                # Targeted repair
                try:
                    repair_raw = _call_with_timeout(
                        ctx.gateway.repair_generate,
                        (
                            f"The generated file '{path}' has a syntax error: {syntax_error}. "
                            "Fix the error and return the corrected file. "
                            "Return strict JSON with keys: path, content, kind ('file')."
                        ),
                        schema_name="PBQStagedFileRepair",
                        context={"file_path": path, "original_content": content, "syntax_error": syntax_error},
                        num_predict=settings.PBQ_STAGE_REFERENCE_FILE_TOKENS,
                    )
                except TimeoutError:
                    repair_raw = {"slm_error": "repair timed out"}
                repaired_content = _extract_file_content(repair_raw, path)
                if repaired_content and not _validate_file_content(path, repaired_content, profile.language):
                    content = repaired_content
                else:
                    stage_errors.append({"stage": "gen_reference_file", "file": path, "error": f"Repair failed: {syntax_error}", "attempt": repair_attempts + 1})
                    return {"ref_repair_attempts": repair_attempts + 1, "stage_errors": stage_errors}

        new_refs = list(existing_refs) + [{"path": path, "content": content, "kind": "file"}]
        if output_dir:
            artifacts.append(_persist_file(output_dir, path, content, "reference"))

        return {
            "reference_files": new_refs,
            "current_ref_index": idx + 1,
            "ref_repair_attempts": 0,
            "generated_artifacts": artifacts,
        }
    return node_gen_reference_file


def _make_node_gen_starter_file(ctx: _RuntimeContext):
    def node_gen_starter_file(state: StagedGenState) -> dict[str, Any]:
        if state.get("failed"):
            return {}
        manifest = state.get("file_manifest", [])
        starter_entries = [e for e in manifest if e.get("role") == "starter"]
        # If no explicit starter entries in manifest, derive from reference entries
        if not starter_entries:
            starter_entries = [
                {**e, "role": "starter"} for e in manifest if e.get("role") == "reference"
            ]
        idx = state.get("current_starter_index", 0)
        if idx >= len(starter_entries):
            return {}

        entry = starter_entries[idx]
        path = entry["path"]
        profile = state["runtime_profile"]
        existing_starters = state.get("starter_files", [])
        existing_refs = state.get("reference_files", [])
        stage_errors = list(state.get("stage_errors", []))
        artifacts = list(state.get("generated_artifacts", []))
        output_dir = state.get("output_dir", "")

        ctx.progress(f"AI: generating starter file {idx + 1}/{len(starter_entries)} ({path}) (4/8)...")

        # Boilerplate — deterministic
        if _is_boilerplate(path, profile.id):
            content = ""
            new_starters = list(existing_starters) + [{"path": path, "content": content, "kind": "file"}]
            if output_dir:
                artifacts.append(_persist_file(output_dir, path, content, "starter"))
            return {
                "starter_files": new_starters,
                "current_starter_index": idx + 1,
                "starter_repair_attempts": 0,
                "generated_artifacts": artifacts,
            }

        # Derive starter from reference — pure Python, no SLM call needed
        ref_content = None
        for rf in existing_refs:
            if rf["path"] == path:
                ref_content = rf["content"]
                break

        if ref_content:
            content = _strip_to_starter(ref_content, profile.language)
        else:
            content = f"# TODO: implement {path}\n"
        logger.info("Generated starter file %s deterministically (no SLM call)", path)

        new_starters = list(existing_starters) + [{"path": path, "content": content, "kind": "file"}]
        if output_dir:
            artifacts.append(_persist_file(output_dir, path, content, "starter"))
        return {
            "starter_files": new_starters,
            "current_starter_index": idx + 1,
            "starter_repair_attempts": 0,
            "generated_artifacts": artifacts,
        }
    return node_gen_starter_file


def _make_node_execute_reference_check(ctx: _RuntimeContext):
    def node_execute_reference_check(state: StagedGenState) -> dict[str, Any]:
        if state.get("failed"):
            return {}
        ctx.progress("Validating reference files (5/8)...")
        ref_files = state.get("reference_files", [])
        profile = state["runtime_profile"]
        stage_errors = list(state.get("stage_errors", []))

        if not ref_files:
            return {
                "failed": True,
                "failure_reason": "No reference files were generated",
                "stage_errors": stage_errors,
            }

        # Syntax check all reference files
        syntax_issues = []
        for rf in ref_files:
            error = _validate_file_content(rf["path"], rf["content"], profile.language)
            if error:
                syntax_issues.append(f"{rf['path']}: {error}")
        if syntax_issues:
            for issue in syntax_issues:
                stage_errors.append({"stage": "execute_reference_check", "file": "", "error": issue})
            return {
                "failed": True,
                "failure_reason": f"Reference files have syntax errors: {'; '.join(syntax_issues[:3])}",
                "stage_errors": stage_errors,
            }
        return {}
    return node_execute_reference_check


def _make_node_gen_public_tests(ctx: _RuntimeContext):
    def node_gen_public_tests(state: StagedGenState) -> dict[str, Any]:
        if state.get("failed"):
            return {}
        ctx.progress("AI: generating public tests (6/8)...")
        spec = state.get("question_spec", {})
        profile = state["runtime_profile"]
        request = state["request"]
        contract = spec.get("behavioral_contract", [])
        manifest = state.get("file_manifest", [])
        test_manifest = [e for e in manifest if e.get("role") == "test_public"]
        test_path = test_manifest[0]["path"] if test_manifest else _default_test_path(profile, "public")

        try:
            raw = _call_with_timeout(
                ctx.gateway.structured_generate,
                (
                    f"Generate public test file '{test_path}' for the PBQ. "
                    "Tests must verify the behavioral contract (not implementation details). "
                    "Include 2-3 test cases covering the happy path and basic validation. "
                    "Return strict JSON with keys: path, content, kind ('file'). "
                    "Test content must be complete, runnable test code with proper imports. "
                    f"Framework rules: {framework_generation_rules(profile)}"
                ),
                schema_name="PBQStagedTestFile",
                context={
                    "test_role": "public",
                    "behavioral_contract": contract,
                    "file_manifest": [{"path": e["path"], "description": e.get("description", "")} for e in manifest if e.get("role") == "reference"],
                    "runtime_profile": profile.to_dict(),
                    "difficulty": request.difficulty,
                },
                num_predict=settings.PBQ_STAGE_PUBLIC_TESTS_TOKENS,
            )
        except TimeoutError as exc:
            raw = {"slm_error": str(exc)}
        content = _extract_file_content(raw, test_path)
        if content is None:
            return {"stage_errors": list(state.get("stage_errors", [])) + [
                {"stage": "gen_public_tests", "file": test_path, "error": "Failed to generate public tests"}
            ]}

        artifacts = list(state.get("generated_artifacts", []))
        output_dir = state.get("output_dir", "")
        if output_dir:
            artifacts.append(_persist_file(output_dir, test_path, content, "public_tests"))
        return {
            "public_tests": [{"path": test_path, "content": content, "kind": "file"}],
            "generated_artifacts": artifacts,
        }
    return node_gen_public_tests


def _make_node_gen_private_tests(ctx: _RuntimeContext):
    def node_gen_private_tests(state: StagedGenState) -> dict[str, Any]:
        if state.get("failed"):
            return {}
        ctx.progress("AI: generating private tests (7/8)...")
        spec = state.get("question_spec", {})
        profile = state["runtime_profile"]
        contract = spec.get("behavioral_contract", [])
        manifest = state.get("file_manifest", [])
        ref_files = state.get("reference_files", [])
        test_manifest = [e for e in manifest if e.get("role") == "test_private"]
        test_path = test_manifest[0]["path"] if test_manifest else _default_test_path(profile, "private")

        # Private tests get reference file summaries to test edge cases
        ref_summary = _interface_summary(ref_files, [e["path"] for e in ref_files])

        try:
            raw = _call_with_timeout(
                ctx.gateway.structured_generate,
                (
                    f"Generate private test file '{test_path}' for the PBQ. "
                    "Private tests must catch common implementation mistakes (mutations). "
                    "Include 3-5 test cases covering edge cases, boundary conditions, and error handling. "
                    "These tests must fail when the implementation has subtle bugs. "
                    "Return strict JSON with keys: path, content, kind ('file'). "
                    f"Framework rules: {framework_generation_rules(profile)}"
                ),
                schema_name="PBQStagedTestFile",
                context={
                    "test_role": "private",
                    "behavioral_contract": contract,
                    "reference_interfaces": ref_summary,
                    "runtime_profile": profile.to_dict(),
                    "difficulty": state["request"].difficulty,
                },
                num_predict=settings.PBQ_STAGE_PRIVATE_TESTS_TOKENS,
            )
        except TimeoutError as exc:
            raw = {"slm_error": str(exc)}
        content = _extract_file_content(raw, test_path)
        if content is None:
            return {"stage_errors": list(state.get("stage_errors", [])) + [
                {"stage": "gen_private_tests", "file": test_path, "error": "Failed to generate private tests"}
            ]}

        artifacts = list(state.get("generated_artifacts", []))
        output_dir = state.get("output_dir", "")
        if output_dir:
            artifacts.append(_persist_file(output_dir, test_path, content, "private_tests"))
        return {
            "private_tests": [{"path": test_path, "content": content, "kind": "file"}],
            "generated_artifacts": artifacts,
        }
    return node_gen_private_tests


def _make_node_gen_mutations(ctx: _RuntimeContext):
    def node_gen_mutations(state: StagedGenState) -> dict[str, Any]:
        if state.get("failed"):
            return {}
        ctx.progress("AI: generating mutations (8/8)...")
        ref_files = state.get("reference_files", [])
        profile = state["runtime_profile"]
        spec = state.get("question_spec", {})
        plan = state.get("feature_plan", {})

        # Pick the main implementation file for mutations
        main_ref = None
        for rf in ref_files:
            if rf["content"].strip():
                main_ref = rf
                break
        if not main_ref:
            return {"stage_errors": list(state.get("stage_errors", [])) + [
                {"stage": "gen_mutations", "file": "", "error": "No reference file with content for mutations"}
            ]}

        try:
            raw = _call_with_timeout(
                ctx.gateway.structured_generate,
                (
                    "Generate 2-3 mutations for the reference solution. Each mutation is an intentionally wrong "
                    "version that private tests should detect. "
                    "Return strict JSON with key: mutations (list of objects with name (string) and "
                    "changes (list of {path, content, kind:'file'})). "
                    "Each change replaces the content of one reference file with a subtly broken version. "
                    "Common mutations: removing validation, off-by-one errors, missing edge case handling."
                ),
                schema_name="PBQStagedMutations",
                context={
                    "reference_file_path": main_ref["path"],
                    "reference_content": main_ref["content"],
                    "behavioral_contract": spec.get("behavioral_contract", []),
                    "mutation_ideas": plan.get("mutation_ideas", []),
                },
                num_predict=settings.PBQ_STAGE_MUTATIONS_TOKENS,
            )
        except TimeoutError as exc:
            raw = {"slm_error": str(exc)}

        mutations_raw = raw.get("mutations")
        if not isinstance(mutations_raw, list) or not mutations_raw:
            return {"stage_errors": list(state.get("stage_errors", [])) + [
                {"stage": "gen_mutations", "file": "", "error": "Failed to generate mutations"}
            ]}

        validated = []
        for i, m in enumerate(mutations_raw):
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or f"mutation_{i + 1}")
            changes = m.get("changes")
            if not isinstance(changes, list) or not changes:
                continue
            valid_changes = []
            for c in changes:
                if isinstance(c, dict) and c.get("path") and c.get("content"):
                    valid_changes.append({"path": c["path"], "content": c["content"], "kind": "file"})
            if valid_changes:
                validated.append({"name": name, "changes": valid_changes})

        return {"mutations": validated}
    return node_gen_mutations


def _make_node_assemble_bundle(ctx: _RuntimeContext):
    def node_assemble_bundle(state: StagedGenState) -> dict[str, Any]:
        """Pure Python assembly — no SLM call."""
        ref_files = state.get("reference_files", [])
        starter_files = state.get("starter_files", [])
        public_tests = state.get("public_tests", [])
        private_tests = state.get("private_tests", [])
        mutations = state.get("mutations", [])
        stage_errors = state.get("stage_errors", [])
        failed = state.get("failed", False)

        if failed or not ref_files:
            return {"failed": True, "failure_reason": state.get("failure_reason", "No usable artifacts generated")}

        # Mark partial failure but still assemble what we have
        generation_mode = "staged_generation"
        if stage_errors:
            generation_mode = "staged_partial_failure"

        return {
            "failed": False,
            "failure_reason": "" if not stage_errors else f"Completed with {len(stage_errors)} stage error(s)",
        }
    return node_assemble_bundle


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _make_route_next_ref(manifest_getter):
    def route_next_ref(state: StagedGenState) -> Literal["gen_reference_file", "gen_starter_file"]:
        manifest = state.get("file_manifest", [])
        ref_entries = [e for e in manifest if e.get("role") == "reference"]
        idx = state.get("current_ref_index", 0)
        repair = state.get("ref_repair_attempts", 0)
        if repair > 0 and repair <= settings.PBQ_STAGE_FILE_REPAIR_MAX:
            return "gen_reference_file"
        if idx < len(ref_entries):
            return "gen_reference_file"
        return "gen_starter_file"
    return route_next_ref


def _make_route_next_starter():
    def route_next_starter(state: StagedGenState) -> Literal["gen_starter_file", "execute_reference_check"]:
        manifest = state.get("file_manifest", [])
        starter_entries = [e for e in manifest if e.get("role") == "starter"]
        if not starter_entries:
            starter_entries = [e for e in manifest if e.get("role") == "reference"]
        idx = state.get("current_starter_index", 0)
        repair = state.get("starter_repair_attempts", 0)
        if repair > 0 and repair <= settings.PBQ_STAGE_FILE_REPAIR_MAX:
            return "gen_starter_file"
        if idx < len(starter_entries):
            return "gen_starter_file"
        return "execute_reference_check"
    return route_next_starter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_file_content(raw: dict[str, Any], expected_path: str) -> str | None:
    """Extract file content from an SLM response dict."""
    if raw.get("slm_error") or raw.get("slm_fallback"):
        return None
    content = raw.get("content")
    if isinstance(content, str) and content.strip():
        return content
    # Try nested structures
    for key in (expected_path, "file", "result"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            c = nested.get("content")
            if isinstance(c, str) and c.strip():
                return c
        elif isinstance(nested, str) and nested.strip():
            return nested
    # Check if the whole response looks like it has content
    for value in raw.values():
        if isinstance(value, str) and len(value) > 20 and "\n" in value:
            return value
    return None


def _default_test_path(profile: RuntimeProfile, kind: str) -> str:
    if profile.id == "react-node20":
        return f"src/App.{kind}.test.jsx"
    if profile.id == "express-node20":
        return f"tests/app.{kind}.test.js"
    return f"tests/test_{kind}.py"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _route_next_ref_then_end(state: StagedGenState) -> Literal["gen_reference_file", "__end__"]:
    """Route reference file loop: continue generating or finish."""
    manifest = state.get("file_manifest", [])
    ref_entries = [e for e in manifest if e.get("role") == "reference"]
    idx = state.get("current_ref_index", 0)
    repair = state.get("ref_repair_attempts", 0)
    if repair > 0 and repair <= settings.PBQ_STAGE_FILE_REPAIR_MAX:
        return "gen_reference_file"
    if idx < len(ref_entries):
        return "gen_reference_file"
    return END


def _build_design_only_graph(ctx: _RuntimeContext) -> Any:
    """Build a graph that runs feature_plan → question_spec → gen_reference_file (loop)."""
    graph = StateGraph(StagedGenState)
    graph.add_node("feature_plan", _make_node_feature_plan(ctx))
    graph.add_node("question_spec", _make_node_question_spec(ctx))
    graph.add_node("gen_reference_file", _make_node_gen_reference_file(ctx))
    graph.add_edge(START, "feature_plan")
    graph.add_edge("feature_plan", "question_spec")
    graph.add_edge("question_spec", "gen_reference_file")
    graph.add_conditional_edges("gen_reference_file", _route_next_ref_then_end)
    return graph.compile()


def build_staged_pbq_graph(ctx: _RuntimeContext) -> Any:
    """Build and compile the LangGraph StateGraph for staged PBQ generation."""
    graph = StateGraph(StagedGenState)

    graph.add_node("feature_plan", _make_node_feature_plan(ctx))
    graph.add_node("question_spec", _make_node_question_spec(ctx))
    graph.add_node("gen_reference_file", _make_node_gen_reference_file(ctx))
    graph.add_node("gen_starter_file", _make_node_gen_starter_file(ctx))
    graph.add_node("execute_reference_check", _make_node_execute_reference_check(ctx))
    graph.add_node("gen_public_tests", _make_node_gen_public_tests(ctx))
    graph.add_node("gen_private_tests", _make_node_gen_private_tests(ctx))
    graph.add_node("gen_mutations", _make_node_gen_mutations(ctx))
    graph.add_node("assemble_bundle", _make_node_assemble_bundle(ctx))

    graph.add_edge(START, "feature_plan")
    graph.add_edge("feature_plan", "question_spec")
    graph.add_edge("question_spec", "gen_reference_file")
    graph.add_conditional_edges("gen_reference_file", _make_route_next_ref(None))
    graph.add_conditional_edges("gen_starter_file", _make_route_next_starter())
    graph.add_edge("execute_reference_check", "gen_public_tests")
    graph.add_edge("gen_public_tests", "gen_private_tests")
    graph.add_edge("gen_private_tests", "gen_mutations")
    graph.add_edge("gen_mutations", "assemble_bundle")
    graph.add_edge("assemble_bundle", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_staged_pbq_bundle(
    fallback: PBQBundle,
    state: PBQState,
    gateway: ModelGateway,
    *,
    on_progress: Callable[[str, str, str], None] | None = None,
    design_only: bool = False,
) -> PBQBundle:
    """Drop-in replacement for _generate_slm_pbq_bundle().

    Runs a LangGraph pipeline that generates each file individually,
    then assembles the result into a PBQBundle.

    When *design_only* is True, only the feature-plan and question-spec
    stages run (no file generation), returning the design and manifest.
    """
    assert state.runtime_profile is not None
    ctx = _RuntimeContext(gateway, on_progress)

    if design_only:
        compiled = _build_design_only_graph(ctx)
    else:
        compiled = build_staged_pbq_graph(ctx)

    initial_state: StagedGenState = {
        "request": state.request,
        "runtime_profile": state.runtime_profile,
        "output_dir": str(state.output_dir),
        "feature_plan": {},
        "question_spec": {},
        "file_manifest": [],
        "reference_files": [],
        "starter_files": [],
        "public_tests": [],
        "private_tests": [],
        "mutations": [],
        "design": {},
        "generated_artifacts": [],
        "current_ref_index": 0,
        "current_starter_index": 0,
        "ref_repair_attempts": 0,
        "starter_repair_attempts": 0,
        "stage_errors": [],
        "failed": False,
        "failure_reason": "",
    }

    start_time = time.monotonic()
    result = compiled.invoke(initial_state)
    elapsed = time.monotonic() - start_time
    logger.info("Staged PBQ generation completed in %.1fs", elapsed)

    # Extract results
    plan = result.get("feature_plan", {})

    if design_only:
        spec = result.get("question_spec", {})
        design_data = result.get("design", {})
        design = prompt_scoped_design(state.runtime_profile, state.request, plan)
        if design_data.get("title"):
            design.title = design_data["title"]
        if isinstance(design_data.get("behavioral_contract"), list) and len(design_data["behavioral_contract"]) >= 3:
            design.behavioral_contract = [str(c) for c in design_data["behavioral_contract"]]
        if design_data.get("scaffolding_strategy"):
            design.scaffolding_strategy = str(design_data["scaffolding_strategy"])

        def _to_entries(items: list[dict]) -> list[FileEntry]:
            return [FileEntry(path=f["path"], content=f["content"], kind=f.get("kind", "file")) for f in items if f.get("content") is not None]

        requirements: dict[str, Any] = {
            "prompt": state.request.prompt,
            "generation_mode": "design_only",
            "ai_feature_plan": plan,
            "staged_question_spec": spec,
            "staged_file_manifest": result.get("file_manifest", []),
            "staged_elapsed_seconds": round(elapsed, 1),
        }
        bundle = PBQBundle(
            requirements=requirements,
            design=design,
            reference_solution=_to_entries(result.get("reference_files", [])),
        )
        bundle = merge_template_starter_files(bundle, state.request)
        bundle = ensure_profile_support_files(bundle, state.runtime_profile)
        return bundle

    ref_files_raw = result.get("reference_files", [])
    starter_files_raw = result.get("starter_files", [])
    public_tests_raw = result.get("public_tests", [])
    private_tests_raw = result.get("private_tests", [])
    mutations_raw = result.get("mutations", [])
    stage_errors = result.get("stage_errors", [])
    failed = result.get("failed", False)

    # If completely failed and no reference files, return failure bundle
    if failed and not ref_files_raw:
        logger.warning("Staged generation failed: %s", result.get("failure_reason", "unknown"))
        return ai_generation_failed_bundle(
            state,
            {"staged_errors": stage_errors, "failure_reason": result.get("failure_reason", "")},
            result.get("failure_reason", "Staged generation failed"),
            plan,
        )

    # Convert raw dicts to FileEntry objects
    def _to_file_entries(items: list[dict]) -> list[FileEntry]:
        return [FileEntry(path=f["path"], content=f["content"], kind=f.get("kind", "file")) for f in items if f.get("content") is not None]

    reference_solution = _to_file_entries(ref_files_raw)
    starter_files = _to_file_entries(starter_files_raw)
    public_tests = _to_file_entries(public_tests_raw)
    private_tests = _to_file_entries(private_tests_raw)

    # Build design
    spec = result.get("question_spec", {})
    design_data = result.get("design", {})
    design = prompt_scoped_design(state.runtime_profile, state.request, plan)
    if design_data.get("title"):
        design.title = design_data["title"]
    if isinstance(design_data.get("behavioral_contract"), list) and len(design_data["behavioral_contract"]) >= 3:
        design.behavioral_contract = [str(c) for c in design_data["behavioral_contract"]]
    if design_data.get("scaffolding_strategy"):
        design.scaffolding_strategy = str(design_data["scaffolding_strategy"])
    design.difficulty_notes.append("Question generated via staged LangGraph pipeline.")

    # Build requirements
    generation_mode = "staged_generation"
    if stage_errors:
        generation_mode = "staged_partial_failure"
    requirements: dict[str, Any] = {
        "prompt": state.request.prompt,
        "generation_mode": generation_mode,
        "ai_feature_plan": plan,
        "staged_question_spec": spec,
        "staged_file_manifest": result.get("file_manifest", []),
        "staged_artifacts": result.get("generated_artifacts", []),
        "staged_elapsed_seconds": round(elapsed, 1),
    }
    if stage_errors:
        requirements["staged_errors"] = stage_errors

    bundle = PBQBundle(
        requirements=requirements,
        design=design,
        starter_files=starter_files,
        public_tests=public_tests,
        private_tests=private_tests,
        reference_solution=reference_solution,
        mutations=mutations_raw,
    )

    bundle = merge_template_starter_files(bundle, state.request)
    bundle = ensure_profile_support_files(bundle, state.runtime_profile)

    try:
        assert_prompt_domain_alignment(bundle, state.request)
    except ValueError:
        logger.warning("Staged bundle failed domain alignment check — returning failure bundle")
        return ai_generation_failed_bundle(
            state,
            {"staged_errors": stage_errors, "domain_drift": True},
            "Generated bundle drifted to template domain instead of prompt domain",
            plan,
        )

    return bundle
