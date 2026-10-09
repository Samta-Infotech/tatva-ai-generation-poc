"""In-memory progress tracking for async PBQ/MCQ generation tasks.

Each operation (generate / validate / regenerate) creates a TaskProgress that
records per-step status.  The API writes to it from a background thread while
the polling endpoint reads it from the asyncio event loop.

Tasks are also written to disk (tasks/ dir) so they survive server restarts.
The UI can still find a completed task by task_id even after a restart.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("tatva.poc.progress")

# Set to the outputs/ directory at startup by api.py
_tasks_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# Step definitions for each operation type
# ---------------------------------------------------------------------------

GENERATE_STEPS: list[tuple[str, str]] = [
    ("load_profile", "Load Runtime Profile"),
    ("analyze", "Analyze Requirements"),
    ("design", "Design PBQ with AI"),
    ("validate_design", "Validate PBQ Design"),
    ("build_workspace", "Build Workspaces"),
    ("execute_tests", "Execute Reference Tests"),
    ("ai_repair_ref", "AI Repair Reference"),
    ("mutation_testing", "Mutation Testing"),
    ("ai_repair_tests", "AI Repair Tests"),
    ("validate_difficulty", "Validate Difficulty"),
    ("finalize", "Finalize Artifacts"),
]

DESIGN_ONLY_STEPS: list[tuple[str, str]] = [
    ("load_profile", "Load Runtime Profile"),
    ("analyze", "Analyze Requirements"),
    ("design", "Design PBQ with AI"),
    ("validate_design", "Validate PBQ Design"),
    ("finalize", "Finalize Artifacts"),
]

VALIDATE_STEPS: list[tuple[str, str]] = [
    ("load_bundle", "Load Saved Bundle"),
    ("build_workspace", "Build Workspaces"),
    ("execute_tests", "Execute Reference Tests"),
    ("ai_repair_ref", "AI Repair Reference"),
    ("mutation_testing", "Mutation Testing"),
    ("ai_repair_tests", "AI Repair Tests"),
    ("validate_difficulty", "Validate Difficulty"),
    ("finalize", "Finalize Artifacts"),
]

REGENERATE_STEPS: list[tuple[str, str]] = [
    ("load_bundle", "Load Saved Bundle"),
    ("build_workspace", "Build Workspaces"),
    ("ai_repair_ref", "AI Repair with AI"),
    ("execute_tests", "Execute Reference Tests"),
    ("mutation_testing", "Mutation Testing"),
    ("ai_repair_tests", "AI Repair Tests"),
    ("validate_difficulty", "Validate Difficulty"),
    ("finalize", "Finalize Artifacts"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProgressStep:
    key: str
    label: str
    status: str = "pending"  # pending | running | done | error | skipped
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    message: str = ""

    def start(self, message: str = "") -> None:
        self.status = "running"
        self.started_at = time.monotonic()
        if message:
            self.message = message

    def done(self, message: str = "") -> None:
        self.status = "done"
        self.finished_at = time.monotonic()
        if message:
            self.message = message

    def error(self, message: str = "") -> None:
        self.status = "error"
        self.finished_at = time.monotonic()
        if message:
            self.message = message

    def skip(self, message: str = "not needed") -> None:
        self.status = "skipped"
        self.finished_at = time.monotonic()
        if message:
            self.message = message

    def elapsed_seconds(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return round(end - self.started_at, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "elapsed": self.elapsed_seconds(),
            "message": self.message,
        }


@dataclass
class TaskProgress:
    task_id: str
    operation: str  # generate | validate | regenerate
    runtime: str
    steps: list[ProgressStep] = field(default_factory=list)
    overall_status: str = "running"  # running | done | error
    result: Optional[dict[str, Any]] = None
    error_message: str = ""
    created_at: float = field(default_factory=time.monotonic)

    def get_step(self, key: str) -> Optional[ProgressStep]:
        return next((s for s in self.steps if s.key == key), None)

    def abort_remaining(self, error: str) -> None:
        for step in self.steps:
            if step.status == "running":
                step.error(error)
            elif step.status == "pending":
                step.skip("aborted")
        self.overall_status = "error"
        self.error_message = error
        _persist(self)

    def to_dict(self) -> dict[str, Any]:
        current = next((s.key for s in self.steps if s.status == "running"), None)
        done_count = sum(1 for s in self.steps if s.status in ("done", "skipped"))
        return {
            "task_id": self.task_id,
            "operation": self.operation,
            "runtime": self.runtime,
            "overall_status": self.overall_status,
            "current_step": current,
            "done_steps": done_count,
            "total_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "result": self.result,
            "error": self.error_message,
        }


# ---------------------------------------------------------------------------
# Thread-safe registry
# ---------------------------------------------------------------------------

_tasks: dict[str, TaskProgress] = {}
_runtime_latest: dict[str, str] = {}  # runtime → latest task_id
_lock = threading.Lock()


def configure(tasks_dir: Path) -> None:
    """Call once at startup with the tasks output directory."""
    global _tasks_dir
    _tasks_dir = tasks_dir
    tasks_dir.mkdir(parents=True, exist_ok=True)


def _persist(task: TaskProgress) -> None:
    """Write task state to disk so it survives restarts. Best-effort."""
    if _tasks_dir is None:
        return
    try:
        path = _tasks_dir / f"{task.task_id}.json"
        path.write_text(json.dumps(task.to_dict(), indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("progress persist failed for %s: %s", task.task_id, exc)


def _load_from_disk(task_id: str) -> Optional[dict[str, Any]]:
    if _tasks_dir is None:
        return None
    path = _tasks_dir / f"{task_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_task(
    task_id: str,
    operation: str,
    runtime: str,
    step_defs: list[tuple[str, str]],
) -> TaskProgress:
    task = TaskProgress(
        task_id=task_id,
        operation=operation,
        runtime=runtime,
        steps=[ProgressStep(key=key, label=label) for key, label in step_defs],
    )
    with _lock:
        _tasks[task_id] = task
        _runtime_latest[runtime] = task_id
    _persist(task)
    return task


def get_task(task_id: str) -> Optional[TaskProgress]:
    with _lock:
        task = _tasks.get(task_id)
    if task is not None:
        return task
    # Fall back to disk for completed tasks from a previous server run
    data = _load_from_disk(task_id)
    if data is None:
        return None
    return _task_from_dict(data)


def get_runtime_task(runtime: str) -> Optional[TaskProgress]:
    with _lock:
        task_id = _runtime_latest.get(runtime)
        if task_id:
            return _tasks.get(task_id)
    return None


def _task_from_dict(data: dict[str, Any]) -> TaskProgress:
    steps = [
        ProgressStep(
            key=s["key"],
            label=s["label"],
            status=s.get("status", "pending"),
            message=s.get("message", ""),
        )
        for s in data.get("steps", [])
    ]
    task = TaskProgress(
        task_id=data["task_id"],
        operation=data.get("operation", "unknown"),
        runtime=data.get("runtime", ""),
        steps=steps,
        overall_status=data.get("overall_status", "unknown"),
        result=data.get("result"),
        error_message=data.get("error", ""),
    )
    return task
