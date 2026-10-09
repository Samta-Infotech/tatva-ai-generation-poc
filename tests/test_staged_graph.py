"""Tests for the staged LangGraph PBQ generation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from config.runtime_profiles import get_runtime_profile
from pbq.schemas import PBQBundle, PBQDesign, PBQRequest
from pbq.staged_graph import (
    StagedGenState,
    _RuntimeContext,
    _extract_file_content,
    _is_boilerplate,
    _persist_file,
    _resume_artifacts,
    _topological_sort,
    _validate_file_content,
    build_staged_pbq_graph,
    generate_staged_pbq_bundle,
)
from pbq.state import PBQState
from shared.schemas import FileEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_request(**overrides) -> PBQRequest:
    defaults = {
        "runtime_profile": "django-sqlite-py312",
        "difficulty": "medium",
        "experience": "mid",
        "duration": 90,
        "prompt": "Build a task management API with projects, tasks, and status tracking",
    }
    defaults.update(overrides)
    return PBQRequest(**defaults)


def _make_state(request: PBQRequest | None = None) -> PBQState:
    request = request or _make_request()
    with tempfile.TemporaryDirectory(prefix="tatva-test-") as tmp:
        state = PBQState(request=request, output_dir=Path(tmp))
        state.runtime_profile = get_runtime_profile(request.runtime_profile)
        yield state


@pytest.fixture
def state():
    yield from _make_state()


@pytest.fixture
def django_state():
    yield from _make_state(_make_request(runtime_profile="django-sqlite-py312"))


@pytest.fixture
def react_state():
    yield from _make_state(_make_request(runtime_profile="react-node20"))


def _mock_gateway(responses: list[dict[str, Any]] | None = None) -> MagicMock:
    """Create a mock ModelGateway that returns predefined responses."""
    gw = MagicMock()
    gw.provider = "ollama"

    if responses is None:
        responses = []

    call_count = {"n": 0}

    def structured_gen(prompt, *, schema_name, context, num_predict=None):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(responses):
            return responses[idx]
        return {"slm_fallback": True, "schema": schema_name}

    def repair_gen(prompt, *, schema_name, context, num_predict=None):
        return {"slm_fallback": True}

    gw.structured_generate = MagicMock(side_effect=structured_gen)
    gw.repair_generate = MagicMock(side_effect=repair_gen)
    return gw


# ---------------------------------------------------------------------------
# Unit tests: _topological_sort
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    def test_sorts_by_dependencies(self):
        manifest = [
            {"path": "app/main.py", "role": "reference", "depends_on": ["app/services.py"]},
            {"path": "app/__init__.py", "role": "reference", "depends_on": []},
            {"path": "app/services.py", "role": "reference", "depends_on": ["app/__init__.py"]},
        ]
        ordered = _topological_sort(manifest, "reference")
        paths = [e["path"] for e in ordered]
        assert paths.index("app/__init__.py") < paths.index("app/services.py")
        assert paths.index("app/services.py") < paths.index("app/main.py")

    def test_detects_cycle(self):
        manifest = [
            {"path": "a.py", "role": "reference", "depends_on": ["b.py"]},
            {"path": "b.py", "role": "reference", "depends_on": ["a.py"]},
        ]
        with pytest.raises(ValueError, match="cycle"):
            _topological_sort(manifest, "reference")

    def test_filters_by_role(self):
        manifest = [
            {"path": "app.py", "role": "reference", "depends_on": []},
            {"path": "test.py", "role": "test_public", "depends_on": []},
        ]
        ordered = _topological_sort(manifest, "reference")
        assert len(ordered) == 1
        assert ordered[0]["path"] == "app.py"

    def test_empty_manifest(self):
        assert _topological_sort([], "reference") == []

    def test_no_dependencies(self):
        manifest = [
            {"path": "a.py", "role": "reference", "depends_on": []},
            {"path": "b.py", "role": "reference", "depends_on": []},
        ]
        ordered = _topological_sort(manifest, "reference")
        assert len(ordered) == 2


# ---------------------------------------------------------------------------
# Unit tests: _extract_file_content
# ---------------------------------------------------------------------------

class TestExtractFileContent:
    def test_direct_content(self):
        raw = {"content": "def hello(): pass\n", "path": "a.py"}
        assert _extract_file_content(raw, "a.py") == "def hello(): pass\n"

    def test_slm_fallback_returns_none(self):
        raw = {"slm_fallback": True}
        assert _extract_file_content(raw, "a.py") is None

    def test_slm_error_returns_none(self):
        raw = {"slm_error": "timeout"}
        assert _extract_file_content(raw, "a.py") is None

    def test_nested_content(self):
        raw = {"file": {"content": "class Foo: pass\n"}}
        assert _extract_file_content(raw, "a.py") == "class Foo: pass\n"

    def test_empty_content_returns_none(self):
        raw = {"content": "   "}
        assert _extract_file_content(raw, "a.py") is None

    def test_truncated_json_returns_none(self):
        raw = {"content": None, "partial": True}
        assert _extract_file_content(raw, "a.py") is None


# ---------------------------------------------------------------------------
# Unit tests: _validate_file_content
# ---------------------------------------------------------------------------

class TestValidateFileContent:
    def test_valid_python(self):
        assert _validate_file_content("a.py", "def foo(): return 1\n", "python") is None

    def test_invalid_python_syntax(self):
        result = _validate_file_content("a.py", "def foo(:\n", "python")
        assert result is not None

    def test_empty_file_rejected(self):
        result = _validate_file_content("a.py", "", "python")
        assert result is not None

    def test_init_py_empty_allowed(self):
        assert _validate_file_content("app/__init__.py", "", "python") is None

    def test_valid_js(self):
        assert _validate_file_content("a.js", "function foo() { return 1; }\n", "javascript") is None

    def test_unbalanced_js(self):
        result = _validate_file_content("a.js", "function foo() { { {", "javascript")
        assert result is not None


# ---------------------------------------------------------------------------
# Unit tests: _is_boilerplate
# ---------------------------------------------------------------------------

class TestIsBoilerplate:
    def test_init_py_is_boilerplate(self):
        assert _is_boilerplate("__init__.py", "django-sqlite-py312")
        assert _is_boilerplate("app/__init__.py", "fastapi-py312")

    def test_manage_py_is_boilerplate_for_django(self):
        assert _is_boilerplate("manage.py", "django-sqlite-py312")
        assert not _is_boilerplate("manage.py", "fastapi-py312")

    def test_source_file_is_not_boilerplate(self):
        assert not _is_boilerplate("app/services.py", "django-sqlite-py312")
        assert not _is_boilerplate("src/App.jsx", "react-node20")


# ---------------------------------------------------------------------------
# Unit tests: persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persist_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = _persist_file(tmp, "app/services.py", "def hello(): pass\n", "reference")
            assert record["path"] == "app/services.py"
            assert record["role"] == "reference"
            assert Path(record["persisted_at"]).is_file()

            resumed = _resume_artifacts(tmp, "reference")
            assert len(resumed) == 1
            assert resumed[0]["path"] == "app/services.py"
            assert resumed[0]["content"] == "def hello(): pass\n"

    def test_resume_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert _resume_artifacts(tmp, "reference") == []

    def test_resume_nonexistent_dir(self):
        assert _resume_artifacts("/nonexistent/path", "reference") == []


# ---------------------------------------------------------------------------
# Integration: file-level retry
# ---------------------------------------------------------------------------

class TestFileLevelRetry:
    def test_retry_on_invalid_response_then_succeed(self):
        gw = _mock_gateway([
            # feature plan
            {"domain_summary": "task management", "entities": ["project", "task"],
             "features": ["create project", "list tasks"], "primary_workflow": "manage tasks",
             "validation_rules": ["name required"], "public_test_focus": "CRUD",
             "private_test_focus": "validation", "mutation_ideas": ["skip validation"],
             "candidate_scope": "services layer"},
            # question spec
            {"title": "Task Manager API", "behavioral_contract": [
                "Create projects", "Add tasks to projects", "Track task status"],
             "scaffolding_strategy": "Django service layer",
             "file_manifest": [
                 {"path": "app/services.py", "role": "reference", "depends_on": [], "description": "Task service"}
             ]},
            # first attempt at reference file — invalid
            {"slm_error": "truncated"},
            # second attempt (repair loop)
            {"path": "app/services.py", "content": "def create_project(name):\n    return {'name': name}\n", "kind": "file"},
            # starter file
            {"path": "app/services.py", "content": "def create_project(name):\n    # TODO: implement\n    pass\n", "kind": "file"},
            # public tests
            {"path": "tests/test_public.py", "content": "def test_create(): assert True\n", "kind": "file"},
            # private tests
            {"path": "tests/test_private.py", "content": "def test_validate(): assert True\n", "kind": "file"},
            # mutations
            {"mutations": [{"name": "skip_validation", "changes": [
                {"path": "app/services.py", "content": "def create_project(name):\n    return {}\n", "kind": "file"}
            ]}]},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            state = PBQState(request=_make_request(), output_dir=Path(tmp))
            state.runtime_profile = get_runtime_profile("django-sqlite-py312")
            fallback = PBQBundle(
                requirements={}, design=PBQDesign("test", [], "", [], []),
                starter_files=[], public_tests=[], private_tests=[],
                reference_solution=[], mutations=[],
            )
            result = generate_staged_pbq_bundle(fallback, state, gw)
            assert isinstance(result, PBQBundle)
            assert result.requirements.get("generation_mode") in {"staged_generation", "staged_partial_failure"}


# ---------------------------------------------------------------------------
# Integration: complete failure
# ---------------------------------------------------------------------------

class TestCompleteFailure:
    def test_all_slm_calls_fail(self):
        gw = _mock_gateway([
            {"slm_error": "connection refused"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            state = PBQState(request=_make_request(), output_dir=Path(tmp))
            state.runtime_profile = get_runtime_profile("django-sqlite-py312")
            fallback = PBQBundle(
                requirements={}, design=PBQDesign("test", [], "", [], []),
            )
            result = generate_staged_pbq_bundle(fallback, state, gw)
            assert isinstance(result, PBQBundle)
            assert result.requirements.get("generation_mode") == "ai_generation_failed_requires_repair"


# ---------------------------------------------------------------------------
# Integration: deterministic boilerplate
# ---------------------------------------------------------------------------

class TestDeterministicBoilerplate:
    def test_init_py_generated_without_slm(self):
        gw = _mock_gateway([
            # feature plan
            {"domain_summary": "test", "features": ["test"]},
            # question spec with __init__.py in manifest
            {"title": "Test", "behavioral_contract": ["a", "b", "c"],
             "scaffolding_strategy": "test",
             "file_manifest": [
                 {"path": "app/__init__.py", "role": "reference", "depends_on": [], "description": "init"},
                 {"path": "app/services.py", "role": "reference", "depends_on": ["app/__init__.py"], "description": "main"},
             ]},
            # Only 1 SLM call for services.py (init.py is boilerplate)
            {"path": "app/services.py", "content": "def main(): return 1\n", "kind": "file"},
            # starter for init (boilerplate, no SLM)
            # starter for services.py
            {"path": "app/services.py", "content": "def main():\n    # TODO\n    pass\n", "kind": "file"},
            # public tests
            {"path": "tests/test_public.py", "content": "def test_x(): assert True\n", "kind": "file"},
            # private tests
            {"path": "tests/test_private.py", "content": "def test_y(): assert True\n", "kind": "file"},
            # mutations
            {"mutations": [{"name": "m1", "changes": [{"path": "app/services.py", "content": "def main(): return 0\n", "kind": "file"}]}]},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            state = PBQState(request=_make_request(), output_dir=Path(tmp))
            state.runtime_profile = get_runtime_profile("django-sqlite-py312")
            fallback = PBQBundle(requirements={}, design=PBQDesign("test", [], "", [], []))
            result = generate_staged_pbq_bundle(fallback, state, gw)
            assert isinstance(result, PBQBundle)
            # __init__.py should have been generated without SLM
            init_files = [f for f in result.reference_solution if f.path == "app/__init__.py"]
            assert len(init_files) >= 1


# ---------------------------------------------------------------------------
# Unit test: graph structure
# ---------------------------------------------------------------------------

class TestGraphStructure:
    def test_graph_compiles(self):
        ctx = _RuntimeContext(MagicMock(), None)
        compiled = build_staged_pbq_graph(ctx)
        assert compiled is not None
