"""Shared helpers for PBQ graph modules.

Extracted from graph.py so that staged_graph.py can reuse them
without circular imports.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from config.pbq_templates import (
    template_generation_context,
    template_locked_paths,
    template_starter_files,
)
from config.runtime_profiles import RuntimeProfile
from pbq.schemas import PBQBundle, PBQDesign, PBQRequest
from pbq.state import PBQState
from runtime.workspace import validate_relative_path
from shared.schemas import FileEntry


def template_context_for_request(request: PBQRequest) -> dict[str, Any] | None:
    if not request.template_slug:
        return None
    return template_generation_context(request.template_slug)


def extract_user_prompt(prompt: str) -> str:
    for line in str(prompt or "").splitlines():
        if line.lower().startswith("user prompt:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    return str(prompt or "").strip()


def framework_generation_rules(profile: RuntimeProfile) -> str:
    if profile.id == "django-sqlite-py312":
        return (
            "Generate a minimal Django + pytest workspace. Include requirements.txt with Django, "
            "djangorestframework, pytest, and pytest-django. Include manage.py, config/__init__.py, "
            "config/settings.py, config/urls.py, pytest.ini, app/__init__.py, app/services.py, "
            "and tests. Tests may import service functions directly; no database migration should be required."
        )
    if profile.id == "fastapi-py312":
        return (
            "Generate a minimal FastAPI + pytest workspace. Include requirements.txt with fastapi, "
            "uvicorn, pytest, and httpx. Include app/__init__.py, app/main.py, app/services.py, "
            "and tests using fastapi.testclient.TestClient."
        )
    if profile.id == "react-node20":
        return (
            "Generate a minimal React + Vite + Vitest workspace. Include package.json, index.html, "
            "src files, public tests, and private tests. package.json must define build as vite build "
            "and test as vitest run. Tests must use @testing-library/react and jsdom."
        )
    return "Generate a minimal workspace that matches the runtime profile exactly."


def safe_file_entries(raw_items: Any, field_name: str) -> list[FileEntry]:
    if isinstance(raw_items, dict):
        raw_items = [
            {**value, "path": value.get("path") or path}
            if isinstance(value, dict)
            else {"path": path, "content": str(value), "kind": "file"}
            for path, value in raw_items.items()
        ]
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"{field_name} must be a non-empty list")
    entries: list[FileEntry] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} entries must be objects")
        kind = str(item.get("kind") or "file")
        if kind != "file":
            raise ValueError(f"{field_name} only supports file entries")
        path = validate_relative_path(str(item.get("path") or ""))
        if path.startswith((".env", "node_modules/", "__pycache__/", ".pytest_cache/", "target/")):
            raise ValueError(f"{field_name} contains blocked path: {path}")
        content = str(item.get("content") or "")
        if not content and path not in {"app/__init__.py", "config/__init__.py"}:
            raise ValueError(f"{field_name} contains empty file: {path}")
        if len(content) > 50000:
            raise ValueError(f"{field_name} file is too large: {path}")
        if field_name in {"reference_solution", "public_tests", "private_tests"}:
            lowered = content.lower()
            blocked_markers = ("testclient(...)", "placeholder", "todo", "not implemented")
            empty_function = re.search(r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*:\s*\n\s*pass\s*(#.*)?$", content, re.MULTILINE)
            if any(marker in lowered for marker in blocked_markers) or empty_function:
                raise ValueError(f"{field_name} contains placeholder implementation: {path}")
        entries.append(FileEntry(path=path, content=content, kind=kind))
    if len(entries) > 32:
        raise ValueError(f"{field_name} contains too many files")
    return entries


def safe_requirements(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, list):
        return {"notes": [str(item) for item in raw_value]}
    if raw_value:
        return {"notes": [str(raw_value)]}
    return {}


def safe_mutations(raw_items: Any) -> list[dict[str, Any]]:
    if isinstance(raw_items, dict):
        raw_items = [
            {"name": name, "changes": changes if isinstance(changes, list) else [changes]}
            for name, changes in raw_items.items()
        ]
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("mutations must be a non-empty list")
    mutations: list[dict[str, Any]] = []
    for index, mutation in enumerate(raw_items):
        if not isinstance(mutation, dict):
            raise ValueError("mutation entries must be objects")
        changes = safe_file_entries(mutation.get("changes"), f"mutations[{index}].changes")
        mutations.append(
            {
                "name": str(mutation.get("name") or f"mutation_{index + 1}"),
                "changes": [entry.to_dict() for entry in changes],
            }
        )
    return mutations


def unwrap_schema_payload(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return raw
    if raw.get("design") or raw.get("starter_files") or raw.get("reference_solution"):
        return raw
    for key in ("PBQExecutableBundle", "PBQExecutableBundleRepair"):
        value = raw.get(key)
        if isinstance(value, dict):
            return {**value, "provider": raw.get("provider"), "model": raw.get("model")}
    for value in raw.values():
        if isinstance(value, dict) and (
            value.get("design") or value.get("starter_files") or value.get("reference_solution")
        ):
            return value
    return raw


def design_from_slm(raw: dict[str, Any], fallback: PBQBundle) -> PBQDesign:
    raw = unwrap_schema_payload(raw)
    raw_design = raw.get("design") if isinstance(raw.get("design"), dict) else raw
    title = str(raw_design.get("title") or fallback.design.title).strip()
    behavioral_contract = raw_design.get("behavioral_contract")
    candidate_freedom = raw_design.get("candidate_freedom")
    difficulty_notes = raw_design.get("difficulty_notes")
    selected_contract = (
        [str(item) for item in behavioral_contract if str(item).strip()]
        if isinstance(behavioral_contract, list) and behavioral_contract
        else fallback.design.behavioral_contract
    )
    if selected_contract and title and not any(title.lower().split()[0] in item.lower() for item in selected_contract):
        selected_contract.append(f"Implement behavior that matches the generated task title: {title}.")
    return PBQDesign(
        title=title or fallback.design.title,
        behavioral_contract=selected_contract,
        scaffolding_strategy=str(raw_design.get("scaffolding_strategy") or fallback.design.scaffolding_strategy),
        candidate_freedom=(
            [str(item) for item in candidate_freedom]
            if isinstance(candidate_freedom, list) and candidate_freedom
            else fallback.design.candidate_freedom
        ),
        difficulty_notes=(
            [str(item) for item in difficulty_notes]
            if isinstance(difficulty_notes, list) and difficulty_notes
            else fallback.design.difficulty_notes
        ),
    )


def prompt_scoped_design(profile: RuntimeProfile, request: PBQRequest, plan: dict[str, Any] | None = None) -> PBQDesign:
    framework_names = {
        "fastapi": "FastAPI",
        "react-vite": "React + Vite",
        "spring-boot": "Spring Boot",
        "django": "Django",
        "flask": "Flask",
        "express": "Express",
    }
    framework_name = framework_names.get(profile.framework, profile.framework.title())
    prompt = request.prompt.strip() or "custom business workflow"
    title_seed = prompt.rstrip(".")
    title = f"{framework_name} PBQ: {title_seed[:80]}"
    features = [str(item) for item in (plan or {}).get("features", []) if str(item).strip()]
    if features:
        contract = [f"Implement {item}." for item in features[:5]]
    else:
        contract = [
            f"Model the core entities and rules for: {prompt}.",
            "Implement the primary user workflow end to end.",
            "Reject invalid inputs with clear errors.",
            "Return externally observable outputs that tests can assert.",
        ]
    return PBQDesign(
        title=title,
        behavioral_contract=contract,
        scaffolding_strategy=(
            f"Generate a compact {framework_name} workspace from the prompt, with starter files for candidates "
            "and a separate reference solution used only for validation."
        ),
        candidate_freedom=[
            f"May edit or create files under {profile.source_root}.",
            "May choose internal names and helper functions as long as public behavior and tests pass.",
            "Must not change runtime commands or dependency policy.",
        ],
        difficulty_notes=[
            f"Requested difficulty: {request.difficulty}",
            f"Target experience: {request.experience}",
            "Scope is selected from the prompt before code generation.",
        ],
    )


def merge_template_starter_files(bundle: PBQBundle, request: PBQRequest) -> PBQBundle:
    if not request.template_slug:
        return bundle
    template_files = template_starter_files(request.template_slug)
    locked = template_locked_paths(request.template_slug)
    by_path = {entry.path: entry for entry in template_files}
    for entry in bundle.starter_files:
        if entry.path in locked:
            continue
        by_path[entry.path] = entry
    bundle.starter_files = list(by_path.values())
    return bundle


def ensure_profile_support_files(bundle: PBQBundle, profile: RuntimeProfile) -> PBQBundle:
    if profile.language == "python" and not _has_file(bundle.starter_files, "requirements.txt"):
        bundle.starter_files.append(FileEntry("requirements.txt", "\n".join(profile.allowed_packages) + "\n"))
        _append_note_once(bundle, "POC added missing requirements.txt from runtime allowed packages.")
    if profile.id == "django-sqlite-py312":
        django_support = [
            FileEntry("manage.py", "import os\nimport sys\n\nif __name__ == '__main__':\n    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n    from django.core.management import execute_from_command_line\n    execute_from_command_line(sys.argv)\n"),
            FileEntry("config/__init__.py", ""),
            FileEntry("config/settings.py", "SECRET_KEY='poc'\nINSTALLED_APPS=[]\nROOT_URLCONF='config.urls'\nDEFAULT_AUTO_FIELD='django.db.models.BigAutoField'\n"),
            FileEntry("config/urls.py", "urlpatterns = []\n"),
            FileEntry("pytest.ini", "[pytest]\nDJANGO_SETTINGS_MODULE = config.settings\npython_files = tests.py test_*.py *_tests.py\n"),
            FileEntry("app/__init__.py", ""),
        ]
        for entry in django_support:
            if not _has_file(bundle.starter_files, entry.path):
                bundle.starter_files.append(entry)
        _append_note_once(bundle, "POC ensured Django runtime support files are present.")
    if profile.id == "fastapi-py312" and not _has_file(bundle.starter_files, "app/__init__.py"):
        bundle.starter_files.append(FileEntry("app/__init__.py", ""))
    if profile.id == "react-node20":
        if not _has_file(bundle.starter_files, "package.json"):
            bundle.starter_files.append(
                FileEntry(
                    "package.json",
                    '{\n'
                    '  "scripts": {"build": "vite build", "test": "vitest run"},\n'
                    '  "dependencies": {"@vitejs/plugin-react": "latest", "@testing-library/react": "latest", '
                    '"jsdom": "latest", "react": "latest", "react-dom": "latest", "vite": "latest", "vitest": "latest"},\n'
                    '  "devDependencies": {}\n'
                    '}\n',
                )
            )
            _append_note_once(bundle, "POC added missing package.json from runtime allowed packages.")
        if not _has_file(bundle.starter_files, "index.html"):
            bundle.starter_files.append(
                FileEntry("index.html", '<div id="root"></div><script type="module" src="/src/App.jsx"></script>\n')
            )
            _append_note_once(bundle, "POC added missing index.html for Vite.")
    return bundle


def ai_generation_failed_bundle(state: PBQState, raw: dict[str, Any], error: str, plan: dict[str, Any]) -> PBQBundle:
    assert state.runtime_profile is not None
    design = prompt_scoped_design(state.runtime_profile, state.request, plan)
    design.difficulty_notes.append(f"AI bundle generation failed schema validation: {error}")
    message = (
        "AI generated an invalid PBQ bundle. Review requirements.slm_bundle_generation and click "
        "Regenerate With AI to repair from this feedback."
    )
    starter: list[FileEntry] = template_starter_files(state.request.template_slug) if state.request.template_slug else []
    reference: list[FileEntry] = []
    public_tests: list[FileEntry] = []
    private_tests: list[FileEntry] = []
    if state.runtime_profile.id in {"django-sqlite-py312", "fastapi-py312"}:
        starter = starter or [
            FileEntry("app/__init__.py", ""),
            FileEntry("app/main.py", "def generation_status():\n    return {'status': 'ai_generation_failed'}\n"),
        ]
        reference = [FileEntry("app/main.py", "def generation_status():\n    return {'status': 'ai_generation_failed'}\n")]
        public_tests = [
            FileEntry(
                "tests/test_generation_failed.py",
                f"def test_ai_generation_must_be_repaired():\n    assert False, {message!r}\n",
            )
        ]
        private_tests = [
            FileEntry(
                "tests/test_generation_failed_private.py",
                f"def test_ai_generation_must_be_repaired_private():\n    assert False, {message!r}\n",
            )
        ]
    elif state.runtime_profile.id == "react-node20":
        starter = starter or [
            FileEntry("src/App.jsx", "export default function App() { return <main>AI generation failed</main>; }\n"),
        ]
        reference = list(starter)
        public_tests = [
            FileEntry(
                "src/App.public.test.jsx",
                f"import {{ test, expect }} from 'vitest';\ntest('ai generation must be repaired', () => {{ expect({message!r}).toBe(''); }});\n",
            )
        ]
        private_tests = [
            FileEntry(
                "src/App.private.test.jsx",
                f"import {{ test, expect }} from 'vitest';\ntest('ai generation must be repaired privately', () => {{ expect({message!r}).toBe(''); }});\n",
            )
        ]
    bundle = PBQBundle(
        requirements={
            "prompt": state.request.prompt,
            "template": template_context_for_request(state.request),
            "ai_feature_plan": plan,
            "slm_bundle_generation": {**raw, "slm_invalid_bundle": error},
            "generation_mode": "ai_generation_failed_requires_repair",
        },
        design=design,
        starter_files=starter,
        public_tests=public_tests,
        private_tests=private_tests,
        reference_solution=reference,
        mutations=[],
    )
    return ensure_profile_support_files(bundle, state.runtime_profile)


def prompt_allows_commerce_terms(prompt: str) -> bool:
    prompt_text = prompt.lower()
    allowed_markers = (
        "cart", "checkout", "commerce", "ecommerce", "e-commerce",
        "inventory", "order", "payment", "price", "pricing",
        "product", "retail", "shop", "store",
    )
    return any(marker in prompt_text for marker in allowed_markers)


def assert_prompt_domain_alignment(bundle: PBQBundle, request: PBQRequest) -> None:
    if prompt_allows_commerce_terms(request.prompt):
        return
    searchable_parts = [
        bundle.design.title,
        *bundle.design.behavioral_contract,
        bundle.design.scaffolding_strategy,
    ]
    for file_group in (bundle.starter_files, bundle.reference_solution, bundle.public_tests, bundle.private_tests):
        searchable_parts.extend(f"{entry.path}\n{entry.content[:4000]}" for entry in file_group)
    searchable = "\n".join(searchable_parts).lower()
    drift_markers = (
        "cart", "discount", "subtotal", "tax_rate",
        "unit_price", "unitprice", "order line", "order/cart",
    )
    if any(marker in searchable for marker in drift_markers):
        raise ValueError(
            "Generated bundle drifted to the old cart/ecommerce example even though the prompt did not ask for it"
        )


def _has_file(files: list[FileEntry], path: str) -> bool:
    return any(entry.path == path for entry in files)


def _append_note_once(bundle: PBQBundle, note: str) -> None:
    if note not in bundle.design.difficulty_notes:
        bundle.design.difficulty_notes.append(note)
