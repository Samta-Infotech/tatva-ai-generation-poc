"""Normalized runtime profiles discovered from the TATVA PBQ runtime images.

Source references inspected read-only:
- candidate-screening-backend-django/core/pbq/manifest.py
- candidate-screening-backend-django/core/pbq/taxonomy.py
- candidate-screening-backend/pbq/local_runner_app.py
- candidate-screening-backend/pbq/images/runtime/*
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RuntimeProfile:
    id: str
    language: str
    framework: str
    runtime_version: str | None
    test_framework: str
    source_root: str
    execution_backend: str
    install_command: str | None
    validation_command: str | None
    test_command: str
    allowed_packages: list[str]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "django-sqlite-py312": RuntimeProfile(
        id="django-sqlite-py312",
        language="python",
        framework="django",
        runtime_version="3.12",
        test_framework="pytest",
        source_root="app",
        execution_backend="isolated_worker",
        install_command="python -m pip install -r requirements.txt",
        validation_command="python manage.py check",
        test_command="python -m pytest -q",
        allowed_packages=["Django", "djangorestframework", "pytest", "pytest-django"],
        metadata={
            "dependency_manager": "pip",
            "dependency_files": ["requirements.txt"],
            "test_roots": ["tests"],
            "runtime_image": "pbq/images/runtime/django-sqlite-py312",
            "actions": ["migrate", "test", "preview"],
        },
    ),
    "fastapi-py312": RuntimeProfile(
        id="fastapi-py312",
        language="python",
        framework="fastapi",
        runtime_version="3.12",
        test_framework="pytest",
        source_root="app",
        execution_backend="isolated_worker",
        install_command="python -m pip install -r requirements.txt",
        validation_command="python -m py_compile app/main.py",
        test_command="python -m pytest -q",
        allowed_packages=["fastapi", "uvicorn", "pytest", "httpx", "sqlalchemy", "aiosqlite"],
        metadata={
            "dependency_manager": "pip",
            "dependency_files": ["requirements.txt"],
            "test_roots": ["tests"],
            "runtime_image": "pbq/images/runtime/fastapi-py312",
            "actions": ["test", "preview"],
        },
    ),
    "flask-py312": RuntimeProfile(
        id="flask-py312",
        language="python",
        framework="flask",
        runtime_version="3.12",
        test_framework="pytest",
        source_root="app",
        execution_backend="isolated_worker",
        install_command=None,
        validation_command="python -m py_compile app/main.py",
        test_command="python -m pytest -q",
        allowed_packages=["Flask", "pytest", "Flask-SQLAlchemy"],
        metadata={
            "dependency_manager": "pip",
            "dependency_files": ["requirements.txt"],
            "test_roots": ["tests"],
            "runtime_image": "pbq/images/runtime/flask-py312",
            "actions": ["test", "preview"],
        },
    ),
    "express-node20": RuntimeProfile(
        id="express-node20",
        language="javascript",
        framework="express",
        runtime_version="20",
        test_framework="jest/supertest",
        source_root="src",
        execution_backend="isolated_worker",
        install_command="npm install",
        validation_command="node --check src/app.js",
        test_command="npm test -- --runInBand",
        allowed_packages=["express", "jest", "supertest", "sequelize", "sqlite3"],
        metadata={
            "dependency_manager": "npm",
            "dependency_files": ["package.json", "package-lock.json"],
            "test_roots": ["tests", "src"],
            "runtime_image": "pbq/images/runtime/express-node20",
            "actions": ["test", "preview"],
        },
    ),
    "react-node20": RuntimeProfile(
        id="react-node20",
        language="javascript",
        framework="react-vite",
        runtime_version="20",
        test_framework="vitest",
        source_root="src",
        execution_backend="isolated_worker",
        install_command="npm install",
        validation_command="npm run build",
        test_command="npm test",
        allowed_packages=[
            "react",
            "react-dom",
            "vite",
            "vitest",
            "jest",
            "jsdom",
            "@testing-library/react",
            "@vitejs/plugin-react",
        ],
        metadata={
            "dependency_manager": "npm",
            "dependency_files": ["package.json", "package-lock.json"],
            "test_roots": ["tests", "src"],
            "runtime_image": "pbq/images/runtime/react-vite-node20",
            "manifest_id": "react-node20",
            "actions": ["test", "preview"],
        },
    ),
    "springboot-java21": RuntimeProfile(
        id="springboot-java21",
        language="java",
        framework="spring-boot",
        runtime_version="21",
        test_framework="maven/junit",
        source_root="src/main/java",
        execution_backend="isolated_worker",
        install_command=None,
        validation_command="mvn -B -q test-compile",
        test_command="mvn -B -q test",
        allowed_packages=[
            "spring-boot-starter-web",
            "spring-boot-starter-test",
            "spring-boot-starter-data-jpa",
            "h2",
        ],
        metadata={
            "dependency_manager": "maven",
            "dependency_files": ["pom.xml"],
            "test_roots": ["src/test/java"],
            "runtime_image": "pbq/images/runtime/springboot-java21",
            "actions": ["test", "preview"],
        },
    ),
}


def get_runtime_profile(profile_id: str) -> RuntimeProfile:
    try:
        return RUNTIME_PROFILES[profile_id]
    except KeyError as exc:
        choices = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(f"Unknown runtime profile '{profile_id}'. Supported: {choices}") from exc


def all_runtime_profiles() -> list[RuntimeProfile]:
    return [RUNTIME_PROFILES[key] for key in sorted(RUNTIME_PROFILES)]
