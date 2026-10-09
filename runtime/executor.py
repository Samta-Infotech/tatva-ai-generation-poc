"""Runtime command execution with honest result capture."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from config.settings import EXECUTION_TIMEOUT_SECONDS
from shared.schemas import CommandResult

# Per-profile runtime cache root — venvs and node_modules are created once
# per profile and reused across all generation runs.
_RUNTIME_CACHE_ROOT = Path(os.getenv("PBQ_RUNTIME_CACHE_ROOT", "/tmp/tatva-pbq-runtimes"))

_NPM_INSTALL_TIMEOUT = 180


def ensure_runtime_venv(profile_id: str, packages: list[str]) -> dict[str, str]:
    """Return env overrides that activate a cached per-profile venv, creating it if needed.

    The venv is keyed by profile_id and its marker file contains the sorted package list.
    If the marker matches, the venv is reused without running pip again.
    """
    environment = _RUNTIME_CACHE_ROOT / profile_id
    python = environment / "bin" / "python"
    marker = environment / ".tatva-runtime-ready"
    expected_marker = "\n".join(sorted(packages))

    if not marker.is_file() or marker.read_text(encoding="utf-8") != expected_marker or not python.is_file():
        _RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        create = subprocess.run(
            (sys.executable, "-m", "venv", str(environment)),
            text=True,
            capture_output=True,
            check=False,
            timeout=EXECUTION_TIMEOUT_SECONDS,
        )
        if create.returncode:
            raise RuntimeError(
                f"Failed to create runtime venv for profile '{profile_id}': {create.stderr.strip()}"
            )
        install = subprocess.run(
            (
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                *packages,
            ),
            text=True,
            capture_output=True,
            check=False,
            timeout=EXECUTION_TIMEOUT_SECONDS,
        )
        if install.returncode:
            raise RuntimeError(
                f"Failed to install packages for profile '{profile_id}': {install.stderr.strip()}"
            )
        marker.write_text(expected_marker, encoding="utf-8")

    return {
        "VIRTUAL_ENV": str(environment),
        "PATH": f"{environment / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }


def ensure_npm_cache(profile_id: str, workspace: Path) -> Path:
    """Run npm install in *workspace* once, caching the resulting node_modules.

    The cache is keyed by profile_id + a hash of the workspace's package.json
    content.  If the cache is valid, node_modules is returned without running
    npm again.
    """
    pkg_json = workspace / "package.json"
    if not pkg_json.is_file():
        raise RuntimeError(f"No package.json in workspace {workspace}")

    pkg_hash = hashlib.sha256(pkg_json.read_bytes()).hexdigest()[:16]
    cache_dir = _RUNTIME_CACHE_ROOT / f"{profile_id}-npm-{pkg_hash}"
    marker = cache_dir / ".tatva-npm-ready"
    node_modules = cache_dir / "node_modules"

    if marker.is_file() and node_modules.is_dir():
        return node_modules

    _RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    shutil.copy2(pkg_json, cache_dir / "package.json")
    lock = workspace / "package-lock.json"
    if lock.is_file():
        shutil.copy2(lock, cache_dir / "package-lock.json")

    install = subprocess.run(
        ("npm", "install", "--prefer-offline", "--no-audit", "--no-fund"),
        cwd=cache_dir,
        text=True,
        capture_output=True,
        check=False,
        timeout=_NPM_INSTALL_TIMEOUT,
    )
    if install.returncode:
        raise RuntimeError(
            f"Failed to npm install for profile '{profile_id}': {install.stderr.strip()}"
        )
    marker.write_text(pkg_hash, encoding="utf-8")
    return node_modules


def inject_npm_cache(node_modules_cache: Path, workspace: Path) -> None:
    """Copy the cached node_modules into a workspace directory."""
    dest = workspace / "node_modules"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(node_modules_cache, dest, symlinks=True)


def run_command(
    command: str,
    *,
    cwd: Path,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    parts = shlex.split(command)
    executable = parts[0] if parts else ""
    command_to_run = command

    # Only remap bare 'python' to sys.executable when no runtime venv is active.
    # When a venv is injected via extra_env, the venv's bin/python is found via PATH.
    if executable == "python" and not extra_env:
        command_to_run = shlex.join([sys.executable, *parts[1:]])
        executable = sys.executable

    # Resolve executable for existence check using the effective PATH.
    effective_env = {**os.environ, **(extra_env or {})}
    effective_path = effective_env.get("PATH")
    if executable and shutil.which(executable, path=effective_path) is None:
        return CommandResult(
            command=command,
            exit_code=None,
            stdout="",
            stderr="",
            duration_ms=0,
            skipped=True,
            skip_reason=f"Executable not found: {executable}",
        )

    start = time.perf_counter()
    # Use Popen + setsid so that on timeout we can kill the entire process
    # group (shell + all its children). subprocess.run with shell=True only
    # kills the shell process on timeout; npm/pip children become orphans,
    # inherit the open pipe, and communicate() blocks until they eventually
    # finish — causing hangs that far exceed the intended timeout.
    proc = subprocess.Popen(
        command_to_run,
        cwd=cwd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=effective_env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return CommandResult(
            command=command,
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
    except subprocess.TimeoutExpired:
        # Kill the entire process group so that child processes (npm, pip,
        # pytest, etc.) are also terminated and don't keep the pipes open.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        return CommandResult(
            command=command,
            exit_code=None,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_ms=int((time.perf_counter() - start) * 1000),
            timed_out=True,
        )
