"""PBQ state container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.runtime_profiles import RuntimeProfile
from pbq.schemas import PBQBundle, PBQRequest
from shared.metrics import Metrics


@dataclass
class PBQState:
    request: PBQRequest
    output_dir: Path
    metrics: Metrics = field(default_factory=Metrics)
    runtime_profile: RuntimeProfile | None = None
    bundle: PBQBundle | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    mutation: dict[str, Any] = field(default_factory=dict)
    difficulty: dict[str, Any] = field(default_factory=dict)
    final: dict[str, Any] = field(default_factory=dict)
    runtime_env: dict[str, str] | None = None

