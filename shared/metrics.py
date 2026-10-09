"""Small in-process metrics recorder for benchmark summaries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Metrics:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    repair_count: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def record_llm_call(self, *, prompt: str, output: str) -> None:
        self.llm_calls += 1
        self.input_tokens += max(1, len(prompt.split()))
        self.output_tokens += max(1, len(output.split()))

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "repair_count": self.repair_count,
            "generation_time_ms": self.elapsed_ms(),
        }

