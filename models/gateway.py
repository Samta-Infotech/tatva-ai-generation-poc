"""Common model gateway with fixture, Ollama, and OpenAI-compatible providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from config import settings
from shared.metrics import Metrics


@dataclass
class ModelGateway:
    provider: str = "fixture"
    generator_model: str = "fixture"
    judge_model: str = "fixture"
    solver_model: str = "fixture"
    repair_model: str = "fixture"
    metrics: Metrics | None = None

    def __post_init__(self) -> None:
        self.provider = settings.MODEL_PROVIDER if self.provider == "fixture" else self.provider
        self.generator_model = _resolve_model(self.generator_model, settings.GENERATOR_MODEL)
        self.judge_model = _resolve_model(self.judge_model, settings.JUDGE_MODEL)
        self.solver_model = _resolve_model(self.solver_model, settings.SOLVER_MODEL)
        self.repair_model = _resolve_model(self.repair_model, settings.REPAIR_MODEL)

    def _record(self, prompt: str, output: Any) -> Any:
        if self.metrics:
            self.metrics.record_llm_call(prompt=prompt, output=json.dumps(output, sort_keys=True))
        return output

    def structured_generate(
        self, prompt: str, *, schema_name: str, context: dict[str, Any], num_predict: int | None = None,
    ) -> dict[str, Any]:
        if self.provider in {"ollama", "openai", "openai-compatible"}:
            fallback = {"schema": schema_name, "provider": self.provider, "context": context, "slm_fallback": True}
            return self._record(
                prompt,
                self._model_json(
                    model=self.generator_model,
                    system="You generate strict JSON only for a question-generation POC.",
                    prompt=(
                        f"{prompt}\n\nSchema name: {schema_name}\n"
                        f"Context JSON:\n{json.dumps(context, indent=2, sort_keys=True)}\n\n"
                        "Return one JSON object. Do not include markdown."
                    ),
                    fallback=fallback,
                    num_predict=num_predict,
                ),
            )
        return self._record(prompt, {"schema": schema_name, "provider": self.provider, "context": context})

    def repair_generate(
        self, prompt: str, *, schema_name: str, context: dict[str, Any], num_predict: int | None = None,
    ) -> dict[str, Any]:
        if self.provider in {"ollama", "openai", "openai-compatible"}:
            fallback = {"schema": schema_name, "provider": self.provider, "context": context, "slm_fallback": True}
            return self._record(
                prompt,
                self._model_json(
                    model=self.repair_model,
                    system="You repair generated programming-question bundles. Return strict JSON only.",
                    prompt=(
                        f"{prompt}\n\nSchema name: {schema_name}\n"
                        f"Context JSON:\n{json.dumps(context, indent=2, sort_keys=True)}\n\n"
                        "Return one JSON object. Do not include markdown."
                    ),
                    fallback=fallback,
                    num_predict=num_predict,
                ),
            )
        return self._record(prompt, {"schema": schema_name, "provider": self.provider, "context": context})

    def judge(self, prompt: str, *, context: dict[str, Any]) -> dict[str, Any]:
        difficulty = str(context.get("difficulty") or context.get("requested_difficulty") or "medium").lower()
        fallback = {
            "predicted_difficulty": difficulty,
            "confidence": 0.82,
            "reasoning_steps": {"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2),
            "concept_count": {"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2),
            "direct_recall": difficulty == "easy",
            "appropriate_for_target": True,
        }
        if self.provider in {"ollama", "openai", "openai-compatible"}:
            judged = self._model_json(
                model=self.judge_model,
                system="You are an independent evaluator. Return strict JSON only.",
                prompt=(
                    f"{prompt}\n\nContext JSON:\n{json.dumps(context, indent=2, sort_keys=True)}\n\n"
                    "Return JSON with predicted_difficulty, confidence, reasoning_steps, "
                    "concept_count, direct_recall, and appropriate_for_target."
                ),
                fallback={**fallback, "slm_fallback": True},
            )
            return self._record(prompt, _normalize_judgement(judged, fallback))
        return self._record(prompt, fallback)

    def solve(self, prompt: str, *, question: str, options: list[dict[str, Any]]) -> dict[str, Any]:
        if self.provider in {"ollama", "openai", "openai-compatible"}:
            fallback = _fixture_solve(options)
            solved = self._model_json(
                model=self.solver_model,
                system="You solve multiple-choice questions. Return strict JSON only.",
                prompt=(
                    f"{prompt}\n\nQuestion:\n{question}\n\n"
                    f"Options JSON:\n{json.dumps(options, indent=2, sort_keys=True)}\n\n"
                    "Return JSON with answer_id and confidence. Do not include the explanation."
                ),
                fallback={**fallback, "slm_fallback": True},
            )
            answer_id = str(solved.get("answer_id") or "").strip()
            valid_ids = {str(option.get("id")) for option in options}
            if answer_id not in valid_ids:
                solved = {**fallback, "slm_invalid_answer": answer_id}
            return self._record(prompt, solved)
        return self._record(prompt, _fixture_solve(options))

    def _model_json(
        self, *, model: str, system: str, prompt: str, fallback: dict[str, Any], num_predict: int | None = None,
    ) -> dict[str, Any]:
        if self.provider == "ollama":
            return self._ollama_json(model=model, system=system, prompt=prompt, fallback=fallback, num_predict=num_predict)
        return self._openai_compatible_json(model=model, system=system, prompt=prompt, fallback=fallback, num_predict=num_predict)

    def _ollama_json(
        self, *, model: str, system: str, prompt: str, fallback: dict[str, Any], num_predict: int | None = None,
    ) -> dict[str, Any]:
        import time as _time

        if not settings.OLLAMA_BASE_URL:
            return {**fallback, "slm_error": "OLLAMA_BASE_URL is not configured"}
        headers = {"Content-Type": "application/json"}
        if settings.OLLAMA_CF_ACCESS_CLIENT_ID and settings.OLLAMA_CF_ACCESS_CLIENT_SECRET:
            headers["CF-Access-Client-Id"] = settings.OLLAMA_CF_ACCESS_CLIENT_ID
            headers["CF-Access-Client-Secret"] = settings.OLLAMA_CF_ACCESS_CLIENT_SECRET
        options = _ollama_options(str(fallback.get("schema") or ""))
        if num_predict is not None:
            options["num_predict"] = num_predict
        wall_clock_limit = getattr(settings, "PBQ_STAGE_FILE_TIMEOUT_SECONDS", 0) if num_predict is not None else 0
        payload = {
            "model": model,
            "stream": True,
            "format": "json",
            "options": options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        last_error = ""
        attempts = max(1, settings.OLLAMA_JSON_RETRIES + 1)
        for _ in range(attempts):
            try:
                parts: list[str] = []
                wall_start = _time.monotonic()
                timed_out = False
                with httpx.Client(timeout=settings.OLLAMA_TIMEOUT_SECONDS, headers=headers) as client:
                    with client.stream("POST", f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if wall_clock_limit and (_time.monotonic() - wall_start) > wall_clock_limit:
                                timed_out = True
                                break
                            if not line.strip():
                                continue
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            token = str((chunk.get("message") or {}).get("content") or "")
                            if token:
                                parts.append(token)
                            if chunk.get("done"):
                                break
                if timed_out:
                    return {**fallback, "slm_error": f"Wall-clock timeout after {wall_clock_limit}s (streamed {len(parts)} tokens)"}
                content = "".join(parts).strip()
            except Exception as exc:
                return {**fallback, "slm_error": str(exc)}
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                last_error = "Ollama response was not valid JSON"
                continue
            if not isinstance(parsed, dict):
                last_error = "Ollama response JSON was not an object"
                continue
            parsed.setdefault("provider", "ollama")
            parsed.setdefault("model", model)
            return parsed
        return {**fallback, "slm_error": last_error or "Ollama response was not valid JSON"}

    def _openai_compatible_json(
        self, *, model: str, system: str, prompt: str, fallback: dict[str, Any], num_predict: int | None = None,
    ) -> dict[str, Any]:
        if not settings.OPENAI_API_KEY:
            return {**fallback, "slm_error": "OPENAI_API_KEY is not configured"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        }
        if settings.OPENAI_ORGANIZATION:
            headers["OpenAI-Organization"] = settings.OPENAI_ORGANIZATION
        if settings.OPENAI_PROJECT:
            headers["OpenAI-Project"] = settings.OPENAI_PROJECT
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": num_predict if num_predict is not None else settings.OPENAI_MAX_COMPLETION_TOKENS,
        }
        try:
            with httpx.Client(timeout=settings.OPENAI_TIMEOUT_SECONDS, headers=headers) as client:
                response = client.post(f"{settings.OPENAI_BASE_URL}/chat/completions", json=payload)
                response.raise_for_status()
                raw = response.json()
        except Exception as exc:
            return {**fallback, "slm_error": str(exc)}
        content = str((((raw.get("choices") or [{}])[0].get("message") or {}).get("content")) or "").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {**fallback, "slm_error": "OpenAI-compatible response was not valid JSON", "slm_raw": content[:500]}
        if not isinstance(parsed, dict):
            return {**fallback, "slm_error": "OpenAI-compatible response JSON was not an object"}
        parsed.setdefault("provider", self.provider)
        parsed.setdefault("model", model)
        return parsed


def _resolve_model(value: str, env_value: str) -> str:
    configured = env_value if value == "fixture" else value
    if configured == "fixture" and settings.MODEL_PROVIDER == "ollama":
        return settings.OLLAMA_MODEL
    if configured == "fixture" and settings.MODEL_PROVIDER in {"openai", "openai-compatible"}:
        return settings.OPENAI_MODEL
    return configured


def _ollama_options(schema_name: str) -> dict[str, int]:
    options = {"num_ctx": settings.OLLAMA_NUM_CTX}
    if schema_name.startswith("PBQ"):
        options["num_predict"] = settings.OLLAMA_PBQ_NUM_PREDICT
    elif schema_name.startswith("MCQ"):
        options["num_predict"] = settings.OLLAMA_MCQ_NUM_PREDICT
    return options


def _fixture_solve(options: list[dict[str, Any]]) -> dict[str, Any]:
    for option in options:
        if option.get("is_correct"):
            return {"answer_id": option["id"], "confidence": 0.88}
    return {"answer_id": options[0]["id"] if options else "", "confidence": 0.25}


def _normalize_judgement(raw: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    difficulty = str(raw.get("predicted_difficulty") or fallback["predicted_difficulty"]).lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = fallback["predicted_difficulty"]
    return {
        **raw,
        "predicted_difficulty": difficulty,
        "confidence": _safe_float(raw.get("confidence"), fallback["confidence"]),
        "reasoning_steps": _safe_int(raw.get("reasoning_steps"), fallback["reasoning_steps"]),
        "concept_count": _safe_int(raw.get("concept_count"), fallback["concept_count"]),
        "direct_recall": bool(raw.get("direct_recall", fallback["direct_recall"])),
        "appropriate_for_target": bool(raw.get("appropriate_for_target", fallback["appropriate_for_target"])),
    }


def _safe_int(value: Any, fallback: int) -> int:
    if isinstance(value, list):
        return len(value) or fallback
    if isinstance(value, dict):
        return len(value) or fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
