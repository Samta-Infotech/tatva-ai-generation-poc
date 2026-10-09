"""Separate MCQ generation and validation graph."""

from __future__ import annotations

from pathlib import Path

from config.settings import MODEL_PROVIDER, OUTPUT_DIR
from mcq.nodes.analyze import analyze_mcq_request
from mcq.nodes.difficulty import create_difficulty_blueprint, validate_difficulty
from mcq.nodes.distractors import validate_distractors
from mcq.nodes.generate import generate_mcq
from mcq.nodes.repair import repair_mcq
from mcq.nodes.solve import independent_solver
from mcq.nodes.validate import validate_answer, validate_mcq_structure
from mcq.schemas import MCQRequest
from mcq.state import MCQState
from models.gateway import ModelGateway
from shared.logging import write_json


def finalize_mcq(state: MCQState) -> MCQState:
    assert state.mcq is not None
    answer_agreement = state.solver.get("answer_id") == state.mcq.correct_answer
    difficulty_match = state.difficulty_judgement.get("predicted_difficulty") == state.request.difficulty
    state.final.update(
        {
            "skill": state.request.skill,
            "topic": state.request.topic,
            "requested_difficulty": state.request.difficulty,
            "predicted_difficulty": state.difficulty_judgement.get("predicted_difficulty"),
            "generator_answer": state.mcq.correct_answer,
            "independent_solver_answer": state.solver.get("answer_id"),
            "answer_agreement": answer_agreement,
            "distractor_score": state.distractors.get("score"),
            "reasoning_depth": state.difficulty_judgement.get("reasoning_steps"),
            "mcq": state.mcq.to_dict(),
            "difficulty_blueprint": state.blueprint.__dict__ if state.blueprint else {},
            "difficulty_validation": state.difficulty_judgement,
            "distractor_validation": state.distractors,
            "repair_count": state.metrics.repair_count,
            "metrics": state.metrics.to_dict(),
            "final_status": "passed" if answer_agreement and difficulty_match else "needs_repair",
        }
    )
    write_json(state.output_dir / "mcq_result.json", state.final)
    return state


def run_mcq_graph(request: MCQRequest, *, output_dir: Path | None = None) -> MCQState:
    target = output_dir or OUTPUT_DIR / "mcq" / request.difficulty
    state = MCQState(request=request, output_dir=target)
    gateway = ModelGateway(provider=MODEL_PROVIDER, metrics=state.metrics)
    state = analyze_mcq_request(state)
    state = create_difficulty_blueprint(state)
    state = generate_mcq(state, gateway)
    ok, issues = validate_mcq_structure(state)
    if not ok:
        state = repair_mcq(state, "; ".join(issues))
    state = independent_solver(state, gateway)
    if not validate_answer(state):
        state = repair_mcq(state, "independent solver disagreed with generator answer")
    state = validate_distractors(state)
    state = validate_difficulty(state, gateway)
    if state.difficulty_judgement.get("predicted_difficulty") != request.difficulty:
        state = repair_mcq(state, "difficulty judge disagreed with requested difficulty")
    state = finalize_mcq(state)
    return state

