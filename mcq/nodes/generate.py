from __future__ import annotations

from mcq.schemas import MCQ, MCQOption
from mcq.state import MCQState
from models.gateway import ModelGateway


def _fixture_mcq(state: MCQState) -> MCQ:
    assert state.blueprint is not None
    difficulty = state.request.difficulty.lower()
    if difficulty == "easy":
        question = f"In {state.request.skill}, which statement best describes a race condition?"
        options = [
            MCQOption("A", "A result that depends on the timing of concurrent operations.", True),
            MCQOption("B", "A syntax error raised before a program starts.", False),
            MCQOption("C", "A loop that always terminates after one iteration.", False),
            MCQOption("D", "A database index used to speed up reads.", False),
        ]
        concepts = ["concurrency"]
        explanation = "The correct option identifies nondeterministic behavior caused by interleaving operations."
    elif difficulty == "medium":
        question = (
            "A Python service updates an in-memory counter from multiple threads with `counter += 1`. "
            "Why can the final value be lower than the number of requests?"
        )
        options = [
            MCQOption("A", "`counter += 1` is a read-modify-write sequence that can interleave without a lock.", True),
            MCQOption("B", "Python integers are always stored in a database transaction.", False),
            MCQOption("C", "The GIL makes all compound operations logically atomic.", False),
            MCQOption("D", "Thread scheduling only changes performance, never correctness.", False),
        ]
        concepts = ["thread safety", "atomicity"]
        explanation = "The operation is compound: load, add, and store can interleave across threads."
    else:
        question = (
            "A Python API uses a ThreadPoolExecutor. Each task reads a shared dict value, calls an I/O API, "
            "then writes an updated value back. A lock is added only around the final write. Which issue remains?"
        )
        options = [
            MCQOption("A", "Lost updates remain possible because the read and write are not protected as one critical section.", True),
            MCQOption("B", "The lock makes the external I/O call execute twice.", False),
            MCQOption("C", "The dict becomes immutable after the first locked write.", False),
            MCQOption("D", "ThreadPoolExecutor serializes all tasks once any lock exists.", False),
        ]
        concepts = ["critical sections", "lost updates", "thread pools"]
        explanation = "Protecting only the write still allows multiple tasks to compute from the same stale value."
    return MCQ(
        question=question,
        options=options,
        correct_answer="A",
        explanation=explanation,
        skills=[state.request.skill],
        concepts=concepts,
        difficulty=difficulty,
        metadata={"topic": state.request.topic, "schema": "tatva-like mcq payload"},
    )


def _mcq_from_slm(raw: dict, state: MCQState, fallback: MCQ) -> MCQ:
    if raw.get("slm_error") or raw.get("slm_fallback"):
        fallback.metadata["slm_generation"] = raw
        return fallback
    question = str(raw.get("question") or raw.get("question_text") or "").strip()
    raw_options = raw.get("options")
    correct_answer = str(raw.get("correct_answer") or raw.get("answer_id") or "").strip()
    if not question or not isinstance(raw_options, list) or len(raw_options) < 2:
        fallback.metadata["slm_generation"] = {**raw, "slm_invalid_schema": True}
        return fallback
    options: list[MCQOption] = []
    for index, item in enumerate(raw_options):
        if isinstance(item, dict):
            option_id = str(item.get("id") or chr(ord("A") + index)).strip()
            text = str(item.get("option_text") or item.get("text") or item.get("label") or "").strip()
            is_correct = bool(item.get("is_correct") or option_id == correct_answer)
        else:
            option_id = chr(ord("A") + index)
            text = str(item).strip()
            is_correct = option_id == correct_answer
        if not text:
            fallback.metadata["slm_generation"] = {**raw, "slm_invalid_schema": True}
            return fallback
        options.append(MCQOption(option_id, text, is_correct))
    if correct_answer and all(option.id != correct_answer for option in options):
        fallback.metadata["slm_generation"] = {**raw, "slm_invalid_answer": correct_answer}
        return fallback
    if sum(1 for option in options if option.is_correct) != 1:
        fallback.metadata["slm_generation"] = {**raw, "slm_invalid_correct_count": True}
        return fallback
    correct = next(option.id for option in options if option.is_correct)
    return MCQ(
        question=question,
        options=options,
        correct_answer=correct,
        explanation=str(raw.get("explanation") or fallback.explanation),
        skills=[str(item) for item in raw.get("skills") or [state.request.skill]],
        concepts=[str(item) for item in raw.get("concepts") or fallback.concepts],
        difficulty=str(raw.get("difficulty") or state.request.difficulty).lower(),
        metadata={
            "topic": state.request.topic,
            "schema": "tatva-like mcq payload",
            "generated_by": "slm",
            "slm_model": raw.get("model"),
        },
    )


def generate_mcq(state: MCQState, gateway: ModelGateway) -> MCQState:
    assert state.blueprint is not None
    fallback = _fixture_mcq(state)
    raw = gateway.structured_generate(
        (
            "Generate one TATVA-style MCQ. Return JSON with question, options, "
            "correct_answer, explanation, skills, concepts, and difficulty. "
            "Options must include id, option_text, and is_correct."
        ),
        schema_name="MCQ",
        context={
            "skill": state.request.skill,
            "topic": state.request.topic,
            "difficulty_blueprint": state.blueprint.__dict__,
        },
    )
    state.mcq = _mcq_from_slm(raw, state, fallback) if gateway.provider == "ollama" else fallback
    return state
