from mcq.graph import run_mcq_graph
from mcq.schemas import MCQRequest


def test_mcq_graph_produces_agreed_answer(tmp_path):
    state = run_mcq_graph(
        MCQRequest(skill="Python", topic="concurrency", difficulty="hard", experience="4-6 years"),
        output_dir=tmp_path,
    )
    assert state.final["answer_agreement"] is True
    assert state.final["final_status"] == "passed"

