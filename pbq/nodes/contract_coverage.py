"""Contract-test coverage validation.

Verifies that:
1. Every behavioral contract item is covered by at least one private test
2. Every private test maps back to a contract item (no surprise requirements)
3. Tests don't check implementation details not described in the contract

This catches the exact bug we saw: tests asserting on dict immutability when
the contract only says "audit changes are tracked".
"""

from __future__ import annotations

import logging
from typing import Any

from models.gateway import ModelGateway
from pbq.state import PBQState

logger = logging.getLogger("tatva.poc.contract_coverage")


def validate_contract_coverage(state: PBQState, gateway: ModelGateway) -> PBQState:
    """AI-powered check that tests align with the behavioral contract."""
    assert state.bundle is not None

    if gateway.provider not in {"ollama", "openai", "openai-compatible"}:
        state.validation["contract_coverage"] = {"skipped": True, "reason": "fixture provider"}
        return state

    contract = state.bundle.design.behavioral_contract
    private_test_contents = {f.path: f.content for f in state.bundle.private_tests}
    public_test_contents = {f.path: f.content for f in state.bundle.public_tests}

    raw = gateway.structured_generate(
        (
            "You are a test quality auditor. Analyze the behavioral contract and the test files.\n\n"
            "For each behavioral contract item, determine which test functions cover it.\n"
            "For each test function, determine which contract item(s) it validates.\n\n"
            "Flag any test that asserts on behavior NOT described in the contract — these are "
            "'unfair tests' that test implementation details rather than specified behavior.\n\n"
            "Flag any contract item that has NO corresponding test — these are 'coverage gaps'.\n\n"
            "Return JSON with:\n"
            "- 'contract_mapping': array of {contract_item, covered_by_tests: [test_name], status: 'covered'|'gap'}\n"
            "- 'test_mapping': array of {test_name, tests_contract_items: [item_index], status: 'fair'|'unfair', "
            "unfair_reason: string|null}\n"
            "- 'coverage_score': float 0-1 (fraction of contract items covered)\n"
            "- 'fairness_score': float 0-1 (fraction of tests that are fair)\n"
            "- 'issues': array of string descriptions of problems found\n"
            "- 'overall_verdict': 'pass'|'has_gaps'|'has_unfair_tests'|'fail'"
        ),
        schema_name="PBQContractCoverage",
        context={
            "behavioral_contract": contract,
            "private_tests": private_test_contents,
            "public_tests": public_test_contents,
            "question_title": state.bundle.design.title,
        },
    )

    coverage = _parse_coverage_result(raw, contract)
    state.validation["contract_coverage"] = coverage
    return state


def _parse_coverage_result(raw: dict[str, Any], contract: list[str]) -> dict[str, Any]:
    """Parse and validate the AI coverage analysis."""
    if raw.get("slm_error") or raw.get("slm_fallback"):
        return {
            "skipped": False,
            "error": raw.get("slm_error", "Unknown error"),
            "verdict": "error",
        }

    coverage_score = _safe_float(raw.get("coverage_score"), 0.0)
    fairness_score = _safe_float(raw.get("fairness_score"), 0.0)
    issues = raw.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    contract_mapping = raw.get("contract_mapping", [])
    if not isinstance(contract_mapping, list):
        contract_mapping = []

    test_mapping = raw.get("test_mapping", [])
    if not isinstance(test_mapping, list):
        test_mapping = []

    gaps = [
        item for item in contract_mapping
        if isinstance(item, dict) and item.get("status") == "gap"
    ]
    unfair = [
        item for item in test_mapping
        if isinstance(item, dict) and item.get("status") == "unfair"
    ]

    verdict = str(raw.get("overall_verdict", "")).lower()
    if verdict not in {"pass", "has_gaps", "has_unfair_tests", "fail"}:
        if gaps and unfair:
            verdict = "fail"
        elif gaps:
            verdict = "has_gaps"
        elif unfair:
            verdict = "has_unfair_tests"
        else:
            verdict = "pass"

    return {
        "skipped": False,
        "coverage_score": coverage_score,
        "fairness_score": fairness_score,
        "contract_items_total": len(contract),
        "contract_items_covered": len(contract) - len(gaps),
        "coverage_gaps": [
            item.get("contract_item", "") for item in gaps if isinstance(item, dict)
        ],
        "unfair_tests": [
            {
                "test": item.get("test_name", ""),
                "reason": item.get("unfair_reason", ""),
            }
            for item in unfair if isinstance(item, dict)
        ],
        "issues": [str(i) for i in issues[:10]],
        "verdict": verdict,
    }


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
