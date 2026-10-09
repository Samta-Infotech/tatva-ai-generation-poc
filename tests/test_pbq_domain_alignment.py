import pytest

from pbq.graph import _bundle_from_slm_raw
from pbq.schemas import PBQBundle, PBQDesign, PBQRequest
from shared.schemas import FileEntry


def _fallback_bundle() -> PBQBundle:
    return PBQBundle(
        requirements={},
        design=PBQDesign(
            title="Fallback",
            behavioral_contract=["Implement the requested behavior."],
            scaffolding_strategy="Keep the workspace small.",
            candidate_freedom=["May edit source files."],
            difficulty_notes=["medium"],
        ),
    )


def test_non_commerce_prompt_rejects_cart_drift():
    raw = {
        "requirements": {},
        "design": {
            "title": "FastAPI cart pricing PBQ",
            "behavioral_contract": ["Price cart lines.", "Apply discount.", "Return tax."],
            "scaffolding_strategy": "Generate cart files.",
            "candidate_freedom": ["May edit app files."],
            "difficulty_notes": ["medium"],
        },
        "starter_files": [{"path": "app/main.py", "kind": "file", "content": "def price_cart():\n    return {}\n"}],
        "reference_solution": [{"path": "app/main.py", "kind": "file", "content": "def price_cart():\n    return {'subtotal': 0}\n"}],
        "public_tests": [{"path": "tests/test_public.py", "kind": "file", "content": "def test_public():\n    assert True\n"}],
        "private_tests": [{"path": "tests/test_private.py", "kind": "file", "content": "def test_private():\n    assert True\n"}],
        "mutations": [
            {
                "name": "broken",
                "changes": [{"path": "app/main.py", "kind": "file", "content": "def price_cart():\n    return None\n"}],
            }
        ],
    }

    with pytest.raises(ValueError, match="old cart/ecommerce example"):
        _bundle_from_slm_raw(
            raw,
            _fallback_bundle(),
            PBQRequest(
                runtime_profile="fastapi-py312",
                difficulty="medium",
                experience="2 years",
                duration=45,
                prompt="Build an authentication system with registration and login",
            ),
        )


def test_commerce_prompt_allows_cart_terms():
    raw = {
        "requirements": {},
        "design": {
            "title": "FastAPI ecommerce cart PBQ",
            "behavioral_contract": ["Add products.", "Price cart.", "Create orders."],
            "scaffolding_strategy": "Generate ecommerce files.",
            "candidate_freedom": ["May edit app files."],
            "difficulty_notes": ["medium"],
        },
        "starter_files": [FileEntry("app/main.py", "def price_cart():\n    return {}\n").to_dict()],
        "reference_solution": [FileEntry("app/main.py", "def price_cart():\n    return {'subtotal': 0}\n").to_dict()],
        "public_tests": [FileEntry("tests/test_public.py", "def test_public():\n    assert True\n").to_dict()],
        "private_tests": [FileEntry("tests/test_private.py", "def test_private():\n    assert True\n").to_dict()],
        "mutations": [
            {
                "name": "broken",
                "changes": [FileEntry("app/main.py", "def price_cart():\n    return None\n").to_dict()],
            }
        ],
    }

    bundle = _bundle_from_slm_raw(
        raw,
        _fallback_bundle(),
        PBQRequest(
            runtime_profile="fastapi-py312",
            difficulty="medium",
            experience="2 years",
            duration=45,
            prompt="Create ecommerce functionality with product, cart, and order workflow",
        ),
    )

    assert bundle.design.title == "FastAPI ecommerce cart PBQ"
