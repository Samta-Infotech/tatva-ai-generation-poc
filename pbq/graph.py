"""Runtime-aware PBQ generation graph.

This mirrors the requested LangGraph shape with deterministic node functions.
The orchestration is intentionally plain Python so the POC remains executable
even before optional LangGraph dependencies are installed.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("tatva.poc.graph")

from config.pbq_templates import (
    get_pbq_template,
    template_generation_context,
    template_locked_paths,
    template_starter_files,
)
from config.runtime_profiles import RuntimeProfile, get_runtime_profile
from config.settings import AI_STEP_TIMEOUT_SECONDS, DESIGN_TIMEOUT_SECONDS, EXECUTE_REF_TIMEOUT_SECONDS, MODEL_PROVIDER, OUTPUT_DIR, PBQ_STAGED_GENERATION
from models.gateway import ModelGateway
from pbq.nodes.design import repair_pbq_design, validate_pbq_design
from pbq.nodes.difficulty import validate_pbq_difficulty
from pbq.nodes.execution import execute_reference
from pbq.nodes.mutation import validate_test_strength
from pbq.nodes.repair import repair_reference, repair_tests
from pbq.nodes.requirements import analyze_pbq_requirements
from pbq.nodes.workspace import generate_candidate_workspace, materialize_reference_workspace
from pbq.schemas import PBQBundle, PBQDesign, PBQRequest
from pbq.state import PBQState
from pbq.graph_helpers import (
    ai_generation_failed_bundle,
    assert_prompt_domain_alignment,
    design_from_slm,
    ensure_profile_support_files,
    extract_user_prompt,
    framework_generation_rules,
    merge_template_starter_files,
    prompt_allows_commerce_terms,
    prompt_scoped_design,
    safe_file_entries,
    safe_mutations,
    safe_requirements,
    template_context_for_request,
    unwrap_schema_payload,
)
from runtime.workspace import overlay, validate_relative_path
from shared.logging import write_json
from shared.schemas import FileEntry

# Aliases preserving the private-name convention used throughout this file.
_template_context_for_request = template_context_for_request
_extract_user_prompt = extract_user_prompt
_framework_generation_rules = framework_generation_rules
_safe_file_entries = safe_file_entries
_safe_requirements = safe_requirements
_safe_mutations = safe_mutations
_unwrap_schema_payload = unwrap_schema_payload
_design_from_slm = design_from_slm
_prompt_scoped_design = prompt_scoped_design
_merge_template_starter_files = merge_template_starter_files
_ensure_profile_support_files = ensure_profile_support_files
_ai_generation_failed_bundle = ai_generation_failed_bundle
_prompt_allows_commerce_terms = prompt_allows_commerce_terms
_assert_prompt_domain_alignment = assert_prompt_domain_alignment


SLM_GENERATED_PBQ_PROFILES = {
    "django-sqlite-py312",
    "fastapi-py312",
    "flask-py312",
    "express-node20",
    "react-node20",
    "springboot-java21",
}


def _apply_request_template(bundle: PBQBundle, request: PBQRequest) -> PBQBundle:
    if not request.template_slug:
        return bundle
    template = get_pbq_template(request.template_slug)
    bundle.starter_files = template_starter_files(request.template_slug)
    bundle.requirements["template"] = {
        "slug": template["slug"],
        "name": template["name"],
        "description": template["description"],
        "runtime_profile": template["runtime_profile"],
        "role": template["role"],
        "stack_labels": list(template.get("stack_labels") or []),
        "source_roots": list(template.get("source_roots") or []),
        "test_roots": list(template.get("test_roots") or []),
        "test_contract": template.get("test_contract") or "",
    }
    bundle.requirements["prompt"] = request.prompt
    return bundle


def _common_design(profile: RuntimeProfile, request: PBQRequest) -> PBQDesign:
    framework_names = {
        "fastapi": "FastAPI",
        "react-vite": "React + Vite",
        "spring-boot": "Spring Boot",
        "django": "Django",
        "flask": "Flask",
        "express": "Express",
    }
    framework_name = framework_names.get(profile.framework, profile.framework.title())
    return PBQDesign(
        title=f"{framework_name} order policy PBQ",
        behavioral_contract=[
            "Accept multiple order/cart lines with quantity and unit price.",
            "Reject invalid quantities and prices with a clear validation error.",
            "Apply the configured high-value discount only when the subtotal threshold is reached.",
            "Return subtotal, discount, tax, and total as externally observable values.",
        ],
        scaffolding_strategy=(
            "Easy gets more completed helpers; medium and hard leave domain modeling and extra files "
            "to the candidate while preserving locked runtime/configuration files."
        ),
        candidate_freedom=[
            f"May edit or create files under {profile.source_root}.",
            "May introduce services, repositories, components, DTOs, or tests where the runtime permits.",
            "Must not change runtime profile, package manager files beyond approved dependencies, or test command.",
        ],
        difficulty_notes=[
            f"Requested difficulty: {request.difficulty}",
            "Difficulty changes scaffolding and behavioral interaction count, not just file count.",
        ],
    )


def _python_cart_files(profile: RuntimeProfile, *, framework: str) -> PBQBundle:
    service_path = "app/services.py"
    starter_service = '''class CartError(ValueError):
    pass


def price_cart(lines, *, discount_threshold=100, discount_rate=0.10, tax_rate=0.08):
    """Return subtotal, discount, tax, and total for order lines.

    TODO: validate every line and implement the pricing policy.
    """
    return {"subtotal": 0.0, "discount": 0.0, "tax": 0.0, "total": 0.0}
'''
    reference_service = '''class CartError(ValueError):
    pass


def _money(value):
    return round(float(value) + 1e-9, 2)


def price_cart(lines, *, discount_threshold=100, discount_rate=0.10, tax_rate=0.08):
    if not isinstance(lines, list) or not lines:
        raise CartError("at least one order line is required")
    subtotal = 0.0
    for index, line in enumerate(lines):
        try:
            quantity = int(line["quantity"])
            unit_price = float(line["unit_price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CartError(f"line {index + 1} is invalid") from exc
        if quantity <= 0:
            raise CartError("quantity must be positive")
        if unit_price < 0:
            raise CartError("unit price must not be negative")
        subtotal += quantity * unit_price
    discount = subtotal * discount_rate if subtotal >= discount_threshold else 0.0
    taxable = subtotal - discount
    tax = taxable * tax_rate
    total = taxable + tax
    return {
        "subtotal": _money(subtotal),
        "discount": _money(discount),
        "tax": _money(tax),
        "total": _money(total),
    }
'''
    public_test = '''from app.services import price_cart


def test_basic_cart_without_discount():
    result = price_cart([{"quantity": 2, "unit_price": 12.5}], discount_threshold=100)
    assert result == {"subtotal": 25.0, "discount": 0.0, "tax": 2.0, "total": 27.0}
'''
    private_test = '''import pytest

from app.services import CartError, price_cart


def test_high_value_discount_and_tax_are_composed():
    result = price_cart([
        {"quantity": 2, "unit_price": 60},
        {"quantity": 1, "unit_price": 30},
    ], discount_threshold=100, discount_rate=0.1, tax_rate=0.05)
    assert result == {"subtotal": 150.0, "discount": 15.0, "tax": 6.75, "total": 141.75}


def test_rejects_invalid_quantity():
    with pytest.raises(CartError):
        price_cart([{"quantity": 0, "unit_price": 10}])
'''
    files = [
        FileEntry("app/__init__.py", ""),
        FileEntry(service_path, starter_service),
        FileEntry("requirements.txt", "\n".join(profile.allowed_packages) + "\n"),
    ]
    if framework == "django":
        files.extend(
            [
                FileEntry("manage.py", "import os\nimport sys\n\nif __name__ == '__main__':\n    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n    from django.core.management import execute_from_command_line\n    execute_from_command_line(sys.argv)\n"),
                FileEntry("config/__init__.py", ""),
                FileEntry("config/settings.py", "SECRET_KEY='poc'\nINSTALLED_APPS=[]\nROOT_URLCONF='config.urls'\nDEFAULT_AUTO_FIELD='django.db.models.BigAutoField'\n"),
                FileEntry("config/urls.py", "urlpatterns = []\n"),
                FileEntry("pytest.ini", "[pytest]\nDJANGO_SETTINGS_MODULE = config.settings\npython_files = tests.py test_*.py *_tests.py\n"),
            ]
        )
    elif framework == "fastapi":
        files.append(
            FileEntry(
                "app/main.py",
                "from fastapi import FastAPI, HTTPException\n"
                "from app.services import CartError, price_cart\n\n"
                "app = FastAPI()\n\n"
                "@app.post('/cart/price')\n"
                "def price(payload: dict):\n"
                "    try:\n"
                "        return price_cart(payload.get('lines', []))\n"
                "    except CartError as exc:\n"
                "        raise HTTPException(status_code=400, detail=str(exc)) from exc\n",
            )
        )
        public_test = '''from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_cart_price_endpoint_without_discount():
    response = client.post("/cart/price", json={"lines": [{"quantity": 2, "unit_price": 12.5}]})
    assert response.status_code == 200
    assert response.json() == {"subtotal": 25.0, "discount": 0.0, "tax": 2.0, "total": 27.0}
'''
        private_test = '''from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_cart_price_endpoint_applies_discount_and_tax():
    response = client.post("/cart/price", json={
        "lines": [
            {"quantity": 2, "unit_price": 60},
            {"quantity": 1, "unit_price": 30},
        ]
    })
    assert response.status_code == 200
    assert response.json() == {"subtotal": 150.0, "discount": 15.0, "tax": 10.8, "total": 145.8}


def test_cart_price_endpoint_rejects_invalid_quantity():
    response = client.post("/cart/price", json={"lines": [{"quantity": 0, "unit_price": 10}]})
    assert response.status_code == 400
    assert "quantity" in response.json()["detail"]
'''
    elif framework == "flask":
        files.append(FileEntry("app/main.py", "from flask import Flask, request, jsonify\nfrom app.services import price_cart\n\napp = Flask(__name__)\n\n@app.post('/cart/price')\ndef price():\n    return jsonify(price_cart((request.get_json() or {}).get('lines', [])))\n"))
    return PBQBundle(
        requirements={},
        design=_common_design(profile, PBQRequest(profile.id, "medium", "", 90, "")),
        starter_files=files,
        public_tests=[FileEntry("tests/test_public_cart.py", public_test)],
        private_tests=[FileEntry("tests/test_private_cart.py", private_test)],
        reference_solution=[FileEntry(service_path, reference_service)],
        mutations=[
            {"name": "missing_discount", "changes": [{"path": service_path, "content": reference_service.replace("discount = subtotal * discount_rate if subtotal >= discount_threshold else 0.0", "discount = 0.0")}]},
            {"name": "allow_zero_quantity", "changes": [{"path": service_path, "content": reference_service.replace("if quantity <= 0:", "if quantity < 0:")}]},
        ],
    )


def _express_files(profile: RuntimeProfile) -> PBQBundle:
    starter = """function priceCart(lines, options = {}) {
  return { subtotal: 0, discount: 0, tax: 0, total: 0 };
}

module.exports = { priceCart };
"""
    reference = """function money(value) {
  return Math.round((Number(value) + Number.EPSILON) * 100) / 100;
}

function priceCart(lines, options = {}) {
  const threshold = options.discountThreshold ?? 100;
  const discountRate = options.discountRate ?? 0.10;
  const taxRate = options.taxRate ?? 0.08;
  if (!Array.isArray(lines) || lines.length === 0) {
    throw new Error('at least one order line is required');
  }
  const subtotal = lines.reduce((sum, line, index) => {
    const quantity = Number(line.quantity);
    const unitPrice = Number(line.unitPrice);
    if (!Number.isInteger(quantity) || quantity <= 0) throw new Error(`line ${index + 1} quantity is invalid`);
    if (!Number.isFinite(unitPrice) || unitPrice < 0) throw new Error(`line ${index + 1} price is invalid`);
    return sum + quantity * unitPrice;
  }, 0);
  const discount = subtotal >= threshold ? subtotal * discountRate : 0;
  const taxable = subtotal - discount;
  const tax = taxable * taxRate;
  return { subtotal: money(subtotal), discount: money(discount), tax: money(tax), total: money(taxable + tax) };
}

module.exports = { priceCart };
"""
    return PBQBundle(
        requirements={},
        design=_common_design(profile, PBQRequest(profile.id, "medium", "", 90, "")),
        starter_files=[
            FileEntry("package.json", '{"name":"tatva-pbq-express","private":true,"scripts":{"test":"jest"},"dependencies":{"express":"5.2.1"},"devDependencies":{"jest":"30.5.1","supertest":"7.2.2"}}\n'),
            FileEntry("src/cart.js", starter),
            FileEntry("src/app.js", "const express = require('express');\nconst { priceCart } = require('./cart');\nconst app = express();\napp.use(express.json());\napp.post('/cart/price', (req, res) => res.json(priceCart(req.body.lines || [])));\nmodule.exports = app;\n"),
        ],
        public_tests=[FileEntry("tests/cart.public.test.js", "const { priceCart } = require('../src/cart');\ntest('prices a simple cart', () => { expect(priceCart([{quantity: 2, unitPrice: 12.5}], {discountThreshold: 100})).toEqual({subtotal:25, discount:0, tax:2, total:27}); });\n")],
        private_tests=[FileEntry("tests/cart.private.test.js", "const { priceCart } = require('../src/cart');\ntest('discount and validation behavior', () => { expect(priceCart([{quantity:2, unitPrice:60},{quantity:1, unitPrice:30}], {discountThreshold:100, discountRate:0.1, taxRate:0.05})).toEqual({subtotal:150, discount:15, tax:6.75, total:141.75}); expect(() => priceCart([{quantity:0, unitPrice:1}])).toThrow(/quantity/); });\n")],
        reference_solution=[FileEntry("src/cart.js", reference)],
        mutations=[{"name": "missing_discount", "changes": [{"path": "src/cart.js", "content": reference.replace("const discount = subtotal >= threshold ? subtotal * discountRate : 0;", "const discount = 0;")}]}],
    )


def _react_files(profile: RuntimeProfile) -> PBQBundle:
    starter = """export function summarizeCart(lines) {
  return { subtotal: 0, discount: 0, total: 0 };
}

export default function CartSummary({ lines = [] }) {
  const summary = summarizeCart(lines);
  return <output aria-label="cart-total">{summary.total}</output>;
}
"""
    reference = """export function summarizeCart(lines, { discountThreshold = 100, discountRate = 0.10 } = {}) {
  if (!Array.isArray(lines) || lines.length === 0) throw new Error('at least one line is required');
  const subtotal = lines.reduce((sum, line) => {
    if (!Number.isInteger(line.quantity) || line.quantity <= 0) throw new Error('quantity is invalid');
    if (!Number.isFinite(line.unitPrice) || line.unitPrice < 0) throw new Error('unit price is invalid');
    return sum + line.quantity * line.unitPrice;
  }, 0);
  const discount = subtotal >= discountThreshold ? subtotal * discountRate : 0;
  return { subtotal, discount, total: Math.round((subtotal - discount) * 100) / 100 };
}

export default function CartSummary({ lines = [] }) {
  const summary = summarizeCart(lines);
  return <output aria-label="cart-total">{summary.total}</output>;
}
"""
    return PBQBundle(
        requirements={},
        design=_common_design(profile, PBQRequest(profile.id, "medium", "", 90, "")),
        starter_files=[
            FileEntry("package.json", '{"name":"tatva-pbq-react","private":true,"type":"module","scripts":{"test":"vitest run","build":"vite build"},"dependencies":{"react":"19.0.0","react-dom":"19.0.0"},"devDependencies":{"@testing-library/react":"16.1.0","@vitejs/plugin-react":"4.3.4","jsdom":"24.1.3","vite":"6.1.0","vitest":"2.1.8"}}\n'),
            FileEntry("index.html", "<div id=\"root\"></div><script type=\"module\" src=\"/src/CartSummary.jsx\"></script>\n"),
            FileEntry("src/CartSummary.jsx", starter),
        ],
        public_tests=[FileEntry("src/CartSummary.public.test.jsx", "import { expect, test } from 'vitest';\nimport { summarizeCart } from './CartSummary.jsx';\ntest('summarizes without discount', () => { expect(summarizeCart([{quantity:2, unitPrice:12.5}], {discountThreshold:100})).toEqual({subtotal:25, discount:0, total:25}); });\n")],
        private_tests=[FileEntry("src/CartSummary.private.test.jsx", "import { expect, test } from 'vitest';\nimport { summarizeCart } from './CartSummary.jsx';\ntest('discounts and rejects invalid quantities', () => { expect(summarizeCart([{quantity:2, unitPrice:60}], {discountThreshold:100, discountRate:0.1})).toEqual({subtotal:120, discount:12, total:108}); expect(() => summarizeCart([{quantity:0, unitPrice:1}])).toThrow(/quantity/); });\n")],
        reference_solution=[FileEntry("src/CartSummary.jsx", reference)],
        mutations=[{"name": "missing_discount", "changes": [{"path": "src/CartSummary.jsx", "content": reference.replace("const discount = subtotal >= discountThreshold ? subtotal * discountRate : 0;", "const discount = 0;")}]}],
    )


def _spring_files(profile: RuntimeProfile) -> PBQBundle:
    service = "src/main/java/com/tatva/pbq/CartPricingService.java"
    starter = """package com.tatva.pbq;

import java.util.List;

public class CartPricingService {
    public record Line(int quantity, double unitPrice) {}
    public record Summary(double subtotal, double discount, double tax, double total) {}

    public Summary price(List<Line> lines, double threshold, double discountRate, double taxRate) {
        return new Summary(0, 0, 0, 0);
    }
}
"""
    reference = """package com.tatva.pbq;

import java.util.List;

public class CartPricingService {
    public record Line(int quantity, double unitPrice) {}
    public record Summary(double subtotal, double discount, double tax, double total) {}

    private double money(double value) {
        return Math.round(value * 100.0) / 100.0;
    }

    public Summary price(List<Line> lines, double threshold, double discountRate, double taxRate) {
        if (lines == null || lines.isEmpty()) throw new IllegalArgumentException("at least one line is required");
        double subtotal = 0;
        for (Line line : lines) {
            if (line.quantity() <= 0) throw new IllegalArgumentException("quantity must be positive");
            if (line.unitPrice() < 0) throw new IllegalArgumentException("unit price must not be negative");
            subtotal += line.quantity() * line.unitPrice();
        }
        double discount = subtotal >= threshold ? subtotal * discountRate : 0;
        double taxable = subtotal - discount;
        double tax = taxable * taxRate;
        return new Summary(money(subtotal), money(discount), money(tax), money(taxable + tax));
    }
}
"""
    return PBQBundle(
        requirements={},
        design=_common_design(profile, PBQRequest(profile.id, "medium", "", 90, "")),
        starter_files=[
            FileEntry("pom.xml", "<project xmlns=\"http://maven.apache.org/POM/4.0.0\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd\"><modelVersion>4.0.0</modelVersion><groupId>com.tatva</groupId><artifactId>pbq</artifactId><version>0.0.1</version><properties><maven.compiler.release>21</maven.compiler.release><junit.jupiter.version>5.11.4</junit.jupiter.version></properties><dependencies><dependency><groupId>org.junit.jupiter</groupId><artifactId>junit-jupiter</artifactId><version>${junit.jupiter.version}</version><scope>test</scope></dependency></dependencies><build><plugins><plugin><groupId>org.apache.maven.plugins</groupId><artifactId>maven-surefire-plugin</artifactId><version>3.5.2</version></plugin></plugins></build></project>\n"),
            FileEntry(service, starter),
        ],
        public_tests=[FileEntry("src/test/java/com/tatva/pbq/CartPricingPublicTest.java", "package com.tatva.pbq;\nimport org.junit.jupiter.api.Test;\nimport java.util.List;\nimport static org.junit.jupiter.api.Assertions.*;\nclass CartPricingPublicTest { @Test void simpleCart() { var s = new CartPricingService().price(List.of(new CartPricingService.Line(2, 12.5)), 100, .1, .08); assertEquals(25.0, s.subtotal()); assertEquals(27.0, s.total()); } }\n")],
        private_tests=[FileEntry("src/test/java/com/tatva/pbq/CartPricingPrivateTest.java", "package com.tatva.pbq;\nimport org.junit.jupiter.api.Test;\nimport java.util.List;\nimport static org.junit.jupiter.api.Assertions.*;\nclass CartPricingPrivateTest { @Test void discountAndValidation() { var svc = new CartPricingService(); var s = svc.price(List.of(new CartPricingService.Line(2, 60), new CartPricingService.Line(1, 30)), 100, .1, .05); assertEquals(15.0, s.discount()); assertEquals(141.75, s.total()); assertThrows(IllegalArgumentException.class, () -> svc.price(List.of(new CartPricingService.Line(0, 1)), 100, .1, .08)); } }\n")],
        reference_solution=[FileEntry(service, reference)],
        mutations=[{"name": "missing_discount", "changes": [{"path": service, "content": reference.replace("double discount = subtotal >= threshold ? subtotal * discountRate : 0;", "double discount = 0;")}]}],
    )


def _apply_slm_pbq_design(bundle: PBQBundle, state: PBQState, gateway: ModelGateway) -> PBQBundle:
    assert state.runtime_profile is not None
    raw = gateway.structured_generate(
        (
            "Generate a runtime-aware PBQ design. Return JSON with title, behavioral_contract, "
            "scaffolding_strategy, candidate_freedom, difficulty_notes, and optional domain. "
            "Do not generate executable code in this response."
        ),
        schema_name="PBQDesign",
        context={
            "prompt": state.request.prompt,
            "difficulty": state.request.difficulty,
            "experience": state.request.experience,
            "duration": state.request.duration,
            "runtime_profile": state.runtime_profile.to_dict(),
        },
    )
    bundle.requirements["slm_design_generation"] = raw
    if raw.get("slm_error") or raw.get("slm_fallback"):
        bundle.design.difficulty_notes.append("SLM design fallback used; see requirements.slm_design_generation.")
        return bundle
    title = str(raw.get("title") or "").strip()
    behavioral_contract = raw.get("behavioral_contract")
    candidate_freedom = raw.get("candidate_freedom")
    difficulty_notes = raw.get("difficulty_notes")
    if title:
        bundle.design.title = title
    if isinstance(behavioral_contract, list) and len(behavioral_contract) >= 3:
        bundle.design.behavioral_contract = [str(item) for item in behavioral_contract]
    if raw.get("scaffolding_strategy"):
        bundle.design.scaffolding_strategy = str(raw["scaffolding_strategy"])
    if isinstance(candidate_freedom, list) and candidate_freedom:
        bundle.design.candidate_freedom = [str(item) for item in candidate_freedom]
    if isinstance(difficulty_notes, list) and difficulty_notes:
        bundle.design.difficulty_notes = [str(item) for item in difficulty_notes]
    bundle.design.difficulty_notes.append("PBQ design generated by SLM; executable starter/reference/test templates remain profile-safe POC fixtures.")
    return bundle


def _bundle_from_slm_raw(raw: dict[str, Any], fallback: PBQBundle, request: PBQRequest | None = None) -> PBQBundle:
    raw = _unwrap_schema_payload(raw)
    if raw.get("slm_error") or raw.get("slm_fallback"):
        raise ValueError(str(raw.get("slm_error") or "SLM fallback returned"))
    bundle = PBQBundle(
        requirements=_safe_requirements(raw.get("requirements")),
        design=_design_from_slm(raw, fallback),
        starter_files=_safe_file_entries(raw.get("starter_files"), "starter_files"),
        public_tests=_safe_file_entries(raw.get("public_tests"), "public_tests"),
        private_tests=_safe_file_entries(raw.get("private_tests"), "private_tests"),
        reference_solution=_safe_file_entries(raw.get("reference_solution"), "reference_solution"),
        mutations=_safe_mutations(raw.get("mutations")),
    )
    if request is not None:
        _assert_prompt_domain_alignment(bundle, request)
    bundle.requirements["generation_mode"] = "slm_executable_bundle"
    return bundle


def _generate_pbq_feature_plan(state: PBQState, gateway: ModelGateway) -> dict[str, Any]:
    assert state.runtime_profile is not None
    template_context = _template_context_for_request(state.request)
    user_prompt_only = _extract_user_prompt(state.request.prompt)
    raw = gateway.structured_generate(
        (
            "Analyze the user prompt and create a PBQ feature plan before writing code. "
            "The selected template is only the framework/runtime scaffold; do not use template or POC wording as the task domain. "
            "Use user_prompt_only as the product/domain requirement. "
            "Think like an assessment designer. Choose the smallest set of realistic features that match "
            "the requested difficulty, experience level, duration, and runtime. Return strict JSON with: "
            "domain_summary, entities, features, primary_workflow, validation_rules, public_test_focus, "
            "private_test_focus, mutation_ideas, and candidate_scope. Do not generate code."
        ),
        schema_name="PBQFeaturePlan",
        context={
            "prompt": state.request.prompt,
            "user_prompt_only": user_prompt_only,
            "difficulty": state.request.difficulty,
            "experience": state.request.experience,
            "duration": state.request.duration,
            "runtime_profile": state.runtime_profile.to_dict(),
            "selected_template": template_context,
            "examples_only_not_fixed": [
                "For ecommerce, likely features may include products, cart, checkout/order, stock/price validation.",
                "For authentication, likely features may include registration, login, token validation, logout/error paths.",
                "For booking, likely features may include availability, reservation, cancellation, conflict validation.",
            ],
        },
    )
    if raw.get("slm_error") or raw.get("slm_fallback"):
        return {"planning_error": raw}
    return raw


def ensure_pbq_runtime_support(bundle: PBQBundle, profile: RuntimeProfile) -> PBQBundle:
    return _ensure_profile_support_files(bundle, profile)


def normalize_pbq_design_from_slm_metadata(bundle: PBQBundle) -> PBQBundle:
    raw = (bundle.requirements or {}).get("slm_bundle_generation")
    if isinstance(raw, dict) and isinstance(raw.get("design"), dict):
        bundle.design = _design_from_slm(raw, bundle)
    return bundle


def _generate_slm_pbq_bundle(
    fallback: PBQBundle,
    state: PBQState,
    gateway: ModelGateway,
    *,
    on_progress: Callable[[str, str, str], None] | None = None,
) -> PBQBundle:
    assert state.runtime_profile is not None
    if on_progress:
        on_progress("design", "update", "AI: planning features (1/2)...")
    plan = _generate_pbq_feature_plan(state, gateway)
    template_context = _template_context_for_request(state.request)
    design_fallback = PBQBundle(
        requirements={"prompt": state.request.prompt, "ai_feature_plan": plan},
        design=_prompt_scoped_design(state.runtime_profile, state.request, plan),
        starter_files=fallback.starter_files,
        public_tests=fallback.public_tests,
        private_tests=fallback.private_tests,
        reference_solution=fallback.reference_solution,
        mutations=fallback.mutations,
    )
    if on_progress:
        on_progress("design", "update", "AI: generating full PBQ bundle (2/2)...")
    raw = gateway.structured_generate(
        (
            "Generate a complete executable PBQ bundle for the requested runtime. "
            "Use the provided feature plan as the source of truth. "
            "The selected template is only the starter scaffold and runtime profile; the task domain must come from user_prompt_only. "
            "Use the selected PBQ template starter files as the project tree/source layout. "
            "Keep locked dependency/runtime files from the selected template unchanged unless tests absolutely require a safe addition. "
            "The task must be based on the user prompt/title, not on a fixed example. "
            "Do not mention cart, order lines, discounts, or tax unless the user prompt asks for those topics. "
            "Return strict JSON with keys: requirements, design, starter_files, reference_solution, "
            "public_tests, private_tests, mutations. Each file entry must have path, content, and kind='file'. "
            "public_tests and private_tests must be test files with source code content, not command objects. "
            "Django and FastAPI tests must import functions/classes directly or define any pytest fixtures they use. "
            "Do not reference undefined fixtures such as cart_service unless you include conftest.py. "
            "For FastAPI tests, import app using `from app.main import app` and initialize `client = TestClient(app)`. "
            "Never use `TestClient(...)`, `...`, TODO, placeholder comments, or `pass` as a function body in reference_solution, public_tests, or private_tests. "
            "The design.behavioral_contract must contain at least three concrete requirements from the user prompt domain. "
            "reference_solution must contain complete replacements for the starter files that need solution code. "
            "mutations must contain at least one intentionally wrong change that private_tests should fail. "
            "Use this shape: {\"design\":{\"title\":\"...\",\"behavioral_contract\":[\"...\"],"
            "\"scaffolding_strategy\":\"...\",\"candidate_freedom\":[\"...\"],\"difficulty_notes\":[\"...\"]},"
            "\"starter_files\":[{\"path\":\"app/main.py\",\"kind\":\"file\",\"content\":\"...\"}],"
            "\"reference_solution\":[{\"path\":\"app/main.py\",\"kind\":\"file\",\"content\":\"...\"}],"
            "\"public_tests\":[{\"path\":\"tests/test_public.py\",\"kind\":\"file\",\"content\":\"...\"}],"
            "\"private_tests\":[{\"path\":\"tests/test_private.py\",\"kind\":\"file\",\"content\":\"...\"}],"
            "\"mutations\":[{\"name\":\"broken_rule\",\"changes\":[{\"path\":\"app/main.py\",\"kind\":\"file\",\"content\":\"...\"}]}]}. "
            "Keep the workspace small, deterministic, and runnable with the runtime commands."
        ),
        schema_name="PBQExecutableBundle",
        context={
            "prompt": state.request.prompt,
            "user_prompt_only": _extract_user_prompt(state.request.prompt),
            "difficulty": state.request.difficulty,
            "experience": state.request.experience,
            "duration": state.request.duration,
            "runtime_profile": state.runtime_profile.to_dict(),
            "framework_rules": _framework_generation_rules(state.runtime_profile),
            "selected_template": template_context,
            "feature_plan": plan,
            "final_status_rules": {
                "reference must pass": state.runtime_profile.validation_command,
                "tests must pass": state.runtime_profile.test_command,
                "private tests must kill mutations": True,
            },
        },
    )
    try:
        bundle = _bundle_from_slm_raw(raw, design_fallback, state.request)
    except ValueError as exc:
        repair_raw = gateway.repair_generate(
            (
                "Your previous PBQ bundle response was invalid. Return one complete corrected executable PBQ bundle. "
                "Use the feature plan and user prompt as the source of truth. Do not fall back to a cart/order task "
                "unless the prompt is about ecommerce/cart/order. Return strict JSON with design, starter_files, "
                "reference_solution, public_tests, private_tests, and mutations."
            ),
            schema_name="PBQExecutableBundleRepair",
            context={
                "invalid_reason": str(exc),
                "previous_response": raw,
                "prompt": state.request.prompt,
                "user_prompt_only": _extract_user_prompt(state.request.prompt),
                "feature_plan": plan,
                "runtime_profile": state.runtime_profile.to_dict(),
                "framework_rules": _framework_generation_rules(state.runtime_profile),
                "selected_template": template_context,
            },
        )
        try:
            bundle = _bundle_from_slm_raw(repair_raw, design_fallback, state.request)
            bundle.requirements["slm_initial_invalid_bundle"] = {**raw, "slm_invalid_bundle": str(exc)}
            raw = repair_raw
        except ValueError as repair_exc:
            return _ai_generation_failed_bundle(state, repair_raw, str(repair_exc), plan)
    bundle.requirements["slm_bundle_generation"] = raw
    bundle.requirements["ai_feature_plan"] = plan
    bundle.design.difficulty_notes.append("Question, starter files, reference solution, tests, and mutations generated by SLM.")
    bundle = _merge_template_starter_files(bundle, state.request)
    return _ensure_profile_support_files(bundle, state.runtime_profile)


def design_pbq(
    state: PBQState,
    gateway: ModelGateway,
    *,
    on_progress: Callable[[str, str, str], None] | None = None,
    design_only: bool = False,
) -> PBQState:
    assert state.runtime_profile is not None
    builders: dict[str, Callable[[RuntimeProfile], PBQBundle]] = {
        "django-sqlite-py312": lambda p: _python_cart_files(p, framework="django"),
        "fastapi-py312": lambda p: _python_cart_files(p, framework="fastapi"),
        "flask-py312": lambda p: _python_cart_files(p, framework="flask"),
        "express-node20": _express_files,
        "react-node20": _react_files,
        "springboot-java21": _spring_files,
    }
    bundle = builders[state.runtime_profile.id](state.runtime_profile)
    bundle.design = _common_design(state.runtime_profile, state.request)
    bundle = _apply_request_template(bundle, state.request)
    if gateway.provider in {"ollama", "openai", "openai-compatible"} and state.runtime_profile.id in SLM_GENERATED_PBQ_PROFILES:
        if PBQ_STAGED_GENERATION:
            from pbq.staged_graph import generate_staged_pbq_bundle
            bundle = generate_staged_pbq_bundle(bundle, state, gateway, on_progress=on_progress, design_only=design_only)
        elif not design_only:
            bundle = _generate_slm_pbq_bundle(bundle, state, gateway, on_progress=on_progress)
    elif gateway.provider in {"ollama", "openai", "openai-compatible"} and not design_only:
        bundle = _apply_slm_pbq_design(bundle, state, gateway)
    state.bundle = bundle
    return state


def load_runtime_profile(state: PBQState) -> PBQState:
    state.runtime_profile = get_runtime_profile(state.request.runtime_profile)
    state.bundle = PBQBundle(requirements={}, design=_common_design(state.runtime_profile, state.request))
    return state


def _render_tree(lines: list[str]) -> str:
    root: dict[str, dict] = {}
    for raw in lines:
        is_dir = raw.endswith("/")
        parts = raw.rstrip("/").split("/")
        node = root
        for index, part in enumerate(parts):
            key = f"{part}/" if is_dir and index == len(parts) - 1 else part
            node = node.setdefault(key, {})

    def walk(node: dict[str, dict], prefix: str = "") -> list[str]:
        output: list[str] = []
        entries = sorted(node)
        for index, name in enumerate(entries):
            last = index == len(entries) - 1
            branch = "`-- " if last else "|-- "
            output.append(f"{prefix}{branch}{name}")
            output.extend(walk(node[name], prefix + ("    " if last else "|   ")))
        return output

    return "\n".join(walk(root)) or "(empty)"


def _files_markdown(title: str, files: list[FileEntry]) -> str:
    blocks = [f"# {title}", ""]
    for entry in files:
        if entry.kind == "directory":
            blocks.extend([f"## {entry.path}/", "Directory", ""])
            continue
        blocks.extend([f"## {entry.path}", "```", entry.content.rstrip(), "```", ""])
    return "\n".join(blocks)


def _execution_markdown(state: PBQState) -> str:
    blocks = ["# Validation Output", ""]
    for item in state.validation.get("reference_execution", []):
        blocks.extend(
            [
                f"## {item.get('command')}",
                f"exit_code: {item.get('exit_code')}",
                f"timed_out: {item.get('timed_out')}",
                f"skipped: {item.get('skipped')}",
                "",
                "### stdout",
                "```",
                str(item.get("stdout") or "").rstrip(),
                "```",
                "",
                "### stderr",
                "```",
                str(item.get("stderr") or item.get("skip_reason") or "").rstrip(),
                "```",
                "",
            ]
        )
    blocks.extend(["# Mutation Test Output", ""])
    for item in state.mutation.get("results", []):
        execution = item.get("execution") or {}
        blocks.extend(
            [
                f"## {item.get('name')}",
                f"killed: {item.get('killed')}",
                f"inconclusive: {item.get('inconclusive')}",
                f"command: {execution.get('command')}",
                f"exit_code: {execution.get('exit_code')}",
                "",
                "### stdout",
                "```",
                str(execution.get("stdout") or "").rstrip(),
                "```",
                "",
                "### stderr",
                "```",
                str(execution.get("stderr") or execution.get("skip_reason") or "").rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(blocks)


def _testcase_summary(results: list[dict[str, Any]], mutation: dict[str, Any]) -> dict[str, Any]:
    commands = [_command_summary(item) for item in results]
    mutation_commands = [_command_summary((item.get("execution") or {}) | {"mutation": item.get("name")}) for item in mutation.get("results", [])]
    failed = [item for item in commands if item["status"] != "passed"]
    failed.extend(item for item in mutation_commands if item["status"] in {"survived", "skipped", "timed_out", "failed"})
    return {
        "status": "passed" if not failed else "failed",
        "commands": commands,
        "mutations": mutation_commands,
        "failed_count": len(failed),
    }


def _command_summary(item: dict[str, Any]) -> dict[str, Any]:
    stdout = str(item.get("stdout") or "")
    stderr = str(item.get("stderr") or item.get("skip_reason") or "")
    exit_code = item.get("exit_code")
    status = "passed"
    if item.get("skipped"):
        status = "skipped"
    elif item.get("timed_out"):
        status = "timed_out"
    elif exit_code not in {0, None}:
        status = "failed"
    if item.get("mutation") and status == "failed":
        status = "killed"
    elif item.get("mutation") and status == "passed":
        status = "survived"
    passed_count = _first_int(r"(\d+)\s+passed", stdout)
    failed_names = sorted(set(re.findall(r"(?:FAILED|ERROR(?: at setup of)?)\s+([A-Za-z0-9_:\-./\[\]]*test[A-Za-z0-9_:\-./\[\]]*)", stdout)))
    if not failed_names:
        failed_names = sorted(set(re.findall(r"def\s+(test_[A-Za-z0-9_]+)", stdout)))
    if status == "passed":
        reason = f"{passed_count} tests passed" if passed_count else "Command completed successfully."
    elif status == "killed":
        reason = "Private tests caught this intentionally broken solution. This is expected and good."
        if failed_names:
            reason += " Failed test(s): " + ", ".join(failed_names[:10])
    elif status == "survived":
        reason = "The intentionally broken solution still passed. Private tests are too weak."
    else:
        reason = stderr.strip() or _first_failure_block(stdout) or stdout.strip()
    return {
        "command": item.get("command"),
        "mutation": item.get("mutation"),
        "status": status,
        "exit_code": exit_code,
        "passed_count": passed_count,
        "failed_tests": failed_names[:20],
        "reason": reason[:2500],
    }


def _first_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


def _first_failure_block(text: str) -> str:
    markers = ("==================================== ERRORS", "=================================== FAILURES", "FAILED ")
    for marker in markers:
        index = text.find(marker)
        if index >= 0:
            return text[index : index + 2500]
    return ""


def _repair_pbq_bundle_with_slm(state: PBQState, gateway: ModelGateway, reason: str) -> PBQState:
    assert state.bundle is not None
    assert state.runtime_profile is not None
    if gateway.provider not in {"ollama", "openai", "openai-compatible"} or state.runtime_profile.id not in SLM_GENERATED_PBQ_PROFILES:
        return state
    raw = gateway.structured_generate(
        (
            "Repair this PBQ bundle using the validation feedback. Return the full corrected PBQ bundle "
            "as strict JSON with the same keys: requirements, design, starter_files, reference_solution, "
            "public_tests, private_tests, mutations. Fix the reference solution and/or tests so the "
            "reference commands pass and private tests catch the mutations. Keep paths safe and keep the "
            "bundle small. public_tests and private_tests must be test files with path/content/kind, "
            "not command objects. Django and FastAPI tests must import functions/classes directly or define "
            "any pytest fixtures they use; do not leave undefined fixtures. For FastAPI tests, import app using "
            "`from app.main import app` and initialize `client = TestClient(app)`. Never use `TestClient(...)`, "
            "`...`, TODO, placeholder comments, or `pass` as a function body in reference_solution, public_tests, or private_tests."
        ),
        schema_name="PBQExecutableBundleRepair",
        context={
            "repair_reason": reason,
            "runtime_profile": state.runtime_profile.to_dict(),
            "framework_rules": _framework_generation_rules(state.runtime_profile),
            "previous_bundle": state.bundle.to_dict(),
            "reference_execution": state.validation.get("reference_execution", []),
            "mutation_results": state.mutation,
        },
    )
    try:
        repaired = _bundle_from_slm_raw(raw, state.bundle, state.request)
    except ValueError as exc:
        try:
            repaired = _partial_repair_bundle_from_slm_raw(raw, state.bundle, state.request)
        except ValueError:
            repaired = None
        if repaired is None:
            state.metrics.repair_count += 1
            state.validation.setdefault("repairs", []).append(f"SLM repair failed schema validation: {exc}")
            state.bundle.requirements["slm_repair_failure"] = {**raw, "slm_invalid_bundle": str(exc)}
            return state
    state.metrics.repair_count += 1
    repaired.requirements["slm_repair_generation"] = raw
    repaired.design.difficulty_notes.append(f"SLM repair applied after validation feedback: {reason}")
    repaired = _merge_template_starter_files(repaired, state.request)
    state.bundle = _ensure_profile_support_files(repaired, state.runtime_profile)
    state.validation.setdefault("repairs", []).append(f"SLM repair applied: {reason}")
    state = generate_candidate_workspace(state)
    state = materialize_reference_workspace(state)
    state = execute_reference(state)
    return state


def repair_pbq_from_validation_feedback(state: PBQState, gateway: ModelGateway, reason: str) -> PBQState:
    return _repair_pbq_bundle_with_slm(state, gateway, reason)


def _partial_repair_bundle_from_slm_raw(
    raw: dict[str, Any], current: PBQBundle, request: PBQRequest | None = None
) -> PBQBundle | None:
    if raw.get("slm_error") or raw.get("slm_fallback"):
        return None
    changed = False
    repaired = PBQBundle(
        requirements={**current.requirements, **_safe_requirements(raw.get("requirements"))},
        design=_design_from_slm(raw, current) if raw.get("design") else current.design,
        starter_files=list(current.starter_files),
        public_tests=list(current.public_tests),
        private_tests=list(current.private_tests),
        reference_solution=list(current.reference_solution),
        mutations=list(current.mutations),
    )
    for field_name in ("starter_files", "reference_solution", "public_tests", "private_tests"):
        if field_name not in raw:
            continue
        try:
            entries = _safe_file_entries(raw.get(field_name), field_name)
        except ValueError:
            continue
        setattr(repaired, field_name, overlay(getattr(repaired, field_name), entries))
        changed = True
    if "mutations" in raw:
        try:
            repaired.mutations = _safe_mutations(raw.get("mutations"))
            changed = True
        except ValueError:
            pass
    if not changed:
        return None
    if request is not None:
        _assert_prompt_domain_alignment(repaired, request)
    return repaired


def _write_pbq_review_packet(state: PBQState) -> Path:
    assert state.bundle is not None
    assert state.runtime_profile is not None
    packet = state.output_dir / "review_packet"
    packet.mkdir(parents=True, exist_ok=True)
    (packet / "QUESTION.md").write_text(
        "\n".join(
            [
                f"# {state.bundle.design.title}",
                "",
                f"Runtime: `{state.runtime_profile.id}`",
                f"Difficulty: `{state.request.difficulty}`",
                "",
                "## Behavioral Contract",
                *(f"- {item}" for item in state.bundle.design.behavioral_contract),
                "",
                "## Scaffolding Strategy",
                state.bundle.design.scaffolding_strategy,
                "",
                "## Candidate Freedom",
                *(f"- {item}" for item in state.bundle.design.candidate_freedom),
                "",
                "## Difficulty Notes",
                *(f"- {item}" for item in state.bundle.design.difficulty_notes),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (packet / "CANDIDATE_TREE.txt").write_text(
        _render_tree(state.validation.get("candidate_workspace_tree", [])),
        encoding="utf-8",
    )
    (packet / "REFERENCE_TREE.txt").write_text(
        _render_tree(state.validation.get("reference_workspace_tree", [])),
        encoding="utf-8",
    )
    (packet / "STARTER_FILES.md").write_text(
        _files_markdown("Starter Files", state.bundle.starter_files),
        encoding="utf-8",
    )
    (packet / "REFERENCE_SOLUTION.md").write_text(
        _files_markdown("Reference Solution", state.bundle.reference_solution),
        encoding="utf-8",
    )
    (packet / "PUBLIC_TESTS.md").write_text(
        _files_markdown("Public Tests", state.bundle.public_tests),
        encoding="utf-8",
    )
    (packet / "PRIVATE_TESTS.md").write_text(
        _files_markdown("Private Tests", state.bundle.private_tests),
        encoding="utf-8",
    )
    (packet / "VALIDATION_OUTPUT.md").write_text(_execution_markdown(state), encoding="utf-8")
    return packet


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(OUTPUT_DIR.parent.resolve()))
    except ValueError:
        return str(path)


def finalize_pbq(state: PBQState) -> PBQState:
    assert state.bundle is not None
    assert state.runtime_profile is not None
    final_status = "passed"
    if not state.validation.get("reference_passed"):
        final_status = "reference_failed"
    elif not state.validation.get("tests_strong_enough"):
        final_status = "weak_tests"
    elif not state.validation.get("difficulty_matches_request"):
        final_status = "difficulty_mismatch"
    review_packet = _write_pbq_review_packet(state)
    state.final = {
        "runtime_profile": state.runtime_profile.to_dict(),
        "question_design": {
            "title": state.bundle.design.title,
            "behavioral_contract": state.bundle.design.behavioral_contract,
            "scaffolding_strategy": state.bundle.design.scaffolding_strategy,
            "candidate_freedom": state.bundle.design.candidate_freedom,
            "difficulty_notes": state.bundle.design.difficulty_notes,
        },
        "requested_difficulty": state.request.difficulty,
        "predicted_difficulty": state.difficulty.get("predicted_difficulty"),
        "requirement_completeness": len(state.bundle.design.behavioral_contract) / 4,
        "reference_files_generated": len(state.bundle.reference_solution),
        "reference_execution_success": state.validation.get("reference_passed", False),
        "mutation_score": state.mutation.get("mutation_score"),
        "candidate_workspace_tree": state.validation.get("candidate_workspace_tree", []),
        "reference_workspace_tree": state.validation.get("reference_workspace_tree", []),
        "execution_results": state.validation.get("reference_execution", []),
        "mutation_results": state.mutation,
        "testcase_summary": _testcase_summary(
            state.validation.get("reference_execution", []),
            state.mutation,
        ),
        "difficulty_validation": state.difficulty,
        "validation_repairs": state.validation.get("repairs", []),
        "repair_count": state.metrics.repair_count,
        "metrics": state.metrics.to_dict(),
        "review_packet_path": _display_path(review_packet),
        "final_status": final_status,
    }
    write_json(state.output_dir / "pbq_bundle.json", state.bundle.to_dict())
    write_json(state.output_dir / "pbq_result.json", state.final)
    return state


def _build_design_only_result(state: PBQState) -> dict[str, Any]:
    bundle = state.bundle
    if not bundle:
        return {"final_status": "design_failed", "error": "No bundle produced"}
    design = bundle.design
    req = bundle.requirements or {}
    plan = req.get("ai_feature_plan", {})
    manifest = req.get("staged_question_spec", {}).get("file_manifest", [])
    files = []
    for section, entries in [
        ("starter_files", bundle.starter_files),
        ("reference_solution", bundle.reference_solution),
        ("public_tests", bundle.public_tests),
        ("private_tests", bundle.private_tests),
    ]:
        for f in entries:
            files.append({"path": f.path, "content": f.content, "section": section})

    return {
        "final_status": "design_only",
        "question_design": {
            "title": design.title,
            "behavioral_contract": design.behavioral_contract,
            "scaffolding_strategy": design.scaffolding_strategy,
            "candidate_freedom": design.candidate_freedom,
            "difficulty_notes": design.difficulty_notes,
        },
        "question": {
            "title": design.title,
            "behavioral_contract": design.behavioral_contract,
            "scaffolding_strategy": design.scaffolding_strategy,
            "difficulty": state.request.difficulty,
            "experience": state.request.experience,
            "runtime_profile": state.request.runtime_profile,
        },
        "feature_plan": {
            "domain_summary": plan.get("domain_summary", ""),
            "entities": plan.get("entities", []),
            "features": plan.get("features", []),
            "primary_workflow": plan.get("primary_workflow", ""),
            "validation_rules": plan.get("validation_rules", []),
        },
        "file_manifest": manifest,
        "files": files,
        "starter_files": [f.path for f in bundle.starter_files],
        "reference_files": [f.path for f in bundle.reference_solution],
        "public_tests": [f.path for f in bundle.public_tests],
        "private_tests": [f.path for f in bundle.private_tests],
    }


def run_pbq_graph(
    request: PBQRequest,
    *,
    output_dir: Path | None = None,
    on_progress: Callable[[str, str, str], None] | None = None,
    design_only: bool = False,
) -> PBQState:
    """Run the full PBQ generation pipeline.

    ``on_progress(step_key, status, message)`` is called before and after each
    major step so callers can track progress.  ``status`` is one of
    "start" | "done" | "error" | "skip".
    """

    def _emit(step: str, status: str, msg: str = "") -> None:
        if on_progress is not None:
            on_progress(step, status, msg)

    target = output_dir or OUTPUT_DIR / "pbq" / request.runtime_profile
    state = PBQState(request=request, output_dir=target)
    gateway = ModelGateway(provider=MODEL_PROVIDER, metrics=state.metrics)

    _emit("load_profile", "start", "Loading runtime profile...")
    state = load_runtime_profile(state)
    _emit("load_profile", "done", f"Loaded {state.runtime_profile.framework}")

    _emit("analyze", "start", "Analyzing requirements...")
    state = analyze_pbq_requirements(state)
    _emit("analyze", "done")

    _emit("design", "start", "Calling AI to design PBQ...")
    _design_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _design_fut = _design_pool.submit(design_pbq, state, gateway, on_progress=on_progress, design_only=design_only)
    try:
        state = _design_fut.result(timeout=DESIGN_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        logger.warning("design_pbq timed out after %ss — AI model did not respond in time", DESIGN_TIMEOUT_SECONDS)
        _design_pool.shutdown(wait=False)
        raise RuntimeError(
            f"AI design step timed out after {DESIGN_TIMEOUT_SECONDS}s. "
            "The model is responding too slowly. Please retry."
        )
    except Exception as exc:
        logger.exception("design_pbq raised unexpectedly: %s", exc)
        _design_pool.shutdown(wait=False)
        raise
    _design_pool.shutdown(wait=False)
    _emit("design", "done", f"Designed: {state.bundle.design.title if state.bundle else ''}")

    _emit("validate_design", "start", "Validating PBQ design...")
    ok, issues = validate_pbq_design(state)
    if not ok:
        state = repair_pbq_design(state, issues)
    _emit("validate_design", "done")

    if design_only:
        state.final = _build_design_only_result(state)
        _emit("finalize", "start", "Building design-only result...")
        _emit("finalize", "done", "Design-only mode — question tree ready")
        return state

    _emit("build_workspace", "start", "Building candidate and reference workspaces...")
    state = generate_candidate_workspace(state)
    state = materialize_reference_workspace(state)
    _emit("build_workspace", "done")

    _emit("execute_tests", "start", "Preparing runtime environment...")
    _exec_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _exec_fut = _exec_pool.submit(execute_reference, state, on_progress=lambda msg: _emit("execute_tests", "update", msg))
    try:
        state = _exec_fut.result(timeout=EXECUTE_REF_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        logger.warning("execute_reference timed out after %ss — marking reference as failed", EXECUTE_REF_TIMEOUT_SECONDS)
        state.validation["reference_passed"] = False
        state.validation.setdefault("reference_execution", []).append(
            {"command": "wall-clock-timeout", "exit_code": None, "timed_out": True, "skipped": False}
        )
    except Exception as exc:
        logger.exception("execute_reference raised unexpectedly: %s", exc)
        state.validation["reference_passed"] = False
    finally:
        # shutdown(wait=False) is critical — without it, the context manager blocks
        # until the thread finishes even after TimeoutError, causing the UI to hang.
        _exec_pool.shutdown(wait=False)
    ref_passed = state.validation.get("reference_passed", False)
    _emit("execute_tests", "done", f"Reference {'passed' if ref_passed else 'failed'}")

    if not ref_passed:
        _emit("ai_repair_ref", "start", "Reference failed — repairing with AI...")
        _repair_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        _repair_fut = _repair_pool.submit(_repair_pbq_bundle_with_slm, state, gateway, "reference execution failed")
        try:
            state = _repair_fut.result(timeout=AI_STEP_TIMEOUT_SECONDS)
        except (concurrent.futures.TimeoutError, Exception) as exc:
            logger.warning("ai_repair_ref timed out or failed: %s", exc)
        finally:
            _repair_pool.shutdown(wait=False)
        if not state.validation.get("reference_passed"):
            state = repair_reference(state)
        _emit("ai_repair_ref", "done")
    else:
        _emit("ai_repair_ref", "skip", "Reference passed — repair not needed")

    _emit("mutation_testing", "start", "Running mutation testing...")
    state = validate_test_strength(state)
    score = state.validation.get("mutation_score", "n/a")
    _emit("mutation_testing", "done", f"Mutation score: {score}")

    if not state.validation.get("tests_strong_enough"):
        _emit("ai_repair_tests", "start", "Tests weak — repairing with AI...")
        _test_repair_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        _test_repair_fut = _test_repair_pool.submit(
            _repair_pbq_bundle_with_slm, state, gateway, "private tests did not catch mutations",
        )
        try:
            state = _test_repair_fut.result(timeout=AI_STEP_TIMEOUT_SECONDS)
        except (concurrent.futures.TimeoutError, Exception) as exc:
            logger.warning("ai_repair_tests timed out or failed: %s", exc)
        finally:
            _test_repair_pool.shutdown(wait=False)
        if state.validation.get("reference_passed"):
            state = validate_test_strength(state)
        if not state.validation.get("tests_strong_enough"):
            state = repair_tests(state)
        _emit("ai_repair_tests", "done")
    else:
        _emit("ai_repair_tests", "skip", "Tests strong enough — repair not needed")

    _emit("validate_difficulty", "start", "Validating difficulty with AI judge...")
    _diff_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _diff_fut = _diff_pool.submit(validate_pbq_difficulty, state, gateway)
    try:
        state = _diff_fut.result(timeout=AI_STEP_TIMEOUT_SECONDS)
    except (concurrent.futures.TimeoutError, Exception) as exc:
        logger.warning("validate_difficulty timed out or failed: %s — using defaults", exc)
    finally:
        _diff_pool.shutdown(wait=False)
    _emit("validate_difficulty", "done")

    _emit("finalize", "start", "Writing artifacts and review packet...")
    state = finalize_pbq(state)
    _emit("finalize", "done", f"Status: {state.final.get('final_status', 'unknown')}")

    return state
