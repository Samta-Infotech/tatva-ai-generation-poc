"""CLI for the TATVA AI generation POC."""

from __future__ import annotations

import argparse
from pathlib import Path

from config.runtime_profiles import all_runtime_profiles
from config.settings import OUTPUT_DIR
from mcq.graph import run_mcq_graph
from mcq.schemas import MCQRequest
from pbq.graph import run_pbq_graph
from pbq.schemas import PBQRequest
from shared.logging import write_json


PBQ_BENCHMARK_PROMPTS = {
    "django-sqlite-py312": "Build a Django order pricing service with validation, discounting, and tax calculation.",
    "fastapi-py312": "Build a FastAPI cart pricing endpoint with validation, discounting, and deterministic totals.",
    "flask-py312": "Build a Flask cart pricing API with validation, discounting, and deterministic totals.",
    "express-node20": "Build an Express cart pricing module and endpoint with validation and discount behavior.",
    "react-node20": "Build a React cart summary component with reusable pricing logic and validation.",
    "springboot-java21": "Build a Spring Boot order pricing service with validation and discount behavior.",
}


def _pbq(args: argparse.Namespace) -> None:
    state = run_pbq_graph(
        PBQRequest(
            runtime_profile=args.runtime,
            difficulty=args.difficulty,
            experience=args.experience,
            duration=args.duration,
            prompt=args.prompt,
        )
    )
    print(state.output_dir / "pbq_result.json")


def _mcq(args: argparse.Namespace) -> None:
    state = run_mcq_graph(
        MCQRequest(
            skill=args.skill,
            topic=args.topic,
            difficulty=args.difficulty,
            experience=args.experience,
        )
    )
    print(state.output_dir / "mcq_result.json")


def _benchmark_pbq(_: argparse.Namespace) -> None:
    results = []
    for profile in all_runtime_profiles():
        request = PBQRequest(
            runtime_profile=profile.id,
            difficulty="hard",
            experience="4-6 years",
            duration=90,
            prompt=PBQ_BENCHMARK_PROMPTS[profile.id],
        )
        state = run_pbq_graph(request, output_dir=OUTPUT_DIR / "benchmarks" / "pbq" / profile.id)
        results.append(state.final)
    summary = {
        "profiles": [item["runtime_profile"]["id"] for item in results],
        "passed": sum(1 for item in results if item["final_status"] == "passed"),
        "results": results,
    }
    write_json(OUTPUT_DIR / "benchmarks" / "pbq_summary.json", summary)
    print(OUTPUT_DIR / "benchmarks" / "pbq_summary.json")


def _benchmark_mcq(_: argparse.Namespace) -> None:
    results = []
    for difficulty in ("easy", "medium", "hard"):
        state = run_mcq_graph(
            MCQRequest(
                skill="Python",
                topic="concurrency",
                difficulty=difficulty,
                experience={"easy": "0-2 years", "medium": "2-4 years", "hard": "4-6 years"}[difficulty],
            ),
            output_dir=OUTPUT_DIR / "benchmarks" / "mcq" / difficulty,
        )
        results.append(state.final)
    summary = {
        "difficulties": [item["requested_difficulty"] for item in results],
        "passed": sum(1 for item in results if item["final_status"] == "passed"),
        "results": results,
    }
    write_json(OUTPUT_DIR / "benchmarks" / "mcq_summary.json", summary)
    print(OUTPUT_DIR / "benchmarks" / "mcq_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TATVA LangGraph-style AI question generation POC")
    sub = parser.add_subparsers(required=True)

    pbq = sub.add_parser("pbq")
    pbq.add_argument("--runtime", required=True)
    pbq.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    pbq.add_argument("--experience", default="4-6 years")
    pbq.add_argument("--duration", type=int, default=90)
    pbq.add_argument("--prompt", required=True)
    pbq.set_defaults(func=_pbq)

    mcq = sub.add_parser("mcq")
    mcq.add_argument("--skill", required=True)
    mcq.add_argument("--topic", required=True)
    mcq.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    mcq.add_argument("--experience", default="4-6 years")
    mcq.set_defaults(func=_mcq)

    bench_pbq = sub.add_parser("benchmark-pbq")
    bench_pbq.set_defaults(func=_benchmark_pbq)

    bench_mcq = sub.add_parser("benchmark-mcq")
    bench_mcq.set_defaults(func=_benchmark_mcq)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

