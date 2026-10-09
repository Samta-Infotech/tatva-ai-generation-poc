# Deliverable Index

- Standalone POC folder: `tatva-ai-generation-poc/`
- Setup and run instructions: `README.md`
- Graph diagrams: `docs/GRAPH_DIAGRAMS.md`
- Runtime configurations: `config/runtime_profiles.py`
- PBQ benchmark results: `outputs/benchmarks/pbq_summary.json`
- MCQ benchmark results: `outputs/benchmarks/mcq_summary.json`
- Candidate workspace trees: each `outputs/benchmarks/pbq/<profile>/pbq_result.json`
- Reference workspace trees: each `outputs/benchmarks/pbq/<profile>/pbq_result.json`
- Public/private tests: each `outputs/benchmarks/pbq/<profile>/pbq_bundle.json`
- Execution results: each `outputs/benchmarks/pbq/<profile>/pbq_result.json`
- Mutation results: each `outputs/benchmarks/pbq/<profile>/pbq_result.json`
- Difficulty validation: PBQ and MCQ result JSON files under `outputs/benchmarks/`
- Token/cost metrics: `metrics` fields in result JSON files
- Retry/repair metrics: `repair_count` fields in result JSON files
- Benchmark comparison: `outputs/benchmarks/pbq_summary.json` and `outputs/benchmarks/mcq_summary.json`
- Final recommendation: `docs/FINAL_RECOMMENDATION.md`
- Reference inspection notes: `docs/REFERENCE_INSPECTION.md`

