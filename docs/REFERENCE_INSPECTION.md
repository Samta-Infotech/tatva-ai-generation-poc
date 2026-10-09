# Reference Inspection Notes

Existing repositories were inspected read-only.

## PBQ Runtime Profiles

`candidate-screening-backend-django/core/pbq/manifest.py` defines more built-in profiles, including Python ML, Rust, and Angular. The concrete runtime image folders present in `candidate-screening-backend/pbq/images/runtime/` cover the six runtime profiles used by this POC:

- `django-sqlite-py312`
- `fastapi-py312`
- `flask-py312`
- `express-node20`
- `react-node20` with image folder `react-vite-node20`
- `springboot-java21`

The POC normalizes these in `config/runtime_profiles.py`.

## PBQ Contracts

The authoring serializer exposes `requirements`, `examples`, `starter_files`, `public_tests`, `private_tests`, `reference_solution`, `test_metadata`, `requested_dependencies`, `project_manifest`, and `runtime_build_id`. Candidate-facing serializers intentionally omit private tests and reference solutions.

The local isolated runner accepts snapshots rather than filesystem paths. It materializes starter files, overlays reference solution files, executes the profile-owned commands, and captures command, exit code, stdout, stderr, duration, timeout, and test summary data.

## Current Generation Flow

`questions/services/pbq_ai_generator.py` uses a family registry and a one-shot provider call with repairs and fallbacks. It already contains important guardrails, including avoiding a single editable solution file for medium/hard LLD-style PBQs and upgrading difficulty when experience implies higher seniority.

## MCQ Schema Shape

MCQ payloads use `question_text`/`question`, `options`, and one correct option represented by `is_correct` in the existing question validation tests. The POC stores an explicit `correct_answer` option id while preserving per-option `is_correct` for compatibility with TATVA-like payloads.

