# TATVA AI Generation POC Flow

## Short Message Format

We have built a FastAPI-based POC UI for AI question generation and validation. The POC can generate PBQ and MCQ questions, use an SLM through Ollama when configured, create candidate and reference workspaces, run validation commands/testcases, perform mutation checks for PBQ quality, repair failed PBQ bundles with validation feedback, and show all outputs in the browser.

The PBQ flow follows a LangGraph-style pipeline: load runtime profile, analyze requirements, generate/design the question, create starter files, create reference solution, run reference validation, run private tests against mutations, repair with SLM feedback when needed, judge difficulty, and finalize the result. The current implementation is plain Python node orchestration that mirrors a LangGraph workflow, so it is simple to run locally while keeping the same graph/node structure.

The UI now shows the generated question, AI/reference solution, candidate folder tree, reference folder tree, public tests, private tests, and validation output. A review packet is also written to `outputs/pbq/<runtime>/review_packet/` so the generated question and solution can be inspected outside the UI.

## How To Run

From the POC folder:

```bash
cd tatva-ai-generation-poc
source .venv/bin/activate
python api.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The backend is FastAPI and runs through Uvicorn.

## SLM Configuration

The POC supports two model providers:

- `fixture`: deterministic local fallback for fast demos and offline testing.
- `ollama`: calls the configured SLM through the Ollama API.
- `openai` / `openai-compatible`: calls a `/chat/completions` API with a Bearer API key.

Set these values in `.env`, not in `.env.example`:

```bash
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=<ollama-url>
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TIMEOUT_SECONDS=120
OLLAMA_JSON_RETRIES=1
OLLAMA_NUM_CTX=4096
OLLAMA_PBQ_NUM_PREDICT=4200
```

The role model variables are optional routing knobs:

```bash
GENERATOR_MODEL=fixture
JUDGE_MODEL=fixture
SOLVER_MODEL=fixture
REPAIR_MODEL=fixture
```

When they are left as `fixture`, the gateway uses the provider's default configured model. For Ollama this means `OLLAMA_MODEL`; for OpenAI-compatible providers this means `OPENAI_MODEL`.

For OpenAI-compatible benchmarking:

```bash
MODEL_PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=<your-api-key>
OPENAI_MODEL=gpt-4o-mini
```

If Cloudflare Access is required, also configure the client id and client secret in `.env`.

The model gateway is implemented in `models/gateway.py`. It sends strict JSON prompts to Ollama using `/api/chat` with `format: "json"`. If the SLM is unavailable or returns invalid JSON, the POC records the error and falls back to deterministic fixture data so the rest of the pipeline can still be tested.

## PBQ Generation Flow

The PBQ graph starts from a `PBQRequest` containing:

- runtime profile, for example `django-sqlite-py312` or `fastapi-py312`;
- requested difficulty;
- target experience;
- duration;
- prompt.

The graph flow is:

1. Load runtime profile
   - Reads framework/runtime details such as language, commands, source root, allowed packages, and validation commands.

2. Analyze requirements
   - Converts the user prompt into requirement metadata.

3. Design PBQ / generate executable bundle
   - For `react-node20`, `django-sqlite-py312`, and `fastapi-py312`, if `MODEL_PROVIDER=ollama`, the POC first asks the SLM to create a feature plan from the user prompt, difficulty, experience, duration, and runtime.
   - The feature plan decides the domain entities, required features, primary workflow, validation rules, public/private test focus, and mutation ideas before code is generated.
   - The second SLM call generates the question design, starter files, reference solution, public tests, private tests, and mutation cases from that feature plan.
   - The prompt/title is the source of truth. For example, an authentication prompt should generate authentication behavior, not cart/pricing behavior.
   - The POC validates the returned JSON, file paths, required sections, and mutation structure before writing anything to disk.
   - If a generated Python bundle forgets `requirements.txt`, or a React bundle forgets `package.json`/`index.html`, the POC adds safe runtime support files from the selected profile.
   - If the SLM drifts back to the old cart/pricing example for a non-commerce prompt, the bundle is rejected and the repair/regeneration flow is used instead of showing the wrong question as valid.
   - If the SLM output remains invalid after repair, the POC records an explicit AI-generation-failed bundle so the UI can show the exact issue and let the user regenerate with AI.
   - For other runtimes, the POC still uses profile-safe fixture templates for executable files and can use the SLM for design metadata.

4. Validate PBQ design
   - Checks that the design has enough behavioral requirements and difficulty notes.
   - If incomplete, the graph repairs the design.

5. Generate candidate workspace
   - Writes starter files only.
   - This is what a candidate would receive.

6. Materialize reference workspace
   - Writes the complete reference solution.
   - Adds public and private tests.

7. Execute reference validation
   - Runs profile-specific validation commands.
   - Example Django commands:

```bash
python manage.py check
python -m pytest -q
```

8. Repair reference if needed
   - If the reference workspace fails and the runtime is React, Django, or FastAPI, the graph sends the failed command output back to the SLM.
   - The SLM can return a corrected full PBQ bundle or partial corrected files.
   - The POC rewrites the candidate/reference workspaces and reruns validation.
   - If still failing, final status becomes `reference_failed`.

9. Mutation validation
   - Applies intentionally wrong versions of the solution.
   - Runs private tests against those wrong versions.
   - If private tests catch the wrong behavior, the mutation is killed.
   - This proves the tests are meaningful, not just passing by accident.
   - If the private tests are weak, the mutation feedback is sent back to the SLM for one repair attempt.

10. Difficulty validation
   - Uses the model gateway/judge to estimate whether the generated task matches the requested difficulty.

11. Finalize result
   - Writes JSON artifacts.
   - Writes the review packet.
   - Sets final status.

Main PBQ implementation:

```text
pbq/graph.py
pbq/nodes/
pbq/schemas.py
pbq/state.py
```

## PBQ Final Status

The PBQ status is decided like this:

- `passed`: reference solution passed and tests were strong enough.
- `reference_failed`: the generated reference workspace failed validation/test commands.
- `weak_tests`: the reference passed, but private tests did not catch mutations.
- `difficulty_mismatch`: generated task did not match the requested difficulty.

If you see `reference_failed`, check:

```text
outputs/pbq/<runtime>/review_packet/VALIDATION_OUTPUT.md
```

The UI also shows this under **Validation Output**.

## PBQ Review Packet

Every generated/validated PBQ writes a review packet:

```text
outputs/pbq/<runtime>/review_packet/
```

It contains:

- `QUESTION.md`: generated PBQ question and behavioral contract.
- `STARTER_FILES.md`: candidate starter files.
- `REFERENCE_SOLUTION.md`: generated reference solution.
- `PUBLIC_TESTS.md`: public tests.
- `PRIVATE_TESTS.md`: hidden/private validation tests.
- `CANDIDATE_TREE.txt`: candidate workspace folder tree.
- `REFERENCE_TREE.txt`: reference workspace folder tree.
- `VALIDATION_OUTPUT.md`: command stdout/stderr and mutation output.

The UI has a **Solution / Review Packet** section with:

- **Show Solution Below**
- **Open Separate Window**

This lets us review the generated AI/reference solution and the whole workspace tree directly in the browser.

## MCQ Generation Flow

The MCQ graph starts from an `MCQRequest` containing:

- skill;
- topic;
- requested difficulty;
- target experience.

The graph flow is:

1. Analyze MCQ request
   - Reads skill/topic/difficulty.

2. Create difficulty blueprint
   - Defines expected reasoning depth, concepts, and whether direct recall is allowed.

3. Generate MCQ
   - If `MODEL_PROVIDER=ollama`, the SLM generates the question/options/answer/explanation as JSON.
   - If SLM fails or returns invalid structure, the POC uses fixture fallback.

4. Validate MCQ structure
   - Ensures the question has valid options and exactly one correct answer.

5. Independent solver check
   - Asks a separate solver path to answer the question.
   - The generated answer must agree with the independent solver.

6. Distractor validation
   - Checks quality of incorrect options.

7. Difficulty validation
   - Judges whether the MCQ matches requested difficulty.

8. Finalize result
   - Writes `outputs/mcq/<difficulty>/mcq_result.json`.

Main MCQ implementation:

```text
mcq/graph.py
mcq/nodes/
mcq/schemas.py
mcq/state.py
```

## FastAPI Layer

The FastAPI app is in:

```text
api.py
```

Important endpoints:

```text
GET  /
GET  /health
GET  /api/profiles
GET  /api/results
GET  /api/slm/status
POST /api/slm/test
POST /api/pbq
POST /api/mcq
POST /api/validate-pbq
POST /api/regenerate-pbq-from-validation
POST /api/validate-mcq
GET  /api/review-packet?runtime=<runtime>
POST /api/benchmark-pbq
POST /api/benchmark-mcq
```

The UI calls these APIs and renders the output in the browser.

## What Validation Proves

PBQ validation proves:

- runtime dependencies can be installed when the profile requires it;
- the generated reference solution can run;
- framework commands pass;
- public/private tests pass;
- private tests catch intentionally broken solutions;
- the workspace tree is created correctly;
- difficulty is checked.
- failed validation feedback can be sent back to the SLM for a repair attempt.

MCQ validation proves:

- question structure is valid;
- only one answer is correct;
- independent solver agrees with the generated answer;
- distractors are reasonable;
- difficulty matches the request.

## Current Demo Path

For a clean demo:

1. Start the server.
2. Open the UI.
3. Select `django-sqlite-py312` or `fastapi-py312`.
4. Click **Run PBQ**.
5. Click **Validate PBQ**.
6. Check **Validation Output**.
7. Open **Solution / Review Packet**.
8. Show the generated question, reference solution, folder trees, and test output.
