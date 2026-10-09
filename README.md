# TATVA AI Generation POC

Standalone POC for AI-driven generation of Project-Based Questions (PBQ) and Multiple-Choice Questions (MCQ). Uses a staged LangGraph pipeline to plan, generate, validate, and repair coding assessments across multiple frameworks.

## Prerequisites

- **Python 3.12+**
- **pip** (comes with Python)
- For full PBQ execution/validation (optional):
  - **Node.js 20+** and **npm** (for Express/React profiles)
  - **Java 21+** and **Maven** (for Spring Boot profile)

## Quick Start

### 1. Clone and set up the virtual environment

```bash
git clone <repo-url>
cd tatva-ai-generation-poc
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

The default `MODEL_PROVIDER=fixture` runs in deterministic demo mode with no external API calls — good for verifying the setup works.

To use an actual SLM/LLM, edit `.env`:

**Ollama (self-hosted SLM):**
```bash
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=https://your-ollama-host
OLLAMA_MODEL=qwen2.5:7b
# If behind Cloudflare Access:
OLLAMA_CF_ACCESS_CLIENT_ID=your-client-id
OLLAMA_CF_ACCESS_CLIENT_SECRET=your-client-secret
```

**OpenAI:**
```bash
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
```

See `.env.example` for the full list of configurable values (timeouts, token limits, context window, retries).

### 3. Start the server

```bash
python3 api.py
```

Open the UI at **http://127.0.0.1:8765** and the API docs at **http://127.0.0.1:8765/docs**.

### 4. Verify SLM connection (if using Ollama/OpenAI)

```bash
curl http://127.0.0.1:8765/api/slm/status
curl -X POST http://127.0.0.1:8765/api/slm/test
```

The UI also has a **Test SLM** button on the main page.

## CLI Usage

Generate a PBQ from the command line:

```bash
python3 main.py pbq \
  --runtime django-sqlite-py312 \
  --difficulty hard \
  --experience "4-6 years" \
  --duration 90 \
  --prompt "Build an e-commerce cart management system"
```

Generate an MCQ:

```bash
python3 main.py mcq \
  --skill Python \
  --topic concurrency \
  --difficulty hard \
  --experience "4-6 years"
```

Run benchmarks:

```bash
python3 main.py benchmark-pbq
python3 main.py benchmark-mcq
```

Outputs are written to `outputs/`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/profiles` | List available runtime profiles |
| GET | `/api/results` | List saved generation results |
| GET | `/api/slm/status` | Check SLM provider connectivity |
| POST | `/api/slm/test` | Run a test generation against the SLM |
| POST | `/api/pbq` | Generate a PBQ |
| POST | `/api/mcq` | Generate an MCQ |
| GET | `/api/task/{task_id}` | Poll task progress (used by the UI) |
| POST | `/api/validate-pbq` | Validate an existing PBQ bundle |
| POST | `/api/regenerate-pbq-from-validation` | Regenerate a PBQ from validation feedback |
| POST | `/api/validate-mcq` | Validate an existing MCQ |
| POST | `/api/benchmark-pbq` | Run the PBQ benchmark suite |
| POST | `/api/benchmark-mcq` | Run the MCQ benchmark suite |
| GET | `/api/artifact?path=...` | Retrieve a saved artifact file |

## Supported Runtime Profiles

| Profile | Language | Framework | Test Runner |
|---------|----------|-----------|-------------|
| `django-sqlite-py312` | Python 3.12 | Django + DRF | pytest |
| `fastapi-py312` | Python 3.12 | FastAPI | pytest |
| `flask-py312` | Python 3.12 | Flask | pytest |
| `express-node20` | Node.js 20 | Express | Jest |
| `react-node20` | Node.js 20 | React + Vite | Vitest |
| `springboot-java21` | Java 21 | Spring Boot | JUnit / Maven |

## Project Structure

```
tatva-ai-generation-poc/
├── api.py                  # FastAPI server and task management
├── ui.py                   # Inline HTML/JS frontend
├── main.py                 # CLI entry point
├── config/
│   ├── settings.py         # Environment-based configuration
│   ├── pbq_templates.py    # PBQ template loader
│   └── runtime_profiles.py # Runtime profile definitions
├── pbq/
│   ├── graph.py            # Main PBQ pipeline orchestration
│   ├── staged_graph.py     # LangGraph staged generation pipeline
│   ├── graph_helpers.py    # Prompt extraction, framework rules
│   ├── schemas.py          # PBQ data models
│   └── nodes/              # Pipeline step implementations
├── mcq/                    # MCQ generation pipeline
├── models/
│   └── gateway.py          # Model gateway (Ollama, OpenAI, fixture)
├── shared/
│   ├── progress.py         # Task progress tracking
│   └── schemas.py          # Shared data models
├── runtime/
│   └── workspace.py        # File materialization and validation
├── tests/                  # Test suite
├── docs/                   # Architecture diagrams
├── examples/               # Example configurations
└── outputs/                # Generated artifacts (git-ignored)
```

## Model Provider Modes

- **`fixture`** — Deterministic, no network calls. Good for development, testing, and auditing the pipeline.
- **`ollama`** — Calls a self-hosted Ollama instance. Supports Cloudflare Access authentication and streaming responses.
- **`openai`** / **`openai-compatible`** — Calls any OpenAI-compatible `/chat/completions` endpoint.

Per-role model overrides (`GENERATOR_MODEL`, `JUDGE_MODEL`, `SOLVER_MODEL`, `REPAIR_MODEL`) let you use different models for different pipeline stages. Leave them as `fixture` to use the provider's default model.

## Honesty Rule

This POC does not fake successful execution. If pytest, npm, Maven, framework packages, or local dependencies are unavailable, the corresponding command result records the actual failure or skip reason.
