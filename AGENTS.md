# AGENTS — Project overview (ai-deals2buy)

This document is a **contributor-oriented map** of the codebase: where things live, which modules do what, how configuration works, and the common dev/test commands.

For the full end-to-end architecture and runtime flow (with diagram), see `docs/README.md`.

---

## Repository layout (src/)

High-level layout under `src/`:

```
src/
├── main.py                    # CLI entrypoint (dotenv, reset memory, optional vector DB build, launch UI)
├── ui/                        # Gradio UI + periodic execution
│   └── app.py                 # App().run() and timer-driven runs + plots/logs/table
├── core/                      # Orchestration + persistence wiring
│   ├── framework.py           # DealAgentFramework (planner selection, memory, Chroma, t-SNE plot data)
│   └── memory.py              # MemoryStore (read/write/reset memory.json)
├── agents/                    # Agent implementations grouped by capability
│   ├── planners/              # workflow planner vs LLM tool-loop planner
│   ├── scanners/              # RSS + page scraping + OpenAI structured deal selection
│   ├── pricing/               # pricing ensemble (frontier RAG + specialist Modal + optional NN)
│   ├── preprocessing/         # text rewrite/normalization prior to pricing
│   └── messaging/             # message crafting + notification send
├── rag/                       # vector store + retrieval + embeddings helpers (Chroma + SentenceTransformers)
├── scraping/                  # RSS + HTML parsing utilities (no model calls)
├── data/                      # Pydantic models used across agents/framework
├── services/                  # External integrations (Modal, notifications)
├── config/                    # Env-backed settings + project constants
└── utils/                     # Shared helpers (logging formatting, visualization)
```

Notes:

- This repo runs as a **regular Python app** (not an installed package). The primary entrypoint is `src/main.py`.
- Persistent state lives outside `src/`:
  - `memory.json` (top-level) stores previously surfaced opportunities.
  - `products_vectorstore/` (top-level) is the persistent Chroma DB directory (collection: `products`).

---

## Key components (where to start reading)

- **Entrypoint**: `src/main.py`
  - Loads `.env` via `python-dotenv`
  - Calls `DealAgentFramework.reset_memory()`
  - Optionally builds the product vector DB (`--build-vectordb`)
  - Launches the Gradio UI (`App().run()`)

- **Orchestrator**: `src/core/framework.py` (`DealAgentFramework`)
  - Loads `memory.json`
  - Opens Chroma `products_vectorstore/` and uses collection `products`
  - Selects planner by `PLANNER_MODE`
  - Runs the planner, persists the best `Opportunity`, computes t-SNE plot data for the UI

- **Planner modes**:
  - `src/agents/planners/planning_agent.py` (**workflow** mode): deterministic scan → price → pick → notify
  - `src/agents/planners/autonomous_planning_agent.py` (**autonomous** mode): LLM tool-calling loop decides next steps

- **Deal acquisition**: `src/agents/scanners/scanner_agent.py`
  - Scrapes DealNews RSS + deal pages (via `src/scraping/`)
  - Uses OpenAI Structured Outputs to normalize “best 5” deals into a Pydantic schema

- **Pricing ensemble**: `src/agents/pricing/ensemble_agent.py`
  - Optional rewrite/normalization via `src/agents/preprocessing/preprocessor.py` (LiteLLM; default `ollama/llama3.2`)
  - Specialist estimate via Modal (`src/agents/pricing/specialist_agent.py` + `src/services/modal/`)
  - Frontier estimate via RAG over Chroma (`src/agents/pricing/frontier_agent.py` + `src/rag/`)
  - Combines into a single “true value” estimate and discount

- **Notifications**: `src/agents/messaging/messaging_agent.py` + `src/services/notifications/pushover.py`
  - Crafts short copy via LiteLLM (Groq)
  - Sends push notifications via Pushover API

---

## Configuration & environment variables

Configuration is intentionally lightweight and environment-backed:

- `src/config/settings.py` defines `Settings` and reads env vars (no extra deps).

Common keys (create a `.env` in repo root; `src/main.py` loads it with `override=True`):

- **OpenAI**: `OPENAI_API_KEY`
  - Used by scanner, frontier estimator, and the autonomous planner.
- **Groq (via LiteLLM)**: `GROQ_API_KEY`
  - Used by `MessagingAgent` for notification copywriting.
- **Pushover**: `PUSHOVER_USER`, `PUSHOVER_TOKEN`
  - Required to actually send push notifications.
- **HuggingFace**: `HF_TOKEN`
  - Used when building the vector DB from a HF dataset.
- **Planner selection**: `PLANNER_MODE=workflow|autonomous`
  - Default is `autonomous`.
- **HF dataset override**: `HF_DATASET_USER` (default: `ed-donner`)
- **Preprocessor model**: `PRICER_PREPROCESSOR_MODEL` (default: `ollama/llama3.2`)

---

## Running locally

### App (Gradio UI)

```bash
python3 src/main.py
```

If you use `uv`:

```bash
uv run src/main.py
```

### Build/populate the vector DB (recommended on fresh runs)

`products_vectorstore/` is a persistent Chroma DB used by the frontier RAG estimator and the UI plot.

```bash
python3 src/main.py --build-vectordb
```

Optional flags:

```bash
# Use the full dataset (slower)
python3 src/main.py --build-vectordb --full-dataset

# Delete and recreate the Chroma collection before ingesting
python3 src/main.py --build-vectordb --force-recreate-vectordb
```

---

## Dependencies & tooling

- **Dependency sources**:
  - `requirements.txt` (pip)
  - `pyproject.toml` + `uv.lock` (uv)

Install via pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install via uv:

```bash
uv sync
```

---

## Tests

Tests are `unittest`-based and mock external services (no network required).

Run all tests:

```bash
python3 -m unittest -v
```

Run individual test modules:

```bash
python3 -m unittest -v tests.test_framework
python3 -m unittest -v tests.test_agents
```

See `tests/README.md` for details on coverage and design goals.

