# ai-deals2buy 

[![CI](https://github.com/aditya-caltechie/ai_deals2buy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aditya-caltechie/ai_deals2buy/actions/workflows/ci.yml?query=branch%3Amain)

An agentic deal-hunting system that estimates a product's "true value" and alerts you when the discount is large enough. It ships with a Gradio UI and a small set of cooperating agents.

Deals are **scraped** and shortlisted; the **ensemble** estimates true price using a **fine-tuned** specialist (Modal), **RAG** retrieval over the product vector DB (frontier), and an optional **neural** model. The pipeline compares those estimates to the scraped deal price to decide if it is a **real** discount, then **notifies** you via push.

## Architecture

![Agent workflow — UI, framework, planner, scanner, ensemble, specialist, frontier (RAG), NN, messaging](docs/images/agent_workflow.svg)

## Quick start

### Requirements

- Python 3.11–3.13 (Python 3.14 is not supported yet; some deps like `onnxruntime` don’t ship 3.14 wheels)
- Optional (depending on which parts you want to run):
  - **Ollama** running locally for preprocessing (default base URL: `http://localhost:11434`)
  - **Modal** configured if you want to use the hosted fine-tuned specialist model
  - **Pushover** credentials if you want push notifications

### Install + run (uv-only)

This repo is standardized on `uv` and ships an `uv.lock` for reproducible installs.

Steps for a fresh clone (uv-only):

#### 1) Install uv (one time on your machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2) Clone + enter repo

```bash
git clone <repo-url>
cd ai-deals2buy
```

#### 3) Sync dependencies

```bash
uv sync
```

#### 4) Configure (.env)

Create a `.env` in the repo root. Common keys:

- OpenAI: `OPENAI_API_KEY` (used by `ScannerAgent`, `FrontierAgent`, `AutonomousPlanningAgent`)
- HuggingFace (vector DB dataset download): `HF_TOKEN` (only needed for `--build-vectordb`)
- Pushover (push notifications): `PUSHOVER_USER`, `PUSHOVER_TOKEN` (only needed to actually send pushes)
- Groq (via LiteLLM): `GROQ_API_KEY` (only needed for notification copywriting via Groq)
- Dataset source override (optional): `HF_DATASET_USER` (defaults to `ed-donner`)
- Planner selection: `PLANNER_MODE=workflow` or `PLANNER_MODE=autonomous`
- Preprocessor model (optional): `PRICER_PREPROCESSOR_MODEL` (default `ollama/llama3.2`)

Notes:

- You can run the UI without Pushover/Groq; you’ll just lose push notifications (and message crafting).
- The ensemble pricer **always** runs three steps in code: preprocessing → specialist (Modal) → frontier (OpenAI RAG), then combines them. If you don’t have Modal/Ollama configured, those calls can fail (see troubleshooting below).

#### 5) Run the app (Gradio UI)

From the repo root:

```bash
uv run python src/main.py
```

#### Do you need to create a venv?

No manual venv needed. `uv sync` will create/manage a project environment (commonly a local `.venv`) automatically, and `uv run ...` runs inside it.

If you want to force the convention explicitly:

```bash
uv venv
uv sync
```

### Build/populate the vector DB (recommended on fresh runs)

The UI's 3D plot reads a persistent Chroma vector DB at `products_vectorstore/` (collection: `products`).

```bash
uv run python src/main.py --build-vectordb
```

Optional flags:

```bash
# Use the full dataset (slower)
uv run python src/main.py --build-vectordb --full-dataset

# Delete and recreate the Chroma collection before ingesting
uv run python src/main.py --build-vectordb --force-recreate-vectordb
```

You can also run the builder directly:

```bash
cd src
uv run python -m rag.vectorstore              # items_lite (default)
uv run python -m rag.vectorstore --full       # items_full
uv run python -m rag.vectorstore --force      # delete and recreate collection first
```

## Models and providers (as implemented)

- Deal selection + summarization: OpenAI `gpt-5-mini` via `openai` SDK (`ScannerAgent`)
- Tool-loop planner (autonomous mode): OpenAI `gpt-5.1` via `openai` SDK (`AutonomousPlanningAgent`)
- Frontier estimator (RAG + reasoning): OpenAI `gpt-5.1` via `openai` SDK (`FrontierAgent`)
- Text preprocessor / rewrite: defaults to local `ollama/llama3.2` via `litellm` (`Preprocessor`)
- Notification copywriting: `groq/openai/gpt-oss-20b` via `litellm` (`MessagingAgent`)
- Specialist estimator: fine-tuned `meta-llama/Llama-3.2-3B` on Modal (`SpecialistAgent`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

## Tech stack

- UI: Gradio, Plotly
- Agents / orchestration: custom agent classes + OpenAI tool calling
- RAG / vector DB: ChromaDB + SentenceTransformers embeddings
- Data ingestion: HuggingFace `datasets` (vector DB build), RSS via `feedparser`, HTML parsing via BeautifulSoup
- Notifications: Pushover (HTTP API), message generation via LiteLLM (Groq)
- ML/vis: scikit-learn (t-SNE for 3D plot), NumPy
- Serving specialist model: Modal + Transformers + PEFT + bitsandbytes

## Tests

This repo uses lightweight `unittest` + `unittest.mock` tests (no extra test dependencies).

From the repo root:

```bash
uv run python -m unittest -v
```

Run a single module:

```bash
uv run python -m unittest -v tests.integration.test_framework
uv run python -m unittest -v tests.unit.test_agents
```
