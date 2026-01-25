# ai-deals2buy (The Price is Right)

An agentic deal-hunting system that estimates a product's "true value" and alerts you when the discount is large enough. It ships with a Gradio UI and a small set of cooperating agents.

For architecture and the full end-to-end flow, see `docs/README.md`. For a contributor-oriented code map, see `AGENTS.md`.

## What it does

On a repeating schedule, the app:

- Scrapes deal RSS feeds (DealNews)
- Uses an LLM to select and summarize the best deals with a clear numeric price
- Estimates a "true value" using an ensemble:
  - Specialist model: a fine-tuned Llama 3.2 3B hosted on Modal
  - Frontier model: an OpenAI model using RAG over a Chroma vector DB
  - Optional local preprocessing via Ollama
- Computes discount = estimated value - deal price
- Sends a push notification when a deal crosses a threshold

This repo runs as a regular Python app (not an installed package). The code lives in `src/`.

## Quick start

### Requirements

- Python 3.11+
- Optional but recommended:
  - Ollama running locally for preprocessing (default base URL: `http://localhost:11434`)
  - Modal configured if you want to use the hosted fine-tuned specialist model

### Install (pip)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install (uv)

If you use `uv`, the repo includes `uv.lock` and a `pyproject.toml` dependency set:

```bash
uv sync
```

### Configure (.env)

Create a `.env` in the repo root. Common keys:

- OpenAI: `OPENAI_API_KEY` (used by `ScannerAgent`, `FrontierAgent`, `AutonomousPlanningAgent`)
- Pushover (push notifications): `PUSHOVER_USER`, `PUSHOVER_TOKEN`
- Groq (via LiteLLM): `GROQ_API_KEY` (recommended if using `groq/openai/gpt-oss-20b`)
- HuggingFace (vector DB dataset download): `HF_TOKEN`
- Dataset source override (optional): `HF_DATASET_USER` (defaults to `ed-donner`)
- Planner selection: `PLANNER_MODE=workflow` or `PLANNER_MODE=autonomous`
- Preprocessor model (optional): `PRICER_PREPROCESSOR_MODEL` (default `ollama/llama3.2`)

Notes:

- You can run the UI without Pushover/Groq, but you'll lose push notifications (and message crafting).
- If you enable the Modal-backed specialist estimator, you must have Modal configured locally (`modal` CLI auth + access to the deployed app).

### Run the app (Gradio UI)

From the repo root:

```bash
python3 src/main.py
```

Or with uv:

```bash
uv run src/main.py
```

### Build/populate the vector DB (recommended on fresh runs)

The UI's 3D plot reads a persistent Chroma vector DB at `products_vectorstore/` (collection: `products`).

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

You can also run the builder directly:

```bash
cd src
python3 -m agents.rag_vectordb          # lite dataset (default)
python3 -m agents.rag_vectordb --full   # full dataset
```

## How it works (at a glance)

When you run `src/main.py` it:

- loads env vars
- resets persisted memory
- optionally builds the vector DB
- launches the Gradio UI

The UI (`src/ui/app.py`) runs the agent pipeline on startup and then every 5 minutes via `gr.Timer`.

Core orchestration happens in `src/core/framework.py`:

- loads previous surfaced opportunities from `memory.json`
- opens Chroma DB `products_vectorstore/` (collection `products`)
- chooses one of two planner modes via `PLANNER_MODE`

### Planner modes

Both modes use the same underlying agents (`ScannerAgent`, `EnsembleAgent`, `MessagingAgent`) but orchestrate them differently:

- Workflow mode (`PLANNER_MODE=workflow`): deterministic pipeline in `src/agents/planners/planning_agent.py`
  - scan -> price top 5 -> pick best -> notify if `discount > PlanningAgent.DEAL_THRESHOLD`
- Tool-loop mode (`PLANNER_MODE=autonomous`, default): LLM function-calling loop in `src/agents/planners/autonomous_planning_agent.py`
  - the planner LLM decides which tool to call next (scan / estimate / notify) until it finishes

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

## Repo layout (high level)

- `src/main.py`: CLI entrypoint (dotenv, reset memory, optional vector DB build, launch UI)
- `src/ui/app.py`: Gradio app + timer-driven runs + 3D embedding visualization
- `src/core/framework.py`: orchestrator, planner selection, Chroma + memory persistence, t-SNE plot data
- `src/agents/`: agent implementations (scanner, planners, pricing ensemble, notifications, vector DB builder)
- `src/services/modal/`: Modal app(s) used to host the fine-tuned specialist model

## Tests

This repo uses lightweight `unittest` + `unittest.mock` tests (no extra test dependencies).

From the repo root:

```bash
python3 -m unittest -v
```

Run a single module:

```bash
python3 -m unittest -v tests.test_framework
python3 -m unittest -v tests.test_agents
```

See `tests/README.md` for details on what is covered.

## Notes / troubleshooting

- The 3D plot uses t-SNE; it needs at least 31 items in the vector DB.
- `memory.json` stores surfaced opportunities so you don't alert on the same deal repeatedly.
- If you don't have Modal configured, the `SpecialistAgent` will fail to connect; use `docs/README.md` to decide whether to stub/disable it for local-only runs.
