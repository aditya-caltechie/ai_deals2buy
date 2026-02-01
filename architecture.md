## Architecture (ai-deals2buy)

This repo is a periodic, agentic deal-hunting system:
- **Acquire deals** (RSS + HTML scrape)
- **Normalize/select the best candidates** (LLM structured output)
- **Estimate “true value”** (ensemble: RAG frontier model + specialist model; optional preprocessing)
- **Notify** when discount is large enough
- **Persist surfaced opportunities** so you don’t get repeated alerts

![Architecture diagram](docs/architecture.svg)

### Main building blocks

- **Entry point**: `src/main.py`
  - Loads `.env`
  - Resets persisted memory (keeps first 2 entries)
  - Optionally builds the vector DB (`--build-vectordb`)
  - Launches the Gradio UI

- **UI layer**: `src/ui/app.py`
  - Runs the pipeline on UI load, then **every 5 minutes** (`gr.Timer`)
  - Shows:
    - a table of opportunities (deal price, estimate, discount, URL)
    - a 3D plot derived from the vector DB (t-SNE)
    - streaming logs (ANSI → HTML reformat)
  - Also supports **manual alerting**: selecting a row triggers `MessagingAgent.alert(...)`

- **Orchestration**: `src/core/framework.py` (`DealAgentFramework`)
  - Loads and writes memory via `src/core/memory.py` (`MemoryStore`)
  - Opens the Chroma persistent DB at `products_vectorstore/` (collection: `products`)
  - Chooses planner via `PLANNER_MODE`:
    - **workflow** → `src/agents/planners/planning_agent.py` (`PlanningAgent`)
    - **autonomous** (default) → `src/agents/planners/autonomous_planning_agent.py` (`AutonomousPlanningAgent`)
  - Produces plot data via `utils/visualization.py` (t-SNE needs \(\ge 31\) vectors)

### Agents (capabilities)

- **Scanner**: `src/agents/scanners/scanner_agent.py`
  - Fetches RSS + deal pages via `src/scraping/rss_scraper.py`
  - Calls OpenAI Structured Outputs to pick/normalize **exactly 5** deals (model: `gpt-5-mini`)

- **Pricing ensemble**: `src/agents/pricing/ensemble_agent.py`
  - **Preprocess** (rewrite/normalize): `src/agents/preprocessing/preprocessor.py` via `litellm` (default model `ollama/llama3.2`)
  - **Frontier estimator (RAG)**: `src/agents/pricing/frontier_agent.py`
    - Retrieves similar products from Chroma (`src/rag/retriever.py`)
    - Calls OpenAI (model: `gpt-5.1`) to estimate a single price
  - **Specialist estimator**: `src/agents/pricing/specialist_agent.py`
    - Calls a Modal-hosted fine-tuned model (`src/services/modal/`)
  - Combines estimates (current weights): \(0.8 \cdot \text{frontier} + 0.2 \cdot \text{specialist}\)

- **Messaging**: `src/agents/messaging/messaging_agent.py`
  - Crafts short push-copy via LiteLLM on Groq (model: `groq/openai/gpt-oss-20b`)
  - Sends notifications via Pushover (`src/services/notifications/pushover.py`)

### State & storage

- **Deal memory**: `src/memory.json`
  - Appends the last surfaced opportunity so future runs can skip already-alerted URLs
  - Note: the code currently persists memory under `src/` (not repo root).

- **Vector DB**: `products_vectorstore/`
  - Persistent Chroma DB used for:
    - RAG retrieval (frontier estimator)
    - UI visualization (t-SNE plot)

### External services (typical)

- **OpenAI**: deal selection + autonomous planning + RAG price estimation
- **Ollama** (optional/local): text preprocessing
- **Modal** (optional): specialist fine-tuned model
- **Groq** (optional): message crafting via LiteLLM
- **Pushover** (optional): push notification delivery

### Related docs

- Full walkthrough (includes Mermaid diagrams and deep details): `docs/README.md`
- End-to-end runtime flow: `workflow-e2e.md`
