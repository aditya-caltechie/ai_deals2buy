## Workflow (end-to-end)

This doc describes what happens when you run the app, from UI startup to notifications and persistence.

![End-to-end workflow](docs/agent_workflow.svg)

### E2E flow (as implemented)

#### 1) Start the app

Run:

```bash
uv run python src/main.py
```

What `src/main.py` does:
- Loads `.env`
- Resets memory (keeps first 2 entries)
- Optionally builds vector DB (if `--build-vectordb` is passed)
- Launches the UI (`App().run()`)

#### 2) UI kicks off runs (startup + timer)

`src/ui/app.py`:
- Executes one run on UI load
- Repeats every **300 seconds (5 minutes)** via `gr.Timer`
- Streams logs while the pipeline runs in a worker thread
- Updates the deals table when the run completes

#### 3) Framework orchestration

Each run calls `DealAgentFramework.run()` (`src/core/framework.py`), which:
- Lazily initializes a planner the first time (based on `PLANNER_MODE`)
- Calls `planner.plan(memory=self.memory)`
- If an `Opportunity` is returned:
  - appends it to memory
  - persists to `src/memory.json`

#### 4) Planner mode selection

`PLANNER_MODE`:
- **workflow**: deterministic pipeline (`PlanningAgent`)
- **autonomous** *(default)*: LLM tool-calling loop (`AutonomousPlanningAgent`)

Both modes use the same underlying agents (scanner, pricing, messaging); the difference is *who* decides the next step (fixed code vs. LLM tool loop).

### Workflow mode (deterministic)

In `src/agents/planners/planning_agent.py`:

1) **Scan**
   - `ScannerAgent.scan(memory)`
   - Scrapes RSS + deal pages, then uses OpenAI Structured Outputs to produce exactly 5 clean deals

2) **Price**
   - For each deal: `EnsembleAgent.price(description)`
   - Computes `discount = estimate - deal_price`

3) **Pick + notify**
   - Chooses the single best discount
   - If `discount > PlanningAgent.DEAL_THRESHOLD`:
     - sends a push notification

### Autonomous mode (tool loop)

In `src/agents/planners/autonomous_planning_agent.py`:

1) Calls an OpenAI chat model with tools enabled:
   - `scan_the_internet_for_bargains()`
   - `estimate_true_value(description)`
   - `notify_user_of_deal(description, deal_price, estimated_true_value, url)`

2) The model decides what tools to call and in what order.

3) On the first notification tool call, the agent records the chosen `Opportunity` and returns it back to the framework for persistence.

### Pricing pipeline (inside `EnsembleAgent`)

For each candidate deal:

1) **Rewrite/normalize text** via `Preprocessor` (LiteLLM; default `ollama/llama3.2`)
2) **Specialist estimate** via Modal (`SpecialistAgent`)
3) **Frontier estimate (RAG)**:
   - vector search over `products_vectorstore/` (Chroma, collection `products`)
   - OpenAI call to estimate price with retrieved examples
4) **Combine** estimates into one value (current weights: 80% frontier / 20% specialist)

### Notification pipeline (inside `MessagingAgent`)

When a deal is worth notifying:
- Crafts a short, exciting message (LiteLLM on Groq, model `groq/openai/gpt-oss-20b`)
- Sends it via Pushover (HTTP API)

### Persistence & “don’t alert twice”

- Memory is a list of past surfaced opportunities stored in `src/memory.json`.
- The scanner filters out already-seen URLs so the same deal isn’t repeatedly selected.

### Manual “alert this row” in the UI

In the Gradio UI, selecting a row triggers `MessagingAgent.alert(opportunity)` immediately (useful for testing notifications or re-sending).

### Related docs

- Component breakdown: `architecture.md`
- Deeper technical walkthrough: `docs/README.md`
