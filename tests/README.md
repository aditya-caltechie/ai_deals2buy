## Tests

This repo uses lightweight **unittest + unittest.mock** tests (no extra dependencies).

### Run

From the repo root:

```bash
python3 -m unittest -v
```

Or run a single file:

```bash
python3 -m unittest -v tests.test_framework
python3 -m unittest -v tests.test_agents
```

Or run a single test method:

```bash
python3 -m unittest -v tests.test_framework.TestDealAgentFramework.test_planner_selection_workflow
```

### What the tests cover

- `tests/test_framework.py`
  - Planner selection via `PLANNER_MODE`
  - Framework `run()` writes to `memory.json` when a planner returns an `Opportunity`
- `tests/test_agents.py`
  - `ScannerAgent` uses OpenAI Structured Outputs (mocked) and returns a `DealSelection`
  - `FrontierAgent` price parsing from an OpenAI response (mocked) + retriever (mocked)
  - `MessagingAgent` crafts text (mocked) and calls Pushover client (mocked)

### Design goals

- No network calls (OpenAI/Groq/Pushover/Modal/DealNews are mocked)
- No dependence on local services (Ollama is mocked)
- Focus on “does the wiring work” (planner selection, agent calls, persistence)

