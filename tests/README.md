## Tests

This repo uses lightweight **unittest + unittest.mock** tests (no extra dependencies).

### Run

From the repo root:

```bash
uv run python -m unittest -v
```

Run unit tests only:

```bash
uv run python -m unittest discover -s tests/unit -p "test_*.py" -v
```

Run integration tests only:

```bash
uv run python -m unittest discover -s tests/integration -p "test_*.py" -v
```

Or run a single module:

```bash
uv run python -m unittest -v tests.integration.test_framework
uv run python -m unittest -v tests.unit.test_agents
```

Or run a single test method:

```bash
uv run python -m unittest -v tests.integration.test_framework.TestDealAgentFramework.test_planner_selection_workflow
```

### What the tests cover

- `tests/integration/test_framework.py`
  - Planner selection via `PLANNER_MODE`
  - Framework `run()` writes to `memory.json` when a planner returns an `Opportunity`
- `tests/unit/test_agents.py`
  - `ScannerAgent` uses OpenAI Structured Outputs (mocked) and returns a `DealSelection`
  - `FrontierAgent` price parsing from an OpenAI response (mocked) + retriever (mocked)
  - `MessagingAgent` crafts text (mocked) and calls Pushover client (mocked)

### Design goals

- No network calls (OpenAI/Groq/Pushover/Modal/DealNews are mocked)
- No dependence on local services (Ollama is mocked)
- Focus on “does the wiring work” (planner selection, agent calls, persistence)

### Conventions

- **Unit tests** (`tests/unit/`): focus on a single module/class with mocks around external dependencies.
- **Integration tests** (`tests/integration/`): exercise wiring across multiple components (e.g., framework + planner selection + persistence) while still mocking network/service calls.

