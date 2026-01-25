# ai-deals2buy

## First-time setup (after cloning)

### Create and activate a virtual environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the repo root and add any required keys (example keys used in notebooks/agents include `HF_TOKEN` and API keys for your LLM provider).

## Run the app (Gradio UI)

From the repo root:

```bash
python3 src/main.py
```

### Populate the vector DB (recommended on fresh runs)

The UI’s 3D plot reads a persistent Chroma vector DB at `products_vectorstore/` (collection: `products`).
On a fresh machine/checkout (or if you deleted `products_vectorstore/`), build it first:

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

### Build the vector DB directly (optional)

```bash
cd src
python3 -m agents.rag_vectordb          # lite dataset (default)
python3 -m agents.rag_vectordb --full   # full dataset
```
