"""
Build/populate the persistent Chroma vector DB used by the Gradio UI plot.

This mirrors the Day 2 notebook flow (simplified):
- load dataset (items_lite / items_full)
- embed item summaries with SentenceTransformer all-MiniLM-L6-v2
- store into Chroma PersistentClient at DB path: products_vectorstore, collection: "products"

Run it directly:
  - from repo root:  uv run src/main.py --build-vectordb
  - or (also from repo root): uv run python src/main.py --build-vectordb
  - or (from src):           uv run -m rag.vectorstore --lite

Notes:
- You need `HF_TOKEN` in your environment/.env to download the dataset.
- The UI 3D plot needs >30 items (TSNE perplexity constraint).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# This repo uses a simple "src/" layout (not an installed package).
# Make imports work whether you run from repo root or via uv.
_SRC_DIR = Path(__file__).resolve().parents[1]  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.models import Item  # noqa: E402
from core.framework import DealAgentFramework  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "products"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _repo_root() -> Path:
    # src/rag/vectorstore.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _db_path() -> str:
    """
    Chroma path used by `DealAgentFramework.DB`. Treat relative paths as repo-root-relative
    so running from other working directories still writes to the same DB.
    """
    db = Path(DealAgentFramework.DB)
    return str(db if db.is_absolute() else (_repo_root() / db))


def _collection_count(collection) -> int:
    try:
        return int(collection.count())
    except Exception:
        # Fallback for older Chroma / edge cases
        try:
            result = collection.get(limit=1)
            ids = result.get("ids") or []
            return 0 if len(ids) == 0 else 1
        except Exception:
            return 0


def build_products_vectordb(
    *,
    dataset: str,
    min_required: int = DealAgentFramework._MIN_SAMPLES_FOR_TSNE,
    force_recreate: bool = False,
    max_items: Optional[int] = None,
    batch_size: int = 1000,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
) -> int:
    """
    Build (or extend) the Chroma collection so it has at least `min_required` items.

    This follows the Day 2 notebook logic closely:
    - documents = item.summary
    - vectors = encoder.encode(documents).astype(float).tolist()
    - metadatas = {"category": item.category, "price": item.price}
    - ids = doc_0..doc_N
    """
    db_path = _db_path()
    client = chromadb.PersistentClient(path=db_path)

    if force_recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(collection_name)
    existing = _collection_count(collection)
    if existing >= min_required and not force_recreate:
        logger.info("Vector DB already populated: %s items (>= %s).", existing, min_required)
        return existing

    logger.info("Loading dataset: %s", dataset)
    train, _val, _test = Item.from_hub(dataset)
    if max_items is not None:
        train = train[:max_items]

    logger.info("Loading embedding model: %s", model_name)
    encoder = SentenceTransformer(model_name)

    # If appending (not recreating), start IDs after the current count to avoid collisions.
    id_start = 0 if force_recreate else max(existing, 0)

    logger.info(
        "Ingesting %s items into Chroma (%s, collection=%s, batch_size=%s)...",
        len(train),
        db_path,
        collection_name,
        batch_size,
    )
    for i in tqdm(range(0, len(train), batch_size)):
        batch = train[i : i + batch_size]
        documents = [item.summary for item in batch]
        vectors = encoder.encode(documents)
        vectors = np.asarray(vectors).astype(float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in batch]

        ids = [f"doc_{j}" for j in range(id_start + i, id_start + i + batch_size)]
        ids = ids[: len(documents)]

        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

    final_count = _collection_count(collection)
    logger.info("Vector DB ready: %s items in '%s'.", final_count, collection_name)
    return final_count


def ensure_products_vectordb(
    *,
    lite_mode: bool = True,
    min_required: int = DealAgentFramework._MIN_SAMPLES_FOR_TSNE,
    max_items: Optional[int] = None,
    force_recreate: bool = False,
) -> int:
    """
    Convenience wrapper used by `src/main.py`.
      dataset = f"{username}/items_lite" if lite_mode else f"{username}/items_full"
    """
    username = os.environ.get("HF_DATASET_USER", "ed-donner")
    dataset = f"{username}/items_lite" if lite_mode else f"{username}/items_full"
    return build_products_vectordb(
        dataset=dataset,
        min_required=min_required,
        force_recreate=force_recreate,
        max_items=max_items,
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Chroma products vector DB (Day 2 flow).")
    p.add_argument("--lite", action="store_true", help="Use items_lite (default).", default=True)
    p.add_argument("--full", action="store_true", help="Use items_full (overrides --lite).")
    p.add_argument("--force", action="store_true", help="Delete and recreate collection first.")
    p.add_argument(
        "--min-required",
        type=int,
        default=DealAgentFramework._MIN_SAMPLES_FOR_TSNE,
        help="Minimum items needed (TSNE needs >30).",
    )
    p.add_argument("--max-items", type=int, default=None, help="Optionally cap ingested items.")
    p.add_argument("--batch-size", type=int, default=1000, help="Ingestion batch size.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [VectorDB] %(message)s")

    args = _parse_args(argv)
    lite_mode = not args.full
    count = ensure_products_vectordb(
        lite_mode=lite_mode,
        min_required=args.min_required,
        max_items=args.max_items,
        force_recreate=args.force,
    )
    if count < args.min_required:
        logger.warning("Only %s items in vector DB; UI plot needs >= %s.", count, args.min_required)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

