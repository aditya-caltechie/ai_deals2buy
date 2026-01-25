"""
Build/populate the persistent Chroma vector DB used by the Gradio UI plot.

The UI (`DealAgentFramework.get_plot_data`) expects:
- Chroma DB path: `products_vectorstore` (relative to repo root)
- Collection name: "products"
- Documents: product summaries
- Metadatas: {"category": ..., "price": ...}

This script mirrors the Day 2 notebook ingestion flow so fresh local runs can
show vector DB data immediately (TSNE requires >30 items).
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import chromadb
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from agents.items import Item
from deal_agent_framework import DealAgentFramework


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorDbConfig:
    db_path: Path
    collection_name: str = "products"
    dataset: str = "ed-donner/items_lite"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 1000
    max_items: Optional[int] = None


def _repo_root() -> Path:
    # src/agents/rag_vectordb.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def default_db_path() -> Path:
    """
    Resolve the default DB path used by `DealAgentFramework.DB`.
    We treat it as relative to the repo root so running from other CWDs still works.
    """
    db = Path(DealAgentFramework.DB)
    return db if db.is_absolute() else (_repo_root() / db)


def _item_document(item: Item) -> str:
    # Prefer the same field as the notebook; fall back gracefully.
    if item.summary and item.summary.strip():
        return item.summary
    if item.full and item.full.strip():
        return item.full
    return f"{item.title} ({item.category})"


def _batched(it: list[Item], batch_size: int) -> Iterable[list[Item]]:
    for i in range(0, len(it), batch_size):
        yield it[i : i + batch_size]


def _safe_collection_count(collection) -> int:
    try:
        return int(collection.count())
    except Exception:
        # Older Chroma versions / edge cases; fall back to a cheap get().
        try:
            result = collection.get(limit=1)
            ids = result.get("ids") or []
            return 0 if len(ids) == 0 else 1
        except Exception:
            return 0


def build_products_vectordb(
    *,
    config: VectorDbConfig,
    min_required: int = DealAgentFramework._MIN_SAMPLES_FOR_TSNE,
    force_recreate: bool = False,
) -> int:
    """
    Ensure the Chroma collection exists and has enough items for TSNE.
    Returns the final collection count.
    """
    config.db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(config.db_path))

    if force_recreate:
        try:
            client.delete_collection(config.collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(config.collection_name)
    existing = _safe_collection_count(collection)
    if existing >= min_required and not force_recreate:
        logger.info(
            "Vector DB already populated: %s items (>= %s).",
            existing,
            min_required,
        )
        return existing

    logger.info("Loading dataset: %s", config.dataset)
    train, _val, _test = Item.from_hub(config.dataset)
    if config.max_items is not None:
        train = train[: config.max_items]

    if len(train) < min_required:
        logger.warning(
            "Dataset has only %s items; UI plot needs >= %s for TSNE.",
            len(train),
            min_required,
        )

    logger.info("Loading embedding model: %s", config.model_name)
    encoder = SentenceTransformer(config.model_name)

    # Avoid collisions if we're re-running ingestion without recreating the DB.
    # We generate a fresh ID namespace based on current count.
    id_offset = 0 if force_recreate else max(existing, 0)

    logger.info(
        "Ingesting %s items into Chroma at %s (collection=%s)...",
        len(train),
        config.db_path,
        config.collection_name,
    )
    for batch_idx, batch in enumerate(tqdm(list(_batched(train, config.batch_size)))):
        documents = [_item_document(item) for item in batch]
        vectors = encoder.encode(documents)
        vectors = np.asarray(vectors, dtype=float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in batch]

        # Stable-ish IDs: prefer item.id if present; otherwise sequential.
        ids = []
        for j, item in enumerate(batch):
            if item.id is not None:
                ids.append(f"doc_{int(item.id)}")
            else:
                ids.append(f"doc_{id_offset + (batch_idx * config.batch_size) + j}")

        # If IDs collide, Chroma will reject; in that case, fall back to append IDs.
        try:
            collection.add(
                ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas
            )
        except Exception:
            ids = [
                f"{_repo_root().name}_doc_{id_offset + (batch_idx * config.batch_size) + j}"
                for j in range(len(batch))
            ]
            collection.add(
                ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas
            )

    final_count = _safe_collection_count(collection)
    logger.info("Vector DB ready: %s items in collection '%s'.", final_count, config.collection_name)
    return final_count


def ensure_products_vectordb(
    *,
    lite_mode: bool = True,
    min_required: int = DealAgentFramework._MIN_SAMPLES_FOR_TSNE,
    max_items: Optional[int] = None,
    force_recreate: bool = False,
) -> int:
    username = os.environ.get("HF_DATASET_USER", "ed-donner")
    dataset = f"{username}/items_lite" if lite_mode else f"{username}/items_full"

    config = VectorDbConfig(
        db_path=default_db_path(),
        dataset=dataset,
        max_items=max_items,
    )
    return build_products_vectordb(
        config=config, min_required=min_required, force_recreate=force_recreate
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the products vector DB for the UI plot.")
    p.add_argument(
        "--lite",
        action="store_true",
        help="Use the lite dataset (default).",
        default=True,
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Use the full dataset (overrides --lite).",
        default=False,
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the Chroma collection first.",
        default=False,
    )
    p.add_argument(
        "--min-required",
        type=int,
        default=DealAgentFramework._MIN_SAMPLES_FOR_TSNE,
        help="Minimum items needed (TSNE needs >30).",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optionally cap number of ingested training items.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [VectorDB] %(message)s")

    args = _parse_args(argv)
    lite_mode = True
    if args.full:
        lite_mode = False

    ensure_products_vectordb(
        lite_mode=lite_mode,
        min_required=args.min_required,
        max_items=args.max_items,
        force_recreate=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

