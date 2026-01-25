from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from rag.embeddings import embed_texts, DEFAULT_EMBEDDING_MODEL


@dataclass
class ChromaRetriever:
    """
    Minimal retriever wrapper around a Chroma collection.
    """

    collection: Any
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    def query_similars(self, description: str, n_results: int = 5) -> Tuple[List[str], List[float]]:
        vector = embed_texts([description], model_name=self.embedding_model)
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(),
            n_results=n_results,
        )
        documents = results["documents"][0][:]
        prices = [m["price"] for m in results["metadatas"][0][:]]
        return documents, prices

