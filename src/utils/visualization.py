from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from sklearn.manifold import TSNE

from config.constants import CATEGORIES, COLORS


def _empty_plot_tuple():
    return [], np.array([]).reshape(0, 3), []


def compute_tsne_plot_data(
    *,
    collection: Any,
    max_datapoints: int = 2000,
    min_samples: int = 31,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Read embeddings/documents from Chroma and compute a 3D t-SNE projection.

    Returns:
      documents: list[str]
      vectors: np.ndarray shaped (n, 3)
      colors: list[str] matching documents
    """
    result = collection.get(
        include=["embeddings", "documents", "metadatas"], limit=max_datapoints
    )
    embeddings = result.get("embeddings")
    if embeddings is None:
        return _empty_plot_tuple()

    vectors = np.asarray(embeddings)
    # Chroma may return []/np.array([]) for empty collections; TSNE also needs >30 samples by default.
    if vectors.size == 0 or vectors.shape[0] < min_samples:
        return _empty_plot_tuple()

    documents = result["documents"]
    categories = [metadata.get("category") for metadata in result.get("metadatas", [])]
    if categories and isinstance(categories[0], list):
        # Chroma format: metadatas = [[{...}, {...}, ...]]
        categories = [m.get("category") for m in categories[0]]

    color_map = {cat: COLORS[i % len(COLORS)] for i, cat in enumerate(CATEGORIES)}
    colors = [color_map.get(cat, "gray") for cat in categories]

    tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
    reduced_vectors = tsne.fit_transform(vectors)
    return documents, reduced_vectors, colors

