from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def get_encoder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: Iterable[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    encoder = get_encoder(model_name)
    vectors = encoder.encode(list(texts))
    return np.asarray(vectors)

