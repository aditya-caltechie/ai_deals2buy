"""
Backward-compatible shim.

Vector DB builder moved to `rag.vectorstore`.
"""

from rag.vectorstore import build_products_vectordb, ensure_products_vectordb, main  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(main())

