#!/usr/bin/env python3
"""
Entry point to run The Price is Right app from the command line.
Replicates the Colab/notebook setup: logging, dotenv, reset memory, then launch the Gradio app.
"""
import logging
import sys

from dotenv import load_dotenv

from core.framework import DealAgentFramework
from ui.app import App
from rag.vectorstore import ensure_products_vectordb

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    load_dotenv(override=True)
    DealAgentFramework.reset_memory()

    # Optional prerequisite step for fresh local runs:
    #   python3 src/main.py --build-vectordb
    argv = set(sys.argv[1:])
    if "--build-vectordb" in argv:
        ensure_products_vectordb(
            lite_mode="--full-dataset" not in argv,
            force_recreate="--force-recreate-vectordb" in argv,
        )

    App().run()
