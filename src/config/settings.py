import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """
    Centralized environment-backed configuration.

    Keep this lightweight (no extra deps) so it works in simple scripts and notebooks.
    """

    planner_mode: str = os.getenv("PLANNER_MODE", "autonomous").strip().lower()

    # API keys / credentials (only some are required depending on enabled features)
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    hf_token: str | None = os.getenv("HF_TOKEN")

    pushover_user: str | None = os.getenv("PUSHOVER_USER")
    pushover_token: str | None = os.getenv("PUSHOVER_TOKEN")

    # Optional overrides
    hf_dataset_user: str = os.getenv("HF_DATASET_USER", "ed-donner")
    pricer_preprocessor_model: str = os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/llama3.2")

