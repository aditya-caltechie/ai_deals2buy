"""
Backward-compatible shim.

The Gradio UI has moved to `ui.app` as part of a directory hierarchy refactor.
This module remains to avoid breaking `from price_is_right import App` imports.
"""

from ui.app import App  # noqa: F401


if __name__ == "__main__":
    App().run()
