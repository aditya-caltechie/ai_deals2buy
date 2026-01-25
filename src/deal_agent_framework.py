"""
Backward-compatible shim.

The orchestration framework has moved to `core.framework` as part of a directory
hierarchy refactor. This module remains to avoid breaking existing imports like:
`from deal_agent_framework import DealAgentFramework`.
"""

from core.framework import DealAgentFramework  # noqa: F401


if __name__ == "__main__":
    DealAgentFramework().run()
