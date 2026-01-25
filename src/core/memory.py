import json
import os
from typing import Iterable, List

from data.models import Opportunity


class MemoryStore:
    """
    Handles persistence of surfaced opportunities to a JSON file (default: memory.json).
    """

    def __init__(self, filename: str = "memory.json"):
        self.filename = filename

    def read(self) -> List[Opportunity]:
        if os.path.exists(self.filename):
            with open(self.filename, "r") as file:
                data = json.load(file)
            return [Opportunity(**item) for item in data]
        return []

    def write(self, opportunities: Iterable[Opportunity]) -> None:
        data = [opportunity.model_dump() for opportunity in opportunities]
        with open(self.filename, "w") as file:
            json.dump(data, file, indent=2)

    def reset_keep_first(self, n: int = 2) -> None:
        """
        Truncate the memory file to the first N items.
        Useful for demo/dev runs to keep context small.
        """
        data = []
        if os.path.exists(self.filename):
            with open(self.filename, "r") as file:
                data = json.load(file)
        truncated = data[:n]
        with open(self.filename, "w") as file:
            json.dump(truncated, file, indent=2)

