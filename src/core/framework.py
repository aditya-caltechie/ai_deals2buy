import logging
import os
import sys
from typing import List

import chromadb
from dotenv import load_dotenv

from data.models import Opportunity
from agents.planners.autonomous_planning_agent import AutonomousPlanningAgent
from agents.planners.planning_agent import PlanningAgent
from core.memory import MemoryStore
from utils.visualization import compute_tsne_plot_data

load_dotenv(override=True)

# Colors for logging
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        init_logging()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory_store = MemoryStore(self.MEMORY_FILENAME)
        self.memory = self.memory_store.read()
        self.collection = client.get_or_create_collection("products")
        self.planner = None

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            mode = os.environ.get("PLANNER_MODE", "autonomous").strip().lower()
            if mode in ("workflow", "planning", "planning_agent", "plan"):
                self.planner = PlanningAgent(self.collection)
                self.log("Using PlanningAgent (workflow mode)")
            else:
                # Default: tool / execution-loop style planner (LLM function-calling)
                self.planner = AutonomousPlanningAgent(self.collection)
                self.log("Using AutonomousPlanningAgent (tool-loop mode)")
            self.log("Agent Framework is ready")

    def write_memory(self) -> None:
        self.memory_store.write(self.memory)

    @classmethod
    def reset_memory(cls) -> None:
        MemoryStore(cls.MEMORY_FILENAME).reset_keep_first(2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        self.init_agents_as_needed()
        logging.info("Kicking off Planning Agent")
        result = self.planner.plan(memory=self.memory)
        logging.info(f"Planning Agent has completed and returned: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    # T-SNE default perplexity is 30; n_samples must be > perplexity
    _MIN_SAMPLES_FOR_TSNE = 31

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection("products")
        return compute_tsne_plot_data(
            collection=collection,
            max_datapoints=max_datapoints,
            min_samples=cls._MIN_SAMPLES_FOR_TSNE,
        )


if __name__ == "__main__":
    DealAgentFramework().run()

