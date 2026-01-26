from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests._testutils import add_src_to_syspath


add_src_to_syspath()


class TestDealAgentFramework(unittest.TestCase):
    def test_planner_selection_workflow(self):
        # Import inside test so patches apply cleanly.
        from core import framework as framework_mod

        fake_collection = object()

        fake_client = MagicMock()
        fake_client.get_or_create_collection.return_value = fake_collection

        with patch.object(framework_mod, "chromadb") as chromadb_mock, patch.dict(
            os.environ, {"PLANNER_MODE": "workflow"}
        ), patch.object(framework_mod, "PlanningAgent") as PlanningAgentMock, patch.object(
            framework_mod, "AutonomousPlanningAgent"
        ) as AutonomousMock:
            chromadb_mock.PersistentClient.return_value = fake_client
            PlanningAgentMock.return_value = MagicMock()
            AutonomousMock.return_value = MagicMock()

            f = framework_mod.DealAgentFramework()
            f.init_agents_as_needed()

            PlanningAgentMock.assert_called_once_with(fake_collection)
            AutonomousMock.assert_not_called()

    def test_run_persists_when_planner_returns_opportunity(self):
        from core import framework as framework_mod
        from data.models import Deal, Opportunity

        fake_collection = object()

        fake_client = MagicMock()
        fake_client.get_or_create_collection.return_value = fake_collection

        with tempfile.TemporaryDirectory() as td:
            memfile = os.path.join(td, "memory.json")

            with patch.object(framework_mod, "chromadb") as chromadb_mock, patch.object(
                framework_mod.DealAgentFramework, "MEMORY_FILENAME", memfile
            ), patch.object(framework_mod, "AutonomousPlanningAgent") as AutonomousMock, patch.dict(
                os.environ, {"PLANNER_MODE": "autonomous"}
            ):
                chromadb_mock.PersistentClient.return_value = fake_client

                opp = Opportunity(
                    deal=Deal(product_description="x", price=10.0, url="u"),
                    estimate=42.0,
                    discount=32.0,
                )
                planner = MagicMock()
                planner.plan.return_value = opp
                AutonomousMock.return_value = planner

                f = framework_mod.DealAgentFramework()
                out = f.run()

                self.assertTrue(out)  # memory list should not be empty
                self.assertEqual(out[-1].discount, 32.0)

                # Memory file should have been written
                self.assertTrue(os.path.exists(memfile))


if __name__ == "__main__":
    unittest.main()

