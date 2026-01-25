from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests._testutils import add_src_to_syspath


add_src_to_syspath()


class TestAgents(unittest.TestCase):
    def test_scanner_agent_uses_openai_structured_outputs(self):
        from agents.scanners.scanner_agent import ScannerAgent
        from scraping.rss_scraper import ScrapedDeal
        from data.models import DealSelection

        # Fake scraped deal list (avoid network)
        fake_deal = SimpleNamespace(
            url="https://example.com/deal",
            describe=lambda: "Title: X\nDetails: Y\nFeatures: Z\nURL: https://example.com/deal",
        )

        # Fake OpenAI structured output response object shape
        parsed = DealSelection(
            deals=[
                {"product_description": "prod", "price": 12.0, "url": "https://x"},
                {"product_description": "prod2", "price": 20.0, "url": "https://y"},
                {"product_description": "prod3", "price": 30.0, "url": "https://z"},
                {"product_description": "prod4", "price": 40.0, "url": "https://a"},
                {"product_description": "prod5", "price": 50.0, "url": "https://b"},
            ]
        )
        fake_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

        fake_openai = MagicMock()
        fake_openai.chat.completions.parse.return_value = fake_response

        with patch("agents.scanners.scanner_agent.OpenAI", return_value=fake_openai), patch.object(
            ScannerAgent, "fetch_deals", return_value=[fake_deal]
        ):
            agent = ScannerAgent()
            result = agent.scan(memory=[])

        self.assertIsInstance(result, DealSelection)
        fake_openai.chat.completions.parse.assert_called()

    def test_frontier_agent_price_parses_number(self):
        from agents.pricing.frontier_agent import FrontierAgent

        fake_collection = object()

        fake_openai = MagicMock()
        fake_openai.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="$123.45"))]
        )

        with patch("agents.pricing.frontier_agent.OpenAI", return_value=fake_openai), patch(
            "agents.pricing.frontier_agent.ChromaRetriever"
        ) as RetrieverMock:
            RetrieverMock.return_value.query_similars.return_value = (["doc1"], [99.0])

            agent = FrontierAgent(fake_collection)
            price = agent.price("some description")

        self.assertAlmostEqual(price, 123.45, places=2)

    def test_messaging_agent_sends_pushover(self):
        from agents.messaging.messaging_agent import MessagingAgent

        fake_completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))]
        )

        with patch("agents.messaging.messaging_agent.completion", return_value=fake_completion), patch(
            "agents.messaging.messaging_agent.PushoverClient"
        ) as PushoverMock, patch.dict(
            os.environ,
            {"PUSHOVER_USER": "u", "PUSHOVER_TOKEN": "t"},
        ):
            pushover = MagicMock()
            PushoverMock.return_value = pushover

            agent = MessagingAgent()
            agent.notify("desc", 10.0, 20.0, "https://deal")

            pushover.send.assert_called()


if __name__ == "__main__":
    unittest.main()

