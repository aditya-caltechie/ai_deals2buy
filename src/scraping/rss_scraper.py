from __future__ import annotations

import logging
import time
from typing import Dict, List, Self

import feedparser
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from scraping.html_parser import extract_deal_snippet

logger = logging.getLogger(__name__)

feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]

# You could also add: "https://www.dealnews.com/c238/Automotive/?rss=1"
# "https://www.dealnews.com/c196/Home-Garden/?rss=1"


class ScrapedDeal:
    """
    A deal retrieved from an RSS feed (plus its fetched deal page).
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        self.title = entry["title"]
        self.summary = extract_deal_snippet(entry["summary"])
        self.url = entry["links"][0]["href"]
        self.details = ""
        self.features = ""

        # DealNews page layouts change over time; avoid crashing the whole run when
        # a specific selector isn't present or a request fails.
        try:
            resp = requests.get(
                self.url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) ai-deals2buy/1.0"
                },
                timeout=20,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")

            # Historically this lived in: <div class="content-section">...</div>
            node = soup.find("div", class_="content-section")
            if node is None:
                # Try common content containers before falling back to a full-page scrape.
                node = soup.find("main") or soup.find("article")

            content = (
                node.get_text(" ", strip=True)
                if node is not None
                else soup.get_text(" ", strip=True)
            )
            content = content.replace(" more", " ").replace("\n", " ").strip()

            if not content:
                # Fallback: at least include RSS summary so LLM can still select deals.
                content = self.summary

            if "Features" in content:
                self.details, self.features = content.split("Features", 1)
            else:
                self.details = content
                self.features = ""
        except Exception as e:
            logger.warning("Failed to scrape deal page %s (%s). Using RSS summary.", self.url, e)
            self.details = self.summary
            self.features = ""

        self.truncate()

    def truncate(self):
        """
        Limit the fields to a sensible length to avoid sending too much info to the model.
        """
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self):
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model.
        """
        return (
            f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"
        )

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve all deals from the selected RSS feeds.
        """
        deals: list[ScrapedDeal] = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                try:
                    deals.append(cls(entry))
                except Exception as e:
                    logger.warning(
                        "Skipping RSS entry due to scrape/parse failure (%s): %s",
                        feed_url,
                        e,
                    )
                time.sleep(0.05)
        return deals

