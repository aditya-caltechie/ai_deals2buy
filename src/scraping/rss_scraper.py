from __future__ import annotations

import time
from typing import Dict, List, Self

import feedparser
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from scraping.html_parser import extract_deal_snippet

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
        stuff = requests.get(self.url).content
        soup = BeautifulSoup(stuff, "html.parser")
        content = soup.find("div", class_="content-section").get_text()
        content = content.replace("\nmore", "").replace("\n", " ")
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
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
                deals.append(cls(entry))
                time.sleep(0.05)
        return deals

