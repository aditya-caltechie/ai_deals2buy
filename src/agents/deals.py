"""
Backward-compatible shim.

Data models moved to `data.models` and RSS scraping moved to `scraping.rss_scraper`.
"""

from data.models import Deal, DealSelection, Opportunity  # noqa: F401
from scraping.rss_scraper import ScrapedDeal  # noqa: F401
