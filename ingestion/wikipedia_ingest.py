"""
FR1: Wikipedia Pageviews Ingestion

Uses the Wikimedia Pageviews API to detect attention spikes.
Pages with pageview counts above a threshold are ingested as attention signals.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

WIKI_PAGEVIEWS_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access"


class WikipediaIngestor(BaseIngestor):
    source_type = "social"

    def fetch(self) -> List[Event]:
        events = []

        # Fetch yesterday's top pageviews
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        date_str = yesterday.strftime("%Y/%m/%d")

        try:
            url = f"{WIKI_PAGEVIEWS_API}/{date_str}"
            resp = requests.get(
                url,
                headers={"User-Agent": "prediction-market-pipeline/1.0"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            articles = data.get("items", [{}])[0].get("articles", [])

            for article in articles:
                title = article.get("article", "").replace("_", " ")
                views = article.get("views", 0)

                # Skip meta pages
                if title.startswith("Special:") or title.startswith("Main Page"):
                    continue

                # Only capture significant spikes
                if views < IngestionConfig.WIKI_PAGEVIEW_THRESHOLD:
                    continue

                content = f"Wikipedia attention spike: {title} received {views:,} pageviews"

                events.append(Event(
                    title=title,
                    content=content,
                    source="wikipedia",
                    source_type=self.source_type,
                    url=f"https://en.wikipedia.org/wiki/{article.get('article', '')}",
                    timestamp=yesterday,
                    content_hash=compute_content_hash(f"wiki-{title}-{date_str}"),
                    signal_role="attention",
                ))

            logger.info(f"Wikipedia: Found {len(events)} pages above {IngestionConfig.WIKI_PAGEVIEW_THRESHOLD} views")

        except Exception as e:
            logger.warning(f"Wikipedia: Failed to fetch pageviews: {e}")

        return events
