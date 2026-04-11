"""
FR1: RSS Feed Ingestion

Pulls headlines and summaries from configured RSS feeds.
Feed URLs are loaded from config — no hardcoded URLs in this file.
"""

import logging
from datetime import datetime
from typing import List

import feedparser

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)


class RSSIngestor(BaseIngestor):
    source_type = "rss"

    def fetch(self) -> List[Event]:
        events = []

        for feed_url in IngestionConfig.RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                source_name = feed.feed.get("title", feed_url).split(" - ")[0].strip()

                for entry in feed.entries:
                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    content = f"{title}. {summary}" if summary else title
                    link = entry.get("link", "")

                    # Parse publication date
                    published = entry.get("published_parsed") or entry.get("updated_parsed")
                    if published:
                        timestamp = datetime(*published[:6])
                    else:
                        timestamp = datetime.utcnow()

                    events.append(Event(
                        title=title,
                        content=content,
                        source=source_name.lower().replace(" ", "_"),
                        source_type=self.source_type,
                        url=link,
                        timestamp=timestamp,
                        content_hash=compute_content_hash(content),
                        signal_role="discovery",
                    ))

                logger.info(f"RSS: Parsed {len(feed.entries)} entries from {source_name}")

            except Exception as e:
                logger.warning(f"RSS: Failed to fetch {feed_url}: {e}")

        return events
