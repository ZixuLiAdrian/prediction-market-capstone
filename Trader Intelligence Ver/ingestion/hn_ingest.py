"""
FR1: Hacker News Ingestion

Fetches top stories from the Hacker News Firebase API (free, no auth required).
Good for tech, AI, startup, and security events.
"""

import logging
from datetime import datetime, timezone
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

HN_TOP_STORIES = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"


class HackerNewsIngestor(BaseIngestor):
    """Ingestor that fetches top stories from the Hacker News Firebase API."""

    source_type = "social"

    def fetch(self) -> List[Event]:
        """Fetch top HN story IDs then retrieve each item, returning them as attention-signal Events."""
        events = []

        try:
            resp = requests.get(HN_TOP_STORIES, timeout=15)
            resp.raise_for_status()
            story_ids = resp.json()[:IngestionConfig.HN_MAX_STORIES]

            for story_id in story_ids:
                try:
                    item_resp = requests.get(HN_ITEM.format(story_id), timeout=10)
                    item_resp.raise_for_status()
                    item = item_resp.json()

                    if not item or item.get("type") != "story":
                        continue

                    title = item.get("title", "")
                    url = item.get("url", "")
                    content = title
                    created = item.get("time", 0)

                    events.append(Event(
                        title=title,
                        content=content,
                        source="hackernews",
                        source_type=self.source_type,
                        url=url,
                        timestamp=datetime.fromtimestamp(created, tz=timezone.utc) if created else datetime.now(timezone.utc),
                        content_hash=compute_content_hash(content),
                        signal_role="attention",
                    ))

                except Exception as e:
                    logger.debug(f"HN: Failed to fetch item {story_id}: {e}")

            logger.info(f"HN: Retrieved {len(events)} top stories")

        except Exception as e:
            logger.warning(f"HN: Failed to fetch top stories: {e}")

        return events
