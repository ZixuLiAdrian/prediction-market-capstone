"""
FR1: Reddit Ingestion

Fetches hot posts from configured subreddits via the Reddit OAuth API.
Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env.
"""

import logging
from datetime import datetime
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE = "https://oauth.reddit.com"


class RedditIngestor(BaseIngestor):
    source_type = "social"

    def __init__(self):
        self._token = None

    def _authenticate(self):
        """Get OAuth bearer token using client credentials."""
        if not IngestionConfig.REDDIT_CLIENT_ID:
            raise RuntimeError("REDDIT_CLIENT_ID not configured")

        resp = requests.post(
            REDDIT_TOKEN_URL,
            auth=(IngestionConfig.REDDIT_CLIENT_ID, IngestionConfig.REDDIT_CLIENT_SECRET),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": IngestionConfig.REDDIT_USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]

    def fetch(self) -> List[Event]:
        events = []

        try:
            self._authenticate()
        except Exception as e:
            logger.warning(f"Reddit: Authentication failed: {e}")
            return events

        headers = {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": IngestionConfig.REDDIT_USER_AGENT,
        }

        for subreddit in IngestionConfig.REDDIT_SUBREDDITS:
            try:
                url = f"{REDDIT_API_BASE}/r/{subreddit}/hot"
                resp = requests.get(
                    url,
                    headers=headers,
                    params={"limit": IngestionConfig.REDDIT_POST_LIMIT},
                    timeout=15,
                )
                resp.raise_for_status()
                posts = resp.json().get("data", {}).get("children", [])

                for post in posts:
                    data = post.get("data", {})
                    title = data.get("title", "")
                    selftext = data.get("selftext", "")[:500]  # cap body length
                    content = f"{title}. {selftext}" if selftext else title
                    permalink = data.get("permalink", "")
                    created_utc = data.get("created_utc", 0)

                    events.append(Event(
                        title=title,
                        content=content,
                        source="reddit",
                        source_type=self.source_type,
                        url=f"https://reddit.com{permalink}" if permalink else "",
                        timestamp=datetime.utcfromtimestamp(created_utc) if created_utc else datetime.utcnow(),
                        content_hash=compute_content_hash(content),
                        signal_role="attention",
                    ))

                logger.info(f"Reddit: Parsed {len(posts)} posts from r/{subreddit}")

            except Exception as e:
                logger.warning(f"Reddit: Failed to fetch r/{subreddit}: {e}")

        return events
