"""
FR1: Prediction Market Ingestion

Fetches existing market listings from Polymarket's public API.
This provides context on what markets already exist, helping the system
avoid generating duplicate questions and identify trending topics.
"""

import logging
from datetime import datetime
from typing import List

import requests

from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

POLYMARKET_API = "https://gamma-api.polymarket.com/markets"


class MarketIngestor(BaseIngestor):
    source_type = "market"

    def fetch(self) -> List[Event]:
        events = []

        try:
            params = {
                "limit": 100,
                "active": "true",
                "order": "volume",
                "ascending": "false",
            }
            resp = requests.get(POLYMARKET_API, params=params, timeout=30)
            resp.raise_for_status()
            markets = resp.json()

            for market in markets:
                question = market.get("question", "")
                description = market.get("description", "")
                content = f"{question}. {description}" if description else question
                end_date = market.get("endDate", "")
                slug = market.get("slug", "")
                url = f"https://polymarket.com/event/{slug}" if slug else ""

                if end_date:
                    try:
                        timestamp = datetime.fromisoformat(end_date.replace("Z", "+00:00")).replace(tzinfo=None)
                    except ValueError:
                        timestamp = datetime.utcnow()
                else:
                    timestamp = datetime.utcnow()

                events.append(Event(
                    title=question,
                    content=content,
                    source="polymarket",
                    source_type=self.source_type,
                    url=url,
                    timestamp=timestamp,
                    content_hash=compute_content_hash(content),
                    signal_role="benchmark",
                ))

            logger.info(f"Market: Retrieved {len(markets)} active markets from Polymarket")

        except Exception as e:
            logger.warning(f"Market: Failed to fetch from Polymarket: {e}")

        return events
