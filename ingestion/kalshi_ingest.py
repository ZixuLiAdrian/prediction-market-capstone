"""
FR1: Kalshi Market Ingestion

Fetches active market listings from Kalshi's public market data API.
Some endpoints are available without authentication.
Provides a second benchmark market universe alongside Polymarket.
"""

import logging
from datetime import datetime
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2/markets"


class KalshiIngestor(BaseIngestor):
    source_type = "market"

    def fetch(self) -> List[Event]:
        events = []

        try:
            params = {
                "limit": IngestionConfig.KALSHI_LIMIT,
                "status": "open",
            }
            resp = requests.get(KALSHI_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            markets = data.get("markets", [])

            for market in markets:
                title = market.get("title", "")
                subtitle = market.get("subtitle", "")
                category = market.get("category", "")
                ticker = market.get("ticker", "")
                close_time = market.get("close_time", "")
                url = f"https://kalshi.com/markets/{ticker}" if ticker else ""

                content = title
                if subtitle:
                    content += f". {subtitle}"
                if category:
                    content += f" [Category: {category}]"

                timestamp = datetime.utcnow()
                if close_time:
                    try:
                        timestamp = datetime.fromisoformat(close_time.replace("Z", "+00:00")).replace(tzinfo=None)
                    except ValueError:
                        pass

                events.append(Event(
                    title=title,
                    content=content,
                    source="kalshi",
                    source_type=self.source_type,
                    url=url,
                    timestamp=timestamp,
                    content_hash=compute_content_hash(content),
                    signal_role="benchmark",
                ))

            logger.info(f"Kalshi: Retrieved {len(markets)} active markets")

        except Exception as e:
            logger.warning(f"Kalshi: Failed to fetch: {e}")

        return events
