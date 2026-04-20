"""
FR1: Congress.gov Ingestion

Fetches recent bills and legislation from the Congress.gov API.
Requires a free API key from api.congress.gov.
"""

import logging
from datetime import datetime, timezone
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

CONGRESS_API = "https://api.congress.gov/v3/bill"


class CongressIngestor(BaseIngestor):
    source_type = "official"

    def fetch(self) -> List[Event]:
        events = []

        if not IngestionConfig.CONGRESS_API_KEY:
            logger.warning("Congress: No API key configured, skipping")
            return events

        try:
            params = {
                "api_key": IngestionConfig.CONGRESS_API_KEY,
                "limit": IngestionConfig.CONGRESS_LIMIT,
                "sort": "updateDate+desc",
                "format": "json",
            }
            resp = requests.get(CONGRESS_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for bill in data.get("bills", []):
                title = bill.get("title", "")
                bill_type = bill.get("type", "")
                bill_number = bill.get("number", "")
                latest_action = bill.get("latestAction", {})
                action_text = latest_action.get("text", "")
                action_date = latest_action.get("actionDate", "")
                congress = bill.get("congress", "")
                url = bill.get("url", "")

                content = f"[{bill_type}{bill_number}, {congress}th Congress] {title}"
                if action_text:
                    content += f". Latest action: {action_text}"

                timestamp = datetime.now(timezone.utc)
                if action_date:
                    try:
                        timestamp = datetime.strptime(action_date, "%Y-%m-%d")
                    except ValueError:
                        pass

                events.append(Event(
                    title=title,
                    content=content,
                    source="congress",
                    source_type=self.source_type,
                    url=url,
                    timestamp=timestamp,
                    content_hash=compute_content_hash(content),
                    signal_role="resolution",
                ))

            logger.info(f"Congress: Retrieved {len(data.get('bills', []))} bills")

        except Exception as e:
            logger.warning(f"Congress: Failed to fetch: {e}")

        return events
