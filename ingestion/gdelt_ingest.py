"""
FR1: GDELT Ingestion

Queries the GDELT DOC 2.0 API (free, no key required) for recent articles
matching configured search terms.
"""

import logging
from datetime import datetime
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


class GDELTIngestor(BaseIngestor):
    source_type = "gdelt"

    def fetch(self) -> List[Event]:
        events = []

        try:
            params = {
                "query": IngestionConfig.GDELT_QUERY,
                "mode": "ArtList",
                "maxrecords": IngestionConfig.GDELT_MAX_RECORDS,
                "format": "json",
            }
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            articles = data.get("articles", [])
            for article in articles:
                title = article.get("title", "")
                url = article.get("url", "")
                source_name = article.get("domain", "gdelt")
                date_str = article.get("seendate", "")

                if date_str:
                    try:
                        timestamp = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ")
                    except ValueError:
                        timestamp = datetime.utcnow()
                else:
                    timestamp = datetime.utcnow()

                content = title  # GDELT DOC API returns titles, not full text

                events.append(Event(
                    title=title,
                    content=content,
                    source=source_name,
                    source_type=self.source_type,
                    url=url,
                    timestamp=timestamp,
                    content_hash=compute_content_hash(content),
                    signal_role="discovery",
                ))

            logger.info(f"GDELT: Retrieved {len(articles)} articles")

        except Exception as e:
            logger.warning(f"GDELT: Failed to fetch: {e}")

        return events
