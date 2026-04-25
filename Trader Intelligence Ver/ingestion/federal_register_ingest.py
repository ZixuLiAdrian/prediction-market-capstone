"""
FR1: Federal Register Ingestion

Fetches recent documents from the Federal Register API (free, no key required).
Covers proposed rules, final rules, notices, and presidential documents.
Great for policy/regulatory prediction markets with clean resolution paths.
"""

import logging
from datetime import datetime, timezone
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

FED_REGISTER_API = "https://www.federalregister.gov/api/v1/documents.json"


class FederalRegisterIngestor(BaseIngestor):
    """Ingestor that fetches recent proposed rules, final rules, and notices from the Federal Register API."""

    source_type = "official"

    def fetch(self) -> List[Event]:
        """Fetch the newest Federal Register documents matching the configured query and return them as Events."""
        events = []

        try:
            params = {
                "per_page": IngestionConfig.FED_REGISTER_PER_PAGE,
                "order": "newest",
                "fields[]": ["title", "abstract", "document_number",
                             "publication_date", "type", "agencies", "html_url"],
            }
            if IngestionConfig.FED_REGISTER_QUERY:
                params["conditions[term]"] = IngestionConfig.FED_REGISTER_QUERY

            resp = requests.get(FED_REGISTER_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for doc in data.get("results", []):
                title = doc.get("title", "")
                abstract = doc.get("abstract", "") or ""
                doc_type = doc.get("type", "")
                agencies = [a.get("name", "") for a in doc.get("agencies", []) if a.get("name")]
                pub_date = doc.get("publication_date", "")
                url = doc.get("html_url", "")

                content = f"[{doc_type}] {title}"
                if abstract:
                    content += f". {abstract[:500]}"
                if agencies:
                    content += f" (Agencies: {', '.join(agencies)})"

                timestamp = datetime.now(timezone.utc)
                if pub_date:
                    try:
                        timestamp = datetime.strptime(pub_date, "%Y-%m-%d")
                    except ValueError:
                        pass

                events.append(Event(
                    title=title,
                    content=content,
                    source="federal_register",
                    source_type=self.source_type,
                    url=url,
                    entities=", ".join(agencies),
                    timestamp=timestamp,
                    content_hash=compute_content_hash(content),
                    signal_role="resolution",
                ))

            logger.info(f"Federal Register: Retrieved {len(data.get('results', []))} documents")

        except Exception as e:
            logger.warning(f"Federal Register: Failed to fetch: {e}")

        return events
