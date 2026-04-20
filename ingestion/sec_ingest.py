"""
FR1: SEC EDGAR Ingestion

Fetches recent filings from SEC EDGAR's EFTS full-text search API.
Free, no key required, but requires a User-Agent header with contact info.
Focuses on 8-K (material events) and Form 4 (insider trading) by default.
"""

import logging
from datetime import datetime
from typing import List

import requests

from config import IngestionConfig
from models import Event
from ingestion.base import BaseIngestor, compute_content_hash

logger = logging.getLogger(__name__)

SEC_EFTS_API = "https://efts.sec.gov/LATEST/search-index"
SEC_FILINGS_API = "https://efts.sec.gov/LATEST/search-index"
SEC_RECENT_API = "https://www.sec.gov/cgi-bin/browse-edgar"


class SECIngestor(BaseIngestor):
    source_type = "official"

    def fetch(self) -> List[Event]:
        events = []
        headers = {"User-Agent": IngestionConfig.SEC_EDGAR_USER_AGENT}

        for form_type in IngestionConfig.SEC_EDGAR_FORM_TYPES:
            try:
                # Use EDGAR full-text search
                url = "https://efts.sec.gov/LATEST/search-index"
                params = {
                    "q": "*",
                    "dateRange": "custom",
                    "startdt": datetime.utcnow().strftime("%Y-%m-%d"),
                    "forms": form_type,
                }

                # Fallback: use the EDGAR RSS feed for recent filings
                rss_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={form_type}&dateb=&owner=include&count={IngestionConfig.SEC_EDGAR_LIMIT}&search_text=&start=0&output=atom"
                resp = requests.get(rss_url, headers=headers, timeout=30)
                resp.raise_for_status()

                # Parse Atom feed
                from xml.etree import ElementTree as ET
                root = ET.fromstring(resp.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                for entry in root.findall("atom:entry", ns):
                    title = entry.findtext("atom:title", "", ns)
                    summary = entry.findtext("atom:summary", "", ns)
                    link_el = entry.find("atom:link", ns)
                    link = link_el.get("href", "") if link_el is not None else ""
                    updated = entry.findtext("atom:updated", "", ns)

                    content = f"[SEC {form_type}] {title}"
                    if summary:
                        content += f". {summary[:500]}"

                    timestamp = datetime.utcnow()
                    if updated:
                        try:
                            timestamp = datetime.fromisoformat(updated.replace("Z", "+00:00")).replace(tzinfo=None)
                        except ValueError:
                            pass

                    events.append(Event(
                        title=title,
                        content=content,
                        source="sec_edgar",
                        source_type=self.source_type,
                        url=link,
                        timestamp=timestamp,
                        content_hash=compute_content_hash(content),
                        signal_role="resolution",
                    ))

                logger.info(f"SEC: Retrieved {form_type} filings")

            except Exception as e:
                logger.warning(f"SEC: Failed to fetch {form_type}: {e}")

        return events
