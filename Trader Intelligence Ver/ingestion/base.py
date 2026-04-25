"""
Base ingestor abstract class.

To add a new data source, subclass BaseIngestor and implement fetch().
The ingest() method handles deduplication and DB insertion automatically.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List

from models import Event
from db.connection import insert_event

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """SHA256 hash of normalized text for deduplication."""
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class BaseIngestor(ABC):
    """Abstract base class for all data source ingestors."""

    source_type: str = ""  # Override in subclass: "rss", "gdelt", "market"

    @abstractmethod
    def fetch(self) -> List[Event]:
        """Fetch events from the data source. Must be implemented by subclass."""
        pass

    def ingest(self) -> int:
        """Fetch, deduplicate, and insert events into DB. Returns count of new events."""
        events = self.fetch()
        new_count = 0

        for event in events:
            if not event.content_hash:
                event.content_hash = compute_content_hash(event.content)

            event_id = insert_event(event)
            if event_id is not None:
                new_count += 1

        logger.info(
            f"[{self.__class__.__name__}] Fetched {len(events)} events, "
            f"inserted {new_count} new (deduplicated {len(events) - new_count})"
        )
        return new_count
