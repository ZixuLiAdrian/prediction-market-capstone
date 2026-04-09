"""
FR3: Event Extractor

Orchestrates the LLM extraction process:
1. Takes a Cluster object
2. Constructs prompt from cluster content
3. Calls LLMClient with schema enforcement
4. Returns a validated ExtractedEvent
"""

import logging
from typing import List, Optional

from models import Cluster, ExtractedEvent
from extraction.llm_client import LLMClient
from extraction.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from extraction.schema import EXTRACTED_EVENT_SCHEMA

logger = logging.getLogger(__name__)


class EventExtractor:
    """Extracts structured event representations from clusters using an LLM."""

    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()

    def extract(self, cluster: Cluster, cluster_id: int) -> Optional[ExtractedEvent]:
        """
        Extract a structured event from a cluster.

        Args:
            cluster: Cluster object containing events.
            cluster_id: Database ID of the cluster (for foreign key).

        Returns:
            ExtractedEvent if successful, None if extraction fails.
        """
        headlines = [e.content for e in cluster.events if e.content]
        sources = [e.source for e in cluster.events]

        if not headlines:
            logger.warning(f"Cluster {cluster_id}: No content to extract from")
            return None

        user_prompt = build_extraction_user_prompt(headlines, sources)

        try:
            result = self.llm_client.call(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=EXTRACTED_EVENT_SCHEMA,
            )

            extracted = ExtractedEvent(
                cluster_id=cluster_id,
                event_summary=result["event_summary"],
                entities=result["entities"],
                time_horizon=result["time_horizon"],
                resolution_hints=result["resolution_hints"],
                raw_llm_response=str(result),
            )

            logger.info(
                f"Cluster {cluster_id}: Extracted event — "
                f"{len(result['entities'])} entities, "
                f"horizon={result['time_horizon']}"
            )
            return extracted

        except RuntimeError as e:
            logger.error(f"Cluster {cluster_id}: Extraction failed — {e}")
            return None

    def extract_batch(self, clusters: List[dict]) -> List[ExtractedEvent]:
        """
        Extract events from multiple clusters.

        Args:
            clusters: List of dicts from db.connection.get_clusters_for_extraction()
                      Each dict has: cluster_id, label, features, events

        Returns:
            List of successfully extracted ExtractedEvent objects.
        """
        results = []

        for cluster_data in clusters:
            cluster = Cluster(
                events=cluster_data["events"],
                features=cluster_data["features"],
                label=cluster_data["label"],
            )
            extracted = self.extract(cluster, cluster_data["cluster_id"])
            if extracted:
                results.append(extracted)

        logger.info(f"Batch extraction: {len(results)}/{len(clusters)} clusters extracted successfully")
        return results
