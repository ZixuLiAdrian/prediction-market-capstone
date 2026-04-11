"""
FR3: Event Extractor

Orchestrates the LLM extraction process:
1. Takes a Cluster object with its computed features
2. Constructs prompt from cluster content + FR2 metadata
3. Calls LLMClient with schema enforcement
4. Returns a validated ExtractedEvent (market-ready event spec)
"""

import logging
from typing import List, Optional

from models import Cluster, ExtractedEvent
from extraction.llm_client import LLMClient
from extraction.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from extraction.schema import EXTRACTED_EVENT_SCHEMA

logger = logging.getLogger(__name__)


class EventExtractor:
    """Extracts structured, market-ready event specifications from clusters using an LLM."""

    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()

    def extract(self, cluster: Cluster, cluster_id: int) -> Optional[ExtractedEvent]:
        """
        Extract a structured event from a cluster.

        Args:
            cluster: Cluster object containing events and computed features.
            cluster_id: Database ID of the cluster (for foreign key).

        Returns:
            ExtractedEvent if successful, None if extraction fails.
        """
        headlines = [e.content for e in cluster.events if e.content]
        sources = [e.source for e in cluster.events]

        if not headlines:
            logger.warning(f"Cluster {cluster_id}: No content to extract from")
            return None

        user_prompt = build_extraction_user_prompt(
            headlines, sources, features=cluster.features
        )

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
                event_type=result.get("event_type", "other"),
                outcome_variable=result.get("outcome_variable", ""),
                candidate_deadlines=result.get("candidate_deadlines", []),
                resolution_sources=result.get("resolution_sources", []),
                tradability=result.get("tradability", "suitable"),
                rejection_reason=result.get("rejection_reason", ""),
                confidence=result.get("confidence", 0.5),
                market_angle=result.get("market_angle", ""),
                contradiction_flag=result.get("contradiction_flag", False),
                contradiction_details=result.get("contradiction_details", ""),
                time_horizon=result.get("time_horizon", ""),
                resolution_hints=result.get("resolution_hints", []),
                raw_llm_response=str(result),
            )

            tradability_str = f"[{extracted.tradability}]" if extracted.tradability == "unsuitable" else ""
            logger.info(
                f"Cluster {cluster_id}: Extracted event — "
                f"type={extracted.event_type}, "
                f"{len(result['entities'])} entities, "
                f"confidence={extracted.confidence:.2f}, "
                f"horizon={extracted.time_horizon} "
                f"{tradability_str}"
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

        suitable = sum(1 for r in results if r.tradability == "suitable")
        logger.info(
            f"Batch extraction: {len(results)}/{len(clusters)} clusters extracted "
            f"({suitable} suitable, {len(results) - suitable} unsuitable)"
        )
        return results
