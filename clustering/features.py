"""
FR2: Cluster Feature Computation

Computes mention velocity, source diversity, and recency for each cluster.
These features are used by FR6 (scoring) downstream.
"""

import logging
from datetime import datetime
from typing import List

from models import Event, Cluster, ClusterFeatures
from config import ClusteringConfig

logger = logging.getLogger(__name__)


def compute_cluster_features(events: List[Event]) -> ClusterFeatures:
    """
    Compute features for a single cluster.

    - mention_velocity: number of events / time span in hours (higher = faster growing)
    - source_diversity: number of distinct sources
    - recency: hours since the most recent event (lower = more recent)
    """
    if not events:
        return ClusterFeatures()

    timestamps = [e.timestamp for e in events]
    sources = {e.source for e in events}

    # Mention velocity: events per hour
    time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
    if time_span < 0.01:  # avoid division by near-zero
        mention_velocity = float(len(events))
    else:
        mention_velocity = len(events) / time_span

    # Source diversity: unique source count
    source_diversity = len(sources)

    # Recency: hours since most recent event
    now = datetime.utcnow()
    most_recent = max(timestamps)
    recency = (now - most_recent).total_seconds() / 3600

    return ClusterFeatures(
        mention_velocity=round(mention_velocity, 4),
        source_diversity=source_diversity,
        recency=round(recency, 4),
    )


def build_clusters(
    label_to_events: dict,
    min_mentions: int = None,
) -> List[Cluster]:
    """
    Build Cluster objects with computed features, filtering by minimum mention count.

    Args:
        label_to_events: Dict mapping cluster label to list of Event objects
        min_mentions: Minimum events required (from config if not specified)

    Returns:
        List of Cluster objects passing the threshold
    """
    min_mentions = min_mentions or ClusteringConfig.CLUSTER_MIN_MENTIONS
    clusters = []

    for label, events in label_to_events.items():
        if len(events) < min_mentions:
            logger.debug(f"Cluster {label}: {len(events)} events < {min_mentions} threshold, skipping")
            continue

        features = compute_cluster_features(events)
        cluster = Cluster(events=events, features=features, label=label)
        clusters.append(cluster)

        logger.info(
            f"Cluster {label}: {len(events)} events, "
            f"velocity={features.mention_velocity}, "
            f"sources={features.source_diversity}, "
            f"recency={features.recency:.1f}h"
        )

    logger.info(f"Built {len(clusters)} clusters (filtered from {len(label_to_events)} by min_mentions={min_mentions})")
    return clusters
