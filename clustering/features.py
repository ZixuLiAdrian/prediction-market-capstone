"""
FR2: Cluster Feature Computation

Computes mention velocity, source diversity, recency, source_role_mix,
coherence_score, and weighted_mention_velocity for each cluster.
Also includes near-duplicate detection to prevent inflated metrics.
"""

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from models import Event, Cluster, ClusterFeatures
from config import ClusteringConfig

logger = logging.getLogger(__name__)


def deduplicate_near_duplicates(
    events: List[Event],
    embeddings: Optional[np.ndarray] = None,
    threshold: float = None,
) -> tuple[List[Event], Optional[np.ndarray]]:
    """
    Remove near-duplicate events from a list using embedding similarity.
    Keeps the first occurrence. If no embeddings provided, returns events as-is.

    Returns (deduped_events, deduped_embeddings). The embeddings array is
    filtered with the same mask so indices stay aligned.

    This prevents syndicated headlines and copied market descriptions
    from inflating mention velocity and cluster size.
    """
    if embeddings is None or len(embeddings) == 0 or len(events) <= 1:
        return events, embeddings

    threshold = threshold or ClusteringConfig.NEAR_DUPLICATE_THRESHOLD
    keep_mask = [True] * len(events)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms

    for i in range(len(events)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(events)):
            if not keep_mask[j]:
                continue
            sim = float(np.dot(normed[i], normed[j]))
            if sim >= threshold:
                keep_mask[j] = False

    mask_arr = np.array(keep_mask)
    deduped = [e for e, keep in zip(events, keep_mask) if keep]
    deduped_emb = embeddings[mask_arr]
    removed = len(events) - len(deduped)
    if removed > 0:
        logger.debug(f"Near-dedup: removed {removed} near-duplicate events (threshold={threshold})")
    return deduped, deduped_emb


def compute_cluster_coherence(embeddings: np.ndarray) -> float:
    """
    Compute average pairwise cosine similarity within a cluster.
    Higher values mean more semantically coherent cluster.
    Returns 0.0 for empty or single-element clusters.
    """
    if embeddings is None or len(embeddings) < 2:
        return 0.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms

    # Compute pairwise cosine similarity matrix
    sim_matrix = np.dot(normed, normed.T)

    # Extract upper triangle (exclude diagonal)
    n = len(embeddings)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    return float(np.mean(pairwise_sims))


def compute_source_role_mix(events: List[Event]) -> dict:
    """Count events by signal_role."""
    roles = Counter(e.signal_role for e in events)
    return dict(roles)


def compute_weighted_mention_velocity(events: List[Event]) -> float:
    """
    Compute source-weight-adjusted mentions per hour.
    Official sources count more than social posts.
    """
    if not events:
        return 0.0

    weights = ClusteringConfig.SOURCE_WEIGHTS
    default_weight = ClusteringConfig.DEFAULT_SOURCE_WEIGHT

    total_weight = sum(weights.get(e.source, default_weight) for e in events)

    timestamps = [e.timestamp for e in events]
    time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
    if time_span < 0.01:
        return total_weight
    return round(total_weight / time_span, 4)


def compute_cluster_features(
    events: List[Event],
    embeddings: Optional[np.ndarray] = None,
) -> ClusterFeatures:
    """
    Compute features for a single cluster.

    - mention_velocity: number of events / time span in hours
    - source_diversity: number of distinct sources
    - recency: hours since the most recent event (lower = more recent)
    - source_role_mix: count of events by signal_role
    - coherence_score: avg pairwise embedding similarity
    - weighted_mention_velocity: source-weight-adjusted mentions per hour
    """
    if not events:
        return ClusterFeatures()

    timestamps = [e.timestamp for e in events]
    sources = {e.source for e in events}

    # Mention velocity: events per hour
    time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
    if time_span < 0.01:
        mention_velocity = float(len(events))
    else:
        mention_velocity = len(events) / time_span

    # Source diversity: unique source count
    source_diversity = len(sources)

    # Recency: hours since most recent event
    now = datetime.now(timezone.utc)
    most_recent = max(timestamps)
    recency = (now - most_recent).total_seconds() / 3600

    # Source role mix
    source_role_mix = compute_source_role_mix(events)

    # Coherence score
    coherence_score = compute_cluster_coherence(embeddings)

    # Weighted mention velocity
    weighted_mention_velocity = compute_weighted_mention_velocity(events)

    return ClusterFeatures(
        mention_velocity=round(mention_velocity, 4),
        source_diversity=source_diversity,
        recency=round(recency, 4),
        source_role_mix=source_role_mix,
        coherence_score=round(coherence_score, 4),
        weighted_mention_velocity=weighted_mention_velocity,
    )


def build_clusters(
    label_to_events: dict,
    min_mentions: int = None,
    label_to_embeddings: dict = None,
) -> List[Cluster]:
    """
    Build Cluster objects with computed features, filtering by minimum mention count.

    Args:
        label_to_events: Dict mapping cluster label to list of Event objects
        min_mentions: Minimum events required (from config if not specified)
        label_to_embeddings: Optional dict mapping cluster label to np.ndarray of embeddings

    Returns:
        List of Cluster objects passing the threshold
    """
    min_mentions = min_mentions or ClusteringConfig.CLUSTER_MIN_MENTIONS
    clusters = []

    for label, events in label_to_events.items():
        if len(events) < min_mentions:
            logger.debug(f"Cluster {label}: {len(events)} events < {min_mentions} threshold, skipping")
            continue

        cluster_embeddings = None
        if label_to_embeddings and label in label_to_embeddings:
            cluster_embeddings = label_to_embeddings[label]

        features = compute_cluster_features(events, cluster_embeddings)
        cluster = Cluster(events=events, features=features, label=label)
        clusters.append(cluster)

        logger.info(
            f"Cluster {label}: {len(events)} events, "
            f"velocity={features.mention_velocity}, "
            f"sources={features.source_diversity}, "
            f"recency={features.recency:.1f}h, "
            f"coherence={features.coherence_score:.3f}, "
            f"roles={features.source_role_mix}"
        )

    logger.info(f"Built {len(clusters)} clusters (filtered from {len(label_to_events)} by min_mentions={min_mentions})")
    return clusters
