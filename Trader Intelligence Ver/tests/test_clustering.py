"""Tests for FR2: Event Clustering."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta, timezone
import numpy as np

from clustering.cluster import ClusterEngine
from clustering.features import (
    compute_cluster_features,
    compute_source_role_mix,
    compute_cluster_coherence,
    compute_weighted_mention_velocity,
    deduplicate_near_duplicates,
)
from models import Event, ClusterFeatures


def _make_events(n, source_prefix="src", hours_ago_start=0, signal_role="discovery"):
    """Helper to create test events."""
    now = datetime.now(timezone.utc)
    return [
        Event(
            content=f"Event {i} about topic",
            source=f"{source_prefix}_{i % 3}",
            source_type="rss",
            timestamp=now - timedelta(hours=hours_ago_start + i),
            signal_role=signal_role,
        )
        for i in range(n)
    ]


def test_cluster_engine_empty():
    """Clustering empty input should return empty dict."""
    engine = ClusterEngine()
    result = engine.cluster(np.array([]), [])
    assert result == {}


def test_cluster_engine_returns_dict():
    """ClusterEngine should return a dict of label -> events."""
    engine = ClusterEngine(eps=0.5, min_samples=2)

    # Create embeddings that form two obvious clusters
    cluster_a = np.tile([1.0, 0.0, 0.0], (5, 1)) + np.random.normal(0, 0.01, (5, 3))
    cluster_b = np.tile([0.0, 1.0, 0.0], (5, 1)) + np.random.normal(0, 0.01, (5, 3))
    embeddings = np.vstack([cluster_a, cluster_b])

    events = _make_events(10)
    result = engine.cluster(embeddings, events)

    assert isinstance(result, dict)
    assert len(result) >= 1


def test_cluster_with_embeddings():
    """cluster_with_embeddings should return both events and embeddings per cluster."""
    engine = ClusterEngine(eps=0.5, min_samples=2)

    cluster_a = np.tile([1.0, 0.0, 0.0], (5, 1)) + np.random.normal(0, 0.01, (5, 3))
    cluster_b = np.tile([0.0, 1.0, 0.0], (5, 1)) + np.random.normal(0, 0.01, (5, 3))
    embeddings = np.vstack([cluster_a, cluster_b])

    events = _make_events(10)
    label_to_events, label_to_embeddings = engine.cluster_with_embeddings(embeddings, events)

    assert isinstance(label_to_events, dict)
    assert isinstance(label_to_embeddings, dict)
    assert set(label_to_events.keys()) == set(label_to_embeddings.keys())

    for label in label_to_events:
        assert len(label_to_events[label]) == label_to_embeddings[label].shape[0]


def test_compute_features_basic():
    """Feature computation should return correct types."""
    events = _make_events(5, hours_ago_start=1)
    features = compute_cluster_features(events)

    assert isinstance(features, ClusterFeatures)
    assert features.mention_velocity > 0
    assert features.source_diversity > 0
    assert features.recency >= 0


def test_source_diversity_count():
    """Source diversity should count unique sources."""
    events = [
        Event(content="a", source="reuters", source_type="rss", timestamp=datetime.now(timezone.utc)),
        Event(content="b", source="bbc", source_type="rss", timestamp=datetime.now(timezone.utc)),
        Event(content="c", source="reuters", source_type="rss", timestamp=datetime.now(timezone.utc)),
    ]
    features = compute_cluster_features(events)
    assert features.source_diversity == 2  # reuters and bbc


def test_empty_events_features():
    """Empty event list should return default features."""
    features = compute_cluster_features([])
    assert features.mention_velocity == 0.0
    assert features.source_diversity == 0
    assert features.recency == 0.0
    assert features.source_role_mix == {}
    assert features.coherence_score == 0.0
    assert features.weighted_mention_velocity == 0.0


def test_source_role_mix():
    """Source role mix should count events by signal_role."""
    events = [
        Event(content="a", source="reuters", source_type="rss", signal_role="discovery", timestamp=datetime.now(timezone.utc)),
        Event(content="b", source="reddit", source_type="social", signal_role="attention", timestamp=datetime.now(timezone.utc)),
        Event(content="c", source="bbc", source_type="rss", signal_role="discovery", timestamp=datetime.now(timezone.utc)),
        Event(content="d", source="sec", source_type="official", signal_role="resolution", timestamp=datetime.now(timezone.utc)),
    ]
    mix = compute_source_role_mix(events)
    assert mix["discovery"] == 2
    assert mix["attention"] == 1
    assert mix["resolution"] == 1


def test_coherence_score_identical():
    """Identical embeddings should have coherence ~1.0."""
    embeddings = np.tile([1.0, 0.0, 0.0], (5, 1))
    score = compute_cluster_coherence(embeddings)
    assert abs(score - 1.0) < 0.01


def test_coherence_score_orthogonal():
    """Orthogonal embeddings should have coherence ~0.0."""
    embeddings = np.eye(5)  # 5 orthogonal vectors
    score = compute_cluster_coherence(embeddings)
    assert abs(score) < 0.01


def test_coherence_score_empty():
    """Empty/single-element should return 0.0."""
    assert compute_cluster_coherence(np.array([])) == 0.0
    assert compute_cluster_coherence(np.array([[1.0, 0.0]])) == 0.0


def test_weighted_mention_velocity():
    """Official sources should produce higher weighted velocity than social sources."""
    now = datetime.now(timezone.utc)
    official_events = [
        Event(content="a", source="federal_register", source_type="official", timestamp=now),
        Event(content="b", source="sec_edgar", source_type="official", timestamp=now - timedelta(hours=1)),
    ]
    social_events = [
        Event(content="a", source="reddit", source_type="social", timestamp=now),
        Event(content="b", source="reddit", source_type="social", timestamp=now - timedelta(hours=1)),
    ]

    official_velocity = compute_weighted_mention_velocity(official_events)
    social_velocity = compute_weighted_mention_velocity(social_events)

    assert official_velocity > social_velocity


def test_near_duplicate_detection():
    """Near-identical embeddings should be deduplicated."""
    events = [
        Event(content="Breaking: Fed cuts rates", source="reuters", source_type="rss", timestamp=datetime.now(timezone.utc)),
        Event(content="Breaking: Fed cuts rates today", source="bbc", source_type="rss", timestamp=datetime.now(timezone.utc)),
        Event(content="Completely different topic about sports", source="espn", source_type="rss", timestamp=datetime.now(timezone.utc)),
    ]
    # Make near-identical embeddings for first two, different for third
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.999, 0.001, 0.0],  # very close to first
        [0.0, 1.0, 0.0],      # very different
    ])

    deduped, deduped_emb = deduplicate_near_duplicates(events, embeddings, threshold=0.99)
    assert len(deduped) == 2  # first and third should remain
    assert deduped_emb.shape[0] == 2  # embeddings filtered to match


def test_near_duplicate_no_embeddings():
    """Without embeddings, should return events unchanged."""
    events = _make_events(5)
    result, result_emb = deduplicate_near_duplicates(events)
    assert len(result) == 5
    assert result_emb is None  # no embeddings provided


def test_features_with_embeddings():
    """Features computation should include coherence when embeddings provided."""
    events = _make_events(5, hours_ago_start=1)
    embeddings = np.tile([1.0, 0.0, 0.0], (5, 1)) + np.random.normal(0, 0.01, (5, 3))

    features = compute_cluster_features(events, embeddings)
    assert features.coherence_score > 0.9  # near-identical embeddings = high coherence
