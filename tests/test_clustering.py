"""Tests for FR2: Event Clustering."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
import numpy as np

from clustering.cluster import ClusterEngine
from clustering.features import compute_cluster_features
from models import Event, ClusterFeatures


def _make_events(n, source_prefix="src", hours_ago_start=0):
    """Helper to create test events."""
    now = datetime.utcnow()
    return [
        Event(
            content=f"Event {i} about topic",
            source=f"{source_prefix}_{i % 3}",
            source_type="rss",
            timestamp=now - timedelta(hours=hours_ago_start + i),
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
    # Should find at least 1 cluster (likely 2)
    assert len(result) >= 1


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
        Event(content="a", source="reuters", source_type="rss", timestamp=datetime.utcnow()),
        Event(content="b", source="bbc", source_type="rss", timestamp=datetime.utcnow()),
        Event(content="c", source="reuters", source_type="rss", timestamp=datetime.utcnow()),
    ]
    features = compute_cluster_features(events)
    assert features.source_diversity == 2  # reuters and bbc


def test_empty_events_features():
    """Empty event list should return default features."""
    features = compute_cluster_features([])
    assert features.mention_velocity == 0.0
    assert features.source_diversity == 0
    assert features.recency == 0.0
