"""
FR2: DBSCAN Clustering

Groups event embeddings into clusters using DBSCAN.
Parameters (eps, min_samples) are configurable via config.py.
"""

import logging
from typing import List, Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from config import ClusteringConfig
from models import Event

logger = logging.getLogger(__name__)


class ClusterEngine:
    """Clusters event embeddings using DBSCAN."""

    def __init__(self, eps: float = None, min_samples: int = None):
        self.eps = eps or ClusteringConfig.DBSCAN_EPS
        self.min_samples = min_samples or ClusteringConfig.DBSCAN_MIN_SAMPLES

    def cluster(
        self, embeddings: np.ndarray, events: List[Event]
    ) -> Dict[int, List[Event]]:
        """
        Cluster embeddings and return a mapping of cluster_label -> events.

        Noise points (label=-1) are excluded from the output.

        Args:
            embeddings: np.ndarray of shape (n_events, embedding_dim)
            events: List of Event objects, same order as embeddings

        Returns:
            Dict mapping cluster label (int) to list of Event objects in that cluster
        """
        if len(embeddings) == 0:
            return {}

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        labels = dbscan.fit_predict(embeddings)

        clusters: Dict[int, List[Event]] = {}
        noise_count = 0

        for label, event in zip(labels, events):
            if label == -1:
                noise_count += 1
                continue
            clusters.setdefault(label, []).append(event)

        n_clusters = len(clusters)
        logger.info(
            f"DBSCAN (eps={self.eps}, min_samples={self.min_samples}): "
            f"{n_clusters} clusters found, {noise_count} noise points excluded"
        )
        return clusters

    def cluster_with_embeddings(
        self, embeddings: np.ndarray, events: List[Event]
    ) -> Tuple[Dict[int, List[Event]], Dict[int, np.ndarray]]:
        """
        Cluster and also return per-cluster embeddings (for coherence computation).

        Returns:
            Tuple of (label_to_events, label_to_embeddings)
        """
        if len(embeddings) == 0:
            return {}, {}

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        labels = dbscan.fit_predict(embeddings)

        label_to_events: Dict[int, List[Event]] = {}
        label_to_embs: Dict[int, List[np.ndarray]] = {}
        noise_count = 0

        for label, event, emb in zip(labels, events, embeddings):
            if label == -1:
                noise_count += 1
                continue
            label_to_events.setdefault(label, []).append(event)
            label_to_embs.setdefault(label, []).append(emb)

        # Convert lists to arrays
        label_to_embeddings = {
            label: np.array(embs) for label, embs in label_to_embs.items()
        }

        n_clusters = len(label_to_events)
        logger.info(
            f"DBSCAN (eps={self.eps}, min_samples={self.min_samples}): "
            f"{n_clusters} clusters found, {noise_count} noise points excluded"
        )
        return label_to_events, label_to_embeddings
