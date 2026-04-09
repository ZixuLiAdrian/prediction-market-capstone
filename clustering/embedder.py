"""
FR2: Text Embedding

Wraps sentence-transformers to produce dense vector representations of event text.
Model name is configurable via config.py.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import ClusteringConfig

logger = logging.getLogger(__name__)


class Embedder:
    """Embeds text using a sentence-transformer model."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or ClusteringConfig.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into dense vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        logger.info(f"Embedded {len(texts)} texts -> shape {embeddings.shape}")
        return embeddings
