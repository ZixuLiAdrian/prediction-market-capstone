"""
Shared data models used across the pipeline.

These dataclasses define the contract between pipeline stages:
- FR1 (Ingestion) produces Event objects
- FR2 (Clustering) produces Cluster objects containing Events
- FR3 (Extraction) produces ExtractedEvent objects from Clusters
- FR4+ (downstream) consumes ExtractedEvent to generate questions, validate, score
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Event:
    """A single ingested event/article from any data source."""
    content: str
    source: str                           # e.g. "reuters", "gdelt", "polymarket"
    source_type: str                      # "rss", "gdelt", "market"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    url: str = ""
    title: str = ""
    entities: str = ""                    # comma-separated entity names
    content_hash: str = ""                # SHA256 for deduplication
    id: Optional[int] = None             # DB-assigned


@dataclass
class ClusterFeatures:
    """Computed features for a cluster, used by FR6 scoring downstream."""
    mention_velocity: float = 0.0         # mentions per hour
    source_diversity: int = 0             # number of unique sources
    recency: float = 0.0                  # hours since most recent event


@dataclass
class Cluster:
    """A group of related events identified by embedding similarity."""
    events: List[Event] = field(default_factory=list)
    features: ClusterFeatures = field(default_factory=ClusterFeatures)
    label: int = -1                       # DBSCAN cluster label
    id: Optional[int] = None


@dataclass
class ExtractedEvent:
    """
    Structured event representation produced by LLM extraction (FR3).

    THIS IS THE HANDOFF CONTRACT FOR FR4:
    Downstream modules (question generation, validation, scoring) consume this object.
    """
    cluster_id: int
    event_summary: str                    # one-paragraph event description
    entities: List[str]                   # key people, orgs, countries
    time_horizon: str                     # e.g. "2-4 weeks", "by June 2025"
    resolution_hints: List[str]           # possible resolution criteria
    id: Optional[int] = None
    raw_llm_response: Optional[str] = None  # for debugging
