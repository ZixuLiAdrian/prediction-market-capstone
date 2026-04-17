"""
Shared data models used across the pipeline.

These dataclasses define the contract between pipeline stages:
- FR1 (Ingestion) produces Event objects
- FR2 (Clustering) produces Cluster objects containing Events
- FR3 (Extraction) produces ExtractedEvent objects from Clusters
- FR4 (Question Generation) produces CandidateQuestion objects from ExtractedEvents
- FR5+ (downstream) consumes CandidateQuestion to validate, score, publish
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import List, Optional


@dataclass
class Event:
    """A single ingested event/article from any data source."""
    content: str
    source: str                           # e.g. "reuters", "gdelt", "polymarket"
    source_type: str                      # "rss", "gdelt", "market"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
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


@dataclass
class CandidateQuestion:
    """
    A prediction market question produced by FR4 (LLM Question Generation).

    THIS IS THE HANDOFF CONTRACT FOR FR5:
    Downstream modules (rule validation, scoring, publishing) consume this object.
    """
    extracted_event_id: int
    question_text: str                     # the market question, ends with "?"
    category: str                          # one of 13 categories (politics, finance, etc.)
    question_type: str                     # "binary" or "multiple_choice"
    options: List[str]                     # ["Yes", "No"] or 3-5 labelled options
    deadline: str                          # ISO date string, e.g. "2025-09-30"
    deadline_source: str                   # URL of official schedule confirming the deadline
    resolution_source: str                 # authoritative org + URL for resolution
    resolution_criteria: str               # plain-language logic for each option
    rationale: str                         # why this question is interesting/novel
    id: Optional[int] = None              # DB-assigned
    raw_llm_response: Optional[str] = None  # full LLM response for debugging


@dataclass
class ValidationResult:
    """
    FR5 output for deterministic rule validation of a CandidateQuestion.
    """
    question_id: int
    is_valid: bool
    flags: List[str]
    clarity_score: float
    id: Optional[int] = None


@dataclass
class ScoredCandidate:
    """
    FR6 output for heuristic scoring and ranking of validated questions.
    """
    question_id: int
    total_score: float
    mention_velocity_score: float
    source_diversity_score: float
    clarity_score: float
    novelty_score: float
    rank: int = 0
    id: Optional[int] = None
