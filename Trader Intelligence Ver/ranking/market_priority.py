"""
Shared popularity-bias heuristics for FR3/FR4 selection.

These helpers softly boost clusters and extracted events that look more like
high-demand prediction market topics (macro, crypto, finance, politics,
sports, major news) while down-ranking low-interest procedural/admin stories.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

from models import ClusterFeatures, Event, ExtractedEvent

_TOPIC_KEYWORDS = {
    "macro": (
        "federal reserve", "fed ", "fomc", "interest rate", "rate cut", "rate hike",
        "cpi", "inflation", "pce", "gdp", "recession", "treasury", "yield",
        "unemployment", "nonfarm payroll", "jobs report", "bls", "fred",
    ),
    "crypto": (
        "bitcoin", "btc", "ethereum", "eth", "solana", "xrp", "dogecoin",
        "stablecoin", "defi", "staking", "token", "spot etf", "crypto",
        "coinbase", "binance", "kraken",
    ),
    "finance": (
        "earnings", "revenue", "guidance", "dividend", "ipo", "merger",
        "acquisition", "buyback", "bankruptcy", "antitrust", "stock", "shares",
        "nasdaq", "s&p 500", "dow", "sec filing", "8-k", "10-q", "10-k",
    ),
    "politics": (
        "election", "primary", "white house", "congress", "senate", "house",
        "governor", "president", "prime minister", "parliament", "cabinet",
        "referendum", "biden", "trump", "democrat", "republican",
    ),
    "sports": (
        "nba", "nfl", "mlb", "nhl", "ncaa", "fifa", "uefa", "world cup",
        "super bowl", "playoff", "playoffs", "championship", "grand slam",
        "olympics", "formula 1", "ufc", "wimbledon",
    ),
    "major_news": (
        "tariff", "sanction", "ceasefire", "war", "conflict", "ai", "artificial intelligence",
        "nato", "china", "taiwan", "israel", "gaza", "ukraine", "earthquake",
        "hurricane", "wildfire", "pandemic", "vaccine", "outbreak",
    ),
}

_TOPIC_EVENT_TYPE_PRIORS = {
    "macro_release": 1.00,
    "crypto": 1.00,
    "election": 0.98,
    "earnings": 0.92,
    "merger": 0.92,
    "sports": 0.98,
    "geopolitics": 0.92,
    "policy": 0.72,
    "court_case": 0.75,
    "tech": 0.82,
    "business": 0.82,
    "health": 0.65,
    "science": 0.60,
    "other": 0.45,
}

_MAJOR_NEWS_SOURCES = {
    "reuters", "bbc", "nyt", "new york times", "wsj", "wall street journal",
    "financial times", "ft", "politico", "npr", "sky", "al jazeera",
}

_BENCHMARK_SOURCES = {"polymarket", "kalshi"}

_PROCEDURAL_PATTERNS = (
    "paperwork reduction act",
    "information collection",
    "notice inviting comments",
    "public comment period",
    "request for information",
    "notice of submission",
    "meeting notice",
    "workshop notice",
    "draft guidance",
    "proposed information collection",
    "extension of a currently approved collection",
)


def _normalize_text(value: Optional[str]) -> str:
    """Lowercase and collapse whitespace; safe for None inputs."""
    text = (value or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def _log_normalize(value: float, scale: float) -> float:
    """Apply log1p normalization so that velocity/diversity values scale sub-linearly up to 1.0."""
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(scale), 1.0)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float to [low, high] (default [0, 1])."""
    return max(low, min(high, value))


def _join_text_parts(parts: Iterable[str]) -> str:
    """Join non-empty string parts into a single space-separated string."""
    return " ".join(part for part in parts if part).strip()


def infer_topic_popularity(text: str, sources: Optional[Iterable[str]] = None) -> tuple[str, float]:
    """
    Return the most likely market-interest topic plus a soft prior in [0, 1].

    This is deterministic and intentionally lightweight so we can bias ranking
    without adding another LLM stage.
    """
    haystack = _normalize_text(text)
    source_list = [_normalize_text(source) for source in (sources or []) if source]
    topic_scores = {topic: 0.0 for topic in _TOPIC_KEYWORDS}

    for topic, keywords in _TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in haystack:
                topic_scores[topic] += 1.0

    if any(source in _MAJOR_NEWS_SOURCES for source in source_list):
        topic_scores["major_news"] += 0.75
    if any(source in _BENCHMARK_SOURCES for source in source_list):
        topic_scores["crypto"] += 0.25
        topic_scores["finance"] += 0.20
        topic_scores["sports"] += 0.15

    best_topic, best_score = max(topic_scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return "other", 0.20
    prior = _clamp(0.20 + (0.18 * min(best_score, 4.0)))
    return best_topic, prior


def compute_procedural_penalty(text: str) -> float:
    """Return a cumulative penalty in [0, 0.72] for each procedural/admin boilerplate pattern found."""
    haystack = _normalize_text(text)
    hits = sum(1 for pattern in _PROCEDURAL_PATTERNS if pattern in haystack)
    return _clamp(hits * 0.18, 0.0, 0.72)


def _compute_cluster_priority_from_text(
    text: str,
    sources: Iterable[str],
    cluster_features: ClusterFeatures,
) -> tuple[str, float]:
    """Compute (topic_label, priority_score) from combined event text, sources, and cluster feature signals."""
    topic, topic_prior = infer_topic_popularity(text, sources)
    procedural_penalty = compute_procedural_penalty(text)
    normalized_sources = [_normalize_text(source) for source in sources]
    benchmark_overlap = 1.0 if any(source in _BENCHMARK_SOURCES for source in normalized_sources) else 0.0
    velocity_score = _log_normalize(cluster_features.weighted_mention_velocity, 12.0)
    diversity_score = _clamp(cluster_features.source_diversity / 6.0)
    recency_score = _clamp(1.0 - (max(cluster_features.recency, 0.0) / 168.0))
    coherence_score = _clamp(cluster_features.coherence_score)

    priority = (
        0.38 * topic_prior
        + 0.24 * velocity_score
        + 0.14 * diversity_score
        + 0.10 * recency_score
        + 0.08 * coherence_score
        + 0.06 * benchmark_overlap
        - procedural_penalty
    )
    return topic, round(priority, 4)


def compute_cluster_priority(cluster_features: ClusterFeatures, events: list[Event]) -> tuple[str, float]:
    """
    Score a cluster for FR3 selection.

    The score is a soft ranking prior. It does not filter anything out.
    """
    combined_text = _join_text_parts(
        _join_text_parts((event.title, event.content, event.entities))
        for event in events
    )
    sources = [event.source for event in events]
    return _compute_cluster_priority_from_text(combined_text, sources, cluster_features)


def compute_extracted_event_priority(
    extracted_event: ExtractedEvent,
    cluster_features: ClusterFeatures,
    sources: Optional[Iterable[str]] = None,
) -> tuple[str, float]:
    """
    Score an extracted event for FR4 selection using the underlying cluster and
    the FR3 output together.
    """
    event_text = _join_text_parts(
        [
            extracted_event.event_summary,
            extracted_event.market_angle,
            extracted_event.outcome_variable,
            extracted_event.event_type,
            " ".join(extracted_event.entities or []),
            " ".join(extracted_event.resolution_sources or []),
        ]
    )
    topic, text_topic_prior = infer_topic_popularity(event_text, sources)
    cluster_topic, cluster_priority = _compute_cluster_priority_from_text(
        event_text,
        list(sources or []),
        cluster_features,
    )

    event_type_prior = _TOPIC_EVENT_TYPE_PRIORS.get(
        _normalize_text(extracted_event.event_type).replace(" ", "_"),
        _TOPIC_EVENT_TYPE_PRIORS["other"],
    )
    contradiction_penalty = 0.12 if extracted_event.contradiction_flag else 0.0
    procedural_penalty = compute_procedural_penalty(event_text) * 0.6
    priority = (
        0.40 * cluster_priority
        + 0.22 * text_topic_prior
        + 0.20 * _clamp(extracted_event.confidence)
        + 0.12 * event_type_prior
        + 0.06 * (1.0 if cluster_topic == topic else 0.0)
        - contradiction_penalty
        - procedural_penalty
    )
    return topic, round(priority, 4)
