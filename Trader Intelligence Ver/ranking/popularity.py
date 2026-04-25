"""
Deterministic popularity priors for FR3/FR4 prioritization.

These helpers bias the pipeline toward topics that tend to perform well on
prediction markets (macro, crypto, finance, politics, sports, major news)
while softly down-ranking procedural/regulatory boilerplate.
"""

from __future__ import annotations

import re
from typing import Iterable

from models import ClusterFeatures, Event, ExtractedEvent

_MACRO_KEYWORDS = {
    "federal reserve", "fed", "fomc", "cpi", "inflation", "pce", "gdp",
    "nonfarm payroll", "jobs report", "unemployment", "treasury", "yield",
    "rate cut", "rate hike", "interest rate", "bls", "fred", "ecb", "boj",
}
_CRYPTO_KEYWORDS = {
    "bitcoin", "btc", "ethereum", "eth", "solana", "xrp", "dogecoin",
    "crypto", "cryptocurrency", "stablecoin", "coinbase", "binance",
    "etf", "token", "defi", "staking",
}
_FINANCE_KEYWORDS = {
    "earnings", "revenue", "profit", "eps", "stock", "share price", "shares",
    "ipo", "merger", "acquisition", "m&a", "guidance", "dividend", "sec",
    "8-k", "10-q", "10-k", "nasdaq", "nyse", "s&p", "dow", "market cap",
}
_POLITICS_KEYWORDS = {
    "election", "white house", "president", "senate", "house", "congress",
    "parliament", "prime minister", "cabinet", "referendum", "campaign",
    "democrat", "republican", "biden", "trump", "supreme court", "bill",
    "governor", "mayor", "vote", "voting",
}
_SPORTS_KEYWORDS = {
    "nba", "nfl", "mlb", "nhl", "fifa", "uefa", "world cup", "super bowl",
    "playoffs", "finals", "championship", "grand slam", "formula 1",
    "premier league", "champions league", "stanley cup", "olympics",
}
_MAJOR_NEWS_KEYWORDS = {
    "war", "sanctions", "ceasefire", "tariff", "earthquake", "wildfire",
    "outbreak", "pandemic", "nato", "united nations", "airstrike",
    "missile", "evacuation", "strike", "hostage",
}
_PROCEDURAL_KEYWORDS = {
    "paperwork reduction act", "public comment period", "request for comments",
    "information collection", "notice of submission", "proposed information collection",
    "federal register notice", "omb approval notice", "docket", "comments invited",
    "extension of a currently approved collection", "information collection request",
}
_MAJOR_NEWS_SOURCES = {
    "reuters", "bbc", "nyt", "new york times", "politico", "ap", "associated press",
    "ft", "financial times", "al jazeera", "npr",
}
_BENCHMARK_SOURCES = {"polymarket", "kalshi"}


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for keyword matching."""
    return " ".join((text or "").strip().lower().split())


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    """Return True if any keyword from the iterable appears as a whole word in the normalized text."""
    haystack = _normalize_text(text)
    for keyword in keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", haystack):
            return True
    return False


def _cluster_text(events: list[Event]) -> str:
    """Concatenate title and content fields from all cluster events into one searchable string."""
    return " ".join(f"{event.title} {event.content}" for event in events if event)


def _topic_popularity_prior(text: str, sources: Iterable[str]) -> float:
    """Return a soft prior in [0.2, 1.0] based on which high-demand topic keywords are found in the text."""
    source_text = " ".join(sources)
    score = 0.2

    if _contains_any(text, _MACRO_KEYWORDS):
        score = max(score, 1.0)
    if _contains_any(text, _CRYPTO_KEYWORDS):
        score = max(score, 0.98)
    if _contains_any(text, _FINANCE_KEYWORDS):
        score = max(score, 0.92)
    if _contains_any(text, _POLITICS_KEYWORDS):
        score = max(score, 0.9)
    if _contains_any(text, _SPORTS_KEYWORDS):
        score = max(score, 0.86)
    if _contains_any(text, _MAJOR_NEWS_KEYWORDS) or _contains_any(source_text, _MAJOR_NEWS_SOURCES):
        score = max(score, 0.8)

    return score


def _procedural_penalty(text: str) -> float:
    """Return 0.35 if the text matches any procedural/admin boilerplate keyword, otherwise 0.0."""
    if _contains_any(text, _PROCEDURAL_KEYWORDS):
        return 0.35
    return 0.0


def _benchmark_overlap_boost(events: list[Event]) -> float:
    """Return 0.08 boost if any event in the cluster came from a benchmark source (Polymarket/Kalshi)."""
    if not events:
        return 0.0
    if any(event.signal_role == "benchmark" or event.source in _BENCHMARK_SOURCES for event in events):
        return 0.08
    return 0.0


def _normalize_velocity(weighted_velocity: float) -> float:
    """Scale weighted mention velocity to [0, 1] using a cap of 15."""
    return min(max(float(weighted_velocity or 0.0) / 15.0, 0.0), 1.0)


def _normalize_diversity(source_diversity: int) -> float:
    """Scale source diversity count to [0, 1] using a cap of 6 unique sources."""
    return min(max(float(source_diversity or 0.0) / 6.0, 0.0), 1.0)


def _normalize_recency(recency_hours: float) -> float:
    """Map recency (hours since most recent event) to [0, 1], decaying linearly over 7 days."""
    recency = max(float(recency_hours or 0.0), 0.0)
    return max(0.0, min(1.0, 1.0 - (recency / 168.0)))


def compute_cluster_priority_score(features: ClusterFeatures, events: list[Event]) -> float:
    """
    Priority score for FR3.

    Combines topic popularity with FR2 signal metrics. Higher is better.
    """
    text = _cluster_text(events)
    sources = [event.source for event in events]
    topic_prior = _topic_popularity_prior(text, sources)
    penalty = _procedural_penalty(text)
    score = (
        0.45 * topic_prior
        + 0.25 * _normalize_velocity(features.weighted_mention_velocity)
        + 0.15 * _normalize_diversity(features.source_diversity)
        + 0.10 * _normalize_recency(features.recency)
        + 0.05 * _benchmark_overlap_boost(events)
        - penalty
    )
    return round(score, 4)


def compute_extracted_event_priority(
    extracted_event: ExtractedEvent,
    cluster_features: ClusterFeatures,
) -> float:
    """
    Priority score for FR4.

    Leans on FR3 structured fields plus the cluster-level popularity signal.
    """
    text = " ".join(
        [
            extracted_event.event_summary,
            extracted_event.event_type,
            extracted_event.outcome_variable,
            extracted_event.market_angle,
            " ".join(extracted_event.entities or []),
            " ".join(extracted_event.resolution_sources or []),
        ]
    )
    topic_prior = _topic_popularity_prior(text, extracted_event.resolution_sources or [])
    penalty = _procedural_penalty(text)
    contradiction_penalty = 0.1 if extracted_event.contradiction_flag else 0.0
    score = (
        0.40 * topic_prior
        + 0.20 * min(max(float(extracted_event.confidence or 0.0), 0.0), 1.0)
        + 0.20 * _normalize_velocity(cluster_features.weighted_mention_velocity)
        + 0.10 * _normalize_diversity(cluster_features.source_diversity)
        + 0.10 * _normalize_recency(cluster_features.recency)
        - penalty
        - contradiction_penalty
    )
    return round(score, 4)
