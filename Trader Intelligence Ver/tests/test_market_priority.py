"""Tests for deterministic popularity bias heuristics used by FR3/FR4 selection."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ranking.market_priority import (
    compute_cluster_priority,
    compute_extracted_event_priority,
    compute_procedural_penalty,
    infer_topic_popularity,
)
from models import ClusterFeatures, Event, ExtractedEvent


def _base_features() -> ClusterFeatures:
    return ClusterFeatures(
        mention_velocity=2.5,
        source_diversity=3,
        recency=6.0,
        coherence_score=0.55,
        weighted_mention_velocity=3.2,
    )


def test_infer_topic_popularity_prefers_crypto_market_text():
    topic, prior = infer_topic_popularity(
        "Bitcoin ETF approval odds jumped after Coinbase and BlackRock updates.",
        sources=["reuters", "polymarket"],
    )
    assert topic == "crypto"
    assert prior > 0.5


def test_compute_procedural_penalty_hits_admin_notice_language():
    penalty = compute_procedural_penalty(
        "Paperwork Reduction Act notice inviting comments on an information collection request."
    )
    assert penalty > 0.3


def test_compute_cluster_priority_favors_macro_over_procedural_notice():
    features = _base_features()
    macro_events = [
        Event(
            title="Fed signals possible rate cut after CPI cools",
            content="Federal Reserve officials discussed inflation, CPI, and the next FOMC meeting.",
            source="reuters",
            source_type="rss",
        )
    ]
    procedural_events = [
        Event(
            title="Agency requests comments on information collection",
            content=(
                "Paperwork Reduction Act notice inviting comments on a proposed "
                "information collection extension and meeting notice."
            ),
            source="federal_register",
            source_type="official",
        )
    ]

    macro_topic, macro_score = compute_cluster_priority(features, macro_events)
    procedural_topic, procedural_score = compute_cluster_priority(features, procedural_events)

    assert macro_topic == "macro"
    assert procedural_topic in {"major_news", "other", "policy"}
    assert macro_score > procedural_score


def test_compute_extracted_event_priority_favors_sports_or_politics_over_procedural_policy():
    features = _base_features()
    sports_event = ExtractedEvent(
        id=1,
        cluster_id=1,
        event_summary="The Celtics advanced to the NBA Finals after a dominant playoff run.",
        entities=["Boston Celtics", "NBA"],
        event_type="sports",
        outcome_variable="NBA Finals winner",
        resolution_sources=["NBA official schedule"],
        confidence=0.86,
        market_angle="High-interest championship outcome with clear resolution.",
    )
    procedural_event = ExtractedEvent(
        id=2,
        cluster_id=2,
        event_summary=(
            "The agency published a public comment period notice under the Paperwork "
            "Reduction Act for an information collection request."
        ),
        entities=["Federal Register"],
        event_type="policy",
        outcome_variable="comment collection complete",
        resolution_sources=["Federal Register notice"],
        confidence=0.86,
        market_angle="Administrative process update.",
    )

    sports_topic, sports_score = compute_extracted_event_priority(
        sports_event,
        features,
        sources=["espn", "polymarket"],
    )
    procedural_topic, procedural_score = compute_extracted_event_priority(
        procedural_event,
        features,
        sources=["federal_register"],
    )

    assert sports_topic == "sports"
    assert procedural_topic in {"major_news", "other", "policy"}
    assert sports_score > procedural_score
