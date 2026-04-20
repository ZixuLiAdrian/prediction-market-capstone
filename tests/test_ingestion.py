"""Tests for FR1: Event Ingestion."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from ingestion.base import compute_content_hash
from models import Event


def test_content_hash_consistency():
    """Same text should produce the same hash."""
    h1 = compute_content_hash("Breaking news: Market update")
    h2 = compute_content_hash("Breaking news: Market update")
    assert h1 == h2


def test_content_hash_normalization():
    """Hash should normalize whitespace and case."""
    h1 = compute_content_hash("  Hello World  ")
    h2 = compute_content_hash("hello world")
    assert h1 == h2


def test_content_hash_different_text():
    """Different text should produce different hashes."""
    h1 = compute_content_hash("Breaking news A")
    h2 = compute_content_hash("Breaking news B")
    assert h1 != h2


def test_event_creation():
    """Event dataclass should be constructable with required fields."""
    event = Event(
        content="Test event content",
        source="test_source",
        source_type="rss",
        title="Test Title",
        url="https://example.com",
        timestamp=datetime(2025, 1, 1),
    )
    assert event.content == "Test event content"
    assert event.source == "test_source"
    assert event.source_type == "rss"
    assert event.id is None


def test_event_default_hash():
    """Event should have empty content_hash by default."""
    event = Event(content="test", source="s", source_type="rss")
    assert event.content_hash == ""


def test_event_signal_role_default():
    """Event should have 'discovery' as default signal_role."""
    event = Event(content="test", source="s", source_type="rss")
    assert event.signal_role == "discovery"


def test_event_signal_role_custom():
    """Event should accept custom signal_role."""
    event = Event(content="test", source="sec_edgar", source_type="official", signal_role="resolution")
    assert event.signal_role == "resolution"

    event2 = Event(content="test", source="reddit", source_type="social", signal_role="attention")
    assert event2.signal_role == "attention"

    event3 = Event(content="test", source="polymarket", source_type="market", signal_role="benchmark")
    assert event3.signal_role == "benchmark"
