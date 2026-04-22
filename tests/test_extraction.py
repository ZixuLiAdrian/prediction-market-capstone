"""Tests for FR3: LLM Event Extraction."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import jsonschema
import pytest

from config import LLMConfig
from extraction.extractor import EventExtractor
from extraction.schema import EXTRACTED_EVENT_SCHEMA, EVENT_TYPE_ENUM
from extraction.prompts import build_extraction_user_prompt
from models import ClusterFeatures


# ---- Helper: a well-formed response matching the expanded schema ----

def _valid_response(**overrides):
    base = {
        "event_summary": "The Federal Reserve announced a 25 basis point rate cut, signaling a shift in monetary policy amid slowing inflation.",
        "entities": ["Federal Reserve", "Jerome Powell", "United States"],
        "event_type": "macro_release",
        "outcome_variable": "Federal funds rate target range",
        "candidate_deadlines": ["2025-06-18", "Q3 2025"],
        "resolution_sources": ["FOMC official statement", "Federal Reserve press conference"],
        "tradability": "suitable",
        "rejection_reason": "",
        "confidence": 0.85,
        "market_angle": "Binary outcome with high public interest and a clear announcement date.",
        "contradiction_flag": False,
        "contradiction_details": "",
        "time_horizon": "2-4 weeks",
        "resolution_hints": [
            "Official Fed statement release",
            "FOMC meeting minutes",
            "Market reaction in bond yields",
        ],
    }
    base.update(overrides)
    return base


# ---- Schema validation tests ----

def test_valid_schema():
    """A well-formed response should pass schema validation."""
    jsonschema.validate(instance=_valid_response(), schema=EXTRACTED_EVENT_SCHEMA)


def test_missing_required_field():
    """Missing a required field should fail validation."""
    invalid = _valid_response()
    del invalid["outcome_variable"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=EXTRACTED_EVENT_SCHEMA)


def test_missing_event_type():
    """Missing event_type should fail validation."""
    invalid = _valid_response()
    del invalid["event_type"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=EXTRACTED_EVENT_SCHEMA)


def test_empty_entities_list():
    """Empty entities list should fail (minItems=1)."""
    invalid = _valid_response(entities=[])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=EXTRACTED_EVENT_SCHEMA)


def test_summary_too_short():
    """Very short summary should fail (minLength=20)."""
    invalid = _valid_response(event_summary="Short")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=EXTRACTED_EVENT_SCHEMA)


def test_extra_fields_rejected():
    """Additional fields beyond schema should fail (additionalProperties=False)."""
    invalid = _valid_response(extra_field="should not be here")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=EXTRACTED_EVENT_SCHEMA)


def test_invalid_event_type():
    """Event type not in enum should fail."""
    invalid = _valid_response(event_type="invalid_category")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=EXTRACTED_EVENT_SCHEMA)


def test_all_event_types_valid():
    """All defined event types should pass validation."""
    for event_type in EVENT_TYPE_ENUM:
        valid = _valid_response(event_type=event_type)
        jsonschema.validate(instance=valid, schema=EXTRACTED_EVENT_SCHEMA)


def test_invalid_tradability():
    """Tradability not in enum should fail."""
    invalid = _valid_response(tradability="maybe")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=EXTRACTED_EVENT_SCHEMA)


def test_confidence_out_of_range():
    """Confidence outside 0.0-1.0 should fail."""
    invalid = _valid_response(confidence=1.5)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=EXTRACTED_EVENT_SCHEMA)

    invalid2 = _valid_response(confidence=-0.1)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid2, schema=EXTRACTED_EVENT_SCHEMA)


def test_unsuitable_event_valid():
    """An unsuitable event with rejection reason should pass validation."""
    valid = _valid_response(
        tradability="unsuitable",
        rejection_reason="Event already resolved; outcome is publicly known.",
        confidence=0.3,
    )
    jsonschema.validate(instance=valid, schema=EXTRACTED_EVENT_SCHEMA)


def test_contradiction_flag_valid():
    """Event with contradiction flag should pass validation."""
    valid = _valid_response(
        contradiction_flag=True,
        contradiction_details="Reuters reports rate cut while Bloomberg reports hold decision.",
    )
    jsonschema.validate(instance=valid, schema=EXTRACTED_EVENT_SCHEMA)


def test_outcome_variable_too_short():
    """Very short outcome variable should fail (minLength=5)."""
    invalid = _valid_response(outcome_variable="CPI")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=EXTRACTED_EVENT_SCHEMA)


def test_empty_candidate_deadlines_valid():
    """Empty candidate_deadlines list should pass (minItems=0)."""
    valid = _valid_response(candidate_deadlines=[])
    jsonschema.validate(instance=valid, schema=EXTRACTED_EVENT_SCHEMA)


# ---- Prompt construction tests ----

def test_prompt_contains_headlines():
    """User prompt should include all provided headlines."""
    headlines = ["Market rally continues", "Fed signals rate pause"]
    prompt = build_extraction_user_prompt(headlines)

    assert "Market rally continues" in prompt
    assert "Fed signals rate pause" in prompt


def test_prompt_includes_sources():
    """User prompt should include source names when provided."""
    headlines = ["Test headline"]
    sources = ["reuters", "bbc"]
    prompt = build_extraction_user_prompt(headlines, sources)

    assert "reuters" in prompt
    assert "bbc" in prompt


def test_prompt_caps_headlines():
    """Prompt should cap headlines at 20 to avoid token limits."""
    headlines = [f"Headline {i}" for i in range(50)]
    prompt = build_extraction_user_prompt(headlines)

    assert "Headline 0" in prompt
    assert "Headline 19" in prompt
    assert "Headline 20" not in prompt


def test_prompt_includes_features():
    """Prompt should include cluster features when provided."""
    headlines = ["Test headline"]
    features = ClusterFeatures(
        mention_velocity=5.0,
        source_diversity=3,
        recency=2.5,
        source_role_mix={"discovery": 2, "attention": 1},
        coherence_score=0.85,
        weighted_mention_velocity=7.5,
    )
    prompt = build_extraction_user_prompt(headlines, features=features)

    assert "5.00 events/hour" in prompt
    assert "3 unique sources" in prompt
    assert "2.5 hours" in prompt
    assert "discovery: 2" in prompt
    assert "attention: 1" in prompt
    assert "0.850" in prompt
    assert "7.50" in prompt


def test_prompt_without_features():
    """Prompt without features should not contain metadata section."""
    headlines = ["Test headline"]
    prompt = build_extraction_user_prompt(headlines)

    assert "Cluster Metadata" not in prompt


def test_prompt_requests_all_fields():
    """Prompt should mention all required output fields."""
    headlines = ["Test headline"]
    prompt = build_extraction_user_prompt(headlines)

    assert "event_type" in prompt
    assert "outcome_variable" in prompt
    assert "tradability" in prompt
    assert "contradiction_flag" in prompt
    assert "confidence" in prompt


def test_event_extractor_uses_stage_specific_model_by_default(monkeypatch):
    """EventExtractor should default to FR3_LLM_MODEL when creating its own client."""
    captured = {}

    class DummyLLMClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("extraction.extractor.LLMClient", DummyLLMClient)
    EventExtractor()

    assert captured["model"] == LLMConfig.FR3_MODEL


def test_event_extractor_allows_model_override(monkeypatch):
    """EventExtractor should allow an explicit per-stage model override."""
    captured = {}

    class DummyLLMClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("extraction.extractor.LLMClient", DummyLLMClient)
    EventExtractor(model="custom-fr3-model")

    assert captured["model"] == "custom-fr3-model"
