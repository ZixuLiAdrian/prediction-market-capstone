"""Tests for FR3: LLM Event Extraction."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import jsonschema
import pytest

from extraction.schema import EXTRACTED_EVENT_SCHEMA
from extraction.prompts import build_extraction_user_prompt


# ---- Schema validation tests ----

def test_valid_schema():
    """A well-formed response should pass schema validation."""
    valid_response = {
        "event_summary": "The Federal Reserve announced a 25 basis point rate cut, signaling a shift in monetary policy amid slowing inflation.",
        "entities": ["Federal Reserve", "Jerome Powell", "United States"],
        "time_horizon": "2-4 weeks",
        "resolution_hints": [
            "Official Fed statement release",
            "FOMC meeting minutes",
            "Market reaction in bond yields",
        ],
    }
    # Should not raise
    jsonschema.validate(instance=valid_response, schema=EXTRACTED_EVENT_SCHEMA)


def test_missing_required_field():
    """Missing a required field should fail validation."""
    invalid_response = {
        "event_summary": "Something happened",
        "entities": ["Someone"],
        # missing time_horizon and resolution_hints
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_response, schema=EXTRACTED_EVENT_SCHEMA)


def test_empty_entities_list():
    """Empty entities list should fail (minItems=1)."""
    invalid_response = {
        "event_summary": "Something happened with significance.",
        "entities": [],
        "time_horizon": "1 week",
        "resolution_hints": ["some hint"],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_response, schema=EXTRACTED_EVENT_SCHEMA)


def test_summary_too_short():
    """Very short summary should fail (minLength=20)."""
    invalid_response = {
        "event_summary": "Short",
        "entities": ["Entity"],
        "time_horizon": "1 week",
        "resolution_hints": ["hint"],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_response, schema=EXTRACTED_EVENT_SCHEMA)


def test_extra_fields_rejected():
    """Additional fields beyond schema should fail (additionalProperties=False)."""
    invalid_response = {
        "event_summary": "A sufficiently long event summary for validation purposes.",
        "entities": ["Entity"],
        "time_horizon": "1 week",
        "resolution_hints": ["hint"],
        "extra_field": "should not be here",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_response, schema=EXTRACTED_EVENT_SCHEMA)


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

    # Should contain headline 0 and 19, but not 20+
    assert "Headline 0" in prompt
    assert "Headline 19" in prompt
    assert "Headline 20" not in prompt
