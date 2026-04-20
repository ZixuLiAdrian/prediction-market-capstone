"""Tests for FR5 deterministic validation."""

import sys
import os
from datetime import datetime, UTC, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import CandidateQuestion
from validation.validator import (
    compute_clarity_score,
    detect_excessive_deadline,
    detect_manipulation_risk,
    detect_minor_involvement,
    detect_pii_exposure,
    detect_prohibited_topic,
    validate_question,
)


def _base_question(**overrides):
    data = {
        "id": 1,
        "extracted_event_id": 10,
        "question_text": "Will the Federal Reserve cut rates by December 31, 2027?",
        "category": "finance",
        "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "December 31, 2027",
        "deadline_source": "2027 FOMC meeting calendar at https://federalreserve.gov/monetarypolicy",
        "resolution_source": "Federal Reserve statement at https://federalreserve.gov/monetarypolicy",
        "resolution_criteria": "Resolves YES if the federal funds target range is reduced by any amount. Resolves NO if held steady or increased.",
        "rationale": "Clear and measurable outcome from an authoritative source.",
    }
    data.update(overrides)
    return CandidateQuestion(**data)


def test_valid_question_returns_valid_result():
    q = _base_question()
    result = validate_question(q)
    assert result.question_id == q.id
    assert result.is_valid is True
    assert result.flags == []
    assert result.clarity_score == 1.0


def test_ambiguous_wording_flag_triggers():
    q = _base_question(
        resolution_criteria=(
            "Resolves YES if there is a significant reduction in policy rates. "
            "Resolves NO otherwise."
        )
    )
    result = validate_question(q)
    assert "ambiguous_wording" in result.flags


def test_weak_resolution_source_flag_triggers():
    q = _base_question(resolution_source="Official source")
    result = validate_question(q)
    assert "weak_resolution_source" in result.flags


def test_weak_resolution_criteria_flag_triggers():
    q = _base_question(resolution_criteria="Outcome if confirmed by reports.")
    result = validate_question(q)
    assert "weak_resolution_criteria" in result.flags


def test_invalid_deadline_window_malformed_and_past():
    malformed = _base_question(deadline="not-a-date")
    malformed_result = validate_question(malformed)
    assert "invalid_deadline_window" in malformed_result.flags

    past_str = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%d")
    past = _base_question(deadline=past_str)
    past_result = validate_question(past)
    assert "invalid_deadline_window" in past_result.flags


def test_unclear_binary_condition_triggers_only_for_binary():
    binary_unclear = _base_question(question_text="The Fed rate cut by year-end?")
    binary_result = validate_question(binary_unclear)
    assert "unclear_binary_condition" in binary_result.flags

    mc_unclear = _base_question(
        question_type="multiple_choice",
        question_text="The Fed rate cut by year-end?",
        options=["No cut", "25 bps cut", "50 bps cut"],
    )
    mc_result = validate_question(mc_unclear)
    assert "unclear_binary_condition" not in mc_result.flags


def test_multiple_flags_reduce_clarity_score_correctly():
    q = _base_question(
        question_text="The Fed likely cuts rates?",
        resolution_source="Official source",
        resolution_criteria="Outcome if confirmed by reports.",
        deadline="bad-date",
    )
    result = validate_question(q)
    # ambiguous_wording + weak_resolution_source + weak_resolution_criteria
    # + invalid_deadline_window + unclear_binary_condition = 5 flags
    assert len(result.flags) == 5
    assert result.clarity_score == 0.0


def test_clarity_score_clamped_to_bounds():
    assert compute_clarity_score([]) == 1.0
    assert compute_clarity_score(["f1", "f2", "f3"]) == pytest.approx(0.4)
    assert compute_clarity_score(["f1", "f2", "f3", "f4", "f5"]) == 0.0
    assert compute_clarity_score(["f1", "f2", "f3", "f4", "f5", "f6"]) == 0.0


# ---- Prohibited topic detection tests ----

def test_detect_prohibited_topic_terrorism():
    result = detect_prohibited_topic(
        "Will a terrorist attack occur in Europe by 2027?",
        "Resolves YES if confirmed by authorities. Resolves NO otherwise.",
    )
    assert result is not None
    assert "terrorism_content" in result


def test_detect_prohibited_topic_illegal_activity():
    result = detect_prohibited_topic(
        "Will drug trafficking across the border increase in 2027?",
        "Resolves YES if DEA reports increase. Resolves NO otherwise.",
    )
    assert result is not None
    assert "illegal_activity" in result


def test_detect_prohibited_topic_death_market():
    result = detect_prohibited_topic(
        "Will the president be assassinated by December 2027?",
        "Resolves YES if confirmed. Resolves NO otherwise.",
    )
    assert result is not None
    # Could match terrorism or death_market depending on term ordering
    assert "terrorism" in result or "death_market" in result or "violence" in result


def test_detect_prohibited_topic_death_market_casualty_count():
    result = detect_prohibited_topic(
        "How many deaths will result from the next hurricane season?",
        "Resolves based on the total number of deaths reported by FEMA.",
    )
    assert result is not None
    assert "death_market" in result


def test_detect_prohibited_topic_clean_question():
    result = detect_prohibited_topic(
        "Will the Federal Reserve cut interest rates by December 2027?",
        "Resolves YES if the fed funds target range is reduced. Resolves NO otherwise.",
    )
    assert result is None


def test_detect_prohibited_topic_violence_forward_looking():
    """Forward-looking questions about severe violence should be flagged."""
    result = detect_prohibited_topic(
        "Will a mass shooting occur in the US by December 2027?",
        "Resolves YES if confirmed. Resolves NO otherwise.",
    )
    assert result is not None


def test_detect_prohibited_topic_violence_past_event_not_flagged():
    """Past-event references in context should not trigger violence flag."""
    result = detect_prohibited_topic(
        "After the 2024 incident, how did policy change?",
        "Resolves YES if new legislation passed. Resolves NO otherwise.",
    )
    assert result is None


# ---- Manipulation risk detection tests ----

def test_detect_manipulation_risk_insider_trading():
    assert detect_manipulation_risk(
        "Will insider trading at Company X be exposed by 2027?"
    ) is True


def test_detect_manipulation_risk_pump_and_dump():
    assert detect_manipulation_risk(
        "Will a pump and dump scheme affect token ABC?"
    ) is True


def test_detect_manipulation_risk_moral_hazard():
    assert detect_manipulation_risk(
        "Will there be a bombing in downtown Chicago?"
    ) is True


def test_detect_manipulation_risk_clean():
    assert detect_manipulation_risk(
        "Will Bitcoin close above $100,000 by December 2027?"
    ) is False


# ---- Minor involvement detection tests ----

def test_detect_minor_involvement_age_pattern():
    assert detect_minor_involvement(
        "Will the 16-year-old athlete win the championship?"
    ) is True


def test_detect_minor_involvement_teen_term():
    assert detect_minor_involvement(
        "Will the teenager break the world record?"
    ) is True


def test_detect_minor_involvement_adult_age():
    assert detect_minor_involvement(
        "Will the 25-year-old player score a hat trick?"
    ) is False


def test_detect_minor_involvement_clean():
    assert detect_minor_involvement(
        "Will the Federal Reserve cut rates?"
    ) is False


def test_detect_minor_involvement_minor_policy_not_flagged():
    """'minor' in 'minor policy change' should NOT trigger the flag."""
    assert detect_minor_involvement(
        "Will there be a minor policy change at the Fed?"
    ) is False


# ---- PII exposure detection tests ----

def test_detect_pii_email():
    assert detect_pii_exposure(
        "Will john.doe@example.com resign by December 2027?"
    ) is True


def test_detect_pii_phone():
    assert detect_pii_exposure(
        "Will the CEO at 555-123-4567 announce a merger?"
    ) is True


def test_detect_pii_ssn():
    assert detect_pii_exposure(
        "Will the person with SSN 123-45-6789 file a claim?"
    ) is True


def test_detect_pii_clean():
    assert detect_pii_exposure(
        "Will Apple stock close above $200 by December 2027?"
    ) is False


# ---- Excessive deadline detection tests ----

def test_detect_excessive_deadline_far_future():
    assert detect_excessive_deadline("December 31, 2040") is True


def test_detect_excessive_deadline_reasonable():
    assert detect_excessive_deadline("December 31, 2027") is False


def test_detect_excessive_deadline_unparseable():
    """Unparseable deadlines should return False (caught by invalid_deadline_window)."""
    assert detect_excessive_deadline("not-a-date") is False


# ---- Integration: validate_question with new checks ----

def test_validate_question_flags_prohibited_topic():
    q = _base_question(
        question_text="Will a terrorist attack occur in London by December 31, 2027?",
    )
    result = validate_question(q)
    assert result.is_valid is False
    assert any("prohibited_topic" in f for f in result.flags)


def test_validate_question_flags_manipulation_risk():
    q = _base_question(
        question_text="Will insider trading be detected at Company X by December 31, 2027?",
    )
    result = validate_question(q)
    assert result.is_valid is False
    assert "manipulation_risk" in result.flags


def test_validate_question_flags_pii():
    q = _base_question(
        question_text="Will user@example.com be promoted by December 31, 2027?",
    )
    result = validate_question(q)
    assert result.is_valid is False
    assert "pii_exposure" in result.flags


def test_validate_question_flags_minor_involvement():
    q = _base_question(
        question_text="Will the 15-year-old prodigy win the tournament by December 31, 2027?",
    )
    result = validate_question(q)
    assert result.is_valid is False
    assert "minor_involvement" in result.flags


def test_validate_question_flags_excessive_deadline():
    q = _base_question(deadline="December 31, 2040")
    result = validate_question(q)
    assert result.is_valid is False
    assert "excessive_deadline" in result.flags

