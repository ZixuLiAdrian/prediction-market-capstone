"""Tests for FR5 deterministic validation."""

import sys
import os
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import CandidateQuestion
from validation.validator import compute_clarity_score, validate_question


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

    past_str = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
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

