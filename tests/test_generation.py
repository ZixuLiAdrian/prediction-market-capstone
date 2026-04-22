"""Tests for FR4: LLM Question Generation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import jsonschema
import pytest

from config import LLMConfig
from generation.schema import CANDIDATE_QUESTIONS_SCHEMA
from generation.prompts import build_generation_user_prompt
from generation.generator import (
    QuestionGenerator,
    _contains_blocked_content,
    _has_garbled_text,
    _is_sportsbook_style_question,
    _normalize_category,
    _normalize_binary_options,
    _repair_deadline_fields,
    _validate_question,
)
from models import ExtractedEvent, CandidateQuestion


# =========================================================
# Fixtures
# =========================================================

@pytest.fixture
def sample_extracted_event():
    return ExtractedEvent(
        id=42,
        cluster_id=7,
        event_summary=(
            "The Federal Reserve signaled a potential interest rate cut at its next "
            "FOMC meeting following lower-than-expected inflation data. Markets reacted "
            "positively with bond yields falling sharply."
        ),
        entities=["Federal Reserve", "Jerome Powell", "United States"],
        time_horizon="4-6 weeks",
        resolution_hints=[
            "Official FOMC statement release",
            "Federal funds rate announcement",
            "CME FedWatch tool probability shift",
        ],
    )


@pytest.fixture
def valid_binary_question():
    return {
        "question_text": "Will the Federal Reserve cut interest rates at its next FOMC meeting?",
        "category": "finance",
        "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "December 18, 2025",
        "deadline_source": "2025 FOMC meeting calendar published at federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "resolution_source": "Federal Reserve official FOMC statement (federalreserve.gov/monetarypolicy)",
        "resolution_criteria": (
            "Resolves YES if the federal funds rate target range is reduced by any amount. "
            "Resolves NO if the rate is held steady or increased."
        ),
        "rationale": "Clear binary outcome with an authoritative official source and a specific deadline.",
        "resolution_confidence": 0.85,
        "resolution_confidence_reason": "Official FOMC statement with clear rate decision",
        "source_independence": 0.9,
        "timing_reliability": 0.8,
        "already_resolved": False,
    }


@pytest.fixture
def valid_mc_question():
    return {
        "question_text": "By how many basis points will the Federal Reserve cut rates at its next FOMC meeting?",
        "category": "finance",
        "question_type": "multiple_choice",
        "options": ["No cut (0 bps)", "25 bps cut", "50 bps cut", "More than 50 bps cut"],
        "deadline": "December 18, 2025",
        "deadline_source": "2025 FOMC meeting calendar published at federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "resolution_source": "Federal Reserve official FOMC statement (federalreserve.gov/monetarypolicy)",
        "resolution_criteria": (
            "Resolves to 'No cut (0 bps)' if the rate is unchanged or raised. "
            "Resolves to '25 bps cut' if the rate is reduced by exactly 25 basis points. "
            "Resolves to '50 bps cut' if the rate is reduced by exactly 50 basis points. "
            "Resolves to 'More than 50 bps cut' if the rate is reduced by more than 50 basis points."
        ),
        "rationale": "Four exhaustive options covering all possible magnitude outcomes with explicit thresholds.",
        "resolution_confidence": 0.85,
        "resolution_confidence_reason": "Official FOMC statement with clear rate decision",
        "source_independence": 0.9,
        "timing_reliability": 0.8,
        "already_resolved": False,
    }


@pytest.fixture
def valid_llm_response(valid_binary_question, valid_mc_question):
    return {"questions": [valid_binary_question, valid_mc_question]}


# =========================================================
# Schema validation tests
# =========================================================

class TestSchema:
    def test_valid_binary_response(self, valid_llm_response):
        """A well-formed LLM response with binary and MC questions should pass."""
        jsonschema.validate(instance=valid_llm_response, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_missing_questions_key(self):
        """Response without 'questions' key should fail."""
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={}, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_empty_questions_list(self):
        """Empty questions list should fail (minItems=1)."""
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"questions": []}, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_too_many_questions(self, valid_binary_question):
        """More than 5 questions should fail (maxItems=5)."""
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                instance={"questions": [valid_binary_question] * 6},
                schema=CANDIDATE_QUESTIONS_SCHEMA,
            )

    def test_invalid_category(self, valid_binary_question):
        """An unlisted category should fail (enum constraint)."""
        bad = dict(valid_binary_question, category="cryptocurrency")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"questions": [bad]}, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_invalid_question_type(self, valid_binary_question):
        """An unlisted question_type should fail."""
        bad = dict(valid_binary_question, question_type="range")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"questions": [bad]}, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_too_few_options(self, valid_mc_question):
        """Fewer than 2 options should fail (minItems=2)."""
        bad = dict(valid_mc_question, options=["Only one"])
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"questions": [bad]}, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_too_many_options(self, valid_mc_question):
        """More than 5 options should fail (maxItems=5)."""
        bad = dict(valid_mc_question, options=["A", "B", "C", "D", "E", "F"])
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"questions": [bad]}, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_question_text_too_short(self, valid_binary_question):
        """Very short question_text should fail (minLength=20)."""
        bad = dict(valid_binary_question, question_text="Short?")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"questions": [bad]}, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_missing_required_field(self, valid_binary_question):
        """Missing a required field (e.g. deadline) should fail."""
        bad = {k: v for k, v in valid_binary_question.items() if k != "deadline"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"questions": [bad]}, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_additional_properties_rejected(self, valid_binary_question):
        """Extra fields not in schema should fail (additionalProperties=False)."""
        bad = dict(valid_binary_question, unexpected_field="oops")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"questions": [bad]}, schema=CANDIDATE_QUESTIONS_SCHEMA)

    def test_all_valid_categories_accepted(self, valid_binary_question):
        """All declared categories should pass schema validation."""
        valid_cats = [
            "politics", "finance", "technology", "geopolitics",
            "science", "health", "business", "sports",
            "energy", "legal", "environment", "space", "other",
        ]
        for cat in valid_cats:
            q = dict(valid_binary_question, category=cat)
            jsonschema.validate(instance={"questions": [q]}, schema=CANDIDATE_QUESTIONS_SCHEMA)


# =========================================================
# Prompt construction tests
# =========================================================

class TestPrompt:
    def test_prompt_includes_event_summary(self):
        prompt = build_generation_user_prompt(
            event_summary="The Fed cut rates by 25bps.",
            entities=["Federal Reserve"],
            time_horizon="2 weeks",
            resolution_hints=["FOMC press release"],
        )
        assert "The Fed cut rates by 25bps." in prompt

    def test_prompt_includes_entities(self):
        prompt = build_generation_user_prompt(
            event_summary="Some event summary text here for testing purposes.",
            entities=["Federal Reserve", "Jerome Powell"],
            time_horizon="1 month",
            resolution_hints=["Official statement"],
        )
        assert "Federal Reserve" in prompt
        assert "Jerome Powell" in prompt

    def test_prompt_includes_resolution_hints(self):
        prompt = build_generation_user_prompt(
            event_summary="Some event summary text here for testing purposes.",
            entities=["Entity"],
            time_horizon="6 weeks",
            resolution_hints=["Vote result published", "Regulatory announcement"],
        )
        assert "Vote result published" in prompt
        assert "Regulatory announcement" in prompt

    def test_prompt_handles_empty_entities(self):
        """Empty entity list should not crash and fall back gracefully."""
        prompt = build_generation_user_prompt(
            event_summary="Some event summary text here for testing purposes.",
            entities=[],
            time_horizon="2 weeks",
            resolution_hints=["some hint"],
        )
        assert "Not specified" in prompt

    def test_prompt_handles_empty_hints(self):
        """Empty resolution hints should not crash."""
        prompt = build_generation_user_prompt(
            event_summary="Some event summary text here for testing purposes.",
            entities=["Entity A"],
            time_horizon="2 weeks",
            resolution_hints=[],
        )
        assert "Not specified" in prompt

    def test_prompt_adds_election_category_guidance(self):
        prompt = build_generation_user_prompt(
            event_summary="A parliamentary election is expected later this year.",
            entities=["Bulgaria"],
            time_horizon="by November 2026",
            resolution_hints=["Official election schedule"],
            event_type="election",
        )
        assert "Category guidance: use category 'politics' for election questions" in prompt
        assert "prefer secretary of state, election commission, official filing pages, or certified results" in prompt

    def test_prompt_adds_geopolitics_source_guidance(self):
        prompt = build_generation_user_prompt(
            event_summary="A ceasefire between two countries may collapse or be extended soon.",
            entities=["Israel", "Lebanon"],
            time_horizon="10 days",
            resolution_hints=["Reuters/AP reporting plus official statements"],
            event_type="geopolitics",
        )
        assert "use named high-credibility outlets plus official statements" in prompt
        assert "prefer observable status/announcement questions over blame or compliance attribution" in prompt


# =========================================================
# Content safety tests
# =========================================================

class TestContentSafety:
    def test_blocked_term_detected(self):
        assert _contains_blocked_content("This question is about that fucker") is True

    def test_clean_text_passes(self):
        assert _contains_blocked_content("Will the Fed cut rates in 2025?") is False

    def test_partial_word_not_blocked(self):
        """'classic' contains 'ass' but should not be blocked (word-boundary check)."""
        assert _contains_blocked_content("This is a classic market setup.") is False

    def test_garbled_text_detected(self):
        """High ratio of non-ASCII characters should be flagged."""
        garbled = "Ẃíll ṫhė Fëd ċüt ṙaṫes?" * 3
        assert _has_garbled_text(garbled) is True

    def test_normal_unicode_passes(self):
        """A few accented characters in an otherwise normal string should be fine."""
        assert _has_garbled_text("Will Société Générale raise its dividend?") is False

    def test_control_characters_flagged(self):
        assert _has_garbled_text("Question\x00with null byte?") is True

    def test_empty_string_garbled(self):
        assert _has_garbled_text("") is True


# =========================================================
# Binary option normalisation tests
# =========================================================

class TestNormalizeBinaryOptions:
    def test_already_normalised(self):
        assert _normalize_binary_options(["Yes", "No"]) == ["Yes", "No"]

    def test_lowercase_normalised(self):
        assert _normalize_binary_options(["yes", "no"]) == ["Yes", "No"]

    def test_uppercase_normalised(self):
        assert _normalize_binary_options(["YES", "NO"]) == ["Yes", "No"]

    def test_true_false_normalised(self):
        assert _normalize_binary_options(["True", "False"]) == ["Yes", "No"]

    def test_reversed_order_normalised(self):
        assert _normalize_binary_options(["No", "Yes"]) == ["Yes", "No"]

    def test_mc_options_unchanged(self):
        opts = ["Option A", "Option B", "Option C"]
        assert _normalize_binary_options(opts) == opts


# =========================================================
# Question-level validation tests
# =========================================================

class TestValidateQuestion:
    def test_valid_binary_passes(self, valid_binary_question):
        assert _validate_question(valid_binary_question) is None

    def test_valid_mc_passes(self, valid_mc_question):
        assert _validate_question(valid_mc_question) is None

    def test_missing_question_mark(self, valid_binary_question):
        bad = dict(valid_binary_question, question_text="Will the Fed cut rates")
        assert _validate_question(bad) is not None

    def test_binary_with_three_options_rejected(self, valid_binary_question):
        bad = dict(valid_binary_question, options=["Yes", "No", "Maybe"])
        assert _validate_question(bad) is not None

    def test_mc_with_two_options_rejected(self, valid_mc_question):
        bad = dict(valid_mc_question, options=["A", "B"])
        assert _validate_question(bad) is not None

    def test_vague_deadline_rejected(self, valid_binary_question):
        bad = dict(valid_binary_question, deadline="soon")
        assert _validate_question(bad) is not None

    def test_empty_deadline_rejected(self, valid_binary_question):
        bad = dict(valid_binary_question, deadline="")
        assert _validate_question(bad) is not None

    def test_missing_deadline_source_rejected(self, valid_binary_question):
        bad = dict(valid_binary_question, deadline_source="")
        assert _validate_question(bad) is not None

    def test_vague_deadline_source_rejected(self, valid_binary_question):
        bad = dict(valid_binary_question, deadline_source="Fed")
        assert _validate_question(bad) is not None

    def test_short_resolution_source_rejected(self, valid_binary_question):
        bad = dict(valid_binary_question, resolution_source="Fed")
        assert _validate_question(bad) is not None

    def test_short_resolution_criteria_rejected(self, valid_binary_question):
        bad = dict(valid_binary_question, resolution_criteria="Yes if cut.")
        assert _validate_question(bad) is not None

    def test_blocked_content_in_question_text(self, valid_binary_question):
        bad = dict(valid_binary_question,
                   question_text="Will that fucker cut rates by December 2025?")
        assert _validate_question(bad) is not None

    def test_empty_option_rejected(self, valid_binary_question):
        bad = dict(valid_binary_question, options=["Yes", ""])
        assert _validate_question(bad) is not None

    def test_past_deadline_can_be_repaired_from_future_candidate_deadline(self, sample_extracted_event, valid_binary_question):
        sample_extracted_event.candidate_deadlines = ["April 22, 2027", "April 1, 2025"]
        bad = dict(valid_binary_question, deadline="April 1, 2025")
        _repair_deadline_fields(bad, sample_extracted_event)
        assert bad["deadline"] == "April 22, 2027"


class TestCategoryAndSportsNormalization:
    def test_normalize_category_maps_election_to_politics(self):
        extracted_event = ExtractedEvent(
            id=1,
            cluster_id=1,
            event_summary="The 2026 Bulgarian parliamentary election is scheduled for later this year.",
            entities=["Bulgaria"],
            event_type="election",
        )
        raw = {"category": "other"}
        assert _normalize_category(raw, extracted_event) == "politics"

    def test_sportsbook_style_filter_flags_exact_score_and_next_game_props(self):
        exact_score = CandidateQuestion(
            extracted_event_id=1,
            question_text="What will be the final score of the Los Angeles Dodgers game against the San Francisco Giants on April 28, 2026?",
            category="sports",
            question_type="multiple_choice",
            options=["1-0", "2-1", "3-2"],
            deadline="April 28, 2026",
            deadline_source="MLB schedule at https://example.com/mlb",
            resolution_source="MLB official box score at https://example.com/boxscore",
            resolution_criteria="Resolves to the listed exact score.",
            rationale="Test",
        )
        next_game_prop = CandidateQuestion(
            extracted_event_id=1,
            question_text="Will Jalen Brunson score at least 25 points in the next New York Knicks game?",
            category="sports",
            question_type="binary",
            options=["Yes", "No"],
            deadline="April 25, 2026",
            deadline_source="NBA schedule at https://example.com/nba",
            resolution_source="NBA official box score at https://example.com/boxscore",
            resolution_criteria="Resolves YES if Brunson scores at least 25 points. Resolves NO otherwise.",
            rationale="Test",
        )
        season_market = CandidateQuestion(
            extracted_event_id=1,
            question_text="Will the Los Angeles Dodgers win at least 90 games in the 2026 MLB season?",
            category="sports",
            question_type="binary",
            options=["Yes", "No"],
            deadline="September 30, 2026",
            deadline_source="MLB season schedule at https://example.com/mlb",
            resolution_source="MLB standings at https://example.com/standings",
            resolution_criteria="Resolves YES if the Dodgers finish with at least 90 wins. Resolves NO otherwise.",
            rationale="Test",
        )

        assert _is_sportsbook_style_question(exact_score) is True
        assert _is_sportsbook_style_question(next_game_prop) is True
        assert _is_sportsbook_style_question(season_market) is False


# =========================================================
# Generator unit tests (no LLM calls — mock the client)
# =========================================================

class MockLLMClient:
    """Minimal mock that returns a pre-built valid response."""

    def __init__(self, response: dict):
        self.response = response

    def call(self, system_prompt, user_prompt, response_schema=None):
        return self.response


class TestQuestionGenerator:
    def test_question_generator_uses_stage_specific_model_by_default(self, monkeypatch):
        """QuestionGenerator should default to FR4_LLM_MODEL when creating its own client."""
        captured = {}

        class DummyLLMClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("generation.generator.LLMClient", DummyLLMClient)
        QuestionGenerator()

        assert captured["model"] == LLMConfig.FR4_MODEL

    def test_question_generator_allows_model_override(self, monkeypatch):
        """QuestionGenerator should allow an explicit FR4 model override."""
        captured = {}

        class DummyLLMClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("generation.generator.LLMClient", DummyLLMClient)
        QuestionGenerator(model="custom-fr4-model")

        assert captured["model"] == "custom-fr4-model"

    def test_generate_returns_candidate_questions(
        self, sample_extracted_event, valid_llm_response
    ):
        generator = QuestionGenerator(llm_client=MockLLMClient(valid_llm_response))
        results = generator.generate(sample_extracted_event)
        assert len(results) == 2
        assert all(isinstance(q, CandidateQuestion) for q in results)

    def test_generate_sets_extracted_event_id(
        self, sample_extracted_event, valid_llm_response
    ):
        generator = QuestionGenerator(llm_client=MockLLMClient(valid_llm_response))
        results = generator.generate(sample_extracted_event)
        for q in results:
            assert q.extracted_event_id == sample_extracted_event.id

    def test_generate_normalises_binary_options(self, sample_extracted_event):
        response = {
            "questions": [
                {
                    "question_text": "Will the Federal Reserve cut interest rates at its next meeting?",
                    "category": "finance",
                    "question_type": "binary",
                    "options": ["yes", "no"],  # lowercase — should be normalised
                    "deadline": "December 18, 2025",
                    "deadline_source": "2025 FOMC meeting calendar at federalreserve.gov/monetarypolicy/fomccalendars.htm",
                    "resolution_source": "Federal Reserve FOMC statement (federalreserve.gov/monetarypolicy)",
                    "resolution_criteria": (
                        "Resolves YES if the federal funds rate is reduced by any amount. "
                        "Resolves NO if the rate is held or increased."
                    ),
                    "rationale": "Clear binary outcome from an official source.",
                    "resolution_confidence": 0.85,
                    "resolution_confidence_reason": "Official FOMC statement",
                    "source_independence": 0.9,
                    "timing_reliability": 0.8,
                    "already_resolved": False,
                }
            ]
        }
        generator = QuestionGenerator(llm_client=MockLLMClient(response))
        results = generator.generate(sample_extracted_event)
        assert results[0].options == ["Yes", "No"]

    def test_generate_skips_event_without_id(self, valid_llm_response):
        event_no_id = ExtractedEvent(
            id=None,
            cluster_id=1,
            event_summary="Some event without a DB id assigned yet.",
            entities=["Entity"],
            time_horizon="2 weeks",
            resolution_hints=["Hint"],
        )
        generator = QuestionGenerator(llm_client=MockLLMClient(valid_llm_response))
        results = generator.generate(event_no_id)
        assert results == []

    def test_generate_filters_invalid_questions(self, sample_extracted_event):
        """Questions that fail post-schema checks should be silently dropped."""
        response = {
            "questions": [
                {
                    "question_text": "Will the Fed cut rates by December 2025?",
                    "category": "finance",
                    "question_type": "binary",
                    "options": ["Yes", "No"],
                    "deadline": "December 18, 2025",
                    "deadline_source": "2025 FOMC calendar at federalreserve.gov/monetarypolicy/fomccalendars.htm",
                    "resolution_source": "Federal Reserve FOMC statement (federalreserve.gov/monetarypolicy)",
                    "resolution_criteria": (
                        "Resolves YES if any cut is announced. Resolves NO otherwise."
                    ),
                    "rationale": "Passes all checks.",
                    "resolution_confidence": 0.85,
                    "resolution_confidence_reason": "Official FOMC statement",
                    "source_independence": 0.9,
                    "timing_reliability": 0.8,
                    "already_resolved": False,
                },
                {
                    # This question has a vague deadline and should be dropped
                    "question_text": "Will something happen in the markets eventually?",
                    "category": "finance",
                    "question_type": "binary",
                    "options": ["Yes", "No"],
                    "deadline": "soon",
                    "deadline_source": "2025 FOMC calendar at federalreserve.gov/monetarypolicy/fomccalendars.htm",
                    "resolution_source": "Federal Reserve official press release (federalreserve.gov/monetarypolicy)",
                    "resolution_criteria": (
                        "Resolves YES if something happens. Resolves NO otherwise."
                    ),
                    "rationale": "Has a vague deadline — should be rejected.",
                    "resolution_confidence": 0.85,
                    "resolution_confidence_reason": "Official source",
                    "source_independence": 0.9,
                    "timing_reliability": 0.8,
                    "already_resolved": False,
                },
            ]
        }
        generator = QuestionGenerator(llm_client=MockLLMClient(response))
        results = generator.generate(sample_extracted_event)
        assert len(results) == 1
        # The valid question (first one) should survive; the vague-deadline one is dropped
        assert "December 2025" in results[0].question_text

    def test_generate_dedupes_near_duplicate_questions_within_one_event(self, sample_extracted_event):
        response = {
            "questions": [
                {
                    "question_text": "Will the ceasefire between Israel and Lebanon be extended beyond its initial term?",
                    "category": "geopolitics",
                    "question_type": "binary",
                    "options": ["Yes", "No"],
                    "deadline": "April 30, 2026",
                    "deadline_source": "Official statement at https://example.com/ceasefire",
                    "resolution_source": "Official source at https://example.com/source",
                    "resolution_criteria": "Resolves YES if the ceasefire is formally extended. Resolves NO otherwise.",
                    "rationale": "Ceasefire extension question.",
                    "resolution_confidence": 0.85,
                    "resolution_confidence_reason": "Official source",
                    "source_independence": 0.9,
                    "timing_reliability": 0.8,
                    "already_resolved": False,
                },
                {
                    "question_text": "Will the cease-fire be extended beyond its initial 10-day term?",
                    "category": "geopolitics",
                    "question_type": "binary",
                    "options": ["Yes", "No"],
                    "deadline": "April 30, 2026",
                    "deadline_source": "Official statement at https://example.com/ceasefire",
                    "resolution_source": "Official source at https://example.com/source",
                    "resolution_criteria": "Resolves YES if the ceasefire is formally extended. Resolves NO otherwise.",
                    "rationale": "Ceasefire extension variant.",
                    "resolution_confidence": 0.85,
                    "resolution_confidence_reason": "Official source",
                    "source_independence": 0.9,
                    "timing_reliability": 0.8,
                    "already_resolved": False,
                },
            ]
        }

        generator = QuestionGenerator(llm_client=MockLLMClient(response))
        results = generator.generate(sample_extracted_event)

        assert len(results) == 1

    def test_generate_filters_sportsbook_style_questions(self):
        sports_event = ExtractedEvent(
            id=99,
            cluster_id=8,
            event_summary="Upcoming Lakers and Dodgers games create immediate sports speculation.",
            entities=["Los Angeles Lakers", "Denver Nuggets", "Los Angeles Dodgers"],
            event_type="sports",
            time_horizon="within 7 days",
            resolution_hints=["Official league box score"],
        )
        response = {
            "questions": [
                {
                    "question_text": "What will be the final score of the Los Angeles Dodgers game against the San Francisco Giants on April 28, 2026?",
                    "category": "sports",
                    "question_type": "multiple_choice",
                    "options": ["1-0", "2-1", "3-2"],
                    "deadline": "April 28, 2026",
                    "deadline_source": "MLB schedule at https://example.com/mlb",
                    "resolution_source": "MLB official box score at https://example.com/boxscore",
                    "resolution_criteria": "Resolves to the listed exact score.",
                    "rationale": "Test rationale.",
                    "resolution_confidence": 0.9,
                    "resolution_confidence_reason": "Official box score",
                    "source_independence": 0.9,
                    "timing_reliability": 0.9,
                    "already_resolved": False,
                },
                {
                    "question_text": "Will the Los Angeles Dodgers win at least 90 games in the 2026 MLB season?",
                    "category": "sports",
                    "question_type": "binary",
                    "options": ["Yes", "No"],
                    "deadline": "September 30, 2026",
                    "deadline_source": "MLB season schedule at https://example.com/mlb",
                    "resolution_source": "MLB standings at https://example.com/standings",
                    "resolution_criteria": "Resolves YES if the Dodgers finish with at least 90 wins. Resolves NO otherwise.",
                    "rationale": "Season-long sports market.",
                    "resolution_confidence": 0.9,
                    "resolution_confidence_reason": "Official standings",
                    "source_independence": 0.9,
                    "timing_reliability": 0.9,
                    "already_resolved": False,
                },
            ]
        }
        generator = QuestionGenerator(llm_client=MockLLMClient(response))
        results = generator.generate(sports_event)

        assert len(results) == 1
        assert "90 games" in results[0].question_text

    def test_generate_handles_llm_failure_gracefully(self, sample_extracted_event):
        """RuntimeError from LLMClient should return empty list, not propagate."""

        class FailingLLMClient:
            def call(self, system_prompt, user_prompt, response_schema=None):
                raise RuntimeError("API rate limit exceeded")

        generator = QuestionGenerator(llm_client=FailingLLMClient())
        results = generator.generate(sample_extracted_event)
        assert results == []

    def test_generate_batch_aggregates_results(self, sample_extracted_event, valid_llm_response):
        event2 = ExtractedEvent(
            id=99,
            cluster_id=8,
            event_summary="Another event for batch testing purposes in the pipeline.",
            entities=["Company X"],
            time_horizon="1 month",
            resolution_hints=["Earnings report"],
        )
        generator = QuestionGenerator(llm_client=MockLLMClient(valid_llm_response))
        results = generator.generate_batch([sample_extracted_event, event2])
        # 2 events × 2 questions each = 4 total
        assert len(results) == 4

    def test_candidate_question_fields_populated(
        self, sample_extracted_event, valid_binary_question
    ):
        """All CandidateQuestion fields should be correctly populated."""
        response = {"questions": [valid_binary_question]}
        generator = QuestionGenerator(llm_client=MockLLMClient(response))
        results = generator.generate(sample_extracted_event)

        q = results[0]
        assert q.question_text == valid_binary_question["question_text"]
        assert q.category == "finance"
        assert q.question_type == "binary"
        assert q.options == ["Yes", "No"]
        assert q.deadline == valid_binary_question["deadline"]
        assert q.deadline_source == valid_binary_question["deadline_source"]
        assert q.resolution_source == valid_binary_question["resolution_source"]
        assert q.resolution_criteria == valid_binary_question["resolution_criteria"]
        assert q.rationale == valid_binary_question["rationale"]
        assert q.extracted_event_id == sample_extracted_event.id
