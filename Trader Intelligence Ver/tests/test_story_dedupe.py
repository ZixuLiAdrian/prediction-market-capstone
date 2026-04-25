"""Tests for story-level dedupe helpers used by FR4."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import CandidateQuestion, ExtractedEvent
from ranking.story_dedupe import dedupe_extracted_events, dedupe_questions, extracted_events_same_story, questions_are_near_duplicates


def test_extracted_events_same_story_detects_overlapping_ceasefire_summaries():
    first = ExtractedEvent(
        id=1,
        cluster_id=10,
        event_summary="A 10-day ceasefire between Israel and Lebanon has gone into effect, with Hezbollah's compliance uncertain.",
        entities=["Israel", "Lebanon", "Hezbollah"],
        event_type="geopolitics",
    )
    second = ExtractedEvent(
        id=2,
        cluster_id=11,
        event_summary="A 10-day ceasefire between Israel and Lebanon has begun, but Hezbollah has not confirmed it will fully comply.",
        entities=["Israel", "Lebanon", "Hezbollah"],
        event_type="geopolitics",
    )

    assert extracted_events_same_story(first, second) is True


def test_dedupe_extracted_events_keeps_first_representative_per_story():
    first = ExtractedEvent(
        id=1,
        cluster_id=10,
        event_summary="A 10-day ceasefire between Israel and Lebanon has gone into effect, with Hezbollah's compliance uncertain.",
        entities=["Israel", "Lebanon", "Hezbollah"],
        event_type="geopolitics",
    )
    second = ExtractedEvent(
        id=2,
        cluster_id=11,
        event_summary="A 10-day ceasefire between Israel and Lebanon has begun, but Hezbollah has not confirmed it will fully comply.",
        entities=["Israel", "Lebanon", "Hezbollah"],
        event_type="geopolitics",
    )
    third = ExtractedEvent(
        id=3,
        cluster_id=12,
        event_summary="Archimedes Tech SPAC Partners II announced a business combination.",
        entities=["Archimedes Tech SPAC Partners II"],
        event_type="business",
    )

    deduped = dedupe_extracted_events([first, second, third])

    assert [event.id for event in deduped] == [1, 3]


def test_questions_are_near_duplicates_handles_small_wording_changes():
    assert questions_are_near_duplicates(
        "Will Hezbollah adhere to the ceasefire for the full 10-day duration?",
        "Will Hezbollah adhere to the current 10-day ceasefire with Israel?",
    ) is True


def test_questions_are_near_duplicates_handles_nominee_wording_variants():
    assert questions_are_near_duplicates(
        "Will the Republican Party announce Brian Montgomery as the nominee for GA-01 by May 19, 2026?",
        "Will Brian Montgomery be nominated as the Republican Party's candidate for GA-01 in the 2026 midterm elections?",
    ) is True


def test_dedupe_questions_keeps_unique_question_shapes():
    questions = [
        CandidateQuestion(
            extracted_event_id=1,
            question_text="Will the ceasefire between Israel and Lebanon be extended beyond its initial term?",
            category="geopolitics",
            question_type="binary",
            options=["Yes", "No"],
            deadline="April 30, 2026",
            deadline_source="Official notice at https://example.com",
            resolution_source="Official source at https://example.com",
            resolution_criteria="Resolves YES if extended. Resolves NO otherwise.",
            rationale="Test",
        ),
        CandidateQuestion(
            extracted_event_id=1,
            question_text="Will the cease-fire be extended beyond its initial 10-day term?",
            category="geopolitics",
            question_type="binary",
            options=["Yes", "No"],
            deadline="April 30, 2026",
            deadline_source="Official notice at https://example.com",
            resolution_source="Official source at https://example.com",
            resolution_criteria="Resolves YES if extended. Resolves NO otherwise.",
            rationale="Test",
        ),
        CandidateQuestion(
            extracted_event_id=1,
            question_text="Will Hezbollah adhere to the ceasefire for the full 10-day duration?",
            category="geopolitics",
            question_type="binary",
            options=["Yes", "No"],
            deadline="April 30, 2026",
            deadline_source="Official notice at https://example.com",
            resolution_source="Official source at https://example.com",
            resolution_criteria="Resolves YES if Hezbollah complies. Resolves NO otherwise.",
            rationale="Test",
        ),
    ]

    deduped = dedupe_questions(questions)

    assert len(deduped) == 2
