"""
FR4: Question Generator

Orchestrates LLM-based question generation:
1. Takes an ExtractedEvent object (from FR3)
2. Constructs a structured prompt from the event's summary, entities, and resolution hints
3. Calls LLMClient with strict schema enforcement
4. Applies post-generation quality filters and content safety checks
5. Returns a list of validated CandidateQuestion objects

Design mirrors FR3's EventExtractor for consistency.
FR5 (Rule Validation) consumes CandidateQuestion objects produced here.
"""

import logging
import re
from typing import List, Optional

from models import ExtractedEvent, CandidateQuestion
from extraction.llm_client import LLMClient
from generation.prompts import GENERATION_SYSTEM_PROMPT, build_generation_user_prompt
from generation.schema import CANDIDATE_QUESTIONS_SCHEMA

logger = logging.getLogger(__name__)

# Words that must never appear in any generated question or option.
# The LLM should never produce these, but this acts as a hard safety net.
_BLOCKED_TERMS: set = {
    "fuck", "shit", "asshole", "bitch", "cunt", "dick", "pussy",
    "nigger", "nigga", "faggot", "retard", "whore", "bastard",
    "damn", "hell",  # too mild to block universally, but added for market context
}
# Only apply the mild terms if they appear standalone (not inside other words)
_MILD_BLOCKED_STANDALONE = {"damn", "hell"}
_BLOCKED_TERMS -= _MILD_BLOCKED_STANDALONE  # remove mild ones from hard block


def _contains_blocked_content(text: str) -> bool:
    """
    Return True if text contains any hard-blocked terms.

    Uses a prefix word-boundary pattern (r"\\bTERM") so root forms catch
    common derivatives: "fuck" also catches "fucker", "fucking", etc.
    The leading \\b prevents false positives from mid-word matches
    (e.g. "classic" does not match "ass").
    """
    lowered = text.lower()
    for term in _BLOCKED_TERMS:
        if re.search(r"\b" + re.escape(term), lowered):
            return True
    return False


def _has_garbled_text(text: str) -> bool:
    """
    Return True if text appears garbled — excessive non-ASCII, control characters,
    or suspiciously high ratio of special characters that suggest encoding issues.
    """
    if not text:
        return True
    # Count printable ASCII + common Unicode letters vs total characters
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii / len(text) > 0.3:
        return True
    # Check for control characters (other than newline/tab)
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text):
        return True
    return False


def _normalize_binary_options(options: List[str]) -> List[str]:
    """
    Normalise binary question options to ["Yes", "No"] if the LLM used
    alternate casing or phrasing (e.g. "yes", "YES", "True", "False").
    """
    if len(options) == 2:
        lower = [o.strip().lower() for o in options]
        yes_variants = {"yes", "true", "correct", "affirmative"}
        no_variants = {"no", "false", "incorrect", "negative"}
        if lower[0] in yes_variants and lower[1] in no_variants:
            return ["Yes", "No"]
        if lower[0] in no_variants and lower[1] in yes_variants:
            return ["Yes", "No"]
    return options


def _validate_question(raw: dict) -> Optional[str]:
    """
    Run post-schema quality checks on a single raw question dict.

    Returns an error string if the question should be rejected, or None if it passes.
    """
    qt = raw.get("question_text", "")
    q_type = raw.get("question_type", "")
    options = raw.get("options", [])
    deadline = raw.get("deadline", "")
    deadline_source = raw.get("deadline_source", "")
    res_source = raw.get("resolution_source", "")
    res_criteria = raw.get("resolution_criteria", "")

    # Must end with a question mark
    if not qt.strip().endswith("?"):
        return f"question_text does not end with '?': {qt[:60]}"

    # Blocked content check across all text fields
    for field_name, value in [
        ("question_text", qt),
        ("resolution_criteria", res_criteria),
        ("resolution_source", res_source),
    ]:
        if _contains_blocked_content(value):
            return f"Blocked content detected in {field_name}"
        if _has_garbled_text(value):
            return f"Garbled text detected in {field_name}"

    # Binary questions must have exactly 2 options
    if q_type == "binary" and len(options) != 2:
        return f"Binary question must have exactly 2 options, got {len(options)}"

    # Multiple-choice must have 3–5 options
    if q_type == "multiple_choice" and not (3 <= len(options) <= 5):
        return f"Multiple-choice question must have 3–5 options, got {len(options)}"

    # All options must be non-empty strings
    for i, opt in enumerate(options):
        if not isinstance(opt, str) or not opt.strip():
            return f"Option {i} is empty or not a string"
        if _contains_blocked_content(opt) or _has_garbled_text(opt):
            return f"Option {i} contains blocked or garbled content"

    # Deadline must not be empty or generic
    vague_deadlines = {"soon", "tbd", "unknown", "n/a", "not specified", "to be determined"}
    if not deadline or deadline.strip().lower() in vague_deadlines:
        return f"Deadline is missing or vague: '{deadline}'"

    # Deadline source must be present
    if not deadline_source or len(deadline_source.strip()) < 10:
        return f"Deadline source is missing or too vague: '{deadline_source}'"

    # Resolution source must reference something concrete
    if len(res_source.strip()) < 10:
        return f"Resolution source is too vague: '{res_source}'"

    # Resolution criteria must be substantive
    if len(res_criteria.strip()) < 20:
        return f"Resolution criteria is too short: '{res_criteria}'"

    return None  # passed all checks


class QuestionGenerator:
    """Generates structured prediction market questions from ExtractedEvent objects."""

    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        extracted_event: ExtractedEvent,
    ) -> List[CandidateQuestion]:
        """
        Generate candidate market questions for a single extracted event.

        Args:
            extracted_event: An ExtractedEvent produced by FR3, with a DB-assigned id.

        Returns:
            List of validated CandidateQuestion objects (may be empty if generation fails).
        """
        if not extracted_event.id:
            logger.warning("ExtractedEvent has no DB id — skipping")
            return []

        user_prompt = build_generation_user_prompt(
            event_summary=extracted_event.event_summary,
            entities=extracted_event.entities,
            time_horizon=extracted_event.time_horizon,
            resolution_hints=extracted_event.resolution_hints,
        )

        try:
            result = self.llm_client.call(
                system_prompt=GENERATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=CANDIDATE_QUESTIONS_SCHEMA,
            )
        except RuntimeError as e:
            logger.error(
                f"ExtractedEvent {extracted_event.id}: Question generation failed — {e}"
            )
            return []

        raw_questions = result.get("questions", [])
        questions = self._validate_and_build(raw_questions, extracted_event.id)

        logger.info(
            f"ExtractedEvent {extracted_event.id}: "
            f"{len(questions)}/{len(raw_questions)} questions passed validation"
        )
        return questions

    def generate_batch(
        self,
        extracted_events: List[ExtractedEvent],
    ) -> List[CandidateQuestion]:
        """
        Generate candidate questions for a list of extracted events.

        Args:
            extracted_events: List of ExtractedEvent objects (from get_extracted_events()).

        Returns:
            Flat list of all validated CandidateQuestion objects across all events.
        """
        all_questions: List[CandidateQuestion] = []

        for event in extracted_events:
            questions = self.generate(event)
            all_questions.extend(questions)

        logger.info(
            f"Batch generation complete: {len(all_questions)} questions from "
            f"{len(extracted_events)} events"
        )
        return all_questions

    def _validate_and_build(
        self,
        raw_questions: list,
        extracted_event_id: int,
    ) -> List[CandidateQuestion]:
        """
        Apply post-schema quality checks and build CandidateQuestion objects.

        Args:
            raw_questions: List of raw question dicts from the LLM response.
            extracted_event_id: DB id of the source ExtractedEvent.

        Returns:
            List of CandidateQuestion objects that passed all quality checks.
        """
        validated: List[CandidateQuestion] = []

        for i, raw in enumerate(raw_questions):
            error = _validate_question(raw)
            if error:
                logger.warning(
                    f"ExtractedEvent {extracted_event_id}, question {i + 1} rejected: {error}"
                )
                continue

            # Normalise binary options casing
            options = raw["options"]
            if raw["question_type"] == "binary":
                options = _normalize_binary_options(options)

            question = CandidateQuestion(
                extracted_event_id=extracted_event_id,
                question_text=raw["question_text"].strip(),
                category=raw["category"],
                question_type=raw["question_type"],
                options=options,
                deadline=raw["deadline"].strip(),
                deadline_source=raw["deadline_source"].strip(),
                resolution_source=raw["resolution_source"].strip(),
                resolution_criteria=raw["resolution_criteria"].strip(),
                rationale=raw["rationale"].strip(),
                raw_llm_response=str(raw),
            )
            validated.append(question)

        return validated
