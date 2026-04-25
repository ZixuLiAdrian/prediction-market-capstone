"""
Deterministic dedupe helpers for FR4 selection and question post-processing.

These helpers keep one news story from flooding the dashboard with near-identical
questions while preserving a small amount of variety.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence

from models import CandidateQuestion, ExtractedEvent

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "with",
    "from", "at", "as", "will", "be", "is", "are", "was", "were", "has", "have",
    "had", "it", "its", "that", "this", "these", "those", "what", "which", "how",
    "many", "within", "next", "current", "before", "after", "into", "about", "than",
    "their", "they", "them", "his", "her", "our", "your", "who", "whom", "if", "all",
    "more", "less", "least", "most", "over", "under", "full", "full", "period",
    "initial", "officially", "publicly", "announce", "announced", "agreement",
}
_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_KEYWORD_ALIASES = {
    "cease-fire": "ceasefire",
    "cease fire": "ceasefire",
    "rate-cut": "ratecut",
    "rate cut": "ratecut",
    "rate-hike": "ratehike",
    "rate hike": "ratehike",
    "nominee": "nominate",
    "nominees": "nominate",
    "nominated": "nominate",
    "nomination": "nominate",
    "candidate": "nominate",
    "candidates": "nominate",
    "elections": "election",
    "midterms": "midterm",
    "mid-term": "midterm",
    "mid term": "midterm",
}


def _normalize_text(text: str) -> str:
    """Lowercase text, apply keyword aliases (e.g. 'ceasefire'), strip punctuation, and collapse whitespace."""
    normalized = (text or "").lower()
    for src, dst in _KEYWORD_ALIASES.items():
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokenize(text: str) -> list[str]:
    """Split normalized text into meaningful tokens, removing stopwords, month words, and short/numeric tokens."""
    tokens = []
    for token in _normalize_text(text).split():
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        if token in _STOPWORDS or token in _MONTH_WORDS:
            continue
        tokens.append(token)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity (intersection over union) between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a: set[str], b: set[str]) -> float:
    """Compute containment similarity (intersection over the smaller set) to catch subset stories."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _event_story_tokens(event: ExtractedEvent) -> set[str]:
    """Build a representative token fingerprint from the top-14 tokens of an ExtractedEvent's key fields."""
    source_text = " ".join(
        [
            event.event_summary,
            event.market_angle,
            event.outcome_variable,
            event.event_type,
            " ".join(event.entities or []),
        ]
    )
    token_counts = Counter(_tokenize(source_text))
    return {token for token, _ in token_counts.most_common(14)}


def extracted_events_same_story(first: ExtractedEvent, second: ExtractedEvent) -> bool:
    """Return True when two extracted events appear to describe the same story."""
    first_tokens = _event_story_tokens(first)
    second_tokens = _event_story_tokens(second)
    similarity = _jaccard(first_tokens, second_tokens)
    containment = _containment(first_tokens, second_tokens)

    first_entities = {token for token in _tokenize(" ".join(first.entities or []))}
    second_entities = {token for token in _tokenize(" ".join(second.entities or []))}
    entity_overlap = len(first_entities & second_entities)

    same_event_type = (first.event_type or "").strip().lower() == (second.event_type or "").strip().lower()

    if similarity >= 0.42:
        return True
    if same_event_type and entity_overlap >= 2 and (similarity >= 0.24 or containment >= 0.42):
        return True
    if containment >= 0.60 and entity_overlap >= 1:
        return True
    if same_event_type and containment >= 0.75:
        return True
    return False


def dedupe_extracted_events(events: Sequence[ExtractedEvent]) -> list[ExtractedEvent]:
    """
    Keep the highest-priority representative from each story group.

    Input order matters: the first event in each duplicate group is kept. Callers
    should pre-sort by desired priority before passing events here.
    """
    unique_events: list[ExtractedEvent] = []
    for event in events:
        if any(extracted_events_same_story(event, existing) for existing in unique_events):
            continue
        unique_events.append(event)
    return unique_events


def _question_tokens(question_text: str) -> set[str]:
    """Return the full token set for a question text string."""
    return set(_tokenize(question_text))


def questions_are_near_duplicates(first_text: str, second_text: str) -> bool:
    """Return True when two question texts are near-duplicates."""
    first_normalized = _normalize_text(first_text)
    second_normalized = _normalize_text(second_text)
    if not first_normalized or not second_normalized:
        return False
    if first_normalized == second_normalized:
        return True

    first_tokens = _question_tokens(first_text)
    second_tokens = _question_tokens(second_text)
    similarity = _jaccard(first_tokens, second_tokens)
    containment = _containment(first_tokens, second_tokens)

    return similarity >= 0.66 or containment >= 0.80


def dedupe_questions(questions: Iterable[CandidateQuestion]) -> List[CandidateQuestion]:
    """Drop near-duplicate questions while preserving input order."""
    unique_questions: list[CandidateQuestion] = []
    for question in questions:
        if any(
            (
                question.question_type == existing.question_type
                and questions_are_near_duplicates(question.question_text, existing.question_text)
            )
            or _normalize_text(question.question_text) == _normalize_text(existing.question_text)
            for existing in unique_questions
        ):
            continue
        unique_questions.append(question)
    return unique_questions
