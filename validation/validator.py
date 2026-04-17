"""
FR5: Deterministic rule validation for candidate questions.
"""

from __future__ import annotations

import re
from datetime import datetime

from models import CandidateQuestion, ValidationResult

_AMBIGUOUS_TERMS = {
    "significant",
    "major",
    "substantial",
    "likely",
    "expected",
    "meaningful",
}

_GENERIC_SOURCE_TERMS = {
    "official source",
    "news reports",
    "media reports",
    "public sources",
    "internet",
    "website",
    "various sources",
    "reputable sources",
    "press",
}

_WEAK_CRITERIA_TERMS = {
    "as expected",
    "as likely",
    "if confirmed",
    "if announced broadly",
    "if widely reported",
    "if official confirmation appears",
}

_BINARY_LEAD_PATTERN = re.compile(
    r"^\s*(will|is|are|was|were|does|do|did|has|have|had|can|could|should|would)\b",
    flags=re.IGNORECASE,
)


def compute_clarity_score(flags: list[str]) -> float:
    score = 1.0 - 0.2 * len(flags)
    return max(0.0, min(1.0, score))


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def detect_ambiguous_wording(question_text: str, resolution_criteria: str) -> bool:
    haystack = f"{question_text or ''} {resolution_criteria or ''}".lower()
    for term in _AMBIGUOUS_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", haystack):
            return True
    return False


def detect_weak_resolution_source(resolution_source: str) -> bool:
    source = _normalize_text(resolution_source)
    if len(source) < 15:
        return True
    if "http://" not in source and "https://" not in source and "." not in source:
        return True
    return any(term in source for term in _GENERIC_SOURCE_TERMS)


def detect_weak_resolution_criteria(resolution_criteria: str) -> bool:
    criteria = _normalize_text(resolution_criteria)
    if len(criteria) < 30:
        return True
    if "resolves yes" not in criteria and "resolves no" not in criteria:
        return True
    return any(term in criteria for term in _WEAK_CRITERIA_TERMS)


def _try_parse_deadline(deadline: str) -> datetime | None:
    value = (deadline or "").strip()
    if not value:
        return None

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def detect_invalid_deadline_window(deadline: str) -> bool:
    parsed = _try_parse_deadline(deadline)
    if parsed is None:
        return True
    return parsed.date() < datetime.utcnow().date()


def detect_unclear_binary_condition(question_text: str, question_type: str) -> bool:
    if question_type != "binary":
        return False
    text = (question_text or "").strip()
    if not text.endswith("?"):
        return True
    return _BINARY_LEAD_PATTERN.search(text) is None


def validate_question(q: CandidateQuestion) -> ValidationResult:
    flags: list[str] = []

    if detect_ambiguous_wording(q.question_text, q.resolution_criteria):
        flags.append("ambiguous_wording")
    if detect_weak_resolution_source(q.resolution_source):
        flags.append("weak_resolution_source")
    if detect_weak_resolution_criteria(q.resolution_criteria):
        flags.append("weak_resolution_criteria")
    if detect_invalid_deadline_window(q.deadline):
        flags.append("invalid_deadline_window")
    if detect_unclear_binary_condition(q.question_text, q.question_type):
        flags.append("unclear_binary_condition")

    clarity_score = compute_clarity_score(flags)
    return ValidationResult(
        question_id=q.id or 0,
        is_valid=len(flags) == 0,
        flags=flags,
        clarity_score=clarity_score,
    )

