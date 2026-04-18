"""
FR5: Deterministic rule validation for candidate questions.

Checks cover:
- Wording quality (ambiguity, clarity)
- Resolution source and criteria strength
- Deadline validity
- Prohibited topics (violence, terrorism, death markets, illegal activity)
- Manipulation risk (moral hazard, insider trading)
- Minor involvement
- PII exposure
- Deadline upper bound
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, UTC, timedelta

from models import CandidateQuestion, ValidationResult

logger = logging.getLogger(__name__)

# ---- Existing term sets ----

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

# ---- New: Prohibited topic terms ----

# Violence, death, and harm
_VIOLENCE_TERMS = {
    "assassinate", "assassinated", "assassination",
    "murder", "murdered", "killing", "killed",
    "die", "dies", "death", "dead",
    "suicide", "self-harm",
    "mass shooting", "school shooting", "shooting",
    "bomb", "bombing", "bombed",
    "attack", "attacked",  # alone too broad; combined with other signals below
    "execute", "executed", "execution",
    "torture", "tortured",
    "kidnap", "kidnapped", "kidnapping",
    "hostage",
}

# Terrorism
_TERRORISM_TERMS = {
    "terrorist", "terrorism", "terror attack",
    "bioterrorism", "cyberterrorism",
    "jihad", "jihadist",
    "extremist attack", "extremist violence",
    "car bomb", "suicide bomb", "suicide bomber",
    "ied", "improvised explosive",
    "anthrax", "sarin", "nerve agent",
    "dirty bomb", "radiological weapon",
}

# Illegal activity
_ILLEGAL_ACTIVITY_TERMS = {
    "drug trafficking", "drug cartel",
    "human trafficking",
    "money laundering",
    "child exploitation", "child abuse", "child pornography",
    "sex trafficking",
    "arms trafficking", "weapons smuggling",
}

# Minor involvement markers
_MINOR_TERMS = {
    "minor", "underage", "juvenile",
    "child", "children",
    "teenager", "teen",
    # "year-old" removed — age-based detection is handled by the regex in
    # detect_minor_involvement() which correctly checks age < 18.
    "high school student", "middle school",
}

# Market manipulation / moral hazard
_MANIPULATION_TERMS = {
    "insider trading", "insider information",
    "front-running", "front running",
    "pump and dump", "pump-and-dump",
    "market manipulation",
    "price manipulation",
    "wash trading", "wash trade",
}

# Combined text for "death market" detection — questions about whether someone will die
_DEATH_MARKET_PATTERNS = [
    re.compile(r"\bwill\b.*\b(die|be killed|be assassinated|be murdered|pass away)\b", re.IGNORECASE),
    re.compile(r"\b(death|assassination|murder)\b.*\bby\b.*\d{4}", re.IGNORECASE),
    re.compile(r"\bnumber of (deaths|casualties|fatalities)\b", re.IGNORECASE),
    re.compile(r"\b(how many|total).*(die|killed|deaths|casualties)\b", re.IGNORECASE),
]

# Patterns suggesting the question participant can influence the outcome
_MORAL_HAZARD_PATTERNS = [
    re.compile(r"\bwill\b.*\b(commit|carry out|perpetrate|launch an attack)\b", re.IGNORECASE),
    re.compile(r"\bwill there be a\b.*(attack|shooting|bombing|riot)\b", re.IGNORECASE),
]

# Max deadline: questions shouldn't resolve more than 5 years out
_MAX_DEADLINE_YEARS = 5


def compute_clarity_score(flags: list[str]) -> float:
    score = 1.0 - 0.2 * len(flags)
    return max(0.0, min(1.0, score))


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


# ---- Original checks ----

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
    return parsed.date() < datetime.now(UTC).date()


def detect_unclear_binary_condition(question_text: str, question_type: str) -> bool:
    if question_type != "binary":
        return False
    text = (question_text or "").strip()
    if not text.endswith("?"):
        return True
    return _BINARY_LEAD_PATTERN.search(text) is None


# ---- New checks ----

def detect_prohibited_topic(question_text: str, resolution_criteria: str) -> str | None:
    """
    Check if the question involves a prohibited topic.
    Returns the specific prohibition reason, or None if clean.
    """
    haystack = _normalize_text(f"{question_text} {resolution_criteria}")

    # Terrorism — check first, highest severity
    for term in _TERRORISM_TERMS:
        if term in haystack:
            return f"terrorism_content ({term})"

    # Illegal activity
    for term in _ILLEGAL_ACTIVITY_TERMS:
        if term in haystack:
            return f"illegal_activity ({term})"

    # Death market patterns (regex-based for more precision)
    for pattern in _DEATH_MARKET_PATTERNS:
        if pattern.search(f"{question_text or ''} {resolution_criteria or ''}"):
            return "death_market"

    # Violence terms — require co-occurrence with question framing
    # ("attack" alone is too broad — "cyberattack on infrastructure" is fine)
    violence_hits = [t for t in _VIOLENCE_TERMS if re.search(r"\b" + re.escape(t) + r"\b", haystack)]
    if violence_hits:
        # Only flag if the question is about whether violence will happen,
        # not if it references a past event in context
        q_lower = (question_text or "").lower()
        if any(q_lower.startswith(w) for w in ("will", "is", "are", "does", "can", "could")):
            # Forward-looking question about violence
            severe_terms = {"assassinate", "assassination", "murder", "mass shooting",
                            "school shooting", "bomb", "bombing", "execute", "execution",
                            "torture", "kidnap", "kidnapping", "hostage"}
            if any(t in severe_terms for t in violence_hits):
                return f"violence_content ({', '.join(violence_hits)})"

    return None


def detect_manipulation_risk(question_text: str) -> bool:
    """
    Check if the question creates moral hazard — where a market participant
    could profit by influencing the outcome, or the question facilitates
    insider trading / market manipulation.
    """
    haystack = _normalize_text(question_text)

    # Direct manipulation terms
    for term in _MANIPULATION_TERMS:
        if term in haystack:
            return True

    # Moral hazard patterns (questions about whether bad acts will happen)
    for pattern in _MORAL_HAZARD_PATTERNS:
        if pattern.search(question_text or ""):
            return True

    return False


def detect_minor_involvement(question_text: str) -> bool:
    """
    Check if the question involves minors in a context that requires
    extra scrutiny (betting on outcomes involving children).
    """
    haystack = _normalize_text(question_text)

    # Look for age patterns like "16-year-old" or "14 year old"
    age_match = re.search(r"\b(\d{1,2})[\s-]?year[\s-]?old\b", haystack)
    if age_match:
        age = int(age_match.group(1))
        if age < 18:
            return True

    # Check for minor-related terms — but exclude common false positives
    # "minor" alone is too broad (e.g. "minor policy change"), so require context
    for term in _MINOR_TERMS:
        if term in ("minor",):
            # Only flag "minor" if followed by person-like context
            if re.search(r"\bminor\b.{0,30}\b(athlete|player|student|child|person|individual)\b", haystack):
                return True
        elif term in haystack:
            # Skip "children" in contexts like "children's hospital" (org name)
            if term == "children" and "children's" in haystack:
                continue
            return True

    return False


def detect_excessive_deadline(deadline: str) -> bool:
    """Check if the deadline is unreasonably far in the future (>5 years)."""
    parsed = _try_parse_deadline(deadline)
    if parsed is None:
        return False  # unparseable deadlines are caught by detect_invalid_deadline_window
    max_date = datetime.now(UTC) + timedelta(days=_MAX_DEADLINE_YEARS * 365)
    # _try_parse_deadline returns naive datetime; compare as dates to avoid tz issues
    return parsed.date() > max_date.date()


def detect_pii_exposure(question_text: str) -> bool:
    """
    Check if the question contains patterns that look like PII
    (phone numbers, email addresses, SSNs, specific home addresses).
    """
    text = question_text or ""

    # Email
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
        return True

    # Phone numbers (US-style)
    if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", text):
        return True

    # SSN pattern
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
        return True

    return False


# ---- Main validation function ----

def validate_question(q: CandidateQuestion) -> ValidationResult:
    flags: list[str] = []

    # --- Prohibited topic checks (highest priority) ---
    prohibited = detect_prohibited_topic(q.question_text, q.resolution_criteria)
    if prohibited:
        flags.append(f"prohibited_topic:{prohibited}")

    if detect_manipulation_risk(q.question_text):
        flags.append("manipulation_risk")

    if detect_minor_involvement(q.question_text):
        flags.append("minor_involvement")

    if detect_pii_exposure(q.question_text):
        flags.append("pii_exposure")

    # --- Quality checks ---
    if detect_ambiguous_wording(q.question_text, q.resolution_criteria):
        flags.append("ambiguous_wording")
    if detect_weak_resolution_source(q.resolution_source):
        flags.append("weak_resolution_source")
    if detect_weak_resolution_criteria(q.resolution_criteria):
        flags.append("weak_resolution_criteria")
    if detect_invalid_deadline_window(q.deadline):
        flags.append("invalid_deadline_window")
    if detect_excessive_deadline(q.deadline):
        flags.append("excessive_deadline")
    if detect_unclear_binary_condition(q.question_text, q.question_type):
        flags.append("unclear_binary_condition")

    clarity_score = compute_clarity_score(flags)
    is_valid = len(flags) == 0

    if not is_valid:
        logger.info(
            f"Validation failed for Q{q.id or '?'}: flags={flags} | "
            f"question={q.question_text[:80]}..."
        )

    return ValidationResult(
        question_id=q.id or 0,
        is_valid=is_valid,
        flags=flags,
        clarity_score=clarity_score,
    )
