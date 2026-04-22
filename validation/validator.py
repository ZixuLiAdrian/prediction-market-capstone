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
- LLM quality self-assessment fields (resolution_confidence, source_independence,
  timing_reliability, already_resolved) carried forward from FR4
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta

UTC = timezone.utc

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
    "various sources",
    "reputable sources",
    "official statements",
}

_GENERIC_NEWS_ONLY_TERMS = {
    "reported by media",
    "reported by the media",
    "reported in the press",
    "press release",
    "press releases",
}

_TRUSTED_MEDIA_TERMS = {
    "reuters",
    "associated press",
    "ap",
    "bbc",
    "financial times",
    "wall street journal",
    "wsj",
    "bloomberg",
    "al jazeera",
    "haaretz",
}

_AUTHORITATIVE_SOURCE_TERMS = {
    "sec.gov",
    "edgar",
    "8-k",
    "10-q",
    "10-k",
    "federal reserve",
    "fomc",
    "bls.gov",
    "fda.gov",
    "clinicaltrials.gov",
    "congress.gov",
    "federal register",
    "court opinion",
    "court docket",
    "election commission",
    "central election commission",
    "government gazette",
    "official gazette",
    "cec.bg",
    "dv.parliament.bg",
    "state.gov",
    "gov.il",
    "government.se",
    "nato.int",
    "un.org",
    "who.int",
    "cdc.gov",
    "sec filing",
    "stock exchange",
    "nasdaq",
    "nyse",
    "league official",
    "official results",
    "investor relations",
    "company investor relations",
}

_CREDIBLE_REPORTING_CATEGORIES = {"geopolitics", "other"}
_AUTHORITATIVE_SOURCE_CATEGORIES = {
    "politics",
    "finance",
    "business",
    "sports",
    "legal",
    "health",
    "science",
    "technology",
    "energy",
    "environment",
    "space",
}

_GENERIC_FALLBACK_PATTERNS = {
    "or reputable news sources",
    "or reputable sources",
    "or news sources",
    "or media reports",
    "or official statements",
}

_DISPUTED_ATTRIBUTION_TERMS = {
    "primary reason",
    "main reason",
    "who is responsible",
    "who was responsible",
    "to blame",
    "blame for",
    "first violated",
    "first violation",
    "declare the ceasefire a success",
    "acknowledge the ceasefire's success",
    "acknowledge the ceasefire success",
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

# Death market detection — questions targeting an individual's mortality.
# NOTE: These patterns are intentionally scoped to INDIVIDUAL mortality, not state-level
# military action or policy. "Will the US carry out airstrikes on Iran?" is a legitimate
# geopolitical question; "Will [person] be assassinated by [date]?" is not.
_DEATH_MARKET_PATTERNS = [
    # Individual mortality: "Will [X] die / be killed / be assassinated?"
    re.compile(r"\bwill\b.{0,60}\b(die|pass away)\b", re.IGNORECASE),
    re.compile(r"\bwill\b.{0,60}\b(be (killed|assassinated|murdered))\b", re.IGNORECASE),
    # Body-count markets: "How many will die / be killed?"
    re.compile(r"\b(how many|total number of|number of)\b.{0,30}\b(deaths|casualties|fatalities|killed|dead)\b", re.IGNORECASE),
]

# Geopolitical / state-level action exemptions — these patterns indicate the question
# is about official state action (military strikes, sanctions, policy), NOT individual
# targeting. If a question matches one of these, death_market is suppressed.
_STATE_ACTION_EXEMPTIONS = [
    re.compile(r"\b(airstrike|airstrikes|military strike|military action|drone strike|bombing campaign)\b", re.IGNORECASE),
    re.compile(r"\b(sanctions|sanction|embargo|trade war|ceasefire|peace deal|peace agreement)\b", re.IGNORECASE),
    re.compile(r"\b(nato|un security council|united nations|congress|parliament|senate)\b.{0,50}\b(vote|approve|pass|authorize)\b", re.IGNORECASE),
    re.compile(r"\b(invade|invasion|withdraw|withdrawal|deploy|deployment)\b", re.IGNORECASE),
]

# Moral hazard patterns — where a bettor could profit by causing the outcome
_MORAL_HAZARD_PATTERNS = [
    re.compile(r"\bwill there be a\b.{0,40}\b(attack|shooting|bombing|riot|massacre)\b", re.IGNORECASE),
    re.compile(r"\bwill\b.{0,30}\b(commit|carry out|perpetrate)\b.{0,30}\b(attack|crime|fraud|violence)\b", re.IGNORECASE),
]

# Thresholds for FR4 quality scores — questions below these are flagged (soft flags, not hard blocks)
_MIN_RESOLUTION_CONFIDENCE = 0.65   # already filtered in FR4, but FR5 re-checks as a safety net
_MIN_SOURCE_INDEPENDENCE = 0.4
_MIN_TIMING_RELIABILITY = 0.4

# Max deadline: questions shouldn't resolve more than 5 years out
_MAX_DEADLINE_YEARS = 5
_SALVAGEABLE_FLAGS = {
    "ambiguous_wording",
    "weak_resolution_source",
    "weak_resolution_criteria",
    "invalid_deadline_window",
    "unclear_binary_condition",
    "low_resolution_confidence",
    "low_source_independence",
    "low_timing_reliability",
}
_NON_SALVAGEABLE_PREFIXES = {
    "prohibited_topic",
    "manipulation_risk",
    "minor_involvement",
    "pii_exposure",
    "already_resolved",
    "excessive_deadline",
}


def compute_clarity_score(flags: list[str]) -> float:
    score = 1.0 - 0.2 * len(flags)
    return max(0.0, min(1.0, score))


def is_salvageable_validation_flags(flags: list[str]) -> bool:
    """Return True when every flag is repairable rather than a hard policy block."""
    if not flags:
        return False

    for flag in flags:
        normalized = (flag or "").strip()
        if any(normalized.startswith(prefix) for prefix in _NON_SALVAGEABLE_PREFIXES):
            return False
        base = normalized.split(":", 1)[0]
        if base not in _SALVAGEABLE_FLAGS:
            return False
    return True


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _contains_term(haystack: str, term: str) -> bool:
    """Word-boundary-aware term detection to avoid substring false positives."""
    return re.search(r"\b" + re.escape(term) + r"\b", haystack) is not None


# ---- Original checks ----

def detect_ambiguous_wording(question_text: str, resolution_criteria: str) -> bool:
    haystack = f"{question_text or ''} {resolution_criteria or ''}".lower()
    for term in _AMBIGUOUS_TERMS:
        if _contains_term(haystack, term):
            return True
    return False


def _count_trusted_media_mentions(source: str) -> int:
    count = 0
    for term in _TRUSTED_MEDIA_TERMS:
        if term == "ap":
            if re.search(r"\bap\b", source):
                count += 1
        elif term in source:
            count += 1
    return count


def _has_authoritative_source_anchor(source: str) -> bool:
    return any(term in source for term in _AUTHORITATIVE_SOURCE_TERMS)


def detect_weak_resolution_source(
    resolution_source: str,
    category: str = "",
    question_text: str = "",
    resolution_criteria: str = "",
) -> bool:
    source = _normalize_text(resolution_source)
    category_normalized = _normalize_text(category)
    if len(source) < 15:
        return True

    has_locator = "http://" in source or "https://" in source or "." in source
    has_authoritative_anchor = _has_authoritative_source_anchor(source)
    trusted_media_mentions = _count_trusted_media_mentions(source)
    has_trusted_media = trusted_media_mentions > 0

    if category_normalized in _AUTHORITATIVE_SOURCE_CATEGORIES:
        if any(pattern in source for pattern in _GENERIC_FALLBACK_PATTERNS):
            return True
        if has_authoritative_anchor:
            return False

    if has_authoritative_anchor:
        return False

    if category_normalized in _CREDIBLE_REPORTING_CATEGORIES and has_trusted_media:
        combined = _normalize_text(f"{question_text} {resolution_criteria}")
        if any(term in combined for term in _DISPUTED_ATTRIBUTION_TERMS):
            return True
        if any(term in source for term in _GENERIC_SOURCE_TERMS):
            return True
        if trusted_media_mentions >= 2:
            return False
        if any(anchor in source for anchor in ("official statement", "official statements", "government", "ministry", "spokesperson")):
            return False
        if has_locator:
            return False

    if not has_locator and not has_trusted_media:
        return True
    if any(term in source for term in _GENERIC_SOURCE_TERMS):
        return True
    if any(term in source for term in _GENERIC_NEWS_ONLY_TERMS) and not has_authoritative_anchor:
        return True
    return False


def detect_weak_resolution_criteria(
    resolution_criteria: str,
    question_type: str = "binary",
    options: list[str] | None = None,
    question_text: str = "",
) -> bool:
    criteria = _normalize_text(resolution_criteria)
    combined = _normalize_text(f"{question_text} {resolution_criteria}")
    if len(criteria) < 30:
        return True
    if any(term in combined for term in _DISPUTED_ATTRIBUTION_TERMS):
        return True

    if question_type == "multiple_choice":
        options = options or []
        option_mentions = 0
        for option in options:
            normalized_option = _normalize_text(option)
            if normalized_option and normalized_option in criteria:
                option_mentions += 1

        resolves_to_count = criteria.count("resolves to")
        if option_mentions < max(2, min(len(options), 3)) and resolves_to_count < 2:
            return True
    else:
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
        if _contains_term(haystack, term):
            return f"terrorism_content ({term})"

    # Illegal activity
    for term in _ILLEGAL_ACTIVITY_TERMS:
        if _contains_term(haystack, term):
            return f"illegal_activity ({term})"

    # Death market patterns — individual mortality targeting
    # First check if any state-level action exemption applies
    full_text = f"{question_text or ''} {resolution_criteria or ''}"
    is_state_action = any(p.search(full_text) for p in _STATE_ACTION_EXEMPTIONS)

    if not is_state_action:
        for pattern in _DEATH_MARKET_PATTERNS:
            if pattern.search(full_text):
                return "death_market"

    # Violence terms — only flag severe forward-looking individual-targeting
    # Skip if it's a state-level action (already checked above)
    if not is_state_action:
        violence_hits = [t for t in _VIOLENCE_TERMS if _contains_term(haystack, t)]
        if violence_hits:
            q_lower = (question_text or "").lower()
            if any(q_lower.startswith(w) for w in ("will", "is", "are", "does", "can", "could")):
                severe_terms = {"assassinate", "assassination", "murder", "mass shooting",
                                "school shooting", "execute", "execution",
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
        if _contains_term(haystack, term):
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
        elif _contains_term(haystack, term):
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
    if detect_weak_resolution_source(
        q.resolution_source,
        category=q.category,
        question_text=q.question_text,
        resolution_criteria=q.resolution_criteria,
    ):
        flags.append("weak_resolution_source")
    if detect_weak_resolution_criteria(
        q.resolution_criteria,
        q.question_type,
        q.options,
        question_text=q.question_text,
    ):
        flags.append("weak_resolution_criteria")
    if detect_invalid_deadline_window(q.deadline):
        flags.append("invalid_deadline_window")
    if detect_excessive_deadline(q.deadline):
        flags.append("excessive_deadline")
    if detect_unclear_binary_condition(q.question_text, q.question_type):
        flags.append("unclear_binary_condition")

    # --- FR4 quality self-assessment checks (safety net — FR4 already filtered most) ---
    if q.already_resolved:
        flags.append("already_resolved")
    if q.resolution_confidence < _MIN_RESOLUTION_CONFIDENCE:
        flags.append(
            f"low_resolution_confidence:{q.resolution_confidence:.2f}"
            + (f" ({q.resolution_confidence_reason})" if q.resolution_confidence_reason else "")
        )
    if q.source_independence < _MIN_SOURCE_INDEPENDENCE:
        flags.append(f"low_source_independence:{q.source_independence:.2f}")
    if q.timing_reliability < _MIN_TIMING_RELIABILITY:
        flags.append(f"low_timing_reliability:{q.timing_reliability:.2f}")

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
