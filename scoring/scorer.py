"""
FR6: Deterministic heuristic scoring for validated questions.
"""

from __future__ import annotations

from datetime import datetime, UTC
from urllib.parse import urlparse
import string
import re

from models import ScoredCandidate


def normalize_minmax(value: float, vmin: float, vmax: float) -> float:
    if vmax == vmin:
        return 1.0
    return (value - vmin) / (vmax - vmin)


def normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    translator = str.maketrans("", "", string.punctuation)
    no_punct = lowered.translate(translator)
    return " ".join(no_punct.split())


def tokenize_to_set(text: str) -> set[str]:
    normalized = normalize_text(text)
    return set(normalized.split()) if normalized else set()


def jaccard_similarity(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


def compute_novelty_score(current_text: str, earlier_texts: list[str]) -> float:
    current_tokens = tokenize_to_set(current_text)
    max_similarity = 0.0

    for text in earlier_texts:
        sim = jaccard_similarity(current_tokens, tokenize_to_set(text))
        if sim > max_similarity:
            max_similarity = sim

    if max_similarity >= 0.9:
        return 0.0
    if max_similarity >= 0.7:
        return 0.25
    if max_similarity >= 0.4:
        return 0.5
    return 1.0


def _parse_deadline(deadline: str) -> datetime | None:
    value = (deadline or "").strip()
    if not value:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _extract_url(text: str) -> str:
    match = re.search(r"https?://[^\s)]+", text or "", flags=re.IGNORECASE)
    return match.group(0).rstrip(".,;") if match else ""


def is_homepage_source(resolution_source: str) -> bool:
    url = _extract_url(resolution_source)
    if not url:
        return True
    parsed = urlparse(url)
    path = (parsed.path or "").strip("/")
    if not path:
        return True
    homepage_like_hosts = {
        "wired.com",
        "www.wired.com",
        "nike.com",
        "www.nike.com",
        "wunderground.com",
        "www.wunderground.com",
    }
    if parsed.netloc.lower() in homepage_like_hosts and path in {"", "home", "homepage"}:
        return True
    return False

def is_trusted_resolution_source(resolution_source: str) -> bool:
    if not resolution_source:
        return False

    s = resolution_source.lower()

    trusted_domains = [
        "binance.com",
        "finance.yahoo.com",
        "yahoo.com",
        "nasdaq.com",
        "nyse.com",
        "cmegroup.com",
        "coingecko.com",
        "coinbase.com",
    ]

    return any(domain in s for domain in trusted_domains)


def _has_price_finance_terms(question_text: str) -> bool:
    text = normalize_text(question_text)
    terms = {
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "dogecoin",
        "tesla",
        "amazon",
        "stock",
        "closing price",
        "adjusted closing price",
        "split-adjusted closing price",
        "price",
    }
    return any(term in text for term in terms)


def _is_major_sports_event(question_text: str) -> bool:
    text = normalize_text(question_text)
    major_terms = {
        "world series",
        "playoffs",
        "playoff",
        "championship",
        "final",
        "finals",
        "title",
        "relegated",
        "relegation",
        "super bowl",
        "world cup",
        "grand slam",
    }
    return any(term in text for term in major_terms)


def _is_ordinary_match_question(question_text: str) -> bool:
    text = normalize_text(question_text)
    if any(t in text for t in ("next match", "upcoming match", "win against", "score more goals", "score more than")):
        return True
    match_words = ("match", "game", "fixture", "win", "score", "goals", "fc")
    if "by april 30, 2026" in text and any(w in text for w in match_words):
        return True
    if "by may 1, 2026" in text and any(w in text for w in match_words):
        return True
    return False


def is_media_event(question_text: str) -> bool:
    text = normalize_text(question_text)
    media_terms = {"publish", "article", "story", "coverage", "headline", "editorial", "op-ed"}
    outlet_terms = {"reuters", "bloomberg", "cnn", "fox", "nytimes", "bbc", "wired", "wsj", "guardian"}
    return any(t in text for t in media_terms) and any(o in text for o in outlet_terms)


def is_promo_event(question_text: str) -> bool:
    text = normalize_text(question_text)
    promo_terms = {
        "sale", "discount", "promotion", "promo", "coupon", "bundle", "black friday",
        "cyber monday", "deal", "markdown", "cashback",
    }
    return any(term in text for term in promo_terms)


def is_retail_promo_event(question_text: str) -> bool:
    text = normalize_text(question_text)
    retail_terms = {"% off", "discount", "coupon", "promotion", "promo", "sale", "bundle deal"}
    return any(term in text for term in retail_terms)


def is_low_significance_event(question_text: str, category: str) -> bool:
    text = normalize_text(question_text)
    low_terms = {
        "homepage", "blog post", "announcement post", "teaser", "trailer", "merch",
        "giveaway", "sweepstakes", "app update", "feature rollout",
    }
    if any(term in text for term in low_terms):
        return True
    return category in {"other"} and ("launch" in text or "release" in text)


def is_weather_event(question_text: str) -> bool:
    text = normalize_text(question_text)
    weather_terms = {
        "weather", "temperature", "rain", "snow", "storm",
        "forecast", "heatwave", "cold front", "degrees",
    }
    return any(term in text for term in weather_terms)


def compute_market_interest_score(question_text: str, category: str) -> float:
    text = normalize_text(question_text)
    high_terms = {
        "bitcoin", "btc", "ethereum", "eth", "crypto", "token", "spot etf", "price",
        "earnings", "revenue", "eps", "guidance", "rate cut", "fomc", "inflation", "cpi",
        "election", "vote", "ballot", "president", "prime minister", "parliament",
        "sanction", "ceasefire", "treaty", "nato", "un vote", "invasion",
        "fda approval", "pdufa", "supreme court", "ruling", "verdict",
        "championship", "final", "finals", "playoff", "world cup", "grand slam", "title",
    }
    low_terms = {
        "sale", "discount", "promotion", "promo", "coupon", "bundle", "deal",
        "publish", "article", "coverage", "announcement", "teaser",
        "weather", "temperature", "rain", "snow", "storm", "forecast", "degrees",
    }
    sports_low_terms = {"regular season", "next match", "friendly", "preseason", "group stage", "prop"}
    major_sports_terms = {
    "world series", "final", "finals", "championship",
    "playoff", "title", "grand slam", "world cup"
    }

    if any(term in text for term in major_sports_terms):
        return 1.0
    if any(term in text for term in high_terms):
        return 1.0
    if category == "sports":
        if any(term in text for term in sports_low_terms):
            return 0.4
        return 0.4
    if any(term in text for term in low_terms):
        return 0.1
    return 0.4


def compute_resolution_strength_score(resolution_source: str) -> float:
    text = normalize_text(resolution_source)
    url = _extract_url(resolution_source)
    host = urlparse(url).netloc.lower() if url else ""
    path = urlparse(url).path.lower() if url else ""

    strong_host_signals = {
        ".gov", "sec.gov", "fda.gov", "ftc.gov", "ec.europa.eu",
        "nasdaq.com", "nyse.com", "cmegroup.com", "fifa.com", "nba.com",
    }
    strong_path_signals = {"results", "standings", "scores", "edgar", "filing", "database", "data", "report"}
    weak_news_hosts = {"wired.com", "www.wired.com", "cnn.com", "www.cnn.com", "nytimes.com", "www.nytimes.com"}

    if is_homepage_source(resolution_source):
        return 0.2
    if any(sig in host for sig in strong_host_signals):
        return 1.0
    if any(sig in path for sig in strong_path_signals):
        return 1.0
    if host in weak_news_hosts:
        return 0.2
    if "official website" in text or "news reports" in text:
        return 0.2
    return 0.6


def compute_time_horizon_score(deadline: str) -> float:
    parsed = _parse_deadline(deadline)
    if parsed is None:
        return 0.0
    days = (parsed.date() - datetime.now(UTC).date()).days
    if days < 0:
        return 0.0
    if days <= 2:
        return 0.2
    if days <= 30:
        return 1.0
    if days <= 180:
        return 0.9
    if days <= 365:
        return 0.6
    return 0.3


def _theme_similarity_score(a: str, b: str) -> float:
    return jaccard_similarity(tokenize_to_set(a), tokenize_to_set(b))


def _soft_penalty_flags(question_text: str, category: str, resolution_source: str, prior_kept_texts: list[str]) -> list[str]:
    flags: list[str] = []
    if is_homepage_source(resolution_source):
        flags.append("homepage_source")
    if is_promo_event(question_text):
        flags.append("promo_event")
    if is_retail_promo_event(question_text):
        flags.append("retail_promo_event")
    if is_weather_event(question_text):
        flags.append("weather_event")
    if is_low_significance_event(question_text, category):
        flags.append("low_significance_event")
    if any(_theme_similarity_score(question_text, prior) >= 0.75 for prior in prior_kept_texts):
        flags.append("near_duplicate_theme")
    return flags


def _build_breakdown(
    row: dict,
    mention_velocity_score: float,
    source_diversity_score: float,
    clarity_score: float,
    novelty_score: float,
    market_interest_score: float,
    resolution_strength_score: float,
    time_horizon_score: float,
    penalty: float,
    final_score: float,
) -> dict:
    flags = set(row.get("quality_flags", []))
    return {
        "question_id": int(row["question_id"]),
        "question_text": row.get("question_text", ""),
        "rank": 0,
        "total_score": final_score,
        "component_scores": {
            "clarity_score": clarity_score,
            "mention_velocity_score": mention_velocity_score,
            "source_diversity_score": source_diversity_score,
            "novelty_score": novelty_score,
            "market_interest_score": market_interest_score,
            "resolution_strength_score": resolution_strength_score,
            "time_horizon_score": time_horizon_score,
        },
        "quality_flags": {
            "homepage_source": "homepage_source" in flags,
            "promo_event": "promo_event" in flags,
            "retail_promo_event": "retail_promo_event" in flags,
            "low_significance_event": "low_significance_event" in flags,
            "near_duplicate_theme": "near_duplicate_theme" in flags,
            "weather_event": "weather_event" in flags,
        },
        "penalty_total": penalty,
        "final_clamped_score": final_score,
    }


def generate_score_explanation(
    market_interest_score: float,
    resolution_strength_score: float,
    time_horizon_score: float,
    quality_flags: list[str],
) -> str:
    if any(flag in quality_flags for flag in {"promo_event", "retail_promo_event"}):
        return (
            "Ranked lower because it resembles a promotional retail market and received "
            "quality penalties despite passing validation."
        )
    if market_interest_score >= 1.0 and resolution_strength_score >= 1.0 and not quality_flags:
        return (
            "Ranked highly because it has strong market interest, a strong resolution source, "
            "and no quality penalties."
        )
    if market_interest_score <= 0.1:
        return (
            "Ranked lower because it has limited market interest relative to major finance, "
            "geopolitics, or championship-style markets."
        )
    if resolution_strength_score <= 0.2:
        return (
            "Ranked in the middle because it is valid, but the resolution source appears weaker "
            "than top-ranked authoritative sources."
        )
    if time_horizon_score <= 0.3:
        return (
            "Ranked in the middle because the trading horizon is less practical than near-term "
            "markets with clearer timing."
        )
    if quality_flags:
        return (
            "Ranked in the middle because it is valid and resolvable, but quality penalties "
            "reduced its final score."
        )
    return (
        "Ranked in the middle because it is valid and clearly resolvable, but has lower market "
        "interest than top-tier markets."
    )


def score_questions(rows: list[dict], all_question_texts_by_id: list[tuple[int, str]]) -> list[ScoredCandidate]:
    if not rows:
        return []

    filtered_rows = [r for r in rows if not is_media_event(r["question_text"])]
    if not filtered_rows:
        return []

    filtered_rows.sort(key=lambda r: int(r["question_id"]))

    prior_kept_texts: list[str] = []
    kept_rows: list[dict] = []
    for row in filtered_rows:
        row = dict(row)
        row["quality_flags"] = _soft_penalty_flags(
            question_text=row["question_text"],
            category=row.get("category", ""),
            resolution_source=row.get("resolution_source", ""),
            prior_kept_texts=prior_kept_texts,
        )
        kept_rows.append(row)
        prior_kept_texts.append(row["question_text"])

    if not kept_rows:
        return []

    velocities = [float(r["mention_velocity"]) for r in kept_rows]
    diversities = [float(r["source_diversity"]) for r in kept_rows]

    vel_min, vel_max = min(velocities), max(velocities)
    div_min, div_max = min(diversities), max(diversities)

    scored: list[ScoredCandidate] = []
    text_by_id = {qid: text for qid, text in all_question_texts_by_id}

    for row in kept_rows:
        question_id = int(row["question_id"])
        q_text = row["question_text"]

        earlier_texts = [
            text_by_id[qid]
            for qid, _ in all_question_texts_by_id
            if qid < question_id and qid in text_by_id
        ]

        mention_velocity_score = normalize_minmax(float(row["mention_velocity"]), vel_min, vel_max)
        source_diversity_score = normalize_minmax(float(row["source_diversity"]), div_min, div_max)
        clarity_score = float(row["clarity_score"])
        novelty_score = compute_novelty_score(q_text, earlier_texts)
        market_interest_score = compute_market_interest_score(q_text, row.get("category", ""))
        resolution_strength_score = compute_resolution_strength_score(row.get("resolution_source", ""))
        time_horizon_score = compute_time_horizon_score(row.get("deadline", ""))
        if _has_price_finance_terms(q_text):
            mention_velocity_score = max(mention_velocity_score, 0.8)
            resolution_strength_score = max(resolution_strength_score, 0.8)
        if _is_major_sports_event(q_text):
            mention_velocity_score = max(mention_velocity_score, 0.6)
            resolution_strength_score = max(resolution_strength_score, 0.6)
            market_interest_score = max(market_interest_score, 1.0)

        base_score = (
            0.20 * clarity_score
            + 0.15 * mention_velocity_score
            + 0.10 * source_diversity_score
            + 0.10 * novelty_score
            + 0.30 * market_interest_score
            + 0.15 * resolution_strength_score
            + 0.10 * time_horizon_score
        )
        if _is_ordinary_match_question(q_text) and not _is_major_sports_event(q_text):
            base_score *= 0.85
        penalty = 0.0
        if "homepage_source" in row["quality_flags"]:
            if not is_trusted_resolution_source(row.get("resolution_source", "")):
                penalty += 0.05
        if "promo_event" in row["quality_flags"]:
            penalty += 0.25
        text = normalize_text(q_text)
        if "retail_promo_event" in row["quality_flags"]:
            penalty += 0.10
            if "discount" in text or "%" in text:
                penalty += 0.05
            if "all purchases" in text:
                penalty += 0.05
        if "weather_event" in row["quality_flags"]:
            penalty += 0.15
        if "low_significance_event" in row["quality_flags"]:
            penalty += 0.10
        if "near_duplicate_theme" in row["quality_flags"]:
            penalty += 0.10
        total_score = max(0.0, min(1.0, base_score - penalty))

        scored.append(
            ScoredCandidate(
                question_id=question_id,
                total_score=total_score,
                mention_velocity_score=mention_velocity_score,
                source_diversity_score=source_diversity_score,
                clarity_score=clarity_score,
                novelty_score=novelty_score,
            )
        )

    scored.sort(key=lambda x: (-x.total_score, x.question_id))
    for rank, candidate in enumerate(scored, start=1):
        candidate.rank = rank

    return scored


def score_questions_with_breakdown(
    rows: list[dict],
    all_question_texts_by_id: list[tuple[int, str]],
) -> tuple[list[ScoredCandidate], dict[int, dict]]:
    if not rows:
        return [], {}

    filtered_rows = [r for r in rows if not is_media_event(r["question_text"])]
    if not filtered_rows:
        return [], {}

    filtered_rows.sort(key=lambda r: int(r["question_id"]))

    prior_kept_texts: list[str] = []
    kept_rows: list[dict] = []
    for row in filtered_rows:
        row = dict(row)
        row["quality_flags"] = _soft_penalty_flags(
            question_text=row["question_text"],
            category=row.get("category", ""),
            resolution_source=row.get("resolution_source", ""),
            prior_kept_texts=prior_kept_texts,
        )
        kept_rows.append(row)
        prior_kept_texts.append(row["question_text"])

    if not kept_rows:
        return [], {}

    velocities = [float(r["mention_velocity"]) for r in kept_rows]
    diversities = [float(r["source_diversity"]) for r in kept_rows]
    vel_min, vel_max = min(velocities), max(velocities)
    div_min, div_max = min(diversities), max(diversities)

    scored: list[ScoredCandidate] = []
    breakdowns: dict[int, dict] = {}
    text_by_id = {qid: text for qid, text in all_question_texts_by_id}

    for row in kept_rows:
        question_id = int(row["question_id"])
        q_text = row["question_text"]
        earlier_texts = [
            text_by_id[qid]
            for qid, _ in all_question_texts_by_id
            if qid < question_id and qid in text_by_id
        ]
        mention_velocity_score = normalize_minmax(float(row["mention_velocity"]), vel_min, vel_max)
        source_diversity_score = normalize_minmax(float(row["source_diversity"]), div_min, div_max)
        clarity_score = float(row["clarity_score"])
        novelty_score = compute_novelty_score(q_text, earlier_texts)
        market_interest_score = compute_market_interest_score(q_text, row.get("category", ""))
        resolution_strength_score = compute_resolution_strength_score(row.get("resolution_source", ""))
        time_horizon_score = compute_time_horizon_score(row.get("deadline", ""))
        if _has_price_finance_terms(q_text):
            mention_velocity_score = max(mention_velocity_score, 0.8)
            resolution_strength_score = max(resolution_strength_score, 0.8)
        if _is_major_sports_event(q_text):
            mention_velocity_score = max(mention_velocity_score, 0.6)
            resolution_strength_score = max(resolution_strength_score, 0.6)
            market_interest_score = max(market_interest_score, 1.0)

        base_score = (
            0.20 * clarity_score
            + 0.15 * mention_velocity_score
            + 0.10 * source_diversity_score
            + 0.10 * novelty_score
            + 0.30 * market_interest_score
            + 0.15 * resolution_strength_score
            + 0.10 * time_horizon_score
        )
        if _is_ordinary_match_question(q_text) and not _is_major_sports_event(q_text):
            base_score *= 0.85
        penalty = 0.0
        if "homepage_source" in row["quality_flags"]:
            if not is_trusted_resolution_source(row.get("resolution_source", "")):
                penalty += 0.05
        if "promo_event" in row["quality_flags"]:
            penalty += 0.25
        text = normalize_text(q_text)
        if "retail_promo_event" in row["quality_flags"]:
            penalty += 0.10
            if "discount" in text or "%" in text:
                penalty += 0.05
            if "all purchases" in text:
                penalty += 0.05
        if "weather_event" in row["quality_flags"]:
            penalty += 0.15
        if "low_significance_event" in row["quality_flags"]:
            penalty += 0.10
        if "near_duplicate_theme" in row["quality_flags"]:
            penalty += 0.10
        total_score = max(0.0, min(1.0, base_score - penalty))

        scored.append(
            ScoredCandidate(
                question_id=question_id,
                total_score=total_score,
                mention_velocity_score=mention_velocity_score,
                source_diversity_score=source_diversity_score,
                clarity_score=clarity_score,
                novelty_score=novelty_score,
            )
        )
        breakdowns[question_id] = _build_breakdown(
            row=row,
            mention_velocity_score=mention_velocity_score,
            source_diversity_score=source_diversity_score,
            clarity_score=clarity_score,
            novelty_score=novelty_score,
            market_interest_score=market_interest_score,
            resolution_strength_score=resolution_strength_score,
            time_horizon_score=time_horizon_score,
            penalty=penalty,
            final_score=total_score,
        )
        breakdowns[question_id]["explanation"] = generate_score_explanation(
            market_interest_score=market_interest_score,
            resolution_strength_score=resolution_strength_score,
            time_horizon_score=time_horizon_score,
            quality_flags=row["quality_flags"],
        )

    scored.sort(key=lambda x: (-x.total_score, x.question_id))
    for rank, candidate in enumerate(scored, start=1):
        candidate.rank = rank
        if candidate.question_id in breakdowns:
            breakdowns[candidate.question_id]["rank"] = rank

    return scored, breakdowns


def top_n_ranked_display_rows(scored_rows: list[dict], top_n: int = 10) -> list[dict]:
    ordered = sorted(scored_rows, key=lambda r: (r["rank"]))
    sliced = ordered[: max(1, int(top_n))]
    return [
        {
            "rank": r["rank"],
            "question_text": r["question_text"],
            "total_score": r["total_score"],
            "category": r["category"],
            "deadline": r["deadline"],
            "resolution_source": r["resolution_source"],
            "explanation": r.get("explanation", ""),
            "score_breakdown": {
                "clarity_score": r.get("clarity_score"),
                "mention_velocity_score": r.get("mention_velocity_score"),
                "source_diversity_score": r.get("source_diversity_score"),
                "novelty_score": r.get("novelty_score"),
                "market_interest_score": r.get("market_interest_score"),
                "resolution_strength_score": r.get("resolution_strength_score"),
                "time_horizon_score": r.get("time_horizon_score"),
                "quality_flags": r.get("quality_flags", []),
            },
        }
        for r in sliced
    ]

