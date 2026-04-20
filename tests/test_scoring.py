"""Tests for FR6 deterministic scoring."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scoring.scorer import (
    _compute_regulatory_penalty,
    _has_regulatory_hard_exclude,
    compute_market_interest_score,
    compute_novelty_score,
    compute_resolution_strength_score,
    compute_time_horizon_score,
    _is_major_sports_event,
    _is_ordinary_match_question,
    is_homepage_source,
    is_media_event,
    is_retail_promo_event,
    is_trusted_resolution_source,
    is_weather_event,
    jaccard_similarity,
    normalize_minmax,
    normalize_text,
    score_questions,
    score_questions_with_breakdown,
    top_n_ranked_display_rows,
    tokenize_to_set,
)


def test_normalize_text_lowercase_remove_punctuation_collapse_whitespace():
    text = "  Hello,   WORLD!!  Fed-rate?\nCut.  "
    assert normalize_text(text) == "hello world fedrate cut"


def test_tokenize_to_set_deterministic():
    text = "Fed rate cut, fed rate"
    assert tokenize_to_set(text) == {"fed", "rate", "cut"}
    assert tokenize_to_set(text) == {"fed", "rate", "cut"}


def test_jaccard_similarity_zero_if_either_set_empty():
    assert jaccard_similarity(set(), {"a"}) == 0.0
    assert jaccard_similarity({"a"}, set()) == 0.0
    assert jaccard_similarity(set(), set()) == 0.0


def test_compute_novelty_score_buckets():
    # >= 0.9 -> 0.0
    assert compute_novelty_score("will fed cut rates", ["will fed cut rates"]) == 0.0

    # >= 0.7 -> 0.25 (4/5 = 0.8)
    assert compute_novelty_score("a b c d e", ["a b c d"]) == 0.25

    # >= 0.4 -> 0.5 (2/5 = 0.4)
    assert compute_novelty_score("a b c d", ["a b e"]) == 0.5

    # < 0.4 -> 1.0
    assert compute_novelty_score("x y z", ["a b c"]) == 1.0


def test_media_event_is_hard_excluded():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Reuters publish an article about Company X by June 30, 2026?",
            "category": "other",
            "deadline": "June 30, 2026",
            "resolution_source": "Reuters at https://www.reuters.com/world/",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        }
    ]
    all_texts = [(1, rows[0]["question_text"])]
    scored = score_questions(rows, all_texts)
    assert scored == []


def test_homepage_source_is_soft_penalty_not_exclusion():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Team A win the championship by December 31, 2026?",
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "League at https://example.com/results/finals",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will Team B win the championship by December 31, 2026?",
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "Example at https://www.wired.com/",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    by_id = {s.question_id: s for s in scored}
    assert 2 in by_id
    assert by_id[2].total_score < by_id[1].total_score


def test_weather_event_penalty_lowers_score():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Bitcoin close above $100,000 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Binance at https://www.binance.com/en/trade/BTC_USDT",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will New York temperature exceed 95 degrees by July 31, 2026?",
            "category": "other",
            "deadline": "July 31, 2026",
            "resolution_source": "NOAA data at https://www.weather.gov/wrh/climate",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    by_id = {s.question_id: s for s in scored}
    assert by_id[2].total_score < by_id[1].total_score


def test_novelty_compares_only_against_earlier_question_ids():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will the Fed cut rates in 2027?",
            "category": "finance",
            "deadline": "December 31, 2027",
            "resolution_source": "Federal Reserve at https://www.federalreserve.gov/monetarypolicy.htm",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will the Fed cut rates in 2027?",
            "category": "finance",
            "deadline": "December 31, 2027",
            "resolution_source": "Federal Reserve at https://www.federalreserve.gov/monetarypolicy.htm",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 3,
            "question_text": "Will Apple report revenue growth in 2027?",
            "category": "business",
            "deadline": "December 31, 2027",
            "resolution_source": "SEC at https://www.sec.gov/edgar/browse/",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [
        (1, "Will the Fed cut rates in 2027?"),
        (2, "Will the Fed cut rates in 2027?"),
        (3, "Will Apple report revenue growth in 2027?"),
    ]

    scored = score_questions(rows, all_texts)
    by_id = {s.question_id: s for s in scored}

    # Q1 has no earlier questions, so novelty should be max novelty.
    assert by_id[1].novelty_score == 1.0
    # Q2 is identical to Q1 (earlier), so novelty should be minimum.
    assert by_id[2].novelty_score == 0.0
    # Q3 compares with Q1/Q2 as earlier; should remain non-zero novelty.
    assert by_id[3].novelty_score in {0.25, 0.5, 1.0}


def test_minmax_normalization_normal_case():
    assert normalize_minmax(5.0, 0.0, 10.0) == 0.5
    assert normalize_minmax(0.0, 0.0, 10.0) == 0.0
    assert normalize_minmax(10.0, 0.0, 10.0) == 1.0


def test_minmax_normalization_equal_min_max_returns_one():
    assert normalize_minmax(5.0, 5.0, 5.0) == 1.0
    assert normalize_minmax(0.0, 0.0, 0.0) == 1.0


def test_market_interest_sports_finals_high_regular_match_medium():
    assert compute_market_interest_score(
        "Will Team A win the World Cup final by July 31, 2026?",
        "sports",
    ) == 1.0
    assert compute_market_interest_score(
        "Will Team A win its next regular season match?",
        "sports",
    ) == 0.4


def test_market_interest_low_bucket_for_promo_and_weather():
    assert compute_market_interest_score(
        "Will Expedia launch a summer discount promotion by June 30, 2026?",
        "other",
    ) == 0.1
    assert compute_market_interest_score(
        "Will Boston weather stay below 30 degrees this week?",
        "other",
    ) == 0.1


def test_resolution_strength_and_time_horizon_scoring():
    assert compute_resolution_strength_score(
        "SEC filings at https://www.sec.gov/edgar/browse/"
    ) == 1.0
    assert compute_resolution_strength_score(
        "Wired at https://www.wired.com/"
    ) == 0.2
    assert compute_time_horizon_score("December 31, 2026") in {0.6, 0.9, 1.0, 0.3, 0.2, 0.0}


def test_source_and_media_detection_helpers():
    assert is_homepage_source("Wired at https://www.wired.com/") is True
    assert is_homepage_source("SEC at https://www.sec.gov/edgar/browse/") is False
    assert is_media_event("Will Reuters publish an article about Tesla this week?") is True
    assert is_media_event("Will Tesla deliver over 500k vehicles this quarter?") is False
    assert is_weather_event("Will NYC temperature exceed 95 degrees tomorrow?") is True
    assert is_weather_event("Will Team A win the championship?") is False
    assert is_trusted_resolution_source("Binance at https://www.binance.com/") is True
    assert is_trusted_resolution_source("Yahoo Finance at https://finance.yahoo.com/") is True
    assert is_trusted_resolution_source("Generic at https://example.com/") is False


def test_total_score_uses_exact_weighted_formula():
    rows = [
        {
            "question_id": 10,
            "question_text": "Will Bitcoin close above $100,000 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Binance at https://www.binance.com/en/trade/BTC_USDT",
            "mention_velocity": 2.0,
            "source_diversity": 4.0,
            "clarity_score": 0.8,
        },
        {
            "question_id": 11,
            "question_text": "Will B happen?",
            "category": "other",
            "deadline": "December 31, 2026",
            "resolution_source": "Example at https://example.com/results",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 0.6,
        },
    ]
    all_texts = [(10, "Will A happen?"), (11, "Will B happen?")]
    scored = score_questions(rows, all_texts)
    by_id = {s.question_id: s for s in scored}

    s10 = by_id[10]
    # New weights: market_interest 25%, resolution_strength 25%, clarity 20%,
    # mention_velocity 10%, novelty 10%, time_horizon 5%, source_diversity 5%
    expected = (
        0.25 * s10.market_interest_score
        + 0.25 * s10.resolution_strength_score
        + 0.20 * s10.clarity_score
        + 0.10 * s10.mention_velocity_score
        + 0.10 * s10.novelty_score
        + 0.05 * s10.time_horizon_score
        + 0.05 * s10.source_diversity_score
    )
    assert s10.total_score == expected


def test_championship_question_ranks_above_ordinary_low_interest_question():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Team A win the championship final by December 31, 2026?",
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "League results at https://example.com/results/finals",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will Team A win its next regular season match?",
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "League fixtures at https://example.com/fixtures",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    assert [s.question_id for s in scored][0] == 1


def test_retail_promo_question_ranks_below_major_championship_question():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Team A win the championship final by December 31, 2026?",
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "League results at https://example.com/results/finals",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will Expedia offer 40% off sale coupon promotion by December 31, 2026?",
            "category": "other",
            "deadline": "December 31, 2026",
            "resolution_source": "Expedia deals at https://example.com/deals/sale",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    assert [s.question_id for s in scored][0] == 1
    assert is_retail_promo_event(rows[1]["question_text"]) is True


def test_trusted_finance_homepage_source_not_penalized_like_non_trusted():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Bitcoin close above $100,000 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Binance at https://www.binance.com/",
            "mention_velocity": 1.0,
            "source_diversity": 1.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will Bitcoin close above $100,000 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Example at https://example.com/",
            "mention_velocity": 1.0,
            "source_diversity": 1.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    by_id = {s.question_id: s for s in scored}
    assert by_id[1].total_score > by_id[2].total_score


def test_finance_crypto_overrides_apply_to_velocity_and_resolution_strength():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Dogecoin closing price exceed $1.00 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Binance at https://www.binance.com/",
            "mention_velocity": 0.1,
            "source_diversity": 1.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will an ordinary event happen by December 31, 2026?",
            "category": "other",
            "deadline": "December 31, 2026",
            "resolution_source": "Example at https://example.com/results",
            "mention_velocity": 10.0,
            "source_diversity": 1.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored, breakdowns = score_questions_with_breakdown(rows, all_texts)
    b1 = breakdowns[1]["component_scores"]
    assert b1["mention_velocity_score"] >= 0.8
    assert b1["resolution_strength_score"] >= 0.8


def test_finance_price_question_ranks_above_ordinary_next_match_when_comparable():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Bitcoin closing price exceed $100,000 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Binance at https://www.binance.com/",
            "mention_velocity": 0.5,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will Team A win its next match by December 31, 2026?",
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "League at https://example.com/fixtures",
            "mention_velocity": 0.5,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    assert [s.question_id for s in scored][0] == 1


def test_score_functions_remain_aligned_after_calibration():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Tesla stock closing price exceed $300 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Yahoo Finance at https://finance.yahoo.com/",
            "mention_velocity": 0.2,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will Team A win its next match by December 31, 2026?",
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "League at https://example.com/fixtures",
            "mention_velocity": 0.2,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored_simple = score_questions(rows, all_texts)
    scored_with_breakdown, _ = score_questions_with_breakdown(rows, all_texts)

    simple_map = {s.question_id: s.total_score for s in scored_simple}
    breakdown_map = {s.question_id: s.total_score for s in scored_with_breakdown}
    assert simple_map == breakdown_map


def test_major_sports_event_helper_detects_world_series_and_relegation():
    assert _is_major_sports_event("Will Team A win the World Series by October 31, 2026?") is True
    assert _is_major_sports_event("Will Club X be relegated by May 1, 2026?") is True


def test_ordinary_match_helper_detects_next_match_and_score_props():
    assert _is_ordinary_match_question("Will Team A win its next match by April 30, 2026?") is True
    assert _is_ordinary_match_question("Will Team A score more goals than Team B on April 30, 2026?") is True


def test_major_sports_event_ranks_above_ordinary_match_when_comparable():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Team A win the World Series by October 31, 2026?",
            "category": "sports",
            "deadline": "October 31, 2026",
            "resolution_source": "MLB at https://www.mlb.com/postseason",
            "mention_velocity": 0.5,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will Team B win its next match by October 31, 2026?",
            "category": "sports",
            "deadline": "October 31, 2026",
            "resolution_source": "League at https://example.com/fixtures",
            "mention_velocity": 0.5,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    assert [s.question_id for s in scored][0] == 1


def test_finance_price_still_ranks_above_ordinary_match_after_major_sports_calibration():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Bitcoin closing price exceed $100,000 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Binance at https://www.binance.com/",
            "mention_velocity": 0.5,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will Team A win its next match by December 31, 2026?",
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "League at https://example.com/fixtures",
            "mention_velocity": 0.5,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    assert [s.question_id for s in scored][0] == 1


def test_score_functions_remain_aligned_after_major_sports_calibration():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Team A win the playoff championship final by December 31, 2026?",
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "League at https://example.com/results/finals",
            "mention_velocity": 0.4,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will Team B win against FC Alpha by April 30, 2026?",
            "category": "sports",
            "deadline": "April 30, 2026",
            "resolution_source": "League at https://example.com/fixtures",
            "mention_velocity": 0.4,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    simple_map = {s.question_id: s.total_score for s in score_questions(rows, all_texts)}
    breakdown_map = {s.question_id: s.total_score for s in score_questions_with_breakdown(rows, all_texts)[0]}
    assert simple_map == breakdown_map


def test_ranking_sorts_desc_then_question_id_and_ranks_sequentially():
    rows = [
        {
            "question_id": 1,
            "question_text": "Q1 unique",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "SEC at https://www.sec.gov/edgar/browse/",
            "mention_velocity": 1.0,
            "source_diversity": 1.0,
            "clarity_score": 0.8,
        },
        {
            "question_id": 2,
            "question_text": "Q2 unique",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "SEC at https://www.sec.gov/edgar/browse/",
            "mention_velocity": 1.0,
            "source_diversity": 1.0,
            "clarity_score": 0.8,
        },
    ]
    all_texts = [(1, "Q1 unique"), (2, "Q2 unique")]
    scored = score_questions(rows, all_texts)

    # With same features and metadata, totals should be equal.
    # Tie-break must be question_id ASC.
    assert [s.question_id for s in scored] == [1, 2]
    assert [s.rank for s in scored] == [1, 2]


def test_score_questions_with_breakdown_contains_components_flags_and_explanation():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Bitcoin close above $100,000 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Binance at https://www.binance.com/en/trade/BTC_USDT",
            "mention_velocity": 2.0,
            "source_diversity": 3.0,
            "clarity_score": 1.0,
        }
    ]
    all_texts = [(1, rows[0]["question_text"])]
    scored, breakdowns = score_questions_with_breakdown(rows, all_texts)

    assert len(scored) == 1
    assert 1 in breakdowns
    b = breakdowns[1]
    assert "component_scores" in b
    assert "quality_flags" in b
    assert "final_clamped_score" in b
    assert "explanation" in b
    assert isinstance(b["explanation"], str) and len(b["explanation"]) > 0


def test_top_n_ranked_display_rows_shape_and_limit():
    rows = [
        {
            "rank": 1,
            "question_text": "Q1",
            "total_score": 0.9,
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Source at https://example.com/data",
            "explanation": "High rank.",
            "clarity_score": 1.0,
            "mention_velocity_score": 1.0,
            "source_diversity_score": 1.0,
            "novelty_score": 1.0,
            "market_interest_score": 1.0,
            "resolution_strength_score": 1.0,
            "time_horizon_score": 0.9,
            "quality_flags": [],
        },
        {
            "rank": 2,
            "question_text": "Q2",
            "total_score": 0.8,
            "category": "sports",
            "deadline": "December 31, 2026",
            "resolution_source": "Source at https://example.com/results",
            "explanation": "Mid rank.",
            "clarity_score": 0.9,
            "mention_velocity_score": 0.8,
            "source_diversity_score": 0.7,
            "novelty_score": 1.0,
            "market_interest_score": 0.4,
            "resolution_strength_score": 1.0,
            "time_horizon_score": 0.9,
            "quality_flags": ["near_duplicate_theme"],
        },
    ]
    top = top_n_ranked_display_rows(rows, top_n=1)
    assert len(top) == 1
    item = top[0]
    assert set(item.keys()) == {
        "rank",
        "question_text",
        "total_score",
        "category",
        "deadline",
        "resolution_source",
        "explanation",
        "score_breakdown",
    }


# ---- Regulatory hard exclusion tests ----

def test_has_regulatory_hard_exclude_prohibited_topic():
    assert _has_regulatory_hard_exclude(["prohibited_topic:terrorism_content (terrorist)"]) is True


def test_has_regulatory_hard_exclude_manipulation_risk():
    assert _has_regulatory_hard_exclude(["manipulation_risk"]) is True


def test_has_regulatory_hard_exclude_pii():
    assert _has_regulatory_hard_exclude(["pii_exposure"]) is True


def test_has_regulatory_hard_exclude_clean():
    assert _has_regulatory_hard_exclude(["ambiguous_wording", "weak_resolution_source"]) is False


def test_has_regulatory_hard_exclude_empty():
    assert _has_regulatory_hard_exclude([]) is False


# ---- Regulatory soft penalty tests ----

def test_compute_regulatory_penalty_minor_involvement():
    assert _compute_regulatory_penalty(["minor_involvement"]) == 0.15


def test_compute_regulatory_penalty_excessive_deadline():
    assert _compute_regulatory_penalty(["excessive_deadline"]) == 0.15


def test_compute_regulatory_penalty_both():
    assert _compute_regulatory_penalty(["minor_involvement", "excessive_deadline"]) == 0.30


def test_compute_regulatory_penalty_no_flags():
    assert _compute_regulatory_penalty([]) == 0.0


def test_compute_regulatory_penalty_ignores_non_regulatory_flags():
    assert _compute_regulatory_penalty(["ambiguous_wording", "weak_resolution_source"]) == 0.0


# ---- Hard exclusion in scoring pipeline ----

def test_regulatory_hard_exclude_removes_question_from_scoring():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Bitcoin close above $100,000 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "Binance at https://www.binance.com/en/trade/BTC_USDT",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
            "validation_flags": ["prohibited_topic:terrorism_content (terrorist)"],
        },
        {
            "question_id": 2,
            "question_text": "Will Apple stock close above $200 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "NASDAQ at https://www.nasdaq.com/market-activity/stocks/aapl",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
            "validation_flags": [],
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    ids = [s.question_id for s in scored]
    assert 1 not in ids
    assert 2 in ids


def test_regulatory_soft_penalty_lowers_score():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will Apple stock close above $200 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "NASDAQ at https://www.nasdaq.com/market-activity/stocks/aapl",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
            "validation_flags": [],
        },
        {
            "question_id": 2,
            "question_text": "Will Google stock close above $200 by December 31, 2026?",
            "category": "finance",
            "deadline": "December 31, 2026",
            "resolution_source": "NASDAQ at https://www.nasdaq.com/market-activity/stocks/googl",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
            "validation_flags": ["minor_involvement"],
        },
    ]
    all_texts = [(1, rows[0]["question_text"]), (2, rows[1]["question_text"])]
    scored = score_questions(rows, all_texts)
    by_id = {s.question_id: s for s in scored}
    assert by_id[2].total_score < by_id[1].total_score


# ---- Expanded trusted source tests ----

def test_trusted_source_government_domains():
    assert is_trusted_resolution_source("BLS at https://www.bls.gov/data/") is True
    assert is_trusted_resolution_source("Congress at https://congress.gov/bill/") is True
    assert is_trusted_resolution_source("Federal Register at https://federalregister.gov/") is True
    assert is_trusted_resolution_source("FRED at https://fred.stlouisfed.org/series/") is True
    assert is_trusted_resolution_source("WHO at https://who.int/data/") is True
    assert is_trusted_resolution_source("UN at https://un.org/en/") is True


def test_trusted_source_sports_bodies():
    assert is_trusted_resolution_source("FIFA at https://www.fifa.com/worldcup") is True
    assert is_trusted_resolution_source("NBA at https://www.nba.com/standings") is True
    assert is_trusted_resolution_source("ESPN at https://www.espn.com/nba/standings") is True


def test_untrusted_source():
    assert is_trusted_resolution_source("Random blog at https://myblog.example.com/") is False


# ---- Updated homepage source tests ----

def test_homepage_source_generic_path_check():
    assert is_homepage_source("Site at https://example.com/") is True
    assert is_homepage_source("Site at https://example.com/home") is True
    assert is_homepage_source("Site at https://example.com/index.html") is True
    assert is_homepage_source("Site at https://example.com/data/results") is False


def test_homepage_source_no_url():
    assert is_homepage_source("Just some text, no URL") is True

