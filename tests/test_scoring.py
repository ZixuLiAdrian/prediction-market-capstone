"""Tests for FR6 deterministic scoring."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scoring.scorer import (
    compute_novelty_score,
    jaccard_similarity,
    normalize_minmax,
    normalize_text,
    score_questions,
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


def test_novelty_compares_only_against_earlier_question_ids():
    rows = [
        {
            "question_id": 1,
            "question_text": "Will the Fed cut rates in 2027?",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 2,
            "question_text": "Will the Fed cut rates in 2027?",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 1.0,
        },
        {
            "question_id": 3,
            "question_text": "Will Apple report revenue growth in 2027?",
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


def test_total_score_uses_exact_weighted_formula():
    rows = [
        {
            "question_id": 10,
            "question_text": "Will A happen?",
            "mention_velocity": 2.0,
            "source_diversity": 4.0,
            "clarity_score": 0.8,
        },
        {
            "question_id": 11,
            "question_text": "Will B happen?",
            "mention_velocity": 1.0,
            "source_diversity": 2.0,
            "clarity_score": 0.6,
        },
    ]
    all_texts = [(10, "Will A happen?"), (11, "Will B happen?")]
    scored = score_questions(rows, all_texts)
    by_id = {s.question_id: s for s in scored}

    s10 = by_id[10]
    expected = (
        0.30 * s10.mention_velocity_score
        + 0.20 * s10.source_diversity_score
        + 0.30 * s10.clarity_score
        + 0.20 * s10.novelty_score
    )
    assert s10.total_score == expected


def test_ranking_sorts_desc_then_question_id_and_ranks_sequentially():
    rows = [
        {
            "question_id": 1,
            "question_text": "Q1 unique",
            "mention_velocity": 1.0,
            "source_diversity": 1.0,
            "clarity_score": 0.8,
        },
        {
            "question_id": 2,
            "question_text": "Q2 unique",
            "mention_velocity": 1.0,
            "source_diversity": 1.0,
            "clarity_score": 0.8,
        },
    ]
    all_texts = [(1, "Q1 unique"), (2, "Q2 unique")]
    scored = score_questions(rows, all_texts)

    # With same feature/clarity and no similar earlier text, both totals are equal.
    # Tie-break must be question_id ASC.
    assert [s.question_id for s in scored] == [1, 2]
    assert [s.rank for s in scored] == [1, 2]

