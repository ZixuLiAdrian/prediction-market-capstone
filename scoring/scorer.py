"""
FR6: Deterministic heuristic scoring for validated questions.
"""

from __future__ import annotations

import string

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


def score_questions(rows: list[dict], all_question_texts_by_id: list[tuple[int, str]]) -> list[ScoredCandidate]:
    if not rows:
        return []

    velocities = [float(r["mention_velocity"]) for r in rows]
    diversities = [float(r["source_diversity"]) for r in rows]

    vel_min, vel_max = min(velocities), max(velocities)
    div_min, div_max = min(diversities), max(diversities)

    scored: list[ScoredCandidate] = []
    text_by_id = {qid: text for qid, text in all_question_texts_by_id}

    for row in rows:
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

        total_score = (
            0.30 * mention_velocity_score
            + 0.20 * source_diversity_score
            + 0.30 * clarity_score
            + 0.20 * novelty_score
        )

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

