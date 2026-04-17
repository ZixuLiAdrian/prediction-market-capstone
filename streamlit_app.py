"""Minimal Streamlit UI for viewing top ranked prediction markets."""

import streamlit as st

from db.connection import get_ranked_scored_questions
from scoring.scorer import (
    compute_market_interest_score,
    compute_resolution_strength_score,
    compute_time_horizon_score,
    generate_score_explanation,
    is_homepage_source,
    is_promo_event,
    is_retail_promo_event,
    is_weather_event,
    is_low_significance_event,
    top_n_ranked_display_rows,
)


def _quality_flags_for_display(question_text: str, category: str, resolution_source: str) -> list[str]:
    flags = []
    if is_homepage_source(resolution_source):
        flags.append("homepage_source")
    if is_promo_event(question_text):
        flags.append("promo_event")
    if is_retail_promo_event(question_text):
        flags.append("retail_promo_event")
    if is_low_significance_event(question_text, category):
        flags.append("low_significance_event")
    if is_weather_event(question_text):
        flags.append("weather_event")
    return flags


def main():
    st.title("Top Ranked Prediction Markets")
    top_n = st.slider("Top N", min_value=1, max_value=50, value=10, step=1)

    rows = get_ranked_scored_questions(limit=top_n)
    if not rows:
        st.info("No scored questions found yet. Run FR6 first, then refresh this page.")
        return

    enriched = []
    for row in rows:
        question_text = row["question_text"]
        category = row["category"]
        deadline = row["deadline"]
        resolution_source = row["resolution_source"]

        market_interest_score = compute_market_interest_score(question_text, category)
        resolution_strength_score = compute_resolution_strength_score(resolution_source)
        time_horizon_score = compute_time_horizon_score(deadline)
        quality_flags = _quality_flags_for_display(question_text, category, resolution_source)
        explanation = generate_score_explanation(
            market_interest_score=market_interest_score,
            resolution_strength_score=resolution_strength_score,
            time_horizon_score=time_horizon_score,
            quality_flags=quality_flags,
        )

        enriched.append(
            {
                "rank": row["rank"],
                "question_text": question_text,
                "total_score": float(row["total_score"]),
                "category": category,
                "deadline": deadline,
                "resolution_source": resolution_source,
                "explanation": explanation,
                "clarity_score": float(row["clarity_score"]),
                "mention_velocity_score": float(row["mention_velocity_score"]),
                "source_diversity_score": float(row["source_diversity_score"]),
                "novelty_score": float(row["novelty_score"]),
                "market_interest_score": market_interest_score,
                "resolution_strength_score": resolution_strength_score,
                "time_horizon_score": time_horizon_score,
                "quality_flags": quality_flags,
            }
        )

    display_rows = top_n_ranked_display_rows(enriched, top_n=top_n)
    for item in display_rows:
        header = f"#{item['rank']} | {item['question_text']}"
        with st.expander(header, expanded=False):
            st.write(f"**Total Score:** {item['total_score']:.4f}")
            st.write(f"**Category:** {item['category']}")
            st.write(f"**Deadline:** {item['deadline']}")
            st.write(f"**Source:** {item['resolution_source']}")
            st.write(f"**Explanation:** {item['explanation']}")

            with st.expander("Score Breakdown", expanded=False):
                breakdown = item["score_breakdown"]
                st.write(f"- clarity_score: {breakdown['clarity_score']:.4f}")
                st.write(f"- mention_velocity_score: {breakdown['mention_velocity_score']:.4f}")
                st.write(f"- source_diversity_score: {breakdown['source_diversity_score']:.4f}")
                st.write(f"- novelty_score: {breakdown['novelty_score']:.4f}")
                st.write(f"- market_interest_score: {breakdown['market_interest_score']:.4f}")
                st.write(f"- resolution_strength_score: {breakdown['resolution_strength_score']:.4f}")
                st.write(f"- time_horizon_score: {breakdown['time_horizon_score']:.4f}")
                st.write(f"- quality_flags: {', '.join(breakdown['quality_flags']) if breakdown['quality_flags'] else 'none'}")


if __name__ == "__main__":
    main()

