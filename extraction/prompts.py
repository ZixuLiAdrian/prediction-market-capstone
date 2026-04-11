"""
FR3: Prompt templates for LLM event extraction.

These prompts instruct the LLM to convert a cluster of related headlines
into a structured, market-ready event specification. Prompts are kept
separate from logic so they can be iterated on independently.
"""

from models import ClusterFeatures

EXTRACTION_SYSTEM_PROMPT = """You are an analyst specializing in prediction markets and event forecasting.

Your task is to analyze a cluster of related news headlines/items and produce a structured, market-ready event specification.

You MUST respond with a valid JSON object containing exactly these fields:

- "event_summary": A one-paragraph summary (2-4 sentences) describing the core event, its significance, and current status.
- "entities": A list of key people, organizations, countries, agencies, or tickers involved.
- "event_type": One of: election, legislation, court_case, macro_release, earnings, merger, policy, crypto, weather, sports, tech, geopolitics, public_health, energy, other.
- "outcome_variable": The specific thing that could change or be measured. Be concrete: "bill passage", "CPI month-over-month change", "CEO resignation", "FDA drug approval", not vague descriptions.
- "candidate_deadlines": A list of possible deadline dates or windows. Use ISO dates when possible (e.g. "2025-07-01"), or descriptive windows ("Q3 2025", "before midterm elections"). Can be empty if truly unknown.
- "resolution_sources": Specific authoritative sources that can verify the outcome. Be precise: "BLS CPI release", "Congress.gov bill status page", "SEC 8-K filing", "official company press release" — not vague like "news reports".
- "tradability": "suitable" if this event could become a clear, resolvable prediction market; "unsuitable" if not.
- "rejection_reason": If unsuitable, explain why (e.g. "already resolved", "too vague for binary market", "no measurable outcome", "purely speculative with no resolution path"). Empty string if suitable.
- "confidence": A number from 0.0 to 1.0 representing your overall confidence in this extraction.
- "market_angle": One sentence explaining WHY this event would make a good prediction market (timing, public interest, binary uncertainty, verifiability).
- "contradiction_flag": true if the headlines/items in this cluster present conflicting or contradictory information, false otherwise.
- "contradiction_details": If contradiction_flag is true, describe the conflicting signals. Empty string otherwise.
- "time_horizon": Expected timeframe for event resolution (e.g., "2-4 weeks", "by Q3 2025", "within 6 months").
- "resolution_hints": A list of concrete, observable criteria that could determine the outcome.

IMPORTANT GUIDELINES:
1. Identify the single core event from the cluster (not multiple separate events).
2. Focus on RESOLVABILITY: can this event produce a clear yes/no or measurable outcome?
3. Official sources (government, regulatory agencies, exchanges) should be preferred for resolution.
4. If items in the cluster disagree, set contradiction_flag to true and explain.
5. Be honest about tradability — not every event cluster is market-worthy.
6. Only use information present in the provided headlines. Mark inferences clearly.

Respond with ONLY the JSON object. No explanation or commentary."""


def build_extraction_user_prompt(
    headlines: list[str],
    sources: list[str] = None,
    features: ClusterFeatures = None,
) -> str:
    """
    Build the user prompt from a cluster's headlines and metadata.

    Args:
        headlines: List of headline/content strings from events in the cluster.
        sources: Optional list of source names for context.
        features: Optional ClusterFeatures to give the LLM context about the cluster.

    Returns:
        Formatted user prompt string.
    """
    headline_block = "\n".join(f"- {h}" for h in headlines[:20])  # cap at 20 to avoid token limits

    source_info = ""
    if sources:
        unique_sources = list(set(sources))
        source_info = f"\nSources: {', '.join(unique_sources)}"

    feature_info = ""
    if features:
        feature_lines = [
            f"- Mention velocity: {features.mention_velocity:.2f} events/hour",
            f"- Source diversity: {features.source_diversity} unique sources",
            f"- Recency: {features.recency:.1f} hours since latest item",
        ]
        if features.source_role_mix:
            role_str = ", ".join(f"{role}: {count}" for role, count in features.source_role_mix.items())
            feature_lines.append(f"- Source role mix: {role_str}")
        if features.coherence_score > 0:
            feature_lines.append(f"- Cluster coherence: {features.coherence_score:.3f}")
        if features.weighted_mention_velocity > 0:
            feature_lines.append(f"- Weighted velocity (official sources count more): {features.weighted_mention_velocity:.2f}")
        feature_info = "\n\nCluster Metadata:\n" + "\n".join(feature_lines)

    return f"""Analyze the following cluster of {len(headlines)} related headlines and extract a structured, market-ready event specification.

Headlines:
{headline_block}
{source_info}{feature_info}

Respond with a JSON object containing: event_summary, entities, event_type, outcome_variable, candidate_deadlines, resolution_sources, tradability, rejection_reason, confidence, market_angle, contradiction_flag, contradiction_details, time_horizon, resolution_hints."""
