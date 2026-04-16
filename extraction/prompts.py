"""
FR3: Prompt templates for LLM event extraction.

These prompts instruct the LLM to convert a cluster of related headlines
into a structured event representation. Prompts are kept separate from
logic so they can be iterated on independently.
"""

EXTRACTION_SYSTEM_PROMPT = """You are an analyst specializing in prediction markets and event forecasting.

Your task is to analyze a cluster of related news headlines and produce a structured event summary.

You MUST respond with a valid JSON object containing exactly these fields:
- "event_summary": A one-paragraph summary (2-4 sentences) describing the core event, its significance, and current status.
- "entities": A list of key people, organizations, or countries involved.
- "time_horizon": The expected timeframe for this event to resolve (e.g., "2-4 weeks", "by Q3 2025", "within 6 months").
- "resolution_hints": A list of concrete, observable criteria that could determine the outcome (e.g., "official announcement", "vote result", "quarterly earnings report").

Focus on:
- Identifying the single core event from the cluster (not multiple separate events)
- Being specific about entities and timelines
- Providing resolution hints that are measurable and verifiable

Respond with ONLY the JSON object. No explanation or commentary."""


def build_extraction_user_prompt(headlines: list[str], sources: list[str] = None) -> str:
    """
    Build the user prompt from a cluster's headlines.

    Args:
        headlines: List of headline/content strings from events in the cluster.
        sources: Optional list of source names for context.

    Returns:
        Formatted user prompt string.
    """
    headline_block = "\n".join(f"- {h}" for h in headlines[:20])  # cap at 20 to avoid token limits

    source_info = ""
    if sources:
        unique_sources = list(set(sources))
        source_info = f"\nSources: {', '.join(unique_sources)}"

    return f"""Analyze the following cluster of {len(headlines)} related headlines and extract a structured event representation.

Headlines:
{headline_block}
{source_info}

Respond with a JSON object containing: event_summary, entities, time_horizon, resolution_hints."""
