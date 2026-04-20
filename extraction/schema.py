"""
FR3: JSON Schema for extracted event output.

This schema enforces the structure of LLM extraction results.
The LLM must return JSON matching this schema, or the response is rejected and retried.

ExtractedEvent is the handoff contract for FR4 — it tells downstream modules
not just what happened, but whether it's market-worthy and how to resolve it.
"""

EVENT_TYPE_ENUM = [
    "election", "legislation", "court_case", "macro_release", "earnings",
    "merger", "policy", "crypto", "weather", "sports", "tech",
    "geopolitics", "public_health", "energy", "other",
]

EXTRACTED_EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_summary": {
            "type": "string",
            "description": "A one-paragraph summary describing the event, its significance, and current status.",
            "minLength": 20,
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key people, organizations, or countries involved in the event.",
            "minItems": 1,
        },
        "event_type": {
            "type": "string",
            "description": "Category of event.",
            "enum": EVENT_TYPE_ENUM,
        },
        "outcome_variable": {
            "type": "string",
            "description": "The specific thing that could change or be measured. E.g. 'bill passage', 'CPI value', 'CEO resignation', 'FDA approval'.",
            "minLength": 5,
        },
        "candidate_deadlines": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Possible deadline dates or windows. E.g. ['2025-07-01', 'Q3 2025', 'before midterm elections'].",
            "minItems": 0,
        },
        "resolution_sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific authoritative sources that can verify the outcome. E.g. ['BLS CPI release', 'Congress.gov bill status', 'SEC 8-K filing'].",
            "minItems": 1,
        },
        "tradability": {
            "type": "string",
            "description": "Whether this event is suitable for a prediction market.",
            "enum": ["suitable", "unsuitable"],
        },
        "rejection_reason": {
            "type": "string",
            "description": "If unsuitable, explain why. E.g. 'already resolved', 'too vague', 'no measurable outcome'. Empty string if suitable.",
        },
        "confidence": {
            "type": "number",
            "description": "Overall confidence in this extraction (0.0 to 1.0).",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "market_angle": {
            "type": "string",
            "description": "One sentence explaining why this event could become a prediction market. E.g. 'Binary outcome with high public interest and a clear deadline.'",
        },
        "contradiction_flag": {
            "type": "boolean",
            "description": "True if sources in the cluster present conflicting information.",
        },
        "contradiction_details": {
            "type": "string",
            "description": "Description of conflicting signals if contradiction_flag is true. Empty string otherwise.",
        },
        "time_horizon": {
            "type": "string",
            "description": "Expected timeframe for event resolution (e.g., '2-4 weeks', 'by June 2025', 'ongoing').",
            "minLength": 3,
        },
        "resolution_hints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "General criteria for determining the event's outcome (kept for backward compatibility).",
            "minItems": 1,
        },
    },
    "required": [
        "event_summary", "entities", "event_type", "outcome_variable",
        "resolution_sources", "tradability", "rejection_reason", "confidence",
        "market_angle", "contradiction_flag", "contradiction_details",
        "time_horizon", "resolution_hints",
    ],
    "additionalProperties": False,
}
