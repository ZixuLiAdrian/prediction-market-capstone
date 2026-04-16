"""
FR3: JSON Schema for extracted event output.

This schema enforces the structure of LLM extraction results.
The LLM must return JSON matching this schema, or the response is rejected and retried.
"""

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
        "time_horizon": {
            "type": "string",
            "description": "Expected timeframe for event resolution (e.g., '2-4 weeks', 'by June 2025', 'ongoing').",
            "minLength": 3,
        },
        "resolution_hints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Possible criteria or indicators for determining the event's outcome.",
            "minItems": 1,
        },
    },
    "required": ["event_summary", "entities", "time_horizon", "resolution_hints"],
    "additionalProperties": False,
}
