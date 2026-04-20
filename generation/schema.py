"""
FR4: JSON schema for LLM question generation output.

The LLM must return a JSON object matching this schema, or the response is
rejected and retried. Strict validation ensures downstream FR5/FR6 always
receive well-formed candidate questions.

Each question includes four quality assessment fields that the LLM self-evaluates:
  - resolution_confidence: how cleanly the outcome can be confirmed
  - resolution_confidence_reason: one-sentence explanation
  - source_independence: how neutral/unbiased the resolution source is
  - timing_reliability: how reliably the source will publish by the deadline
  - already_resolved: whether the event has already concluded (auto-rejects)
"""

CANDIDATE_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "description": "List of candidate prediction market questions generated from the event.",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "question_text": {
                        "type": "string",
                        "description": "The market question, professionally worded, ending with a question mark.",
                        "minLength": 20,
                        "maxLength": 500,
                    },
                    "category": {
                        "type": "string",
                        "description": "Thematic category of the question.",
                        "enum": [
                            "politics", "finance", "technology", "geopolitics",
                            "science", "health", "business", "sports",
                            "energy", "legal", "environment", "space", "other",
                        ],
                    },
                    "question_type": {
                        "type": "string",
                        "description": "Whether the question has a binary (Yes/No) or multiple-choice outcome.",
                        "enum": ["binary", "multiple_choice"],
                    },
                    "options": {
                        "type": "array",
                        "description": "Mutually exclusive, collectively exhaustive answer options.",
                        "minItems": 2,
                        "maxItems": 5,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 200,
                        },
                    },
                    "deadline": {
                        "type": "string",
                        "description": "Specific resolution deadline tied to a real scheduled event, e.g. 'December 10, 2025'.",
                        "minLength": 5,
                        "maxLength": 100,
                    },
                    "deadline_source": {
                        "type": "string",
                        "description": "Official published schedule or calendar that establishes the deadline date, with URL.",
                        "minLength": 10,
                        "maxLength": 500,
                    },
                    "resolution_source": {
                        "type": "string",
                        "description": "Specific, authoritative, publicly accessible source for resolution.",
                        "minLength": 10,
                        "maxLength": 300,
                    },
                    "resolution_criteria": {
                        "type": "string",
                        "description": "Precise, unambiguous rule determining which option wins.",
                        "minLength": 20,
                        "maxLength": 1000,
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief explanation of why this is a good prediction market question.",
                        "minLength": 10,
                        "maxLength": 500,
                    },
                    # ---- Quality assessment fields ----
                    "resolution_confidence": {
                        "type": "number",
                        "description": (
                            "How cleanly can the outcome be confirmed from the resolution_source? "
                            "1.0 = definitive unambiguous result published by independent body. "
                            "0.7 = authoritative but may need minor interpretation. "
                            "0.4 = delayed, inaccessible, or potentially contested. "
                            "0.0 = controlled by an interested party or likely suppressed."
                        ),
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "resolution_confidence_reason": {
                        "type": "string",
                        "description": "One sentence explaining the resolution_confidence score.",
                        "minLength": 10,
                        "maxLength": 300,
                    },
                    "source_independence": {
                        "type": "number",
                        "description": (
                            "Is the resolution source independent of the parties in the question? "
                            "1.0 = fully independent (government stats agency, exchange, treaty org). "
                            "0.7 = mostly independent (major newswire with editorial standards). "
                            "0.3 = party reports its own outcome (company earnings — usable but noted). "
                            "0.0 = subject of the question is the sole source with direct interest."
                        ),
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "timing_reliability": {
                        "type": "number",
                        "description": (
                            "How reliably will the resolution source publish a result by the deadline? "
                            "1.0 = fixed calendar date (FOMC meeting, scheduled earnings, election day). "
                            "0.7 = usually on time, minor delay risk. "
                            "0.3 = process can be delayed (legislation, regulatory review). "
                            "0.0 = no scheduled timeline, outcome may never formally occur."
                        ),
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "already_resolved": {
                        "type": "boolean",
                        "description": (
                            "True if this event has already concluded as of today — "
                            "deadline passed or outcome is already publicly known. "
                            "Set true to auto-reject this question."
                        ),
                    },
                },
                "required": [
                    "question_text", "category", "question_type", "options",
                    "deadline", "deadline_source", "resolution_source",
                    "resolution_criteria", "rationale",
                    "resolution_confidence", "resolution_confidence_reason",
                    "source_independence", "timing_reliability", "already_resolved",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}
