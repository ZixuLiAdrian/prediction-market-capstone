"""
FR4: JSON schema for LLM question generation output.

The LLM must return a JSON object matching this schema, or the response is
rejected and retried. Strict validation ensures downstream FR5/FR6 always
receive well-formed candidate questions.
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
                            "politics",
                            "finance",
                            "technology",
                            "geopolitics",
                            "science",
                            "health",
                            "business",
                            "sports",
                            "energy",
                            "legal",
                            "environment",
                            "space",
                            "other",
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
                },
                "required": [
                    "question_text",
                    "category",
                    "question_type",
                    "options",
                    "deadline",
                    "deadline_source",
                    "resolution_source",
                    "resolution_criteria",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}
