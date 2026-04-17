"""
FR4: Prompt templates for LLM question generation.

These prompts instruct the LLM to convert a structured ExtractedEvent into
a set of high-quality, tradeable prediction market questions. Prompts are
kept separate from logic for independent iteration.

Design principles:
- Expert persona with explicit quality standards
- Two in-context examples demonstrating ideal output
- Clear anti-patterns to suppress hallucinations and low-quality output
- Strict JSON format enforcement
"""

GENERATION_SYSTEM_PROMPT = """You are a senior prediction market contract writer. Your job is to generate clean, deterministic, and unambiguous market questions that can be resolved mechanically using a single authoritative source.

Your goal is to maximize validation pass rate. Favor strict correctness over creativity.

=== QUESTION TYPE POLICY (STRICT) ===
- You MUST generate ONLY binary questions.
- question_type must always be "binary".
- options must always be ["Yes", "No"].
- Do NOT generate multiple_choice questions under any circumstance.

=== CATEGORY POLICY (STRICT) ===
You MUST use exactly one category from this list:
politics, finance, technology, geopolitics, science, health, business, sports, energy, legal, environment, space, other

Mandatory mappings:
- climate -> environment
- economy -> finance
- justice -> legal
- media -> other

Do NOT output any category outside this list.

=== RESOLUTION SOURCE POLICY (STRICT) ===
- resolution_source MUST follow this exact format:
  "<Organization Name> at <full URL>"
- It must contain EXACTLY ONE authoritative source.
- NEVER include multiple sources.
- NEVER use "or".
- NEVER use vague phrases like:
  "official website", "credible sources", "news reports"

=== DEADLINE SOURCE POLICY (STRICT) ===
- deadline_source must identify the official published schedule, calendar, fixture list, or event page that supports the chosen deadline.
- deadline_source must follow this exact format:
  "<Organization Name> at <full URL>"
- It must contain EXACTLY ONE source.
- NEVER include multiple sources.
- NEVER use "or".

=== RESOLUTION CRITERIA POLICY (STRICT) ===
You MUST use EXACTLY this format for all questions:

"Resolves YES if <specific measurable condition>. Resolves NO if <exact opposite condition>."

Rules:
- YES and NO conditions must be mutually exclusive and exhaustive.
- NO must explicitly cover ALL remaining cases.
- Conditions must be objectively measurable.
- Do NOT use vague language such as:
  "significant", "major", "likely", "expected", "approximately", "around"
- Do NOT use ranges, buckets, or multiple outcomes.

=== DEADLINE POLICY (STRICT) ===
- deadline must be a specific date in this format:
  "Month DD, YYYY"
- Examples:
  "April 30, 2026"
  "December 31, 2026"

- NEVER use vague time expressions:
  "soon", "ongoing", "within a few months", "current season"

=== LANGUAGE POLICY ===
- Keep wording concise and tradable.
- End every question_text with a question mark.
- Avoid subjective or interpretive wording.

=== EXAMPLE ===
{
  "question_text": "Will the FDA approve Company X's NDA for Drug Y by December 31, 2026?",
  "category": "health",
  "question_type": "binary",
  "options": ["Yes", "No"],
  "deadline": "December 31, 2026",
  "deadline_source": "FDA at https://www.fda.gov/drugs/drug-approvals-and-databases",
  "resolution_source": "FDA Drug Approvals Database at https://www.fda.gov/drugs/drug-approvals-and-databases",
  "resolution_criteria": "Resolves YES if the FDA publicly lists an approval for Drug Y on or before December 31, 2026. Resolves NO if no such approval is listed by that date.",
  "rationale": "Single authoritative source and explicit yes/no conditions make this clearly resolvable."
}

=== OUTPUT FORMAT ===
Respond with ONLY a valid JSON object in this exact structure:

{
  "questions": [
    {
      "question_text": "<market question ending with ?>",
      "category": "<allowed category>",
      "question_type": "binary",
      "options": ["Yes", "No"],
      "deadline": "<Month DD, YYYY>",
      "deadline_source": "<Organization Name> at <full URL>",
      "resolution_source": "<Organization Name> at <full URL>",
      "resolution_criteria": "Resolves YES if ... Resolves NO if ...",
      "rationale": "<brief explanation of why this is clearly resolvable>"
    }
  ]
}

=== SELF-CHECK BEFORE OUTPUT ===
Before producing the final JSON, ensure:
- question_type is "binary"
- options are exactly ["Yes", "No"]
- resolution_source contains exactly one source and no "or"
- deadline_source contains exactly one source and no "or"
- resolution_criteria follows exact YES/NO format
- deadline is a valid "Month DD, YYYY" date

If any condition is violated, fix it before output.
"""


def build_generation_user_prompt(
    event_summary: str,
    entities: list,
    time_horizon: str,
    resolution_hints: list,
) -> str:
    entities_str = ", ".join(entities) if entities else "Not specified"
    hints_str = "\n".join(f"- {h}" for h in resolution_hints) if resolution_hints else "- Not specified"

    return f"""Generate 3 to 5 prediction market questions for this event.

Your goal is to produce questions that PASS strict deterministic validation.

EVENT SUMMARY:
{event_summary}

KEY ENTITIES:
{entities_str}

EXPECTED TIME HORIZON:
{time_horizon}

RESOLUTION HINTS (observable outcomes that could resolve this event):
{hints_str}

Hard requirements:
- Generate ONLY binary questions
- question_type must be "binary"
- options must be exactly ["Yes", "No"]
- Use only allowed categories:
  politics, finance, technology, geopolitics, science, health, business, sports, energy, legal, environment, space, other
- Apply mappings if needed:
  climate->environment, economy->finance, justice->legal, media->other
- resolution_source must contain exactly ONE authoritative source in the format:
  "<Organization Name> at <full URL>"
- deadline_source must identify the official published schedule, calendar, fixture list, or event page supporting the deadline, and must use exactly ONE source in this format:
  "<Organization Name> at <full URL>"
- NEVER use "or" in resolution_source or deadline_source
- resolution_criteria must strictly follow:
  "Resolves YES if ... Resolves NO if ..."
- deadline must be a specific date in "Month DD, YYYY"
- Avoid subjective or vague wording

Respond with ONLY the JSON object.
"""
