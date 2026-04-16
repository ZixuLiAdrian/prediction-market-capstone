# Prediction Market Pipeline

A Generative AI system for automated prediction market design. The pipeline ingests real-time event data from multiple sources, clusters related signals, and uses LLMs to extract structured event representations — enabling prediction market platforms to discover and design markets faster and with less ambiguity.

Built for MATH 5470: Mathematics of Generative AI (Columbia University).

## Architecture

The system is a six-stage sequential pipeline:

```
[RSS / GDELT / Markets] → Event Ingestion (FR1) → Event Clustering (FR2) → LLM Extraction (FR3)
                          → Question Generation (FR4) → Rule Validation (FR5) → Scoring (FR6) → Dashboard (FR7)
```

- **FR1-FR3** (implemented): Data processing and AI extraction
- **FR4** (implemented): LLM-based candidate question generation
- **FR5-FR7** (planned): Rule validation, scoring, and user interface

## Setup

### Prerequisites
- Python 3.10+
- PostgreSQL

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd prediction-market-pipeline

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your DB credentials and LLM API key

# Initialize database
psql -d prediction_markets -f db/schema.sql
```

### Configuration

All parameters are configured via `.env` (see `.env.example` for full list):

| Variable | Description | Default |
|---|---|---|
| `DB_HOST` | PostgreSQL host | `localhost` |
| `LLM_PROVIDER` | LLM backend (`groq` or `gemini`) | `groq` |
| `LLM_API_KEY` | API key for the LLM provider | — |
| `RSS_FEEDS` | Comma-separated RSS feed URLs | BBC, Reuters, Politico |
| `DBSCAN_EPS` | DBSCAN epsilon parameter | `0.35` |
| `CLUSTER_MIN_MENTIONS` | Minimum events per cluster | `3` |

## Running the Pipeline

```bash
# Run full pipeline (FR1 → FR2 → FR3 → FR4)
python pipeline.py

# Run specific stages
python pipeline.py --stage 1      # FR1 only
python pipeline.py --stage 1-2    # FR1 and FR2
python pipeline.py --stage 3      # FR3 only
python pipeline.py --stage 4      # FR4 only
python pipeline.py --stage 3-4    # FR3 and FR4
```

Logs are written to both stdout and `pipeline.log`.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
prediction-market-pipeline/
├── config.py                  # Centralized configuration from .env
├── models.py                  # Shared data models (Event, Cluster, ExtractedEvent)
├── pipeline.py                # Pipeline orchestrator with composable stages
├── db/
│   ├── schema.sql             # PostgreSQL schema (FR1-FR7 tables)
│   └── connection.py          # DB helpers
├── ingestion/                 # FR1: Event Ingestion
│   ├── base.py                # BaseIngestor ABC (extend for new sources)
│   ├── rss_ingest.py          # RSS feed ingestion
│   ├── gdelt_ingest.py        # GDELT API ingestion
│   └── market_ingest.py       # Polymarket ingestion
├── clustering/                # FR2: Event Clustering
│   ├── embedder.py            # Sentence-transformer embedding
│   ├── cluster.py             # DBSCAN clustering
│   └── features.py            # Cluster feature computation
├── extraction/                # FR3: LLM Event Extraction
│   ├── llm_client.py          # Reusable LLM wrapper (retry + schema validation)
│   ├── prompts.py             # Prompt templates
│   ├── schema.py              # JSON schema for extraction output
│   └── extractor.py           # Extraction orchestrator
├── generation/                # FR4: LLM Question Generation
│   ├── prompts.py             # System prompt + user prompt builder
│   ├── schema.py              # JSON schema for question generation output
│   └── generator.py           # Generation orchestrator + content safety filters
├── tests/                     # Unit tests
└── sample_outputs/            # Evidence of execution
```

## Implemented Features (FR1-FR3)

### FR1: Event Ingestion
- RSS feed ingestion from configurable feed URLs (feedparser)
- GDELT DOC API integration for structured event data
- Polymarket public API for existing market listings
- SHA256-based content deduplication
- Extensible via `BaseIngestor` abstract class

### FR2: Event Clustering
- Dense text embeddings via sentence-transformers (all-MiniLM-L6-v2)
- DBSCAN clustering with configurable parameters
- Cluster feature computation: mention velocity, source diversity, recency
- Threshold-based filtering

### FR3: LLM Event Extraction
- Structured event extraction via Groq/Gemini LLM APIs
- Enforced JSON schema output (event_summary, entities, time_horizon, resolution_hints)
- Automatic retry on malformed responses
- Reusable `LLMClient` class for downstream FR4 integration

### FR4: LLM Question Generation
- Generates 3–5 candidate prediction market questions per extracted event
- Supports both **binary** (Yes/No) and **multiple-choice** (3–5 options) question types
- Each question includes: `question_text`, `category`, `question_type`, `options`, `deadline`, `resolution_source`, `resolution_criteria`, `rationale`
- Two-layer quality enforcement:
  1. **JSON schema validation** — strict structure check on every LLM response (schema enforced by `LLMClient`)
  2. **Post-generation content filter** — word-boundary profanity check, garbled-text detection, deadline vagueness check, option count validation, question-mark enforcement
- In-context few-shot examples in system prompt for consistently high output quality
- Idempotent: re-running FR4 skips already-processed events
- `CandidateQuestion` model in `models.py` is the handoff contract for FR5/FR6

#### FR5 Integration Guide
FR5 (Rule Validation) should consume `CandidateQuestion` objects from the DB:
```python
from db.connection import get_candidate_questions
from models import CandidateQuestion

# Get all unvalidated questions
questions: list[CandidateQuestion] = get_candidate_questions()

# Each question exposes:
#   q.question_text        — the market question (always ends with ?)
#   q.question_type        — "binary" or "multiple_choice"
#   q.options              — list of answer strings (2 for binary, 3-5 for MC)
#   q.deadline             — explicit resolution deadline string
#   q.resolution_source    — named authoritative source
#   q.resolution_criteria  — precise per-option resolution rule
#   q.category             — thematic category (for scoring signals)
#   q.extracted_event_id   — FK to extracted_events → clusters (for FR6 features)
```
Insert validation results using the stub `validation_results` table in `db/schema.sql`.

## Planned Features (FR5-FR7)

Teammates should implement these by extending the pipeline:

- **FR5: Rule Validation** — Deterministic checks on `CandidateQuestion` objects (deadline presence, binary wording, resolution source, ambiguity flags)
- **FR6: Heuristic Scoring** — Weighted scoring: `S = w1·MentionVelocity + w2·SourceDiversity + w3·ClarityScore + w4·NoveltyScore`
- **FR7: Dashboard** — Streamlit UI reading from `scored_candidates` table

To add a new stage, append to `STAGES` in `pipeline.py`.

## Known Limitations

- GDELT DOC API returns headlines only (not full article text)
- Clustering quality depends on DBSCAN parameter tuning for the specific data distribution
- LLM extraction quality varies by model; Groq free tier has rate limits
- No real-time streaming; batch processing only
- Requires PostgreSQL to be running locally

## Team Responsibilities

| Member | Responsibility |
|---|---|
| Zixu Li | FR1-FR3 (Ingestion, Clustering, LLM Extraction) |
| Jack Jia | FR4 (LLM Question Generation) |
| Jia Herng Yap | FR5-FR7 (Rule Validation, Scoring, Dashboard) |
