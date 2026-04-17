# Prediction Market Pipeline

A Generative AI system for automated prediction market design. The pipeline ingests real-time event data from multiple sources, clusters related signals, and uses LLMs to extract structured event representations — enabling prediction market platforms to discover and design markets faster and with less ambiguity.

Built for MATH 5470: Mathematics of Generative AI (Columbia University).

## Architecture

The system is a six-stage sequential pipeline:

```
[RSS / GDELT / Markets] → Event Ingestion (FR1) → Event Clustering (FR2) → LLM Extraction (FR3)
                          → Question Generation (FR4) → Rule Validation (FR5) → Scoring (FR6) → Dashboard (FR7)
```

- **FR1-FR4** (implemented): Data ingestion, clustering, extraction, and question generation
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
python pipeline.py --stage 1-4    # Full pipeline
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
├── models.py                  # Shared data models (Event, Cluster, ExtractedEvent, CandidateQuestion)
├── pipeline.py                # Pipeline orchestrator with composable stages
├── db/
│   ├── schema.sql             # PostgreSQL schema (FR1-FR7 tables)
│   └── connection.py          # DB helpers
├── ingestion/                 # FR1: Event Ingestion
│   ├── base.py                # BaseIngestor ABC (extend for new sources)
│   ├── rss_ingest.py          # RSS feed ingestion (30 feeds: 15 news + 15 Reddit)
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
│   ├── schema.py              # JSON schema for question output (13 categories, binary/MC)
│   ├── prompts.py             # System prompt with few-shot examples and novelty guidance
│   └── generator.py           # QuestionGenerator with content safety and validation
├── tests/                     # Unit tests (51 tests covering FR1-FR4)
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
- Generates 1-3 candidate prediction market questions per extracted event
- Supports **binary** (Yes/No) and **multiple-choice** (3-5 options) question types
- 13-category classification: politics, finance, technology, geopolitics, science, health, business, sports, energy, legal, environment, space, other
- Each question includes a verifiable `resolution_source` (authoritative org + URL) and `deadline_source` (official schedule URL confirming the deadline)
- Few-shot prompting with diverse domain examples (Fed policy, FDA approvals, earnings, NATO)
- Novelty-focused: prioritizes underserved domains (health, climate, space, supply chain, geopolitics)
- Two-layer quality enforcement: JSON schema validation + post-generation content filter
- Content safety filter: blocks profanity (prefix word-boundary regex), garbled text (>30% non-ASCII), vague deadlines
- Idempotent: skips already-processed events on re-run
- Expanded FR1 data sources: 30 RSS feeds (15 major news outlets + 15 Reddit subreddits) and broadened GDELT keywords across 14 domains

## Planned Features (FR5-FR7)

Teammates should implement these by extending the pipeline:

- **FR5: Rule Validation** — Deterministic checks on `CandidateQuestion` objects
- **FR6: Heuristic Scoring** — Weighted scoring using cluster features (mention_velocity, source_diversity, clarity, novelty)
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
| Jia Herng Yap | TBD |
