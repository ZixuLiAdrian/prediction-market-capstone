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
- **FR4-FR7** (planned): Question generation, validation, scoring, and user interface

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
# Run full pipeline (FR1 → FR2 → FR3)
python pipeline.py

# Run specific stages
python pipeline.py --stage 1      # FR1 only
python pipeline.py --stage 1-2    # FR1 and FR2
python pipeline.py --stage 3      # FR3 only
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

## Planned Features (FR4-FR7)

Teammates should implement these by extending the pipeline:

- **FR4: Question Generation** — Use `LLMClient` to generate candidate market questions from `ExtractedEvent`
- **FR5: Rule Validation** — Deterministic checks on candidate questions
- **FR6: Heuristic Scoring** — Weighted scoring using cluster features
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
| Jack Jia | TBD |
| Jia Herng Yap | TBD |
