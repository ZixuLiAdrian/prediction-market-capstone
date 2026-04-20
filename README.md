# Prediction Market Pipeline

A Generative AI system for automated prediction market design. The pipeline ingests real-time event data from multiple sources, clusters related signals, and uses LLMs to extract structured, market-ready event specifications — enabling prediction market platforms to discover and design markets faster and with less ambiguity.

Built for MATH 5470: Mathematics of Generative AI (Columbia University).

## Architecture

The system is a six-stage sequential pipeline:

```
[13 Data Sources] → Event Ingestion (FR1) → Event Clustering (FR2) → LLM Extraction (FR3)
                     → Question Generation (FR4) → Rule Validation (FR5) → Scoring (FR6) → Dashboard (FR7)
```

- **FR1-FR4** (implemented): Data ingestion, clustering, extraction, and question generation
- **FR5-FR7** (planned): Rule validation, scoring, and user interface

### Source Architecture

Events are classified by **signal role** — how they contribute to market generation:

| Role | Purpose | Sources |
|---|---|---|
| **discovery** | Find emerging events | RSS feeds, GDELT |
| **attention** | Detect public interest spikes | Reddit, Hacker News, Wikipedia Pageviews |
| **resolution** | Anchor markets to verifiable outcomes | Federal Register, Congress.gov, SEC EDGAR, BLS, FRED, EIA |
| **benchmark** | Compare against existing markets | Polymarket, Kalshi |

Official sources are weighted higher than social posts in cluster scoring, so an SEC filing counts more than a Reddit post.

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
# Edit .env with your DB credentials, LLM API key, and optional source API keys

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
| `REDDIT_CLIENT_ID` | Reddit OAuth client ID | — |
| `CONGRESS_API_KEY` | Congress.gov API key | — |
| `FRED_API_KEY` | FRED API key | — |
| `DBSCAN_EPS` | DBSCAN epsilon parameter | `0.35` |
| `CLUSTER_MIN_MENTIONS` | Minimum events per cluster | `3` |
| `NEAR_DUPLICATE_THRESHOLD` | Embedding similarity threshold for near-duplicate detection | `0.92` |

Most official-source APIs are free. See `.env.example` for registration links.

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
├── config.py                          # Centralized configuration from .env
├── models.py                          # Shared data models (Event, Cluster, ExtractedEvent, CandidateQuestion)
├── pipeline.py                        # Pipeline orchestrator with composable stages
├── db/
│   ├── schema.sql                     # PostgreSQL schema (FR1-FR7 tables)
│   └── connection.py                  # DB helpers
├── ingestion/                         # FR1: Event Ingestion (13 sources)
│   ├── base.py                        # BaseIngestor ABC
│   ├── rss_ingest.py                  # RSS feed ingestion
│   ├── gdelt_ingest.py                # GDELT API ingestion
│   ├── reddit_ingest.py               # Reddit API ingestion
│   ├── hn_ingest.py                   # Hacker News API ingestion
│   ├── wikipedia_ingest.py            # Wikipedia Pageviews ingestion
│   ├── federal_register_ingest.py     # Federal Register API ingestion
│   ├── congress_ingest.py             # Congress.gov API ingestion
│   ├── sec_ingest.py                  # SEC EDGAR ingestion
│   ├── bls_ingest.py                  # BLS API ingestion
│   ├── fred_ingest.py                 # FRED API ingestion
│   ├── eia_ingest.py                  # EIA API ingestion
│   ├── market_ingest.py               # Polymarket ingestion
│   └── kalshi_ingest.py               # Kalshi ingestion
├── clustering/                        # FR2: Event Clustering
│   ├── embedder.py                    # Sentence-transformer embedding
│   ├── cluster.py                     # DBSCAN clustering
│   └── features.py                    # Cluster features, near-dedup, source weights
├── extraction/                        # FR3: LLM Event Extraction
│   ├── llm_client.py                  # Reusable LLM wrapper (smart retry + schema validation)
│   ├── prompts.py                     # Prompt templates (with FR2 metadata injection)
│   ├── schema.py                      # JSON schema for market-ready event specs
│   └── extractor.py                   # Extraction orchestrator
├── generation/                        # FR4: LLM Question Generation
│   ├── schema.py                      # JSON schema for question output (13 categories, binary/MC)
│   ├── prompts.py                     # System prompt with few-shot examples and novelty guidance
│   └── generator.py                   # QuestionGenerator with content safety and validation
├── validation/                        # FR5: Rule Validation
│   └── validator.py                   # Deterministic checks (prohibited topics, PII, deadlines)
├── scoring/                           # FR6: Heuristic Scoring
│   └── scorer.py                      # 7-component weighted scoring with regulatory penalties
├── tests/                             # Unit tests (174 tests covering FR1-FR6)
└── sample_outputs/                    # Evidence of execution
```

## Implemented Features (FR1-FR3)

### FR1: Event Ingestion
- **13 data sources** across 4 categories:
  - **News/Discovery**: RSS feeds (feedparser), GDELT DOC API
  - **Social/Attention**: Reddit API, Hacker News API, Wikipedia Pageviews
  - **Official/Resolution**: Federal Register, Congress.gov, SEC EDGAR, BLS, FRED, EIA
  - **Market/Benchmark**: Polymarket, Kalshi
- Each event tagged with `signal_role` (discovery, attention, resolution, benchmark)
- SHA256-based content deduplication
- Extensible via `BaseIngestor` abstract class
- Graceful failure: each ingestor fails independently without stopping the pipeline

### FR2: Event Clustering
- Dense text embeddings via sentence-transformers (all-MiniLM-L6-v2)
- **Near-duplicate detection** before clustering (embedding similarity threshold)
- DBSCAN clustering with configurable parameters
- Enhanced cluster features:
  - **Mention velocity** — events per hour
  - **Weighted mention velocity** — source-weight-adjusted (official > social)
  - **Source diversity** — number of distinct sources
  - **Source role mix** — breakdown by signal role (discovery/attention/resolution/benchmark)
  - **Coherence score** — average pairwise embedding similarity within cluster
  - **Recency** — hours since most recent event
- **Per-source weights**: official sources (Federal Register, SEC, BLS) count 2.5-3x more than social posts
- Threshold-based filtering

### FR3: LLM Event Extraction
- Produces **market-ready event specifications**, not just summaries
- Expanded output schema:
  - `event_type`: categorization (election, legislation, macro_release, etc.)
  - `outcome_variable`: what can be measured/predicted
  - `candidate_deadlines`: possible resolution dates
  - `resolution_sources`: specific authoritative sources for verification
  - `tradability`: suitable/unsuitable assessment with rejection reason
  - `confidence`: 0.0-1.0 extraction confidence score
  - `market_angle`: why this event could become a prediction market
  - `contradiction_flag`: whether cluster sources disagree
- FR2 cluster metadata (velocity, diversity, role mix, coherence) injected into prompts
- **Smart retry**: validation errors sent back to LLM for repair, not blind retry
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

- **FR4: Question Generation** — LLM-based generation of candidate market questions from `ExtractedEvent`, with content safety checks and quality gates
- **FR5: Rule Validation** — Deterministic checks on `CandidateQuestion` objects (prohibited topics, PII, manipulation risk, deadline validity)
- **FR6: Heuristic Scoring** — 7-component weighted scoring (market_interest, resolution_strength, clarity, mention_velocity, novelty, time_horizon, source_diversity) with regulatory hard/soft exclusions
- **FR7: Dashboard** — Streamlit UI reading from `scored_candidates` table

To add a new stage, append to `STAGES` in `pipeline.py`.

## Known Limitations

- GDELT DOC API returns headlines only (not full article text)
- Some APIs require free registration (Congress.gov, FRED, EIA) — pipeline skips unconfigured sources
- Reddit requires OAuth app credentials
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
