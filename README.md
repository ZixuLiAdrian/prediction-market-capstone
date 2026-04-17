# Prediction Market Question Generation & Ranking Pipeline

Batch pipeline that converts real-world events into candidate prediction-market questions, validates them with deterministic rules, ranks them using heuristic scoring, and persists results in PostgreSQL.

Stages FR1–FR6 are executed in `pipeline.py`. A Streamlit UI reads ranked outputs from the database (read-only; not part of the pipeline).

## Architecture

```
[RSS | GDELT | Polymarket] → FR1 ingest → FR2 cluster → FR3 extract → FR4 generate
  → FR5 validate → FR6 score & rank → PostgreSQL → Streamlit (read-only UI)
```
All stages write intermediate outputs to PostgreSQL; downstream stages read from the database rather than passing data in-memory.
- **Pipeline (FR1–FR6):** writes intermediate and final rows through `db/connection.py` into the schema in `db/schema.sql`.
- **Database:** source of truth between runs.
- **Streamlit:** optional viewer on top of stored scores; does not execute FR1–FR6.

## Functional stages

| Stage | Role |
|--------|------|
| **FR1** | Ingest events from RSS, GDELT, and Polymarket; dedupe by content hash; store `Event` rows. |
| **FR2** | Embed event text, cluster related events and compute features (velocity, source diversity, recency), and persist `Cluster` rows. |
| **FR3** | Call the configured LLM to extract structured fields per cluster; validate JSON against the extraction schema; store `ExtractedEvent` rows. |
| **FR4** | Call the LLM to generate candidate questions from extracted events; schema-validated output; store `CandidateQuestion` rows. |
| **FR5** | Run deterministic checks on each candidate; attach flags, validity, and a clarity score; persist `ValidationResult`. |
| **FR6** | Score and rank validated questions with deterministic heuristics; persist ranked rows in `scored_candidates` and log component breakdowns. |

## Implemented features (FR1–FR4)

### FR1: Event ingestion

- RSS feed ingestion via feedparser
- GDELT DOC API integration
- Polymarket API ingestion
- SHA256-based deduplication
- Extensible `BaseIngestor` abstraction

### FR2: Event clustering

- Text embeddings via sentence-transformers
- DBSCAN clustering to group related events
- Feature computation: mention velocity, source diversity, recency

### FR3: LLM event extraction

- Structured extraction via Groq or Gemini APIs (configured in `.env`)
- JSON schema enforcement on model output
- Retry handling for malformed responses
- Reusable `LLMClient` abstraction

### FR4: LLM question generation

- Generates candidate prediction-market questions per extracted event
- Supports binary and multiple-choice formats
- Category classification across predefined domains
- Each question includes verifiable resolution and deadline sources for downstream validation
- JSON schema validation with post-generation content filtering
- Idempotent processing (skips events already processed for generation)

### FR5: Deterministic validation
- Enforces strict validation rules on generated questions before downstream use
- Validates presence and format of required fields (question, type, options, category)
- Ensures resolution criteria are clear, objective, and time-bounded
- Verifies resolution_source and deadline_source are provided and well-formed URLs
- Filters out ambiguous, vague, or non-verifiable questions
- Deduplicates similar or overlapping questions across events
- Rule-based validation ensures deterministic and reproducible outputs

### FR6: Scoring and ranking
- Assigns quality scores to validated questions using heuristic-based metrics
- Scoring factors include clarity, specificity, verifiability, and timeliness
- Rewards questions with strong resolution criteria and reliable sources
- Penalizes vague wording, weak sources, or missing fields
- Ranks questions within each event to prioritize high-quality candidates
- Outputs ranked question sets for downstream display or storage
- Deterministic scoring ensures consistent results for identical inputs

## Streamlit UI

`streamlit_app.py` loads top-N **ranked scored questions** from PostgreSQL, shows question text, category, deadline, source, total score, a **plain-language explanation**, and an expandable **numeric score breakdown** (clarity, cluster-style components, market interest, resolution strength, time horizon, quality flags). The Streamlit UI is included in `requirements.txt`; launch it after FR6 has written scores.
This provides a quick inspection layer for evaluating ranking quality without rerunning the pipeline.

## Setup

**Prerequisites:** Python 3.10+, PostgreSQL.

```bash
git clone <repo-url>
cd prediction-market-capstone

pip install -r requirements.txt

cp .env.example .env
# Then edit .env with your real local DB credentials and LLM_API_KEY (see below).

# Create the database if it does not exist yet (name must match DB_NAME in .env).
createdb prediction_markets

psql -d prediction_markets -f db/schema.sql
```

**Secrets and local configuration**

- Copy the template: `cp .env.example .env`
- Edit `.env` and set your **local** PostgreSQL settings (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) and your **LLM provider API key** (`LLM_API_KEY`, plus `LLM_PROVIDER` / `LLM_MODEL` if you change them). Values in `.env.example` are placeholders for documentation only.
- **`.env` is local-only and must not be committed** (it is gitignored). Only `.env.example` should live in the repository as the shared template.

## Configuration

All runtime parameters are configured via `.env` (see `.env.example` for the full list). Below are the key parameters used to control system behavior.

### Database

| Variable | Description | Example |
| --- | --- | --- |
| `DB_HOST` | PostgreSQL host (typically `localhost` for local runs) | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | Database name used by the pipeline | `prediction_markets` |
| `DB_USER` | Database username | `postgres` |
| `DB_PASSWORD` | Database password | `(set locally)` |

### LLM

| Variable | Description | Example |
| --- | --- | --- |
| `LLM_PROVIDER` | LLM backend for FR3–FR4 (`groq` or `gemini`) | `groq` |
| `LLM_MODEL` | Model name for the selected provider | `llama-3.1-8b-instant` |
| `LLM_API_KEY` | API key for the LLM provider | `(your key)` |

### Ingestion and clustering

| Variable | Description | Example |
| --- | --- | --- |
| `RSS_FEEDS` | Comma-separated RSS feed URLs for FR1 (optional; if unset, defaults are loaded from `config.py`) | Comma-separated URLs |
| `DBSCAN_EPS` | DBSCAN neighborhood radius; higher values yield looser clusters | `0.35` |
| `CLUSTER_MIN_MENTIONS` | Minimum events required for a cluster to be kept after feature computation | `3` |

### Notes

- `.env` is required for the pipeline to run; missing or wrong values will cause connection or API failures.
- `.env` must not be committed (it is gitignored).
- Tune ingestion and clustering parameters to trade off output quality, diversity, and runtime.

Pipeline stage wiring is defined in `pipeline.py` (`STAGES`).

## How to run

**Tests**

```bash
python -m pytest tests/ -q
```

**Pipeline**

```bash
# FR1 → FR6 (all stages)
python pipeline.py

# Selected stages (examples)
python pipeline.py --stage 4    # FR4 only
python pipeline.py --stage 5    # FR5 only
python pipeline.py --stage 6    # FR6 only
python pipeline.py --stage 1-2  # inclusive range
```

Logs go to stdout and `pipeline.log`.

**Streamlit (after FR6 has written scores)**

```bash
python -m streamlit run streamlit_app.py
```

## Configurable event cap (`MAX_EVENTS`)

In `pipeline.py`, **`MAX_EVENTS`** limits how many **extracted events** are passed into **FR4 (LLM question generation)**:

```python
events[:MAX_EVENTS]
```

That slice is applied **only when** `MAX_EVENTS` is not `None` (after extracted events are loaded from PostgreSQL). If `MAX_EVENTS` is `None`, every pending extracted event is eligible for FR4. Use a **small integer** (e.g. `3`) for faster smoke tests or demos; use **`None`** when you want the full backlog processed. Update the constant in `pipeline.py` before running.

## Project structure

```
prediction-market-capstone/
├── config.py                  # Configuration from .env
├── models.py                  # Core data models
├── pipeline.py                # Pipeline orchestration (FR1–FR6)
├── streamlit_app.py           # UI layer (reads from DB)
├── db/
│   ├── schema.sql             # PostgreSQL schema
│   └── connection.py          # DB helpers
├── ingestion/                 # FR1
├── clustering/                # FR2
├── extraction/                # FR3 (LLM)
├── generation/                # FR4 (LLM)
├── validation/                # FR5
├── scoring/                   # FR6
├── tests/                     # Unit tests covering FR1–FR6
└── sample_outputs/            # Example outputs
```

## Known limitations

- GDELT DOC responses are headline-oriented; body text is not ingested from that API.
- Cluster quality depends on embedding model and DBSCAN settings for the current corpus.
- LLM stages (FR3–FR4) depend on provider rate limits and model behavior.
- Processing is batch-oriented; no real-time ingestion or incremental updates.
- PostgreSQL must be available for pipeline and UI reads.

## Extending the pipeline

Append a `(name, callable)` entry to **`STAGES`** in `pipeline.py`, add any new tables in `db/schema.sql`, and wire persistence in `db/connection.py` as needed.

## Team responsibilities

| Member | Responsibility |
| --- | --- |
| Zixu Li | FR1–FR3 (Ingestion, Clustering, LLM Extraction) |
| Jack Jia | FR4 (LLM Question Generation) |
| Jia Herng Yap | FR5–FR6 (Validation, Scoring and Ranking) |